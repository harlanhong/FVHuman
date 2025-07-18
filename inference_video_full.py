import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path
from collections import OrderedDict
import copy
import json
import glob
import wandb
from datetime import datetime
import diffusers
import numpy as np
import decord
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision.utils import save_image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from einops import rearrange
from decord import VideoReader
import pdb
from datasets.video_anyone import VideoDataset
from pipelines.pipeline_pose2vid_ms_attn_add_cross_pose_envproj_spatial_selection import Pose2VideoPipeline
from models.model_anyone import AnyoneModel_MutilRef_attn_add_CrossPose_envproj_spatial_selection as AnyoneModel
from models.pose_guider import SpatialAttn3 as SpatialAttn
from models.pose_guider import CrossPoseGuider

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d_multi_pose_guided_attn_add_crosspose_spatial_selection import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControlv4 as ReferenceAttentionControl
from models.bright_proj import EnvProjModel
from utils.util import seed_everything, delete_additional_ckpt, compute_snr
from utils.video_utils import save_videos_grid, save_videos_from_pil, concat_pil, resize_tensor_frames
from utils.video_level_evaluation import video_level_evaluation
from utils.pkg import instantiate_from_config
from utils.util import calculate_brightness,calculate_contrast
from datasets.data_utils import calculate_global_bbox_with_margin
from decord import VideoReader
from dwpose_tools.extract_dwpose_from_vid import online_process_image,online_process_video
warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")
def sinusoidal_encode(value, d_model):
    """
    使用正弦和余弦函数将一个标量值编码成多维向量
    :param value: 需要编码的标量值
    :param d_model: 编码的维度
    :return: 编码后的多维向量
    """
    # 创建一个位置索引数组
    position = np.arange(d_model // 2)
    
    # 计算编码
    angles = value / (10000 ** (position / (d_model // 2)))
    
    # 计算正弦和余弦
    encoded_vector = np.zeros(d_model)
    encoded_vector[0::2] = np.sin(angles)
    encoded_vector[1::2] = np.cos(angles)
    
    return encoded_vector

@torch.no_grad()
def log_validation(
    cfg,
    vae,
    image_enc,
    model,
    scheduler,
    device,
    width,
    height,
    seed=42,
    dtype=torch.float32,
    save_dir=None,
):
    # logger.info("Running validation ...")
    unwrap_model = model
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    pose_guider = unwrap_model.pose_guider
    attn_guider = unwrap_model.attn_guider
    env_proj = unwrap_model.env_proj

    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    pipeline = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        attn_guider = attn_guider,
        env_proj = env_proj,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)

    ref_img_path_list = cfg.ref_imgs_path
    tgt_vid_path = cfg.target_image_path[0]
   
    name = os.path.basename(tgt_vid_path).split('.')[0]
    # if os.path.exists(os.path.join(args.save_dir,f"{name}.mp4")):
    #     continue
    ref_image_pil_list = []
    ref_pose_pil_list = []
    for img_path in ref_img_path_list:
        if img_path.endswith('.jpg'):
            ref_img_pil = Image.open(img_path)
            ref_pose_pil = online_process_image(ref_img_pil)
        else:
            ref_img_pil = Image.fromarray(decord.VideoReader(img_path)[0].asnumpy())
            ref_pose_pil = online_process_image(ref_img_pil)
        ref_image_pil_list.append(ref_img_pil)
        ref_pose_pil_list.append(ref_pose_pil)
    ref_image_h,ref_image_w =  1278,720
    
    tgt_video = VideoReader(tgt_vid_path)
    tgt_pose_list = online_process_video(tgt_vid_path)
    
    tgt_image_list = [Image.fromarray(img.asnumpy()) for img in tgt_video]
    tgt_pose_list = tgt_pose_list[:300]
    tgt_image_list = tgt_image_list[:300]
    brightness_list = [calculate_brightness(img) for img in tgt_image_list]
    contrast_list = [calculate_contrast(img) for img in tgt_image_list]
    brightness = np.mean(brightness_list)
    contrast = np.mean(contrast_list)
    brightness = sinusoidal_encode(brightness,128)
    contrast = sinusoidal_encode(contrast,128)
    env_code = torch.tensor(np.concatenate((brightness,contrast),0)).view(1,-1)
    w, h = 512,512
    generator = torch.manual_seed(seed)
    ref_pose_pil_list = [img.resize((w, h),Image.Resampling.BILINEAR) for img in ref_pose_pil_list]
    ref_image_pil_list = [img.resize((w, h),Image.Resampling.BILINEAR) for img in ref_image_pil_list]
    tgt_pose_list = [img.resize((w, h),Image.Resampling.BILINEAR) for img in tgt_pose_list]
    
    video_tensor = pipeline(
        ref_image_pil_list,
        tgt_pose_list,
        ref_pose_pil_list,
        env_code,
        w,
        h,
        len(tgt_pose_list),
        cfg.validation.denoising_steps,
        cfg.validation.guidance_scale,
        AttnControl = ReferenceAttentionControl,
        generator=generator,
    ).videos
                
    video_tensor = resize_tensor_frames(
        video_tensor, (ref_image_h, ref_image_w)
    )
    ref_pose_pil_list = [img.resize((ref_image_w, ref_image_h),Image.Resampling.BILINEAR) for img in ref_pose_pil_list]
    tgt_image_list = [img.resize((ref_image_w, ref_image_h),Image.Resampling.BILINEAR) for img in tgt_image_list]
    ref_image_pil_list = [img.resize((ref_image_w, ref_image_h),Image.Resampling.BILINEAR) for img in ref_image_pil_list]
    tgt_pose_list = [img.resize((ref_image_w, ref_image_h),Image.Resampling.BILINEAR) for img in tgt_pose_list]
    video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()
    video_pil_lst = []
    for frame_idx, image_tensor in enumerate(video_tensor):
        result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
        
        result_pil_lst = ref_image_pil_list+ [tgt_pose_list[frame_idx], result_img_pil, tgt_image_list[frame_idx]]
        concated_pil = concat_pil(result_pil_lst)
        video_pil_lst.append(concated_pil)
    save_videos_from_pil(video_pil_lst,args.save_name, fps=25)
    print(f"save to {args.save_name}")
       
    del pipeline
    torch.cuda.empty_cache()

def main(cfg):

    if cfg.seed is not None:
        seed_everything(cfg.seed)
        
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )
        
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs
        ),
    ).to(device="cuda")
    
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")
    envproj = EnvProjModel(
        input_dim=256,
        latent_dim=768,
    ).to(device="cuda")

    # guidance_encoder_group = setup_guidance_encoder(cfg)
    if cfg.pose_guider_pretrain:
        pose_guider = CrossPoseGuider(
            conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256), 
            conditioning_channels=3,
        ).to(device="cuda")
        # load pretrained controlnet-openpose params for pose_guider
        controlnet_openpose_state_dict = torch.load(cfg.controlnet_openpose_path)
        state_dict_to_load = {}
        for k in controlnet_openpose_state_dict.keys():
            if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                new_k = k.replace("controlnet_cond_embedding.", "")
                # skip first conv layer
                # if new_k != "conv_in.weight":
                state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
        miss, _ = pose_guider.load_state_dict(state_dict_to_load, strict=False)
        logger.info(f"Missing key for pose guider: {len(miss)}")
    else:
        pose_guider = CrossPoseGuider(
            conditioning_embedding_channels=320,
        ).to(device="cuda")
    attn_guider = SpatialAttn(
            conditioning_embedding_channels=320,
        ).to(device="cuda")
    
   
   
    
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )   
    
    model = AnyoneModel(
        reference_unet,
        denoising_unet,
        pose_guider,
        attn_guider,
        envproj,
        reference_control_writer,
        reference_control_reader,
    )
    
    
    checkpoint_path = args.checkpoint_path
    checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_state_dict)
    device = torch.device("cuda")
    sample_dicts = log_validation(
        cfg=cfg,
        vae=vae,
        image_enc=image_enc,
        model=model,
        scheduler=val_noise_scheduler,
        device=device,
        width=cfg.data.params.image_size,
        height=cfg.data.params.image_size,
        seed=cfg.seed,
        save_dir=args.save_name,
    )
   

if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    os.environ["WANDB_API_KEY"] = config.wandb_key
    main(config)   
        