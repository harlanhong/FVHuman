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

import wandb
from datetime import datetime
import diffusers
import numpy as np
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
from utils.video_utils import save_videos_grid, save_videos_from_pil, concat_pil
from utils.video_level_evaluation import video_level_evaluation
from utils.pkg import instantiate_from_config
from utils.util import calculate_brightness,calculate_contrast
from datasets.data_utils import calculate_global_bbox_with_margin

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
    accelerator,
    width,
    height,
    seed=42,
    dtype=torch.float32,
    save_dir=None,
):
    logger.info("Running validation ...")
    unwrap_model = copy.deepcopy(accelerator.unwrap_model(model))
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
    pipeline = pipeline.to(accelerator.device)

    all_test_cases = []
    for data_meta_path in cfg.valdata.test_case:
        all_test_cases.extend(json.load(open(data_meta_path, "r")))
    
  
    current_device_id = torch.cuda.current_device()
    pil_images = []
    if cfg.wandb_project == "debug" or args.debug:
        all_test_cases = all_test_cases[:2]
    res_pil_cases = []
    pose_pil_cases = []
    ref_pil_cases = []
    tgt_pil_cases = []
    for idx, test_case_sample in enumerate(all_test_cases):
        # if idx > 1: break
            if idx % accelerator.num_processes == current_device_id:
                try:
                    test_case_refs = test_case_sample['ref'][:cfg.data.params.ref_num]
                    test_case_tgt = test_case_sample['target']
                    video_paths = [ref["video_path"] for ref in test_case_refs]
                    ref_pose_vid_paths = [ref["kps_path"] for ref in test_case_refs]
                    bboxes_paths = [vp.replace('videos','bboxes').replace('mp4','npy') for vp in video_paths]
                    ref_idx_list =  [ref["idx"] for ref in test_case_refs]
                    ref_mask_vid_paths = [video.replace('videos','masks') for video in video_paths]
                    
                    kps_pth = test_case_tgt["kps_path"]
                    video_path_tgt = test_case_tgt["video_path"]
                    bbox_fp_tgt = video_path_tgt.replace('videos','bboxes').replace('mp4','npy')
                    mask_pth_gt = video_path_tgt.replace('videos','masks')
                    
                    rgb_video_list = [VideoReader(os.path.join(cfg.data.params.root,video_path)) for video_path in video_paths]
                    rgb_pose_vid_list = [VideoReader(os.path.join(cfg.data.params.root,video_path)) for video_path in ref_pose_vid_paths]
                    mask_vid_list = [VideoReader(os.path.join(cfg.data.params.root,video_path)) for video_path in ref_mask_vid_paths]
                    bboxes_list = [np.load(os.path.join(cfg.data.params.root,bp)).squeeze(1) for bp in bboxes_paths]
                    
                    
                    rgb_video_tgt = VideoReader(os.path.join(cfg.data.params.root,video_path_tgt))
                    kps_video_tgt = VideoReader(os.path.join(cfg.data.params.root,kps_pth))
                    mask_video_tgt = VideoReader(os.path.join(cfg.data.params.root,mask_pth_gt))
                    bbox_tgt = np.load(os.path.join(cfg.data.params.root,bbox_fp_tgt)).squeeze(1)
                    
                    
                    ref_img_list = [rgb_video[ref_idxs[0]] for rgb_video, ref_idxs in zip(rgb_video_list,ref_idx_list)]
                    ref_mask_list = [mask_video[ref_idxs[0]].asnumpy()/255 for  mask_video, ref_idxs in zip(mask_vid_list,ref_idx_list)]
                    ref_pose_img_list = [rgb_video[ref_idxs[0]] for rgb_video, ref_idxs in zip(rgb_pose_vid_list,ref_idx_list)]
                    ref_img_pil_list = [Image.fromarray(ref_img.asnumpy()*ref_mask.astype(np.uint8)) for ref_img,ref_mask in zip(ref_img_list,ref_mask_list)]
                    ref_pose_pil_list = [Image.fromarray(ref_img.asnumpy()) for ref_img in ref_pose_img_list]
                    ref_img_pil_case_list = []
                    ref_pose_pil_case_list = []
                    for img_pil,pose_pil,bbox in zip(ref_img_pil_list,ref_pose_pil_list,bboxes_list):
                        x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bbox,img_pil.size[0],img_pil.size[1])
                        ref_img_pil_case_list.append(img_pil.crop((x1_min, y1_min, x2_max, y2_max)).resize((512,512),Image.Resampling.BILINEAR))
                        ref_pose_pil_case_list.append(pose_pil.crop((x1_min, y1_min, x2_max, y2_max)).resize((512,512),Image.Resampling.BILINEAR))
                    
                
                    tgt_vidpil_lst = []
                    tgt_guid_vid_list = []
                    brightness_list = []
                    contrast_list = []
                    end = min(30,len(rgb_video_tgt))
                    x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bbox_tgt,rgb_video_tgt[0].shape[0],rgb_video_tgt[0].shape[1])
                    
                    for index in range(0, end):
                        img = rgb_video_tgt[index]
                        mask_tgt = mask_video_tgt[index].asnumpy()/255
                        pil_img = Image.fromarray(img.asnumpy()*mask_tgt.astype(np.uint8))
                        tgt_guid_pil = Image.fromarray(kps_video_tgt[index].asnumpy())
                        pil_img = pil_img.crop((x1_min, y1_min, x2_max, y2_max)).resize((512,512),Image.Resampling.BILINEAR)
                        tgt_guid_pil = tgt_guid_pil.crop((x1_min, y1_min, x2_max, y2_max)).resize((512,512),Image.Resampling.BILINEAR)
                        tgt_vidpil_lst.append(pil_img)
                        tgt_guid_vid_list.append(tgt_guid_pil)
                        brightness_list.append(calculate_brightness(pil_img))
                        contrast_list.append(calculate_contrast(pil_img))
                    brightness = np.mean(brightness_list)
                    contrast = np.mean(contrast_list)
                    brightness = sinusoidal_encode(brightness,128)
                    contrast = sinusoidal_encode(contrast,128)
                    env_code = torch.tensor(np.concatenate((brightness,contrast),0)).view(1,-1)
                    
                    w, h = 512,512
                    generator = torch.manual_seed(seed)
                    video_tensor = pipeline(
                        ref_img_pil_case_list,
                        tgt_guid_vid_list,
                        ref_pose_pil_case_list,
                        env_code,
                        w,
                        h,
                        len(tgt_guid_vid_list),
                        cfg.validation.denoising_steps,
                        cfg.validation.guidance_scale,
                        AttnControl = ReferenceAttentionControl,
                        generator=generator,
                    ).videos
                    video_tensor = video_tensor[0, ...].permute(1, 2, 3, 0).cpu().numpy()
                    video_pil_lst = []
                    for frame_idx, image_tensor in enumerate(video_tensor):
                        result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
                        result_pil_lst = ref_img_pil_case_list+ [tgt_guid_vid_list[frame_idx], result_img_pil, tgt_vidpil_lst[frame_idx]]
                        concated_pil = concat_pil(result_pil_lst)
                        video_pil_lst.append(concated_pil)
                    save_videos_from_pil(video_pil_lst, os.path.join(save_dir,f"{idx}.mp4"), fps=6)
                    print(f"save to {os.path.join(save_dir,f'{idx}.mp4')}")
                    pil_images.append({"name": f"{idx}", "video": video_pil_lst})
                except Exception as e:
                    print(e)   
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipeline
    torch.cuda.empty_cache()
    return pil_images

def train_val_function(
    accelerator, weight_dtype, vae, train_noise_scheduler, image_enc,
    trainable_params, optimizer, lr_scheduler, train_loss, model, batch, cfg, mode="train"
):
    if mode == "train":
        model.train()
    else:
        model.eval()
    with accelerator.accumulate(model):
        # Convert videos to latent space
        pixel_values_vid = batch["tgt_vid"].to(weight_dtype)
        env_code = batch["env_code"]
        
        with torch.no_grad():
            video_length = pixel_values_vid.shape[1]
            pixel_values_vid = rearrange(
                pixel_values_vid, "b f c h w -> (b f) c h w"
            )
            latents = vae.encode(pixel_values_vid).latent_dist.sample()
            latents = rearrange(
                latents, "(b f) c h w -> b c f h w", f=video_length
            )
            latents = latents * 0.18215    

        noise = torch.randn_like(latents)
        if cfg.noise_offset > 0:
            noise += cfg.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1, 1),
                device=latents.device,
            )
        bsz = latents.shape[0]
        # Sample a random timestep for each video
        timesteps = torch.randint(
            0,
            train_noise_scheduler.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        uncond_fwd = random.random() < cfg.uncond_ratio
        num_ref = random.randint(2,len(batch["ref_img"]))
        batch["ref_img"] = batch["ref_img"][:num_ref]
        batch["ref_pose_img"] = batch["ref_pose_img"][:num_ref]
        batch["clip_img"] = batch["clip_img"][:num_ref]
        
        tgt_guid_videos = batch["tgt_guid_vid"]  # (bs, f, c, H, W)
        tgt_guid_videos = tgt_guid_videos.transpose(
            1, 2
        )  # (bs, c, f, H, W)
        ref_latents_list = []
        image_prompt_embeds_list = []
        for ref_idx, (single_ref_img,single_clip_img) in enumerate(zip(batch["ref_img"],batch["clip_img"])):
            clip_image_list = []
            ref_image_list = []
            for batch_idx, (ref_img, clip_img) in enumerate(
                zip(
                    single_ref_img,
                    single_clip_img,
                )
            ):
                if uncond_fwd:
                    clip_image_list.append(torch.zeros_like(clip_img))
                else:
                    clip_image_list.append(clip_img)
                ref_image_list.append(ref_img)
            with torch.no_grad():
                ref_img = torch.stack(ref_image_list, dim=0).to(
                    dtype=vae.dtype, device=vae.device
                )
                ref_image_latents = vae.encode(
                    ref_img
                ).latent_dist.sample()  # (bs, d, 64, 64)
                ref_image_latents = ref_image_latents * 0.18215

                clip_img = torch.stack(clip_image_list, dim=0).to(
                    dtype=image_enc.dtype, device=image_enc.device
                )
                clip_image_embeds = image_enc(
                    clip_img.to("cuda", dtype=weight_dtype)
                ).image_embeds
                image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                ref_latents_list.append(ref_image_latents)
                image_prompt_embeds_list.append(image_prompt_embeds)
                
        # add noise 
        noisy_latents = train_noise_scheduler.add_noise(
            latents, noise, timesteps
        )

        # Get the target for loss depending on the prediction type
        if train_noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif train_noise_scheduler.prediction_type == "v_prediction":
            target = train_noise_scheduler.get_velocity(
                latents, noise, timesteps
            )
        else:
            raise ValueError(
                f"Unknown prediction type {train_noise_scheduler.prediction_type}"
            )

        model_pred = model(
            noisy_latents,
            timesteps,
            ref_latents_list,
            image_prompt_embeds_list,
            tgt_guid_videos,
            batch["ref_pose_img"],
            env_code,
            uncond_fwd=uncond_fwd,
        )

        if cfg.snr_gamma == 0:
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )
        else:
            snr = compute_snr(train_noise_scheduler, timesteps)
            if train_noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                torch.stack(
                    [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0]
                / snr
            )
            loss = F.mse_loss(
                model_pred.float(), target.float(), reduction="none"
            )
            loss = (
                loss.mean(dim=list(range(1, len(loss.shape))))
                * mse_loss_weights
            )
            loss = loss.mean()

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
        train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

        if mode == "train":
            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    trainable_params,
                    cfg.solver.max_grad_norm,
                )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    return train_loss, loss

def load_stage1_state_dict(
    denoising_unet,
    reference_unet,
    pose_guider,
    attn_guider,
    env_proj,
    stage1_ckpt_dir, stage1_ckpt_step="latest",
):
    if stage1_ckpt_step == "latest":
        ckpt_files = sorted(os.listdir(stage1_ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0]))
        latest_pth_name = (Path(stage1_ckpt_dir) / ckpt_files[-1]).stem
        stage1_ckpt_step = int(latest_pth_name.split("-")[-1])
    
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    pose_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"pose_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    attn_guider.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"attn_guider-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    env_proj.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"env_proj-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    logger.info(f"Loaded stage1 models from {stage1_ckpt_dir}, step={stage1_ckpt_step}")

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="wandb",
        project_dir=cfg.wandb_log_dir,
        kwargs_handlers=[kwargs],
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    if accelerator.is_main_process:
        save_dir = os.path.join(cfg.output_dir, args.exp_name)
        if cfg.wandb_project == "debug":
            if os.path.exists(save_dir):
                try:
                    os.system(f"rm -r {save_dir}")
                except:
                    pass
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
        
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
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
    
    load_stage1_state_dict(
        denoising_unet,
        reference_unet,
        pose_guider,
        attn_guider,
        envproj,
        cfg.stage1_ckpt_dir,
        cfg.stage1_ckpt_step,
    )
        
    # Freeze
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)  
    pose_guider.requires_grad_(False) 
    attn_guider.requires_grad_(False) 
    for name, module in denoising_unet.named_modules():
        if "motion_modules" in name:
            for params in module.parameters():
                params.requires_grad = True
    
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
    
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate
        
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    ) 
    
    train_dataset = instantiate_from_config(cfg.data)
    train_dataset[0]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )
    val_dataset = instantiate_from_config(cfg.valdata)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.train_bs, shuffle=False, num_workers=16
    )


    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )
    
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.wandb_project,
            init_kwargs={"wandb": {"name": args.exp_name + "_" + run_time, "entity": cfg.wandb_entity, "dir":cfg.wandb_log_dir}},
        )
        config_dict = OmegaConf.to_container(cfg)
        wandb.config.update(config_dict)
        
    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")
    
    global_step, first_epoch = 0, 0
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = f"{cfg.output_dir}/{args.exp_name}/checkpoints"
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch    
  
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    
    
    
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            train_loss, loss = train_val_function(
                accelerator, weight_dtype, vae, train_noise_scheduler, image_enc,
                trainable_params, optimizer, lr_scheduler, train_loss, model, batch, cfg, mode="train"
            )
            if global_step % cfg.validation.val_loss_steps == 0:
                accelerator.wait_for_everyone()
                val_loss = 0.
                for _, val_batch in enumerate(val_dataloader):
                    with torch.no_grad():
                        reference_control_reader.clear()
                        reference_control_writer.clear()
                        val_loss, _ = train_val_function(
                                accelerator, weight_dtype, vae, val_noise_scheduler, image_enc,
                                trainable_params, optimizer, lr_scheduler, val_loss, model, val_batch, cfg, mode="val"
                            )
                val_loss_avg = val_loss/len(val_dataloader)
                
                accelerator.log({"val_loss": val_loss_avg}, step=global_step)
            
            save_dir = f'{cfg.output_dir}/{args.exp_name}'
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0                                                                
                #　save checkpoints
                if global_step % cfg.validation.validation_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)  
                        
                #  sanity check
                if global_step == 1:
                    ref_forcheck = [img*0.5+0.5 for img in batch['ref_img']]
                    img_forcheck = batch['tgt_vid'] * 0.5 + 0.5
                    guid_forcheck = batch['tgt_guid_vid']
                    ref_forcheck = [img.unsqueeze(2).repeat(1, 1, cfg.data.sample_frames, 1, 1) for img in ref_forcheck]
                    img_forcheck = rearrange(img_forcheck, 'b f c h w -> b c f h w')
                    guid_forcheck = rearrange(guid_forcheck, 'b f c h w -> b c f h w')
                    video_forcheck = torch.cat(ref_forcheck+[img_forcheck, guid_forcheck], dim=0).cpu()
                    save_videos_grid(video_forcheck, f'{cfg.output_dir}/{args.exp_name}/sanity_check/data-{global_step:06d}-rank{accelerator.device.index}.mp4', fps=30, n_rows=3)
    
                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:  
                    image_save_dir = f"{os.path.join(save_dir, 'validation')}/{global_step:06d}/"
                    if not os.path.exists(image_save_dir):
                        os.makedirs(image_save_dir,exist_ok=True)
                    sample_dicts = log_validation(
                        cfg=cfg,
                        vae=vae,
                        image_enc=image_enc,
                        model=model,
                        scheduler=val_noise_scheduler,
                        accelerator=accelerator,
                        width=cfg.data.params.image_size,
                        height=cfg.data.params.image_size,
                        seed=cfg.seed,
                        save_dir=image_save_dir,
                    )
                    
                    # for sample_id, sample_dict in enumerate(sample_dicts):
                    #     sample_name = sample_dict["name"]
                    #     img = sample_dict["video"]
                    #     # print(img[0].size[0], img[0].size[1])
                    #     if img[0].size[0] >= img[0].size[1]:
                    #         img_path = f"{image_save_dir}{global_step:06d}-{sample_name}-higher.mp4"
                    #     else:
                    #         img_path = f"{image_save_dir}{global_step:06d}-{sample_name}-wider.mp4"
                    #     save_videos_from_pil(img, img_path, fps=6)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        images_to_log_higher = []
                        images_to_log_wider = []
                        images_to_evaluation = []
                        for img_path in os.listdir(image_save_dir):
                            if img_path.endswith("-higher.mp4"):
                                wandb_img = wandb.Video(image_save_dir+img_path, caption=f"{global_step:06d}-{img_path}")
                                images_to_log_higher.append(wandb_img)
                            elif img_path.endswith("-wider.mp4"):
                                wandb_img = wandb.Video(image_save_dir+img_path, caption=f"{global_step:06d}-{img_path}")
                                images_to_log_wider.append(wandb_img)
                            images_to_evaluation.append(image_save_dir+img_path)
                            
                        video_eval_dict = video_level_evaluation(images_to_evaluation,5)
                        accelerator.log({"lpips": video_eval_dict["lpips"]}, step=global_step)
                        accelerator.log({"l1_error": video_eval_dict["l1_error"]}, step=global_step)
                        accelerator.log({"psnr": video_eval_dict["psnr"]}, step=global_step)
                        accelerator.log({"fvd": video_eval_dict["fvd"]}, step=global_step)
                        accelerator.log({"movie": video_eval_dict["movie"]}, step=global_step)
                        wandb.log({
                                    "images_higher": images_to_log_higher,
                                    "images_wider": images_to_log_wider,
                                    "iteration": global_step
                                })
                        unwrap_model = accelerator.unwrap_model(model)
                        save_checkpoint(
                            unwrap_model,
                            save_dir,
                            "net",
                            global_step,
                            total_limit=2,
                            )
                    torch.cuda.empty_cache()

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 2,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break                  

    accelerator.wait_for_everyone()
    accelerator.end_training()

def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    mm_state_dict = OrderedDict()
    state_dict = model.state_dict()
    # for key in state_dict:
    #     if "motion_module" in key:
    #         mm_state_dict[key] = state_dict[key]

    torch.save(state_dict, save_path)
    
    
if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--debug", action='store_true')
    
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    os.environ["WANDB_API_KEY"] = config.wandb_key
    main(config)   
        