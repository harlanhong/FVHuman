import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path
import json
import time
import pdb
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
from decord import VideoReader
import copy
# modify the following for different experiments
from datasets.image_anyone import ImageDataset
from pipelines.pipeline_pose2image_ms_attn_add_cross_pose_envproj_spatial_selection import Pose2ImagePipeline
from models.model_anyone import AnyoneModel_MutilRef_attn_add_CrossPose_envproj_spatial_selection as AnyoneModel
from models.pose_guider import SpatialAttn3 as SpatialAttn
from models.pose_guider import CrossPoseGuider

from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d_multi_pose_guided_attn_add_crosspose_spatial_selection import UNet3DConditionModel
from models.mutual_self_attention import ReferenceAttentionControlv4 as ReferenceAttentionControl
from models.bright_proj import EnvProjModel
from utils.util import seed_everything, delete_additional_ckpt, compute_snr, ensure_file_written
from utils.image_level_evaluation import image_level_evaluation
from utils.pkg import instantiate_from_config
from datasets.data_utils import calculate_global_bbox_with_margin
warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")
logger = get_logger(__name__, log_level="INFO")
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3,_read_s3_to_bytesio
from utils.util import calculate_brightness,calculate_contrast
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
    '''
    imporvements:
        1. inference speed
        2. validation samples
            some training samples, 
                same id, same pose
            some validation samples.
                same id, different pose, 
                same id, out of domain pose
                different id (not seen in the dataset)
            input:
                list of video name, reference id, target id. 
        todo:
        metrics, including loss.
    '''

    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    pose_guider = unwrap_model.pose_guider
    attn_guider = unwrap_model.attn_guider
    env_proj = unwrap_model.env_proj

    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    
    pipeline = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=copy.deepcopy(reference_unet),
        denoising_unet=copy.deepcopy(denoising_unet),
        pose_guider=pose_guider,
        attn_guider=attn_guider,
        env_proj=env_proj,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(accelerator.device)

    all_test_cases = []
    for data_meta_path in cfg.valdata.test_case:
        all_test_cases.extend(json.load(open(data_meta_path, "r")))
    all_test_cases = [case for case in all_test_cases if case['mode']=='test']
    current_device_id = torch.cuda.current_device()
    pil_images = []
    if cfg.wandb_project == "debug" or args.debug:
        all_test_cases = all_test_cases[:2]
    res_pil_cases = []
    pose_pil_cases = []
    ref_pil_cases = []
    tgt_pil_cases = []
    frame_idx = [0,40,80,120,160,200,240]
    for idx, test_case_sample in enumerate(all_test_cases):
        # if idx > 1: break
        try:
            if idx % accelerator.num_processes == current_device_id:
            
                test_case_refs = test_case_sample['data'][:cfg.data.params.ref_num]
                test_case_tgt = test_case_sample['data'][-1]

                ref_video_paths = [ref["video_path"] for ref in test_case_refs]
                ref_pose_vid_paths = [ref["kps_path"] for ref in test_case_refs]
                
                kps_pth = test_case_tgt["kps_path"]
                video_path_tgt = test_case_tgt["video_path"]
                
                ref_video_list = [VideoReader(os.path.join(cfg.data.params.root,video_path)) for video_path in ref_video_paths]
                ref_pose_vid_list = [VideoReader(os.path.join(cfg.data.params.root,video_path)) for video_path in ref_pose_vid_paths]
                
                rgb_video_tgt = VideoReader(os.path.join(cfg.data.params.root,video_path_tgt))
                kps_video = VideoReader(os.path.join(cfg.data.params.root,kps_pth))
                
                # w, h = width,height
                # ref_idx_list = test_case_sample['ref_idx']
                for indx in frame_idx:
                    ref_img_list = [rgb_video[indx] for rgb_video in ref_video_list]
                    ref_pose_img_list = [rgb_video[indx] for rgb_video in ref_pose_vid_list]
                    ref_img_pil_list = [Image.fromarray(ref_img.asnumpy()) for ref_img in ref_img_list]
                    ref_pose_pil_list = [Image.fromarray(ref_img.asnumpy()) for ref_img in ref_pose_img_list]
                    tgt_img = rgb_video_tgt[indx]
                    tgt_img_pil = Image.fromarray(tgt_img.asnumpy())
                    pose_img = kps_video[indx]
                    pose_img_pil = Image.fromarray(pose_img.asnumpy())
                    brightness = calculate_brightness(tgt_img_pil)
                    contrast = calculate_contrast(tgt_img_pil)
                    brightness = sinusoidal_encode(brightness,128)
                    contrast = sinusoidal_encode(contrast,128)
                    env_code = torch.tensor(np.concatenate((brightness,contrast),0)).view(1,-1)
                    w, h = 512,512
                    generator = torch.manual_seed(seed)
                    image = pipeline(
                        ref_img_pil_list,
                        pose_img_pil,
                        ref_pose_pil_list,
                        env_code,
                        w,
                        h,
                        cfg.validation.denoising_steps,
                        cfg.validation.guidance_scale,
                        AttnControl = ReferenceAttentionControl,
                        generator=generator,
                    ).images
                    image = image[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
                    res_img_pil_case = Image.fromarray((image * 255).astype(np.uint8)).resize((512,512),Image.Resampling.BILINEAR)
                    res_img_pil_case = res_img_pil_case.resize((w, h))
                    pose_img_pil = pose_img_pil.resize((w, h))
                    num_ref = len(ref_img_pil_list)
                    canvas = Image.new("RGB", (w * (3+num_ref), h), "white")
                    for cind, ref_pil in enumerate(ref_img_pil_list):
                        canvas.paste(ref_pil, (cind*w, 0))
                    canvas.paste(pose_img_pil, (w*num_ref, 0))
                    canvas.paste(res_img_pil_case, (w * (num_ref+1), 0))
                    canvas.paste(tgt_img_pil, (w * (num_ref+2), 0))
                    canvas.save(os.path.join(save_dir,f"{idx}-{indx}.png"))
                    # pil_images.append({"name": f"{idx}-{indx}", "img": canvas})

        except Exception as e:
            print(e)
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del pipeline
    torch.cuda.empty_cache()
    # return pil_images


def train_val_function(
        accelerator, weight_dtype, vae, train_noise_scheduler, image_enc,
        trainable_params, optimizer, lr_scheduler, train_loss, model, batch, cfg, mode="train"
    ):
    if mode == "train":
        model.train()
    else:
        model.eval()
    with accelerator.accumulate(model):
        pixel_values = batch["tgt_img"].to(weight_dtype)
        with torch.no_grad():
            # print(vae.device, pixel_values.device)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
            latents = latents * 0.18215
            
        noise = torch.randn_like(latents)
        if cfg.noise_offset > 0.0:
            noise += cfg.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1, 1),
                device=noise.device,
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
        ref_latents_list = []
        image_prompt_embeds_list = []
        num_ref = random.randint(2,len(batch["ref_img"]))
        batch["ref_img"] = batch["ref_img"][:num_ref]
        batch["ref_pose_img"] = batch["ref_pose_img"][:num_ref]
        batch["clip_img"] = batch["clip_img"][:num_ref]
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
                
        noisy_latents = train_noise_scheduler.add_noise(
            latents, noise, timesteps
        )

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
            batch["tgt_guid"].unsqueeze(2),
            batch["ref_pose_img"],
            batch["env_code"],
            uncond_fwd,
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


def main(cfg):    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    wandb_log_dir = os.path.join(cfg.output_dir, args.exp_name,'wandb')
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="wandb",
        project_dir=wandb_log_dir,
        kwargs_handlers=[kwargs],
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    save_dir = os.path.join(cfg.output_dir, args.exp_name)
    validation_dir = os.path.join(save_dir, 'validation')
    if accelerator.is_main_process:
        if cfg.wandb_project == "debug":
            if os.path.exists(save_dir):
                try:
                    os.system(f"rm -r {save_dir}")
                except:
                    pass
        os.makedirs(save_dir, exist_ok=True)
        # os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
        os.makedirs(validation_dir, exist_ok=True)
        
        # save config, script
        # shutil.copy(args.config, os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml'))
        # shutil.copy(os.path.abspath(__file__), os.path.join(save_dir, 'sanity_check'))
           
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
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },
    ).to(device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")    
    envproj = EnvProjModel(
        input_dim=256,
        latent_dim=768,
    ).to(device="cuda")

    # guidance_encoder_group = setup_guidance_encoder(cfg)
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
    
    # Freeze some modules
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    pose_guider.requires_grad_(True)
            
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
            init_kwargs={"wandb": {"name": args.exp_name + "_" + run_time, "entity": cfg.wandb_entity, "dir":wandb_log_dir}},
        )
        config_dict = OmegaConf.to_container(cfg)
        wandb.config.update(config_dict)

    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")
    
    global_step, first_epoch = 0, 0
    
    try:
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
    except Exception as e:
        logger.info(f"No checkpoint found, start from scratch")
        
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    # Training Loop
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.
        for _, batch in enumerate(train_dataloader):
            reference_control_reader.clear()
            reference_control_writer.clear()
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
                # print(val_loss_avg, global_step)
                accelerator.log({"val_loss": val_loss_avg}, step=global_step)

            # Logging
            save_dir = f"{cfg.output_dir}/{args.exp_name}"
            
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
                # check data
                if global_step == 1:
                    img_forcheck = batch['tgt_img'] * 0.5 + 0.5
                    ref_forcheck = [img*0.5+0.5 for img in batch['ref_img']]
                    # guid_forcheck = list(torch.chunk(batch['tgt_guid'], batch['tgt_guid'].shape[1]//3, dim=1))
                    guid_forcheck = batch['tgt_guid']
                    # print(batch['tgt_guid'].shape, batch['tgt_img'].shape)
                    batch_forcheck = torch.cat(ref_forcheck+[img_forcheck, guid_forcheck], dim=0)
                    save_image(batch_forcheck, f'{cfg.output_dir}/{args.exp_name}/sanity_check/data-{global_step:06d}-rank{accelerator.device.index}.png', nrow=4)
                # log validation                      
                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:
                    # device_id = accelerator.device
                    image_save_dir = f"{validation_dir}/{global_step:06d}/"
                    if not os.path.exists(image_save_dir):
                        os.makedirs(image_save_dir,exist_ok=True)
                    sample_dicts = log_validation(
                        cfg=cfg,
                        vae=vae,
                        image_enc=image_enc,
                        model=model,
                        scheduler=val_noise_scheduler,
                        accelerator=accelerator,
                        width=cfg.data.params.img_size[0],
                        height=cfg.data.params.img_size[0],
                        seed=cfg.seed,
                        save_dir=image_save_dir,
                    )
                        
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        images_to_log = []
                        images_to_evaluation = []
                        for img_path in os.listdir(image_save_dir):
                            wandb_img = wandb.Image(image_save_dir+img_path, caption=f"{global_step:06d}-{img_path}")
                            images_to_log.append(wandb_img)
                            images_to_evaluation.append(image_save_dir+img_path)
                        eval_dict = image_level_evaluation(images_to_evaluation,num_pil = cfg.data.params.ref_num+3)
                        accelerator.log({"lpips": eval_dict["lpips"]}, step=global_step)
                        accelerator.log({"l1_error": eval_dict["l1_error"]}, step=global_step)
                        accelerator.log({"psnr": eval_dict["psnr"]}, step=global_step)
                        wandb.log({
                                    "images_higher": images_to_log[:6],
                                    "iteration": global_step
                                })
                        unwrap_model = accelerator.unwrap_model(model)
                        save_checkpoint(
                            unwrap_model.denoising_unet,
                            save_dir,
                            "denoising_unet",
                            global_step,
                            total_limit=1,
                            )
                        save_checkpoint(
                            unwrap_model.reference_unet,
                            save_dir,
                            "reference_unet",
                            global_step,
                            total_limit=1,
                            )
                        save_checkpoint(
                            unwrap_model.pose_guider,
                            save_dir,
                            "pose_guider",
                            global_step,
                            total_limit=1,
                            )  
                        save_checkpoint(
                            unwrap_model.attn_guider,
                            save_dir,
                            "attn_guider",
                            global_step,
                            total_limit=1,
                            )  
                        save_checkpoint(
                            unwrap_model.env_proj,
                            save_dir,
                            "env_proj",
                            global_step,
                            total_limit=1,
                            )  
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 1,
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

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)           
    
    
if __name__ == "__main__":
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage1.yaml")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--exp_name", type=str)
    
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    os.environ["WANDB_API_KEY"] = config.wandb_key
    main(config)              