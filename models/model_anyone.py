import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.pose_guider import PoseGuider
import pdb
class AnyoneModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class AnyoneModel_with_bg(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        bg_image_latents,
        clip_image_embeds,
        clip_bg_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_unet(
                bg_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_bg_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=torch.cat([clip_image_embeds,clip_bg_embeds],1)
        ).sample

        return model_pred


class AnyoneModel_no_cfg(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        # if not uncond_fwd:
        ref_timesteps = torch.zeros_like(timesteps)
        self.reference_unet(
            ref_image_latents,
            ref_timesteps,
            encoder_hidden_states=clip_image_embeds,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class AnyoneModelNoReference(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        clip_image_embeds,
        pose_img,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        # if not uncond_fwd:
        #     ref_timesteps = torch.zeros_like(timesteps)
        #     self.reference_unet(
        #         ref_image_latents,
        #         ref_timesteps,
        #         encoder_hidden_states=clip_image_embeds,
        #         return_dict=False,
        #     )
        #     self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        return model_pred

class AnyoneModel_MutilRef(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_fea = self.pose_guider(pose_cond_tensor)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred

class AnyoneModel_MutilRef_CrossPose(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        pose_fea = [self.pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append(torch.stack([fea[i] for fea in pose_fea]).mean(0))
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea_fuse,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred

#不做mean，保持原样
class AnyoneModel_MutilRef_CrossPose_attn(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        pose_fea = [self.pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append([fea[i] for fea in pose_fea])
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea_fuse,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred

#不做mean，保持原样
class AnyoneModel_MutilRef_CrossPose_attn_add_tgt_pose(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider: PoseGuider,
        cross_pose_guider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.cross_pose_guider = cross_pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        pose_fea = [self.cross_pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        
        tgt_pose_fea = self.pose_guider(pose_cond_tensor)
        
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append([fea[i] for fea in pose_fea])
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=pose_fea_fuse,
            tgt_fea = tgt_pose_fea,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred

#pose guided只用tgt
class AnyoneModel_MutilRev2f_attn_add_CrossPose(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider,
        attn_guider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.attn_guider = attn_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        attn_fea = [self.attn_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        attn_fea_fuse = []
        for i in range(len(attn_fea[0])):
            attn_fea_fuse.append([fea[i] for fea in attn_fea])
        
        pose_fea = self.pose_guider(pose_cond_tensor)
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            attn_feat=attn_fea_fuse,
            pose_feat = pose_fea,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred


class AnyoneModel_MutilRef_attn_add_CrossPose(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider,
        attn_guider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.attn_guider = attn_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        uncond_fwd: bool = False,
    ):  
        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        attn_fea = [self.attn_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        attn_fea_fuse = []
        for i in range(len(attn_fea[0])):
            attn_fea_fuse.append([fea[i] for fea in attn_fea])
        
        pose_fea = [self.pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append(torch.stack([fea[i] for fea in pose_fea]).mean(0))
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            attn_feat=attn_fea_fuse,
            pose_feat = pose_fea_fuse,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred


class AnyoneModel_MutilRef_attn_add_CrossPose_envproj(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider,
        attn_guider,
        env_proj,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.attn_guider = attn_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.env_proj = env_proj
    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        env_code,
        uncond_fwd: bool = False,
    ):  
        env_code = self.env_proj(env_code.to(dtype=pose_img.dtype)).to(device="cuda").unsqueeze(1)

        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        attn_fea = [self.attn_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        attn_fea_fuse = []
        for i in range(len(attn_fea[0])):
            attn_fea_fuse.append([fea[i] for fea in attn_fea])
        
        pose_fea = [self.pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append(torch.stack([fea[i] for fea in pose_fea]).mean(0))
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            attn_feat=attn_fea_fuse,
            pose_feat = pose_fea_fuse,
            encoder_hidden_states=torch.cat(clip_image_embeds_list+[env_code],-2),
        ).sample

        return model_pred

class AnyoneModel_MutilRef_attn_add_CrossPose_spatial_selection(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider,
        attn_guider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.attn_guider = attn_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
    def update_control(self, reader, writer):
        self.reference_control_writer = writer
        self.reference_control_reader = reader
        
    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        uncond_fwd: bool = False,
    ):  

        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        attn_feas = [self.attn_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea = [self.pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append(torch.stack([fea[i] for fea in pose_fea]).mean(0))
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            attn_feat=attn_feas,
            pose_feat = pose_fea_fuse,
            encoder_hidden_states=torch.cat(clip_image_embeds_list,-2),
        ).sample

        return model_pred

#加上一个spatial attn
class AnyoneModel_MutilRef_attn_add_CrossPose_envproj_spatial_selection(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        pose_guider,
        attn_guider,
        env_proj,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.attn_guider = attn_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.env_proj = env_proj
    def update_control(self, reader, writer):
        self.reference_control_writer = writer
        self.reference_control_reader = reader
        
    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents_list,
        clip_image_embeds_list,
        pose_img,
        pose_ref_list,
        env_code,
        uncond_fwd: bool = False,
    ):  
        env_code = self.env_proj(env_code.to(dtype=pose_img.dtype)).to(device="cuda").unsqueeze(1)

        pose_cond_tensor = pose_img.to(device="cuda")
        pose_ref_tensor_list = [pose_ref.to(device="cuda") for pose_ref in pose_ref_list]
        attn_feas = [self.attn_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea = [self.pose_guider(pose_cond_tensor, pose_ref) for pose_ref in pose_ref_tensor_list]
        pose_fea_fuse = []
        for i in range(len(pose_fea[0])):
            pose_fea_fuse.append(torch.stack([fea[i] for fea in pose_fea]).mean(0))
        
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            for ref_image_latents,clip_image_embeds in zip(ref_image_latents_list,clip_image_embeds_list):
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=clip_image_embeds,
                    return_dict=False,
                )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            attn_feat=attn_feas,
            pose_feat = pose_fea_fuse,
            encoder_hidden_states=torch.cat(clip_image_embeds_list+[env_code],-2),
        ).sample

        return model_pred
