import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path
import time
import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pdb
def ensure_file_written(file_path, retries=10, delay=0.1):
    for i in range(retries):
        if os.path.exists(file_path):
            image = Image.open(file_path)
            if image is not None:
                return True
            else:
                print(i)
        time.sleep(delay)
    return False

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr
    
def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps
def numpy_to_video(tensor, output_video_file, audio_source=None, fps=25):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_file (str): The file path where the output video will be saved.
        audio_source (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 25 fps.
    """
    tensor = np.clip(tensor, 0, 255).astype(
        np.uint8
    )  # to [0, 255]

    def make_frame(t):
        # get index
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]
    new_video_clip = VideoClip(make_frame, duration=tensor.shape[0] / fps)
    if audio_source is not None:
        audio_clip = AudioFileClip(audio_source).subclip(0, tensor.shape[0] / fps)
        new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_file, fps=fps, audio_codec='aac')

def calculate_brightness(image):
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)
    for index in range(scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)
    return 1 if brightness == 255 else brightness / scale

def calculate_contrast(image):
    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    pixels = sum(histogram)
    mean = sum(i * w for i, w in enumerate(histogram)) / pixels
    contrast = sum((i - mean) ** 2 * w for i, w in enumerate(histogram)) / pixels
    return contrast ** 0.5
def attention_map_to_image(attention_map, img_pil):
    # 如果 attention_map 是 torch.Tensor，则将其转换为 numpy 数组
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()
    elif isinstance(attention_map, np.ndarray):
        pass  # 已经是 numpy 数组
    else:
        raise TypeError("attention_map 必须是 torch.Tensor 或 numpy.ndarray 类型")
    # 确保 attention_map 是二维数组（n x n）
    if attention_map.ndim > 2:
        # 如果 attention_map 有多于两个维度，例如 (heads, n, n)
        # 可以对多头取平均，或者选择特定的头
        attention_map = attention_map.mean(axis=0)

    # 归一化到 [0, 1]
    attention_map -= attention_map.min()
    attention_map /= attention_map.max()

    # 将 attention_map 调整为与原始图像相同的尺寸
    attention_map_img = Image.fromarray(np.uint8(attention_map * 255))
    attention_map_img = attention_map_img.resize(img_pil.size, resample=Image.BILINEAR)

    # 将 attention_map 转换为 numpy 数组
    attention_map = np.array(attention_map_img)

    # 应用颜色映射
    colored_attention = plt.get_cmap('jet')(attention_map / 255.0)[:, :, :3]
    colored_attention = np.uint8(colored_attention * 255)
    colored_attention = Image.fromarray(colored_attention)

    # 将原始图像和 attention_map 叠加
    blended = Image.blend(img_pil.convert('RGB'), colored_attention, alpha=0.5)
    return blended

def all_attention_map_to_image(attention_map, img_pil_list):
    attention_map = torch.cat([fea[0,0] for fea in attention_map],0)
    # 如果 attention_map 是 torch.Tensor，则将其转换为 numpy 数组
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach().cpu().numpy()
    elif isinstance(attention_map, np.ndarray):
        pass  # 已经是 numpy 数组
    else:
        raise TypeError("attention_map 必须是 torch.Tensor 或 numpy.ndarray 类型")
    

    # 归一化到 [0, 1]
    attention_map -= attention_map.min()
    attention_map /= attention_map.max()

    # 将 attention_map 调整为与原始图像相同的尺寸
    res_img_list = []
    for attn_map,img_pil in zip(attention_map,img_pil_list):
        attention_map_img = Image.fromarray(np.uint8(attn_map * 255))
        attention_map_img = attention_map_img.resize(img_pil.size, resample=Image.BILINEAR)

        # 将 attention_map 转换为 numpy 数组
        attention_map = np.array(attention_map_img)

        # 应用颜色映射
        colored_attention = plt.get_cmap('jet')(attention_map / 255.0)[:, :, :3]
        colored_attention = np.uint8(colored_attention * 255)
        colored_attention = Image.fromarray(colored_attention)

        # 将原始图像和 attention_map 叠加
        blended = Image.blend(img_pil.convert('RGB'), colored_attention, alpha=0.5)
        res_img_list.append(blended)
    return res_img_list