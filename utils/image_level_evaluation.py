import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
import lpips
import time
from PIL import Image

loss_fn_alex = lpips.LPIPS(net='alex')

def lpips_metric(video1, video2):
    video1_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video1]
    video2_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video2]
    video1_rgb = np.array(video1_rgb)
    video2_rgb = np.array(video2_rgb)
    video1_torch = torch.tensor(video1_rgb).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    video2_torch = torch.tensor(video2_rgb).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
    
    lpips_values = []
    for frame1, frame2 in zip(video1_torch, video2_torch):
        lpips_values.append(loss_fn_alex(frame1.unsqueeze(0), frame2.unsqueeze(0)).item())
    return np.mean(lpips_values)
    

def mean_l1_pixel_error(video1, video2):
    return np.mean(np.abs(video1 - video2))


def psnr(video1, video2):
    mse = np.mean((video1 - video2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def mean_l1_pixel_error(video1, video2):
    return np.mean(np.abs(video1 - video2))


def psnr(video1, video2):
    mse = np.mean((video1 - video2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def image_level_evaluation(input_list, num_pil=None, verbose=False, output_dir="./debug/"):
    if verbose and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    psnr_value = 0
    mean_l1_error = 0
    lpips_value = 0
    #print(input_list)
    for image_path in tqdm(input_list):
        gt_list = []
        res_list = []
        #print(image_path)
        image = Image.open(image_path)
        image = np.array(image)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        H, W, _ = image.shape
        w = W // num_pil
        gt = image[:, (num_pil-1)*w:num_pil*w, :]
        
        res = image[:, (num_pil-2)*w:(num_pil-1)*w, :]
        gt_list.append(gt)
        res_list.append(res)

        if verbose:
            for i, (gt_patch, res_patch) in enumerate(zip(gt, res)):
                gt_path = os.path.join(output_dir, f"gt_{os.path.basename(image_path).split('.')[0]}_{i}.png")
                res_path = os.path.join(output_dir, f"res_{os.path.basename(image_path).split('.')[0]}_{i}.png")
                cv2.imwrite(gt_path, gt_patch)
                cv2.imwrite(res_path, res_patch)
        
        gt_array = np.array(gt_list)
        res_array = np.array(res_list)
        
        psnr_value += psnr(gt_array, res_array)
        mean_l1_error += mean_l1_pixel_error(gt_array, res_array)
        lpips_value += lpips_metric(gt_array, res_array)
        
    psnr_value /= len(input_list)
    lpips_value /= len(input_list)
    mean_l1_error /= len(input_list)
    if verbose:
        print(f'PSNR: {psnr_value}')
        print(f'Mean L1 Pixel Error: {mean_l1_error}')
        print(f'LPIPS: {lpips_value}')
        
    return {
        "l1_error": mean_l1_error,
        "psnr": psnr_value,
        "lpips": lpips_value,
    }
    
# input_list = [
#     "/mnt/localssd/outputs/wandb/debug/000001/000001-0-wider.png",
#     "/mnt/localssd/outputs/wandb/debug/000001/000001-1-higher.png",
#     "/mnt/localssd/outputs/wandb/debug/000001/000001-2-wider.png",
#     "/mnt/localssd/outputs/wandb/debug/000001/000001-3-wider.png",
# ]
# dict = image_level_evaluation(input_list, True)