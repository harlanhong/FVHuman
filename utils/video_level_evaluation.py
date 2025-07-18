import os
from tqdm import tqdm
import numpy as np
import cv2
import torch
import lpips
import time
from PIL import Image
from decord import VideoReader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from .pytorch_i3d import InceptionI3d
from scipy import linalg
import io
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        state_dict = torch.load(buffer, map_location=device)
        return state_dict
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
        
def frechet_distance(samples_A, samples_B):
    A_mu = np.mean(samples_A, axis=0)
    A_sigma = np.cov(samples_A, rowvar=False)
    B_mu = np.mean(samples_B, axis=0)
    B_sigma = np.cov(samples_B, rowvar=False)
    try:
        frechet_dist = calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
    except ValueError:
        frechet_dist = 1e+10
    return frechet_dist

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    #print(mu1[0], mu2[0])
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    #print(sigma1[0], sigma2[0])
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

loss_fn_alex = lpips.LPIPS(net='alex')

model_path='utils/rgb_charades.pt'
i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(157)
state_dict = load_model(model_path)
i3d.load_state_dict(state_dict)
i3d.cuda()
i3d.eval()
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

def get_fvd(video1, video2):
    frames = [Image.fromarray(frame) for frame in video1]
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    frames = frames.unsqueeze(0).cuda()
    frames = frames.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        # print(frames.shape)
        features = i3d.extract_features(frames)
        feature_vector_1 = features.cpu().numpy()
        # print("Feature vector shape:", feature_vector_1.shape)
    
    frames = [Image.fromarray(frame) for frame in video2]
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    frames = frames.unsqueeze(0).cuda()
    frames = frames.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        # print(frames.shape)
        features = i3d.extract_features(frames)
    feature_vector_2 = features.cpu().numpy()
    # print("Feature vector shape:", feature_vector_2.shape)
    return feature_vector_1, feature_vector_2


def movie2(video1, video2):
    temporal_diff = np.mean((video1[:, 1:] - video1[:, :-1]) - (video2[:, 1:] - video2[:, :-1]) ** 2)
    spatial_diff = np.mean((video1 - video2) ** 2)
    return 0.5 * (temporal_diff + spatial_diff)

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


def video_level_evaluation(input_list, num_vid = 3,verbose=False, output_dir="./debug/"):
    if verbose and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    psnr_value = 0
    mean_l1_error = 0
    lpips_value = 0
    movie = 0
    fvd = 0
    #print(input_list)
    all_gt = []
    all_target = []
    for video_path in tqdm(input_list):
        gt_list = []
        res_list = []
        gt_pil_list = []
        res_pil_list = []
        #print(image_path)
        video_reader = VideoReader(video_path)
        for index in range(len(video_reader)):
            img = video_reader[index]
            image = Image.fromarray(img.asnumpy()).resize((512, 512*num_vid))
            res = np.array(image)[-2*512:-512]
            if res.shape[2] == 4:
                res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
            elif res.shape[2] == 3:
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            gt = np.array(image)[-512:]
            if gt.shape[2] == 4:
                gt = cv2.cvtColor(gt, cv2.COLOR_RGBA2RGB)
            elif gt.shape[2] == 3:
                gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
                
            gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            # res_rgb = res_rgb - res_rgb
            # print(image.shape, gt.shape, res.shape)
            gt_list.append(gt)
            res_list.append(res)
            gt_pil_list.append(gt_rgb)
            res_pil_list.append(res_rgb)

        if verbose:
            for i, (gt_patch, res_patch) in enumerate(zip(gt_list, res_list)):
                gt_path = os.path.join(output_dir, f"gt_{os.path.basename(video_path).split('.')[0]}_{i}.jpg")
                res_path = os.path.join(output_dir, f"res_{os.path.basename(video_path).split('.')[0]}_{i}.jpg")
                # print(gt_patch.shape, )
                cv2.imwrite(gt_path, gt_patch)
                cv2.imwrite(res_path, res_patch)
        
        gt_array = np.array(gt_list)
        res_array = np.array(res_list)
        # print(gt_array.shape, res_array.shape)
        
        psnr_value += psnr(gt_array, res_array)
        mean_l1_error += mean_l1_pixel_error(gt_array, res_array)
        lpips_value += lpips_metric(gt_array, res_array)
        movie += movie2(gt_array, res_array)
        
        gt_fea, res_fea = get_fvd(gt_pil_list, res_pil_list)
        all_gt.append(gt_fea.squeeze(0).squeeze(2).squeeze(2))
        all_target.append(res_fea.squeeze(0).squeeze(2).squeeze(2))
    
    all_gt = np.concatenate(all_gt, axis=1)
    all_target = np.concatenate(all_target, axis=1)
    fvd = frechet_distance(all_gt, all_target)
    psnr_value /= len(input_list)
    lpips_value /= len(input_list)
    mean_l1_error /= len(input_list)
    movie /= len(input_list)
    
    if verbose:
        print(f'PSNR: {psnr_value}')
        print(f'Mean L1 Pixel Error: {mean_l1_error}')
        print(f'LPIPS: {lpips_value}')
        print(f'MOVIE: {movie}')
        print(f"FVD: {fvd}")
        
    return {
        "l1_error": mean_l1_error,
        "psnr": psnr_value,
        "lpips": lpips_value,
        "movie": movie,
        "fvd": fvd,
    }
    
# input_list = [
#     "/mnt/localssd/outputs/wandb/debug-vfi/000001/000001-0-higher.mp4",
#     "/mnt/localssd/outputs/wandb/debug-vfi/000001/000001-1-higher.mp4",
# ]
# dict = video_level_evaluation(input_list, True)