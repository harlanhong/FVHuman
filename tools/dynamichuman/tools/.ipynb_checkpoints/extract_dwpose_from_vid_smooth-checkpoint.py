import concurrent.futures
import os
import random
from pathlib import Path

import numpy as np

from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil

# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4


import numpy as np

def temporal_smooth_pose(pose_list, window_size=3):
    """
    Smooths the pose data over time using a moving average filter.
    
    :param pose_list: List of pose dictionaries containing 'bodies', 'hands', and 'faces' data.
    :param window_size: Size of the moving average window.
    :return: A list of smoothed pose dictionaries.
    """
    smoothed_pose_list = []
    num_poses = len(pose_list)
    
    # Prepare to iterate over each pose component
    for i in range(num_poses):
        if i < window_size // 2 or i >= num_poses - window_size // 2:
            # For border cases, we just use the original data (could be improved with different padding strategies)
            smoothed_pose_list.append(pose_list[i])
        else:
            smoothed_pose = {}
            # Perform smoothing for each component
            for part in ['bodies', 'hands', 'faces']:
                if part == 'bodies':
                    smoothed_data = {}
                    window_data = np.array([pose_list[j][part]["candidate"] for j in range(i - window_size // 2, i + window_size // 2 + 1)])
                    smoothed_data["candidate"] = np.mean(window_data, axis=0)
                    smoothed_data["subset"] = pose_list[i][part]["subset"]
                else:
                    smoothed_data = {}
                    window_data = np.array([pose_list[j][part] for j in range(i - window_size // 2, i + window_size // 2 + 1)])
                    smoothed_data = np.mean(window_data, axis=0)
                smoothed_pose[part] = smoothed_data
            smoothed_pose_list.append(smoothed_pose)
    
    return smoothed_pose_list

import numpy as np
from scipy.stats import norm

def temporal_smooth_pose_gaussian(pose_list, window_size=3, sigma=1):
    """
    Smooths the pose data over time using a Gaussian weighted moving average filter.
    
    :param pose_list: List of pose dictionaries containing 'bodies', 'hands', and 'faces' data.
    :param window_size: Size of the moving average window.
    :param sigma: Standard deviation of the Gaussian distribution used for the weights.
    :return: A list of smoothed pose dictionaries.
    """
    smoothed_pose_list = []
    num_poses = len(pose_list)
    
    # Calculate Gaussian weights
    weights = norm.pdf(np.linspace(-window_size // 2, window_size // 2, window_size), 0, sigma)
    weights /= weights.sum()  # Normalize weights so they sum to 1

    # Prepare to iterate over each pose component
    for i in range(num_poses):
        if i < window_size // 2 or i >= num_poses - window_size // 2:
            # For border cases, we just use the original data (could be improved with different padding strategies)
            smoothed_pose_list.append(pose_list[i])
        else:
            smoothed_pose = {}
            # Perform smoothing for each component
            for part in ['bodies', 'hands', 'faces']:
                if part == 'bodies':
                    smoothed_data = {}
                    # Gather data for the window
                    window_data = np.array([pose_list[j][part]["candidate"] for j in range(i - window_size // 2, i + window_size // 2 + 1)])
                    # Apply Gaussian weighted average
                    weighted_avg = np.tensordot(window_data, weights, axes=([0], [0]))
                    smoothed_data["candidate"] = weighted_avg
                    smoothed_data["subset"] = pose_list[i][part]["subset"]
                else:
                    window_data = np.array([pose_list[j][part] for j in range(i - window_size // 2, i + window_size // 2 + 1)])
                    # Apply Gaussian weighted average
                    weighted_avg = np.tensordot(window_data, weights, axes=([0], [0]))
                    smoothed_data = weighted_avg
                smoothed_pose[part] = smoothed_data
            smoothed_pose_list.append(smoothed_pose)
    
    return smoothed_pose_list

import numpy as np
from scipy.signal import medfilt

def temporal_smooth_pose_median(pose_list, window_size=5):
    """
    Smooths the pose data over time using a median filter.
    
    :param pose_list: List of pose dictionaries containing 'bodies', 'hands', and 'faces' data.
    :param window_size: Size of the median filter window.
    :return: A list of smoothed pose dictionaries.
    """
    smoothed_pose_list = []
    num_poses = len(pose_list)
    
    # Prepare to iterate over each pose component
    for i in range(num_poses):
        smoothed_pose = {}
        # Perform smoothing for each component
        for part in ['bodies', 'hands', 'faces']:
            if part == 'bodies':
                smoothed_data = {}
                # Gather data for the window across all poses for 'candidate'
                candidate_data = np.array([pose_list[j][part]["candidate"] for j in range(max(0, i - window_size // 2), min(num_poses, i + window_size // 2 + 1))])
                # Apply median filter across the first dimension
                smoothed_data["candidate"] = np.median(candidate_data, axis=0)
                smoothed_data["subset"] = pose_list[i][part]["subset"]  # Copy the subset without smoothing
            else:
                # Gather data for the window across all poses for hands and faces
                part_data = np.array([pose_list[j][part] for j in range(max(0, i - window_size // 2), min(num_poses, i + window_size // 2 + 1))])
                # Apply median filter across the first dimension
                smoothed_data = np.median(part_data, axis=0)
            smoothed_pose[part] = smoothed_data
        smoothed_pose_list.append(smoothed_pose)
    
    return smoothed_pose_list


import numpy as np
from scipy.signal import savgol_filter

def temporal_smooth_pose_savgol(pose_list, window_size=7, poly_order=2):
    """
    Smooths the pose data over time using a Savitzky-Golay filter.
    
    :param pose_list: List of pose dictionaries containing 'bodies', 'hands', and 'faces' data.
    :param window_size: Size of the Savitzky-Golay filter window; must be a positive odd integer.
    :param poly_order: Order of the polynomial used to fit the samples; must be less than window_size.
    :return: A list of smoothed pose dictionaries.
    """
    smoothed_pose_list = []
    num_poses = len(pose_list)
    
    # Ensure the window size is odd and greater than the polynomial order
    if window_size % 2 == 0:
        window_size += 1  # Make window size odd if it is not
    
    # Prepare to iterate over each pose component
    for i in range(num_poses):
        smoothed_pose = {}
        # Perform smoothing for each component
        for part in ['bodies', 'hands', 'faces']:
            if part == 'bodies':
                smoothed_data = {}
                # Gather data for the window across all poses for 'candidate'
                candidate_data = np.array([pose_list[j][part]["candidate"] for j in range(num_poses)])
                # Apply Savitzky-Golay filter for each coordinate dimension
                for dim in range(candidate_data.shape[-1]):
                    candidate_data[:, :, dim] = savgol_filter(candidate_data[:, :, dim], window_size, poly_order, axis=0)
                smoothed_data["candidate"] = candidate_data[i]  # Get the filtered data for the current frame
                smoothed_data["subset"] = pose_list[i][part]["subset"]  # Copy the subset without smoothing
            else:
                part_data = np.array([pose_list[j][part] for j in range(num_poses)])
                # Apply Savitzky-Golay filter for each coordinate dimension
                for dim in range(part_data.shape[-1]):
                    part_data[:, :, dim] = savgol_filter(part_data[:, :, dim], window_size, poly_order, axis=0)
                smoothed_data = part_data[i]  # Get the filtered data for the current frame
            smoothed_pose[part] = smoothed_data
        smoothed_pose_list.append(smoothed_pose)
    
    return smoothed_pose_list


# Example usage:
# Assuming pose_list is your list of poses like the one defined in the provided code snippet.
# smoothed_pose_list = temporal_smooth_pose(pose_list)

    

def process_single_video(video_path, detector, root_dir, save_dir):
    relative_path = os.path.relpath(video_path, root_dir)
    print(relative_path, video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    if os.path.exists(out_path):
        return

    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)
    frames = read_frames(video_path)
    kps_results = []
    pose_pos = []
    pose_id = []

    face_pos = []
    hands_pos = []
    # hands_id = []

    pose_list = [] 
    body_score_list = []
    for i, frame_pil in enumerate(frames):
        pose, H, W, input_image, image_resolution, body_score = detector.smooth_image(frame_pil)
        pose_list.append(pose)
        body_score_list.append(body_score)
        #print(pose)
        #print(bug)
    smoothed_pose_list = temporal_smooth_pose_savgol(pose_list)
    for _ in range(30):
        smoothed_pose_list = temporal_smooth_pose_savgol(smoothed_pose_list)
    for _ in range(5):
        smoothed_pose_list = temporal_smooth_pose_median(smoothed_pose_list)
    for _ in range(3):
        smoothed_pose_list = temporal_smooth_pose(smoothed_pose_list)
    #print(smoothed_pose_list[3]["bodies"]["candidate"], pose_list[3]["bodies"]["candidate"])
    #print(bug)
    for i, smoothed_pose in enumerate(smoothed_pose_list):
        body_score = body_score_list[i]
        result, score, pose = detector.process_smoothed_pose(smoothed_pose, H, W, input_image, image_resolution, body_score)
        score = np.mean(score, axis=-1)
        kps_results.append(result)
        

        pose_pos.append(pose["bodies"]["candidate"])
        pose_id.append(pose["bodies"]["subset"])
        face_pos.append(pose["faces"])
        hands_pos.append(pose["hands"])
        
        # print(pose["bodies"]["candidate"].shape, pose["bodies"]["subset"].shape)
        # print(result, score)
    pose_pos = np.array(pose_pos) # 300*18*2
    pose_id = np.array(pose_id) #300*18
    # print(pose_pos.shape, pose_id.shape)
    
    
    # print(out_path)
    hands_pos = np.array(hands_pos)
    face_pos = np.array(face_pos)
    # print(hands_pos.shape, face_pos.shape)
    base_name = os.path.basename(out_path)
    np.save(out_path.replace(base_name, base_name[:-4]+"_pose_pos.npy"), pose_pos)
    np.save(out_path.replace(base_name, base_name[:-4]+"_pose_id.npy"), pose_id)
    np.save(out_path.replace(base_name, base_name[:-4]+"_face_pos.npy"), face_pos)
    np.save(out_path.replace(base_name, base_name[:-4]+"_hands_pos.npy"), hands_pos)
    # np.save(out_path, pose_pos)
    save_videos_from_pil(kps_results, out_path, fps=fps)


def process_batch_videos(video_list, detector, root_dir, save_dir):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, root_dir, save_dir)


if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str)
    parser.add_argument(
        "--save_dir", type=str, help="Path to save extracted pose videos"
    )
    parser.add_argument("-j", type=int, default=2, help="Num workers")
    args = parser.parse_args()
    num_workers = args.j
    if args.save_dir is None:
        save_dir = args.video_root + "_dwpose_smooth"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    # collect all video_folder paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_root):
        if "dwpose" in dirs: continue
        if "dwpose" in root: continue   
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    # split into chunks,
    batch_size = (len(video_mp4_paths) + num_workers - 1) // num_workers 
    print(f"Num videos: {len(video_mp4_paths)} {batch_size = }")
    video_chunks = [
        video_mp4_paths[i : i + batch_size]
        for i in range(0, len(video_mp4_paths), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks):
            # init detector
            gpu_id = gpu_ids[i % len(gpu_ids)]
            detector = DWposeDetector()
            # torch.cuda.set_device(gpu_id)
            detector = detector.to(f"cuda:{gpu_id}")

            futures.append(
                executor.submit(
                    process_batch_videos, chunk, detector, args.video_root, save_dir
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
