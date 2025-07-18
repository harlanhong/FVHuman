import concurrent.futures
import os
import random
from pathlib import Path

import numpy as np

from dwpose import DWposeDetector
from utils.util import get_fps, read_frames, save_videos_from_pil

# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4


def process_single_video(video_path, detector, root_dir, save_dir):
    relative_path = os.path.relpath(video_path, root_dir)
    # print(relative_path, video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    # print(out_path)
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
    
    for i, frame_pil in enumerate(frames):
        result, score, pose = detector(frame_pil, detect_resolution=512, image_resolution=512)
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
        print(f"Process {i}/{len(video_list)} video {video_path}")
        process_single_video(video_path, detector, root_dir, save_dir)


if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, default="/mnt/localssd/champ_trainning_sample/rawdata/")
    parser.add_argument(
        "--save_dir", type=str, help="Path to save extracted pose videos"
    )
    parser.add_argument("--num_workers", type=int, default=2, help="Num workers")
    parser.add_argument("--job_idx", type=int, default=2, help="Num workers")
    args = parser.parse_args()
    num_workers = args.num_workers
    job_idx = args.job_idx
    if args.save_dir is None:
        save_dir = args.video_root + "_dwpose3"
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    # collect all video_folder paths
    os.makedirs(args.video_root+"_dwpose3/", exist_ok=True)
    existing_dwpose_files = os.listdir(args.video_root+"_dwpose3/")
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_root):
        if "dwpose" in dirs: continue
        if "dwpose" in root: continue 
        for name in files:
            if name.endswith(".mp4"):
                # print(name)
                # print(existing_dwpose_files)
                exist_flap = False
                for dwpfiles in existing_dwpose_files:
                    if name in dwpfiles:
                        exist_flap = True
                        break
                if exist_flap:
                    continue
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
    # init detector
    gpu_id = gpu_ids[job_idx % len(gpu_ids)]
    detector = DWposeDetector()
    # torch.cuda.set_device(gpu_id)
    detector = detector.to(f"cuda:{gpu_id}")
    process_batch_videos(video_chunks[job_idx],detector,args.video_root, save_dir)
    