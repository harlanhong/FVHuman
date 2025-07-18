import concurrent.futures
import os
import random
from pathlib import Path
import sys
import numpy as np
# 获取脚本的目录，并将父目录添加到 sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from dwpose import DWposeDetector
import argparse
import glob
from tqdm import tqdm
import pdb
import cv2
from utils.util import get_fps, read_frames, save_videos_from_pil
from PIL import Image
# Extract dwpose mp4 videos from raw videos
# /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_dwpose/*/*.mp4


def process_single_video(video_path, detector,out_path):
    if os.path.exists(out_path):
        return
   
    fps = get_fps(video_path)
    frames = read_frames(video_path)
    kps_results = []
    for i, frame_pil in enumerate(frames):
        result, score = detector(frame_pil)
        score = np.mean(score, axis=-1)
        kps_results.append(result)

    save_videos_from_pil(kps_results, out_path, fps=fps)
    
def process_single_image(image_path, detector,out_path):
    frame_pil = Image.open(image_path)
    image = cv2.imread(image_path)
    result = detector(image)
    pdb.set_trace()
    cv2.imwrite(out_path,result)

def process_batch_videos(video_list, detector, save_path):
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, root_dir, save_dir)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-i", "--input_dir", type=str,required=True, help="Directory containing videos")
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")
    args = parser.parse_args()
    video_files = sorted(glob.glob(args.input_dir+'/*/images_lr/*/*.jpg'))
    allocated_fps = [video_files[i] for i in range(len(video_files)) if i % args.parallelism == args.rank]
    detector = DWposeDetector()
    # torch.cuda.set_device(gpu_id)
    # detector = detector.cuda()
    pbar = tqdm(total=len(allocated_fps), desc="processing ...")
    for video_file in allocated_fps:
        pbar.update(1)
        # 生成与视频同一目录下的 pose.npz 文件的路径
        output_npz_file = video_file.replace('images_lr','dwpose')
        if os.path.exists(output_npz_file):
            continue
        os.makedirs(os.path.dirname(output_npz_file),exist_ok=True)
        # 处理视频并保存结果
        # try:
        process_single_image(video_file, detector,output_npz_file)
        # except Exception as e:
        #     print(e)
    