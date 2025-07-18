import numpy as np
import torch
import cv2
import imageio
import pdb
import sys
from segment_anything import sam_model_registry, SamPredictor

import os
import argparse
import sys
from tqdm import tqdm
import glob

import json
from decord import VideoReader,cpu
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3,_read_s3_to_bytesio
a_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(a_folder_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='', help='path of dataset')
    args = parser.parse_args()
    return args

def process_video(predictor,video_path, bbox_path,output_path):
   
    # 读取视频
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    bboxes = np.load(bbox_path)
    # 准备保存视频
    frame_idx = 0
    mask_list=[]
    for frame in tqdm(vr):
        frame = frame.asnumpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_idx < len(bboxes):
            input_box = bboxes[frame_idx]
        else:
            break  # 如果 bboxes 列表中的数据不足以覆盖所有帧，则停止处理
        predictor.set_image(frame_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(input_box)[None, :],
            multimask_output=False,
        )
        mask_list.append(masks[0])
        frame_idx+=1
    np.savez(output_path, mask=mask_list)

def generate_mp4(filename):
    save_name = filename.replace('s3://zhanxu-public/fating/final_ted_v2_mask','/mnt/localssd/final_ted_v2_mask2').replace('npz','mp4')
    if os.path.exists(save_name):
        return
    os.makedirs(os.path.dirname(save_name),exist_ok=True)
    ref_mask_reader = np.load(_read_s3_to_bytesio(filename))['mask']
    frames = [sample.astype(np.uint8)*255 for sample in ref_mask_reader]

    imageio.mimsave(save_name, frames, fps=25, codec='libx264', macro_block_size=None)

    print(f'Video saved to {save_name}')

def main():
   
    
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    args = parser.parse_args()

    fp = 's3://zhanxu-public/fating/final_ted_v2_mask/'
    
    video_files = []
    dir_list = list_s3_dir(fp)
    for dirname in tqdm(dir_list):
        file_list = list_s3_dir(dirname)
        video_files+=file_list
    allocated_fps = [video_files[i] for i in range(len(video_files)) if i % args.parallelism == args.rank]
   
    pbar = tqdm(total=len(allocated_fps), desc="processing ...")
    for video_file in allocated_fps:
        pbar.update(1)
        try:
            generate_mp4(video_file)
        except Exception as e:
            print(e)
if __name__ == "__main__":
    main()



