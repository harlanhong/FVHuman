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
import imageio
import json
from decord import VideoReader,cpu
a_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(a_folder_path)
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3,_read_s3_to_bytesio
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
    frames = [sample.astype(np.uint8)*255 for sample in mask_list]
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', macro_block_size=None)
    # bytesio = BytesIO()
    # image.save(bytesio, format=s3_url.split(".")[-1].upper())
    # _write_bytesio_to_s3(bytesio, s3_url, s3_client)

    
def main():
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    # parser.add_argument("-i", "--input_dir", type=str,
    #                     required=True, help="Directory containing videos")
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    args = parser.parse_args()

    # video_files = sorted(glob.glob('/projects/p32321/fating/dataset/CelebV-HQ/35666/*/*.mp4'))
    video_files = sorted(glob.glob('/mnt/localssd/final_ted_dataset/videos/*/*.mp4'))
    # video_files = sorted(glob.glob('/projects/p32296/fating/dataset/HDTF-Processed/*/video.mp4'))
    allocated_fps = [video_files[i] for i in range(len(video_files)) if i % args.parallelism == args.rank]
    sam_checkpoint = "/sensei-fs/users/fhong/src/pose2video/tools/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    pbar = tqdm(total=len(allocated_fps), desc="processing ...")
    for video_file in allocated_fps:
        pbar.update(1)
        # 生成与视频同一目录下的 pose.npz 文件的路径
        output_npz_file = video_file.replace('videos','masks')
        if os.path.exists(output_npz_file):
            continue
        os.makedirs(os.path.dirname(output_npz_file),exist_ok=True)
        # 处理视频并保存结果
        bbox_path = video_file.replace('videos','bboxes').replace('mp4','npy')
        # try:
        process_video(predictor, video_file, bbox_path,output_npz_file)
        # except Exception as e:
            # os.system(f'rm {bbox_path}')
            # print(bbox_path)
if __name__ == "__main__":
    main()



