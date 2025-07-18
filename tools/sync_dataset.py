
import glob
from tqdm import tqdm
import os
import numpy as np
import shutil
import pdb
os.makedirs('/mnt/localssd/ted_dataset/final_ted_v2',exist_ok=True)
os.makedirs('/mnt/localssd/ted_dataset/final_ted_v2_dwpose',exist_ok=True)
os.makedirs('/mnt/localssd/ted_dataset/final_ted_v2_mask',exist_ok=True)
os.makedirs('/mnt/localssd/ted_dataset/final_ted_v2_bbox',exist_ok=True)
videos = glob.glob('/mnt/localssd/final_ted_v2/*/*.mp4')

for vid_fp in tqdm(videos):
    ref_kps_path = vid_fp.replace('final_ted_v2','final_ted_v2_dwpose')
    ref_mask_path =  vid_fp.replace('final_ted_v2','final_ted_v2_mask2')
    bbox_video_path =vid_fp.replace('final_ted_v2','final_ted_v2_bbox').replace('mp4','npy')
    if os.path.exists(ref_kps_path) and os.path.exists(ref_mask_path) and os.path.exists(bbox_video_path):
        new_vid_fp = vid_fp.replace('/mnt/localssd/final_ted_v2','/mnt/localssd/ted_dataset/final_ted_v2')
        os.makedirs(os.path.dirname(new_vid_fp),exist_ok=True)
        shutil.copy(vid_fp,new_vid_fp)
        new_fp = ref_kps_path.replace('/mnt/localssd/final_ted_v2_dwpose','/mnt/localssd/ted_dataset/final_ted_v2_dwpose')
        os.makedirs(os.path.dirname(new_fp),exist_ok=True)
        shutil.copy(ref_kps_path,new_fp)
        
        new_fp = ref_mask_path.replace('/mnt/localssd/final_ted_v2_mask2','/mnt/localssd/ted_dataset/final_ted_v2_mask2')
        os.makedirs(os.path.dirname(new_fp),exist_ok=True)
        shutil.copy(ref_mask_path,new_fp)
        
        new_fp = bbox_video_path.replace('/mnt/localssd/final_ted_v2_bbox','/mnt/localssd/ted_dataset/final_ted_v2_bbox')
        os.makedirs(os.path.dirname(new_fp),exist_ok=True)
        shutil.copy(bbox_video_path,new_fp)