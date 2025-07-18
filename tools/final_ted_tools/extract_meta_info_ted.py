import argparse
import json
import os
from decord import VideoReader
from tqdm import tqdm
import pdb
import glob
import random
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str,default='final_ted')
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

video_dir = '/projects/p32321/fating/dataset/final_ted_mini_set/videos'
root_fp = os.path.dirname(video_dir)

vids = sorted(glob.glob(video_dir+'/*/*.mp4'))
# collect all video_folder paths
video_mp4_paths = set()
all_frames = 0
for vid in tqdm(vids):
    video = VideoReader(vid)
    all_frames += len(video)
num_test = max(1, len(vids) // 100)

# print(video_mp4_paths)
print(len(vids))
print(all_frames/30/3600)

ids = sorted(glob.glob( video_dir+'/*'))
random.shuffle(ids)

num_test = max(1, len(ids) // 100)

train_data = ids[:-num_test]
test_data = ids[-num_test:]

meta_infos = []
for idx, vid_folder_path in enumerate(ids):
    
    mode = "test" if vid_folder_path in test_data else "train"
    vids = sorted(glob.glob(vid_folder_path+'/*.mp4'))
    
    folder_name = os.path.basename(vid_folder_path)
    folder_dict=[]
    for vid_fp in vids: 
        dwpose_fp = vid_fp.replace('videos','dwpose').replace('.mp4','_weighted_heatmap.mp4')
        mask_fp = vid_fp.replace('videos','masks')
        if os.path.exists(dwpose_fp) and os.path.exists(mask_fp) and os.path.exists(vid_fp):
            folder_dict.append({
            "video_path": os.path.relpath(vid_fp, root_fp),
            "kps_path": os.path.relpath(dwpose_fp, root_fp),
            "mask_path": os.path.relpath(mask_fp, root_fp),
            })
    random.shuffle(folder_dict)
    meta_infos.append({
        'id':folder_name,
        'data':folder_dict,
        "mode": mode
    })

json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
