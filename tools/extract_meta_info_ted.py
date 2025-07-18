import argparse
import json
import os
from decord import VideoReader
from tqdm import tqdm
import pdb
import glob
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
parser = argparse.ArgumentParser()
parser.add_argument("--pose_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

root_fp = os.path.dirname(args.pose_path)
if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

video_dir = args.pose_path.replace('_dwpose','')

pose_vids = sorted(glob.glob( args.pose_path+'/*'))
# collect all video_folder paths
video_mp4_paths = set()
all_frames = 0
for folder_fp in tqdm(pose_vids):
    vids = sorted(glob.glob(folder_fp+'/*'))
    for vid_fp in vids:
        if vid_fp.endswith(".mp4"):
            try:
                video = VideoReader(vid_fp)
                all_frames += len(video)
            except:
                continue
# video_mp4_paths = [vid.replace('_dwpose','') for vid in pose_vids]
num_test = max(1, len(pose_vids) // 100)

# print(video_mp4_paths)
print(len(pose_vids))
print(all_frames/30/3600)
print(f"Num test videos: {num_test}")

test_indices = set(range(0, len(pose_vids), len(pose_vids) // num_test))

meta_infos = []
for idx, pose_folder_path in enumerate(pose_vids):
    video_folder_path = pose_folder_path.replace('_dwpose','')
    mode = "test" if idx in test_indices else "train"
    vids = sorted(glob.glob(pose_folder_path+'/*.mp4'))
    if len(vids)<3:
        continue
    folder_name = os.path.basename(pose_folder_path)
    folder_dict=[]
    for pose_fp in vids: 
        vid_fp = pose_fp.replace('_dwpose','')
        folder_dict.append({
        "video_path": os.path.relpath(vid_fp, root_fp),
        "kps_path": os.path.relpath(pose_fp, root_fp),
    })
    meta_infos.append({
        'id':folder_name,
        'data':folder_dict,
        "mode": mode
    })

json.dump(meta_infos, open(f"./data/{args.meta_info_name}_meta.json", "w"))
