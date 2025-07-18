import argparse
import json
import os
from decord import VideoReader
from tqdm import tqdm
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion
parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str)
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--meta_info_name", type=str)

args = parser.parse_args()

if args.meta_info_name is None:
    args.meta_info_name = args.dataset_name

pose_dir = args.root_path + "_dwpose"

# collect all video_folder paths
video_mp4_paths = set()
all_frames = 0
counter = 0
for name in tqdm(os.listdir(args.root_path)):
    if name.endswith(".mp4"):
        try:
            video = VideoReader(os.path.join(args.root_path, name))
            video = VideoReader(os.path.join(pose_dir, name.replace(".mp4", "_weighted_heatmap.mp4")))
            video = VideoReader(os.path.join(pose_dir, name.replace(".mp4", "_hands_face.mp4")))
            
            counter += 1 
        except:
            continue
        if counter % 5 == 0:
            video_mp4_paths.add(os.path.join(args.root_path, name))
            all_frames += len(video)
video_mp4_paths = list(video_mp4_paths)
num_test = max(1, len(video_mp4_paths) // 1000)

# print(video_mp4_paths)
print(len(video_mp4_paths))
print(all_frames/30/3600)
print(f"Num test videos: {num_test}")

test_indices = set(range(0, len(video_mp4_paths), len(video_mp4_paths) // num_test))

meta_infos = []
for idx, video_mp4_path in enumerate(video_mp4_paths):
    relative_video_name = os.path.relpath(video_mp4_path, args.root_path)
    kps_path = os.path.join(pose_dir, relative_video_name.replace(".mp4", "_weighted_heatmap.mp4"))
    mask_path = os.path.join(pose_dir, relative_video_name.replace(".mp4", "_hands_face.mp4"))
    mode = "test" if idx in test_indices else "train"
    meta_infos.append({
        "video_path": video_mp4_path,
        "kps_path": kps_path,
        "mask_path": mask_path,
        "mode": mode
    })
json.dump(meta_infos, open(f"./data/{args.meta_info_name}.json", "w"))
