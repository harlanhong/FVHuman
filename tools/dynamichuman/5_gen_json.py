import argparse
import json
import os
from decord import VideoReader
from tqdm import tqdm
import pdb
import random
import glob
# python tools/extract_meta_info.py --root_path /path/to/video_dir --dataset_name fashion


fp = '/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman'

ids = glob.glob(fp+'/*')

test_set = random.sample(ids,5)


meta_infos = []
for idx, id_fp in enumerate(ids):
    folder_name = os.path.basename(id_fp)
    mode = "test" if id_fp in test_set else "train"
    folder_dict=[]
    cam_list = glob.glob(id_fp+'/*')
    for cam in cam_list: 
        folder_dict.append({
        "video_path": os.path.join(cam, 'crop_video_clip.mp4'),
        "kps_path": os.path.join(cam, 'dwpose/weighted_heatmap.mp4'),
    })
    meta_infos.append({
        'id':folder_name,
        'data':folder_dict,
        "mode": mode
    })

json.dump(meta_infos, open(f"./data/dyhuman_meta.json", "w"))
