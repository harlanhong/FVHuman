import argparse
import json
import os
from decord import VideoReader
from tqdm import tqdm
import pdb
import glob
import random
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3
import numpy as np
def sample_elements(lst,num):
    # 等间隔采样出 20 个元素
    indices = np.linspace(0, len(lst) - 1, num=num, dtype=int)
    sampled_elements = [lst[i] for i in indices]
    return sampled_elements

folder_list = list_s3_dir('s3://zhanxu-public/fating/test_mvhuman_data/')
id_list = sorted([fp for fp in folder_list if 'tar.gz' not in fp])
    # video_files = []
    # for f1 in tqdm(folder_list):
    #     cam_list = sorted(list_s3_dir(f1))
    #     cam_list = sample_elements(cam_list,15)

test_ist = random.sample(id_list, 5)

print(f"Num test videos: {test_ist}")

meta_infos = []
for idx, id_s3_url in enumerate(id_list):
    img_s3 = id_s3_url+'images_lr/'
    dwpose_s3 = id_s3_url+'dwpose/'
    dwpose_cam_list = sorted(list_s3_dir(dwpose_s3))
    # img_cam_list = sample_elements(dwpose_cam_list,15)
    mask_cam_list = [fp.replace('dwpose','fmask_lr') for fp in dwpose_cam_list]
    img_cam_list = [fp.replace('dwpose','images_lr') for fp in dwpose_cam_list]
    annots_cam_list = [fp.replace('dwpose','annots').replace('jpg','json') for fp in dwpose_cam_list]
    data = []
    for img,mask,dwpose,annots in zip(img_cam_list,mask_cam_list,dwpose_cam_list,annots_cam_list):
        ele = {}
        ele['video_path'] = img
        ele['mask_path'] = mask
        ele['kps_path'] = dwpose
        ele['annots'] = annots
        data.append(ele)
    ele_dict = {}
    ele_dict['id'] = idx
    ele_dict['data'] = data
    if id_s3_url in test_ist:
        ele_dict['mode'] = 'test'
    else:
        ele_dict['mode'] = 'train'
    meta_infos.append(ele_dict)
json.dump(meta_infos, open(f"./data/mvhuman_meta.json", "w"))
