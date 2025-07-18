import json
import numpy as np
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3,_read_s3_to_bytesio
import os
import random
import pdb
samples = []
id_list = ['s3://zhanxu-public/fating/test_mvhuman_data/101262/','s3://zhanxu-public/fating/test_mvhuman_data/102228/','s3://zhanxu-public/fating/test_mvhuman_data/101903/','s3://zhanxu-public/fating/test_mvhuman_data/101234/','s3://zhanxu-public/fating/test_mvhuman_data/102234/']
for i in range(5):
    for id in id_list:
        img_folder = os.path.join(id,'images_lr/')
        masks_forlder = os.path.join(id,'fmask_lr/')
        dwposes_forlder = os.path.join(id,'dwpose/')
        annots_forlder = os.path.join(id,'annots/')
        cameras = list_s3_dir(dwposes_forlder)
        select_dwpose_cameras = random.sample(cameras, 11)
        ref = []
        for imgca in select_dwpose_cameras[:-1]:
            imgs = list_s3_dir(imgca)
            select_img = random.sample(imgs, 1)[0]
            ref.append({
                "kps_path": select_img,
                "image_path": select_img.replace('dwpose','images_lr').replace('png','jpg'),
                "mask_path": select_img.replace('dwpose','fmask_lr').replace('.png','_fmask.png'),
                "annots_path": select_img.replace('dwpose','annots').replace('png','json'),
            })
        target_camera = select_dwpose_cameras[-1]
        imgs = list_s3_dir(target_camera)
        select_img = random.sample(imgs, 1)[0]
        sample = {
                "target": {
                    "kps_path": select_img,
                    "image_path": select_img.replace('dwpose','images_lr').replace('png','jpg'),
                    "mask_path": select_img.replace('dwpose','fmask_lr').replace('.png','_fmask.png'),
                    "annots_path": select_img.replace('dwpose','annots').replace('png','json'),
                },
                "ref": ref
            }
        samples.append(sample)
# 转换为 JSON 格式并打印
json_output = json.dumps(samples, indent=4)
print(json_output)

# 保存到文件
with open("/projects/p32296/fating/src/novel-view-human-master/data/mvhuman_s1_test.json", "w") as f:
    f.write(json_output)