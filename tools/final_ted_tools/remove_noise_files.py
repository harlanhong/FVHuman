import os
import numpy as np
import pdb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--root",  type=str)
args = parser.parse_args()

root = args.root
with open('delete_file.txt','r') as f:
    data = f.readlines()

data = [da.strip() for da in data]
npy_prefix_list = ['_face_pos.npy','_hands_pos.npy','_pose_id.npy','_pose_pos.npy','_pose_score.npy']

for fp in data:
    vid = os.path.join(root,fp)
    if os.path.isdir(vid):
        os.system(f'rm -rf {vid}')
        dwpose = vid.replace('videos','dwpose')
        masks = vid.replace('videos','masks')
        npy = vid.replace('videos','npy')
        bboxes = vid.replace('videos','bboxes')
        
        os.system(f'rm -rf {dwpose}')
        os.system(f'rm -rf {masks}')
        os.system(f'rm -rf {npy}')
        os.system(f'rm -rf {bboxes}')
    else:
        dwpose1 = os.path.join(root,fp.replace('videos','dwpose').replace('.mp4','_hands_face.mp4'))
        dwpose2 = os.path.join(root,fp.replace('videos','dwpose').replace('.mp4','_weighted_heatmap.mp4'))
        mask = os.path.join(root,fp.replace('videos','masks'))
        npy_list = [os.path.join(root,fp.replace('videos','npy').replace('.mp4',comp)) for comp in npy_prefix_list]
        bbox = os.path.join(root,fp.replace('videos','bboxes').replace('.mp4','.npy'))
        os.system(f'rm {vid}')
        os.system(f'rm {dwpose1}')
        os.system(f'rm {dwpose2}')
        os.system(f'rm {mask}')
        os.system(f'rm {bbox}')
        for comp in npy_list:
            os.system(f'rm {comp}') 
    print(fp)