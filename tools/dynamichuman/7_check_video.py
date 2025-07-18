import os
import glob
from tqdm import tqdm
import numpy as np
import pdb
import decord
if __name__ == '__main__':
    vid_fps = glob.glob('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/*/*')
    for vfp in tqdm(vid_fps):
        coor_fp = os.path.join(vfp,'coor.npy')
        mask_fp = os.path.join(vfp,'mask_clip.mp4')
        video_fp = os.path.join(vfp,'video_clip.mp4')
        
        [min_x,min_y,max_x,max_y] = np.load(coor_fp)
        save_mask_fp =  os.path.join(vfp,'crop_mask_clip.mp4')
        save_video_fp =  os.path.join(vfp,'crop_video_clip.mp4')
        try:
            vr = decord.VideoReader(save_mask_fp)
            vr = decord.VideoReader(save_video_fp)
        except Exception as e:
            print(vfp)
            # width = int(max_x - min_x)
            # height = int(max_y - min_y)
            # x = round(min_x)
            # y = round(min_y)
            # crop = f'crop={width}:{height}:{x}:{y}'
            # os.system(f'ffmpeg -y -i {mask_fp} -vf "{crop}" -c:a copy {save_mask_fp}')
            # os.system(f'ffmpeg -y -i {video_fp} -vf "{crop}" -c:a copy {save_video_fp}')
            # print(f'process {vfp}')
        