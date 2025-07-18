import decord
import glob
import os
from tqdm import tqdm
import numpy as np
def detect_mask():
    videos = glob.glob('/mnt/localssd/final_ted_dataset/masks/*/*.mp4')
    for vid in tqdm(videos):
        try:
            tmp=decord.VideoReader(vid)
        except Exception as e:
            print(vid)
            os.system(f'rm {vid}')

def detetct_bbox():
    bboxes = glob.glob('/mnt/localssd/final_ted_dataset/bboxes/*/*.npy')
    for bbox in tqdm(bboxes):
        try:
            tmp=np.load(bbox)
        except Exception as e:
            print(bbox)
            os.system(f'rm {bbox}')

if __name__ == '__main__':
    detetct_bbox()