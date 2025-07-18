import os
import cv2
import numpy as np
import glob
import decord
from tqdm import tqdm
ids = glob.glob("/projects/p32321/fating/dataset/final_ted_dataset_v2/videos/*")
print('total videos: ', len(ids))
n_t = 0
n_v = 0
for id in tqdm(ids):
    videos = glob.glob(os.path.join(id, '*.mp4'))
    n_v += len(videos)
    for video in videos:
        vr = decord.VideoReader(video)
        n_t += len(vr)/25
print('total videos: ', n_v)
print('total times: ', n_t)
