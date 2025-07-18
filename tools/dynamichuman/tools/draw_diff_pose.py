import concurrent.futures
import os
import random
from pathlib import Path
import torch
import numpy as np
import av
import importlib
import os.path as osp
import shutil
import sys
from pathlib import Path
import numpy as np
import cv2
import math
import matplotlib
from decord import VideoReader
from tqdm import tqdm
from controlnet_aux.util import HWC3, resize_image
from PIL import Image
import gc
import shutil
import subprocess 

eps = 0.01

def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps
    
    
def draw_pose(pose_pos, pose_id, face, hands, body_score, face_score, hands_score, H, W):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, pose_pos, pose_id, body_score)
    canvas = draw_handpose(canvas, hands, hands_score)
    # canvas = draw_facepose(canvas, face, face_score)
    canvas = HWC3(canvas)
    return canvas


def draw_bodypose(canvas, candidate, subset, body_score):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        # [3, 17],
        # [6, 18],
    ]

    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    for i, limb in enumerate(limbSeq):
        for n in range(len(subset)):
            index = subset[n][np.array(limb) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            avg_score = (body_score[limb[0] - 1] + body_score[limb[1] - 1]) / 2
            color = (np.array(colors[i]) * avg_score).astype(int).tolist()
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(canvas, polygon, color)

    canvas = (canvas * 0.6).astype(np.uint8)
    # position
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            color = (np.array(colors[i]) * body_score[i]).astype(int).tolist()
            cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)
    
    return canvas


def draw_handpose(canvas, all_hand_peaks, hands_score):
    H, W, C = canvas.shape

    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    for peaks, score_list in zip(all_hand_peaks, hands_score):
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            avg_score = (score_list[e[0]] + score_list[e[1]]) / 2
            color = (matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * avg_score * 255).astype(int).tolist()
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness=2,
                )

        for i, keypoint in enumerate(peaks):
            x, y = keypoint
            x = int(x * W)
            y = int(y * H)
            color = (np.array([0, 0, 255]) * score_list[i]).astype(int).tolist()
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, color, thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, face_score):
    H, W, C = canvas.shape
    for lmks, score_list in zip(all_lmks, face_score):
        lmks = np.array(lmks)
        for i, lmk in enumerate(lmks):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            brightness = int(score_list[i] * 255)  # Access the correct score
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (brightness, brightness, brightness), thickness=-1)
    return canvas
    

def process_video(save_dir, video_name):
    pose_pos = np.load(save_dir.replace("dwpose3", "dwpose2") + video_name + "_pose_pos.npy")
    pose_id = np.load(save_dir.replace("dwpose3", "dwpose2") + video_name + "_pose_id.npy")
    face = np.load(save_dir.replace("dwpose3", "dwpose2") + video_name + "_face_pos.npy")
    hands = np.load(save_dir.replace("dwpose3", "dwpose2") + video_name + "_hands_pos.npy")
    whole_score = np.load(save_dir.replace("dwpose3", "dwpose2") + video_name + "_pose_score.npy")
    body_score = np.ones_like(whole_score[:, :18])
    face_score = np.ones_like(whole_score[:, 24:92].reshape(-1, 1, 68))
    hands_score = np.ones_like(whole_score[:, 92:].reshape(-1, 2, 21))
    
    video_reader = VideoReader(save_dir.replace("dwpose3", "dwpose2")+video_name+".mp4")
    fps = round(video_reader.get_avg_fps())
    h, w, _ = video_reader[0].asnumpy().shape
    frames = pose_pos.shape[0]
    
    heatmaps = []
    limbs = []
    weighted_heatmaps = []
    oris = []
    hands_map = []
    for i in tqdm(range(frames)):
        ori = draw_pose(pose_pos[i], pose_id[i], face[i], hands[i], body_score[i], face_score[i], hands_score[i], h, w)
        ori = Image.fromarray(ori)
        oris.append(ori)
        if i % 60 == 0:  # Adjust this value based on your system's memory capacity
            gc.collect()
            
    # save_videos_from_pil(hands_map, save_dir + video_name + "_hands_face.mp4", fps=fps)
    save_videos_from_pil(oris, save_dir + video_name + "_weighted_heatmap_diff.mp4", fps=fps)
