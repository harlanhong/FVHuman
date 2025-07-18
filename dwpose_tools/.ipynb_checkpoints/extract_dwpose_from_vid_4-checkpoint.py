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


def draw_pose_limb_mask(pose_pos, pose_id, face, hands, body_score, face_score, hands_score, H, W):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    mask = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    mask = draw_connected_limbs(mask, pose_pos, pose_id, body_score)
    canvas = HWC3(mask)
    return canvas


def draw_connected_limbs(mask, candidate, subset, body_score, thickness=16):
    H, W, _ = mask.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    
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

    for i, limb in enumerate(limbSeq):
        for n in range(len(subset)):
            index = subset[n][np.array(limb) - 1]
            if -1 in index:
                continue
            point1 = candidate[index[0].astype(int)]
            point2 = candidate[index[1].astype(int)]
            x1, y1, score1 = point1[0], point1[1], body_score[limb[0] - 1]
            x2, y2, score2 = point2[0], point2[1], body_score[limb[1] - 1]

            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)
            avg_score = (score1 + score2) / 2

            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(mask, (x1, y1), (x2, y2), (int(avg_score * 255), int(avg_score * 255), int(avg_score * 255)), thickness)
    
    return mask
    

def draw_pose_mask(pose_pos, pose_id, face, hands, body_score, face_score, hands_score, H, W):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    mask = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # mask2 = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # print(pose_pos.shape, pose_id.shape, face.shape, hands.shape)
    
    mask = draw_handpose_mask(mask, hands, hands_score)
    mask = draw_facepose_mask(mask, face, face_score)
    mask2 = mask.copy()
    mask = draw_bodypose_mask(mask, pose_pos, pose_id, body_score)
    canvas = HWC3(mask)
    return mask, mask2


def draw_bodypose_mask(mask, candidate, subset, body_score):
    H, W, _ = mask.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            score = body_score[i]  # Assuming body_score is an array of scores
            # print(x,y,W,H, candidate.shape, subset.shape)
            x = int(x * W)
            y = int(y * H)
            color = int(score * 255)
            cv2.circle(mask, (int(x), int(y)), 16, (color, color, color), thickness=-1)  # Set thickness to 1 instead of -1  
    return mask


def draw_handpose_mask(mask, all_hand_peaks, hands_score):
    H, W, _ = mask.shape
    
    for peaks, score_list in zip(all_hand_peaks, hands_score):
        peaks = np.array(peaks)
        
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            score = score_list[i]  # Assuming hands_score is a list of score lists
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                color = int(score * 255)
                cv2.circle(mask, (x, y), 16, (color, color, color), thickness=-1)  # Set thickness to 1 instead of -1
    return mask


def draw_facepose_mask(mask, all_lmks, face_score):
    H, W, _ = mask.shape
    for lmks, score_list in zip(all_lmks, face_score):
        lmks = np.array(lmks)
        for i, lmk in enumerate(lmks):
            x, y = lmk
            score = score_list[i]  # Assuming face_score is a list of score lists
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                color = int(score * 255)
                cv2.circle(mask, (x, y), 16, (color, color, color), thickness=-1)  # Set thickness to 1 instead of -1
    return mask


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
    body_score = whole_score[:, :18]
    face_score = whole_score[:, 24:92].reshape(-1, 1, 68)
    hands_score = whole_score[:, 92:].reshape(-1, 2, 21)
    
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
        _, hand_map = draw_pose_mask(pose_pos[i], pose_id[i], face[i], hands[i], body_score[i], face_score[i], hands_score[i], h, w)
        limb = draw_pose_limb_mask(pose_pos[i], pose_id[i], face[i], hands[i], body_score[i], face_score[i], hands_score[i], h, w)
        ori = draw_pose(pose_pos[i], pose_id[i], face[i], hands[i], body_score[i], face_score[i], hands_score[i], h, w)
        # heatmap = Image.fromarray(heatmap)
        # limb = Image.fromarray(limb)
        ori = Image.fromarray(ori)
        hand_map = Image.fromarray(hand_map)
        # heatmaps.append(heatmap)
        hands_map.append(hand_map)
        # limbs.append(limb)
        oris.append(ori)
        # Clear memory
        if i % 60 == 0:  # Adjust this value based on your system's memory capacity
            gc.collect()
            
    # save_videos_from_pil(heatmaps, save_dir + video_name + "_heatmap.mp4", fps=fps)
    # save_videos_from_pil(limbs, save_dir + video_name + "_limb.mp4", fps=fps)
    save_videos_from_pil(hands_map, save_dir + video_name + "_hands_face.mp4", fps=fps)
    save_videos_from_pil(oris, save_dir + video_name + "_weighted_heatmap.mp4", fps=fps)
    os.system(f'aws s3 cp {save_dir + video_name + "_hands_face.mp4"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_hands_face.mp4"}')
    os.system(f'aws s3 cp {save_dir + video_name + "_weighted_heatmap.mp4"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_weighted_heatmap.mp4"}')


def process_single_video(
    video_name='videoxxx', 
    detector='detector', 
    root_dir="/path/to/video/", 
    save_dir="/path/to/video/save/", 
    finetune_data=False
):
    print("here")
    video_path = root_dir + video_name + ".mp4"
    video_reader = VideoReader(video_path)
    video_length = len(video_reader)
    print(f"video_length: {video_length}")
    fps = round(video_reader.get_avg_fps())
    print(f"FPS: {fps}")

    kps_results = []
    pose_pos = []
    pose_id = []
    pose_score = []
    face_pos = []
    hands_pos = []

    for i in range(video_length):
        frame_pil = Image.fromarray(video_reader[i].asnumpy())
        result, score, pose = detector(frame_pil, detect_resolution=512, image_resolution=512)
        kps_results.append(result)
        pose_pos.append(pose["bodies"]["candidate"])
        pose_id.append(pose["bodies"]["subset"])
        pose_score.append(score[0])
        face_pos.append(pose["faces"])
        hands_pos.append(pose["hands"])
        if i % 60 == 0:  # Adjust this value based on your system's memory capacity
            gc.collect()

    pose_pos = np.array(pose_pos) # 300*18*2
    pose_id = np.array(pose_id) #300*18
    pose_score = np.array(pose_score)
    hands_pos = np.array(hands_pos)
    face_pos = np.array(face_pos)

    np.save(save_dir + video_name + "_pose_score.npy", pose_score)
    np.save(save_dir + video_name + "_pose_pos.npy", pose_pos)
    np.save(save_dir + video_name + "_pose_id.npy", pose_id)
    np.save(save_dir + video_name + "_face_pos.npy", face_pos)
    np.save(save_dir + video_name + "_hands_pos.npy", hands_pos)
    save_videos_from_pil(kps_results, save_dir + video_name + ".mp4", fps=fps)

    os.system(f'aws s3 cp {save_dir + video_name + "_pose_score.npy"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_pose_score.npy"}')
    os.system(f'aws s3 cp {save_dir + video_name + "_pose_pos.npy"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_pose_pos.npy"}')
    os.system(f'aws s3 cp {save_dir + video_name + "_pose_id.npy"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_pose_id.npy"}')
    os.system(f'aws s3 cp {save_dir + video_name + "_face_pos.npy"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_face_pos.npy"}')
    os.system(f'aws s3 cp {save_dir + video_name + "_hands_pos.npy"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + "_hands_pos.npy"}')
    os.system(f'aws s3 cp {save_dir + video_name + ".mp4"} {save_dir.replace("/mnt/localssd/", "s3://zhanxu-public/haiyangl/") + video_name + ".mp4"}')
    gc.collect()
    if finetune_data:
        process_video(save_dir, video_name)

def process_video_with_gpu(video_name, video_dir, save_dir, device_id):
    # print(device_id)
    torch.cuda.set_device(device_id)
    # print("sb2")
    detector = DWposeDetector().to(f"cuda:{device_id}")
    # print("sb")
    process_single_video(video_name, detector, video_dir, save_dir, True)
    # del detector

def update_progress_bar(*args):
    progress_bar.update()
    
if __name__ == "__main__":
    import argparse
    from dwpose import DWposeDetector
    # from concurrent.futures import ThreadPoolExecutor, as_completed
    from multiprocessing import Pool, cpu_count
    
    parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="/mnt/localssd/motionx/video/fitness/")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument(
        "--save_dir", type=str, default="/mnt/localssd/motionx/video/fitness/_dwpose/"
    )
    parser.add_argument(
        "--max_workers", type=int, default=32, help="Maximum number of threads to use."
    )
    
    args = parser.parse_args()  
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
 
    todo_video = []
    existing_video = os.listdir(args.save_dir)
    # print(existing_video)
    for video_name in os.listdir(args.video_dir):
        if video_name.endswith(".mp4"):
            if args.resume:
                if video_name.replace(".mp4", "_weighted_heatmap.mp4") in existing_video:
                    continue
            todo_video.append(video_name[:-4])
    
    print("lens: ", len(todo_video))
    
    num_gpus = torch.cuda.device_count()
    with tqdm(total=len(todo_video), desc="Processing videos") as progress_bar:
        with Pool(processes=args.max_workers) as pool:
            for i, video_name in enumerate(todo_video):
                device_id = i % num_gpus
                pool.apply_async(
                    process_video_with_gpu, 
                    (video_name, args.video_dir, args.save_dir, device_id), 
                    callback=update_progress_bar
                )
            pool.close()
            pool.join()  