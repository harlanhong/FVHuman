import cv2
import numpy as np
import glob
import pdb
import torch
from tqdm import tqdm
import torchvision.transforms as T
from decord import VideoReader
import imageio
from moviepy.editor import AudioFileClip, VideoClip
import os
import argparse
# from utils.util import numpy_to_video
def numpy_to_video(tensor, output_video_file, audio_source=None, fps=25):
    """
    Converts a Tensor with shape [c, f, h, w] into a video and adds an audio track from the specified audio file.

    Args:
        tensor (Tensor): The Tensor to be converted, shaped [c, f, h, w].
        output_video_file (str): The file path where the output video will be saved.
        audio_source (str): The path to the audio file (WAV file) that contains the audio track to be added.
        fps (int): The frame rate of the output video. Default is 25 fps.
    """
    tensor = np.clip(tensor, 0, 255).astype(
        np.uint8
    )  # to [0, 255]

    def make_frame(t):
        # get index
        frame_index = min(int(t * fps), tensor.shape[0] - 1)
        return tensor[frame_index]
    new_video_clip = VideoClip(make_frame, duration=tensor.shape[0] / fps)
    if audio_source is not None:
        audio_clip = AudioFileClip(audio_source).subclip(0, tensor.shape[0] / fps)
        new_video_clip = new_video_clip.set_audio(audio_clip)
    new_video_clip.write_videofile(output_video_file, fps=fps, audio_codec='aac')

def crop_face_region_with_bbox(bbox: np.array, frames: np.array) -> np.array:
    frames = torch.from_numpy(frames / 255.0).permute(0, 3, 1, 2).float()

    dets = {}
    bbox_union = np.array([
        np.min(bbox[:, 0], axis=0),
        np.min(bbox[:, 1], axis=0),
        np.max(bbox[:, 2], axis=0),
        np.max(bbox[:, 3], axis=0)]
    )
    
    dets["s"] = np.max(np.stack([(bbox_union[3] - bbox_union[1]), (bbox_union[2] - bbox_union[0])]), axis=0) // 2
    dets["y"] = (bbox_union[1] + bbox_union[3]) // 2  # crop center x
    dets["x"] = (bbox_union[0] + bbox_union[2]) // 2  # crop center y

    bs = int((dets["s"] * 1.05))  # Detection box size -- Adding a extra 5% buffer
    face_det_scale=0.25
    face_det_factor = int(bs * face_det_scale)
    bsi = int(bs + 2 * face_det_factor)  # Pad videos by this amount

    frames = T.Pad(bsi, fill=0, padding_mode='constant')(frames)
    my = int(dets["y"] + bsi)  # BBox center Y
    mx = int(dets["x"] + bsi)  # BBox center X

    # # symmetrical padding
    # y1 = my - bs - face_det_factor
    # y2 = my + bs + face_det_factor
    # x1 = mx - bs - face_det_factor
    # x2 = mx + bs + face_det_factor

    y1 = my - bs
    y2 = my + bs + 2 * face_det_factor
    x1 = mx - bs - face_det_factor
    x2 = mx + bs + face_det_factor

    faces = frames[
        :, :,
        int(y1) : int(y2),
        int(x1) : int(x2),
    ]

    bbox = bbox + bsi
    bbox = bbox - np.array([x1, y1, x1, y1])

    bbox = np.round(bbox).astype(int)
    faces = (faces.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return faces, bbox
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
    )
    parser.add_argument("-p", "--parallelism", default=1,
                        type=int, help="Level of parallelism")
    parser.add_argument("-r", "--rank", default=0, type=int,
                        help="Rank for distributed processing")

    args = parser.parse_args()

    videos = glob.glob('/mnt/localssd/ted_dataset/final_ted_v2/*/*.mp4')
    allocated_fps = [videos[i] for i in range(len(videos)) if i % args.parallelism == args.rank]
    
    for vid in tqdm(allocated_fps):
        try:
            bbox_video_path = vid.replace('final_ted_v2','final_ted_v2_bbox').replace('mp4','npy')
            bboxes = np.load(bbox_video_path,allow_pickle=True).squeeze(1)
            bboxes = bboxes[5:-5]
            vr = VideoReader(vid)
            # 获取视频的第一个帧（只为了获得尺寸信息）
            frame = vr[0]
            
            # 获取帧的尺寸
            height, width, _ = frame.shape
            crop_vid, new_bbox = crop_face_region_with_bbox(bboxes,vr[5:-5].asnumpy())
            new_vid_fp = bbox_video_path.replace('ted_dataset','ted_dataset2')
            os.makedirs(os.path.dirname(new_vid_fp),exist_ok=True)
            new_vid_fp = vid.replace('ted_dataset','ted_dataset2')
            os.makedirs(os.path.dirname(new_vid_fp),exist_ok=True)
            # imageio.mimsave(new_vid_fp, crop_vid, fps=25, codec='libx264', macro_block_size=None)
            numpy_to_video(crop_vid,new_vid_fp)
            
            dwpose_vid = vid.replace('final_ted_v2','final_ted_v2_dwpose')
            vr = VideoReader(dwpose_vid,width=width, height=height)
            crop_vid,_ = crop_face_region_with_bbox(bboxes,vr[5:-5].asnumpy())
            new_vid_fp = dwpose_vid.replace('ted_dataset','ted_dataset2')
            os.makedirs(os.path.dirname(new_vid_fp),exist_ok=True)
            numpy_to_video(crop_vid,new_vid_fp)
            
            mask_vid = vid.replace('final_ted_v2','final_ted_v2_mask2')
            vr = VideoReader(mask_vid)
            crop_vid,_ = crop_face_region_with_bbox(bboxes,vr[5:-5].asnumpy())
            new_vid_fp = mask_vid.replace('ted_dataset','ted_dataset2')
            os.makedirs(os.path.dirname(new_vid_fp),exist_ok=True)
            numpy_to_video(crop_vid,new_vid_fp)
        except Exception as e:
            print(e)