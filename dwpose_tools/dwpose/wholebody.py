# https://github.com/IDEA-Research/DWPose
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .onnxdet import inference_detector
from .onnxpose import inference_pose

ModelDataPathPrefix = Path("/mnt/localssd/pretrained_weights")


class Wholebody:
    def __init__(self, device="cuda:0"):
        device_id = int(device.split(":")[1]) if "cuda" in device else None

        if device_id is not None:
            providers = ["CUDAExecutionProvider"]
            provider_options = [{"device_id": device_id}]
        else:
            providers = ["CPUExecutionProvider"]
            provider_options = [{}] 
            
        onnx_det = Path("/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/dwpose_tools/yolox_l.onnx")
        onnx_pose = Path("/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/dwpose_tools/dw-ll_ucoco_384.onnx") 

        self.session_det = ort.InferenceSession(
            path_or_bytes=onnx_det, providers=providers, provider_options=provider_options
        )
        self.session_pose = ort.InferenceSession(
            path_or_bytes=onnx_pose, providers=providers, provider_options=provider_options
        )

    def __call__(self, oriImg):
        det_result = inference_detector(self.session_det, oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)
        # print(keypoints.shape, scores.shape)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3
        ).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
        # print(keypoints.shape, scores.shape)
        return keypoints, scores
