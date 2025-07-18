# no bbox crop
import json
import random
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import numpy as np
import cv2
import torchvision.transforms.functional as F


class PairedRandomMaxSquareCropResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2, img3):
        width, height = img1.size
        img3 = transforms.functional.resize(img3,(height, width))
        min_dim = min(width, height)
        if width > height:
            x = random.randint(0, width - min_dim)
            y = 0
        else:
            x = 0
            y = random.randint(0, height - min_dim)
        
        # Crop both images to a square of size min_dim x min_dim
        img1 = img1.crop((x, y, x + min_dim, y + min_dim))
        img2 = img2.crop((x, y, x + min_dim, y + min_dim))
        img3 = img3.crop((x, y, x + min_dim, y + min_dim))
        # print("3", img1.size, img2.size, img3.size)
        
        img1 = transforms.functional.resize(img1, self.size)
        img2 = transforms.functional.resize(img2, self.size)
        img3 = transforms.functional.resize(img3, self.size)
        # print("4", img1.size, img2.size, img3.size)
        return img1, img2, img3
        

class ImageDataset(Dataset):
    def __init__(
        self,
        img_size,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
        aug_type="Resize",
        mode="train",
    ):
        super().__init__()

        self.img_size = img_size
        self.img_size_single = img_size[0]
        self.sample_margin = sample_margin
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        # self.vid_meta = vid_meta
        self.vid_meta = [item for item in vid_meta if item.get("mode") == mode]
        self.clip_image_processor = CLIPImageProcessor()
        self.aug_type = aug_type
        if self.aug_type == "Resize":
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size_single, self.img_size_single)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.cond_transform = transforms.Compose([
                transforms.Resize((self.img_size_single, self.img_size_single)),
                transforms.ToTensor(),
            ])
        
        elif self.aug_type == "Padding":
            self.transform = transforms.Compose([
                transforms.Lambda(self.resize_long_edge),
                transforms.Lambda(self.padding_short_edge),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.cond_transform = transforms.Compose([
                transforms.Lambda(self.resize_long_edge),
                transforms.Lambda(self.padding_short_edge),
                transforms.ToTensor(),
            ])

        elif self.aug_type == "RandomResizeCrop": # center crop
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(self.img_size_single, self.img_size_single), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.cond_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(self.img_size_single, self.img_size_single), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        elif self.aug_type == "RandomCrop":
            self.paired_transform = PairedRandomMaxSquareCropResize((self.img_size_single, self.img_size_single))
            self.transform = transforms.Compose(
                [   
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.cond_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.img_size_single / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.img_size_single, self.img_size_single
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
       
        video_length = len(video_reader)
        if len(video_reader) != len(kps_reader): video_length = min(len(video_reader),len(kps_reader))
        
        margin = min(self.sample_margin, video_length)
        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())
        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())
        
        state = torch.get_rng_state()
        if self.aug_type == "RandomCrop":
            tgt_img_pil, ref_img_pil, tgt_pose_pil = self.paired_transform(tgt_img_pil, ref_img_pil, tgt_pose_pil)
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]
        
        sample = dict(
            # video_dir=video_path,
            tgt_img=tgt_img,
            tgt_guid=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_img=clip_image,
        )
        return sample

    def __len__(self):
        return len(self.vid_meta)