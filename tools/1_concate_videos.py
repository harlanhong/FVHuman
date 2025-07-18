import os
import cv2
from pathlib import Path
import numpy as np
import pdb
def concat_videos(input_folders, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取第一个文件夹中所有视频文件名
    video_names = []
    for video_file in os.listdir(input_folders[0]):
        if video_file.endswith('.mp4'):
            video_names.append(video_file)
    w,h = 512,512
    # 对每个视频文件名进行处理
    for video_name in video_names:
        video_paths = []
        # 检查所有输入文件夹中是否都有该视频
        for folder in input_folders:
            video_path = os.path.join(folder, video_name)
            if os.path.exists(video_path):
                video_paths.append(video_path)
            else:
                print(f"警告: 在文件夹 {folder} 中未找到视频 {video_name}")
                continue
        if len(video_paths) != len(input_folders):
            print(f"警告: {video_name} 在部分文件夹中未找到,跳过处理")
            continue
            
        # 读取所有视频
        caps = [cv2.VideoCapture(vp) for vp in video_paths]
        
        # 获取第一个视频的属性
        fps = int(caps[0].get(cv2.CAP_PROP_FPS))
        frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # 创建输出视频写入器
        output_path = os.path.join(output_folder, video_name)
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, 
                            (w * len(video_paths), h))
        
        # 逐帧拼接
        for _ in range(frame_count):
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(frame, (w, h)))
                
            if len(frames) == len(caps):
                # 水平拼接所有帧
                concat_frame = np.hstack(frames)
                out.write(concat_frame)
            else:
                break
                
        # 释放资源
        out.release()
        for cap in caps:
            cap.release()
            
        print(f"已保存拼接视频: {output_path}")

if __name__ == "__main__":
    # 示例用法
    input_folders = [
        "ablation_result/baseline_2ref_ted_full_crop",
        "ablation_result/baseline_2ref_ted_full_2ref_crop",
        "ablation_result/baseline_2ref_ted_full_4ref_crop",
        "ablation_result/baseline_2ref_ted_full_5ref_crop",
        "/apdcephfs_cq8/share_1367250/harlanhong/src/human_cmp/ted_test_set/gt"
    ]
    output_folder = "ablation_result/ted_mref_concat"
    
    concat_videos(input_folders, output_folder)
