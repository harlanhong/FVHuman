import random
import json
import os
import glob
import decord
import imageio
def generate_samples(root,ids):
    samples = []
    for id in ids:
        vids_fp = os.path.join(root,'videos',id)
        kps_fp = os.path.join(root,'dwpose',id)
        masks_fp = os.path.join(root,'masks',id)
        vids = glob.glob(vids_fp+'/*.mp4')
        print(vids)
        selected_videos = random.choices(vids, k=3)       
        target = selected_videos[0]
        ref_videos = [selected_videos[j] for j in range(len(selected_videos)) if j != 0]
        
        ref = []
        for ref_video in ref_videos:
            # 使用 decord 打开视频
            vr = decord.VideoReader(ref_video)
            # 获取帧数
            frame_count = len(vr)
            ref.append({
                "video_path": ref_video,
                "kps_path": ref_video.replace('videos','dwpose').replace('.mp4','_weighted_heatmap.mp4'),
                "mask_path": ref_video.replace('videos','masks'),
                "idx":  random.sample(list(range(frame_count)), 3)
            })
        
        # 抽取 target 的 idx
        # 使用 decord 打开视频
        vr = decord.VideoReader(target)
            # 获取帧数
        frame_count = len(vr)
        sample = {
            "target": {
                "video_path": target,
                "kps_path": target.replace('videos','dwpose').replace('.mp4','_weighted_heatmap.mp4'),
                "mask_path": target.replace('videos','masks'),
                "idx": random.sample(list(range(frame_count)), 3)
            },
            "ref": ref
        }
        
        samples.append(sample)
    
    return samples

# 生成样本
samples = generate_samples('/projects/p32321/fating/dataset/final_ted_mini_set',['000645','000548','000620'])

# 转换为 JSON 格式并打印
json_output = json.dumps(samples, indent=4)
print(json_output)

# 保存到文件
with open("/projects/p32296/fating/src/novel-view-human-master/data/mini_tests.json", "w") as f:
    f.write(json_output)
