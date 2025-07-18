import cv2
import os
import glob
import pdb
import decord
from tqdm import tqdm
def images_to_video(image_folder, output_video_file, fps=30):
    # 获取文件夹中的所有图像文件
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 确保图像按名称排序
    images.sort()
    
    # 检查是否有图像文件
    if not images:
        print("没有找到图像文件。")
        return
    
    # 读取第一张图像以获取视频的宽度和高度
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    
    # 定义视频编码和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    
    # 遍历所有图像并写入视频
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)  # 写入视频帧
    
    # 释放视频写入对象
    video.release()
    print(f"视频已保存到 {output_video_file}")

if __name__ == "__main__":
    
    mask_fps = glob.glob('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/*/*/pha')
    # images_to_video('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/1080_Kungfu_Fan_Single_m12/cam_58/images','/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/1080_Kungfu_Fan_Single_m12/cam_58/video_clip.mp4',25)
    for mfp in tqdm(mask_fps):
        dir_fp = os.path.dirname(mfp)
        vid_fp = os.path.join(dir_fp,'video_clip.mp4')
        try:
            fps = decord.VideoReader(vid_fp).get_avg_fps()
        except Exception as e:
            print(vid_fp)
            continue
        save_fp = os.path.join(dir_fp,'mask_clip.mp4')
        if os.path.exists(save_fp):
            continue
        os.system(f"ffmpeg -framerate 25 -i {mfp}/%06d.png -c:v libx264 -pix_fmt yuv420p {save_fp}")

        # images_to_video(mfp,save_fp,fps)
    