import cv2
import numpy as np
import pdb
import glob
import os
from tqdm import tqdm
def extract_human_from_video(mask_dir, buffer_ratio=0.2):
    """
    从输入视频中提取人类部分并保存为新视频。

    :param input_video_path: 输入视频文件的路径
    :param output_video_path: 输出视频文件的路径
    :param buffer_ratio: 边界框的缓冲比例，默认是0.2
    """
    # 读取输入视频
    frames_fp = glob.glob(mask_dir+'/*.png')

    # 创建视频写入对象

    # 初始化边界框的最小和最大坐标
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    width,height=None, None
    # 第一次遍历视频以计算边界框
    for fr_fp in frames_fp:
        mask_frame = cv2.imread(fr_fp)
        if width is None:
            height, width, _ = mask_frame.shape
        # 假设mask_frame是掩模帧，值为1的区域表示人类

        # 找到轮廓
        contours, _ = cv2.findContours(mask_frame[:,:,0].astype(np.uint8) , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 获取最大轮廓
            max_contour = max(contours, key=cv2.contourArea)
            # 计算边界框
            x, y, w, h = cv2.boundingRect(max_contour)
            # 更新边界框坐标
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)


    # 计算带有缓冲区的边界框
    frame_width = max_x - min_x
    frame_height = max_y - min_y

    # 增加缓冲区
    min_x -= frame_width * buffer_ratio
    min_y -= frame_height * buffer_ratio
    max_x += frame_width * buffer_ratio
    max_y += frame_height * buffer_ratio
    min_y = max(min_y,0)
    min_x = max(min_x,0)
    max_x = min(max_x,width)
    max_y = min(max_y,height)
    return [min_x,min_y,max_x,max_y]

# 调用示例
# extract_human_from_video('path_to_your_video.mp4', 'output_video.mp4', buffer_ratio=0.2)
if __name__ == '__main__':
    cams_fp = glob.glob('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/*/*/pha')
    rst = {}
    for cfp in tqdm(cams_fp):
        print(f'process {cfp}')
        save_fp = os.path.dirname(cfp)+'/coor.npy'
        if os.path.exists(save_fp):
            continue
        coor = extract_human_from_video(cfp)
        np.save(save_fp, np.array(coor))