import cv2
import numpy as np
import pdb
def extract_human_from_video(mask_fp,crop_video_list, output_video_path, buffer_ratio=0.2):
    """
    从输入视频中提取人类部分并保存为新视频。

    :param input_video_path: 输入视频文件的路径
    :param output_video_path: 输出视频文件的路径
    :param buffer_ratio: 边界框的缓冲比例，默认是0.2
    """
    # 读取输入视频
    cap = cv2.VideoCapture(mask_video)

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编码方式

    # 创建视频写入对象

    # 初始化边界框的最小和最大坐标
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # 第一次遍历视频以计算边界框
    while cap.isOpened():
        ret, mask_frame = cap.read()
        if not ret:
            break
        
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

    cap.release()

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
    
    # 重新读取视频并裁剪每一帧
    for crop_video in crop_video_list:
        cap = cv2.VideoCapture(crop_video)
        out = cv2.VideoWriter(crop_video.replace(), fourcc, fps,  (int(max_x - min_x), int(max_y - min_y)))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 裁剪帧
            cropped_frame = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
            
            # 确保裁剪区域在帧的边界内
            cropped_frame = cv2.resize(cropped_frame, (int(max_x - min_x), int(max_y - min_y)))

            # 写入新视频
            out.write(cropped_frame)

        cap.release()
        out.release()

    print("新视频已保存:", output_video_path)

# 调用示例
# extract_human_from_video('path_to_your_video.mp4', 'output_video.mp4', buffer_ratio=0.2)
if __name__ == '__main__':
    
extract_human_from_video('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/1080_Dance_Jazz_Single_c22/cam_2/mask_clip.mp4','tmp.mp4')
