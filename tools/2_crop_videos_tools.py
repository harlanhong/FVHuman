import os
import subprocess
import pdb
import cv2
import decord
def crop_video_horizontally_part(input_video, output_path, part_index, total_parts):
    """将视频水平切分并保存第i部分
    
    Args:
        input_video: 输入视频路径
        output_path: 输出视频路径
        part_index: 需要截取的部分索引(从0开始)
        total_parts: 总共划分的份数
    """
    # 检查参数有效性
    if part_index >= total_parts:
        raise ValueError(f"part_index ({part_index}) 必须小于 total_parts ({total_parts})")
        
    # 使用ffprobe获取视频信息
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {input_video}'
    output = subprocess.check_output(cmd, shell=True).decode()
    width, height = map(int, output.strip().split('x'))
    
    # 计算每一部分的宽度和起始位置
    part_width = width // total_parts
    x_offset = part_index * part_width
    
    # 创建输出文件夹(如果需要)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 裁剪指定部分
    cmd = (f'ffmpeg -i {input_video} -filter:v '
           f'"crop={part_width}:{height}:{x_offset}:0" '
           f'-c:v libx264  {output_path} -y')
    
    subprocess.call(cmd, shell=True)
    print(f'已保存第{part_index+1}部分视频: {output_path}')

def split_video_horizontally_part(input_video, output_folder, total_parts):
    """将视频水平切分并保存第i部分
    
    Args:
        input_video: 输入视频路径
        output_path: 输出视频路径
        part_index: 需要截取的部分索引(从0开始)
        total_parts: 总共划分的份数
    """
    # 检查参数有效性
    
    # 使用ffprobe获取视频信息
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 {input_video}'
    output = subprocess.check_output(cmd, shell=True).decode()
    width, height = map(int, output.strip().split('x'))
    
    # 计算每一部分的宽度和起始位置
    part_width = width // total_parts
    os.makedirs(output_folder, exist_ok=True)
    for i in range(total_parts):
        output_path = os.path.join(output_folder, f'{i}.mp4')
        x_offset = i * part_width

        # 裁剪指定部分
        cmd = (f'ffmpeg -i {input_video} -filter:v '
            f'"crop={part_width}:{height}:{x_offset}:0" '
            f'-c:v libx264  {output_path} -y')
        
        subprocess.call(cmd, shell=True)
        print(f'已保存第{i+1}部分视频: {output_path}')

def crop_video_with_position(input_video, output_path, x, y, width, height):
    """根据指定位置和大小裁剪视频
    
    Args:
        input_video: 输入视频路径
        output_path: 输出视频路径
        x: 裁剪起始x坐标
        y: 裁剪起始y坐标  
        width: 裁剪宽度
        height: 裁剪高度
    """
    # 创建输出文件夹(如果需要)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 使用ffmpeg裁剪视频
    cmd = (f'ffmpeg -i {input_video} -filter:v '
           f'"crop={width}:{height}:{x}:{y}" '
           f'-c:v libx264 {output_path} -y')
    
    subprocess.call(cmd, shell=True)
    print(f'已保存裁剪后的视频: {output_path}')

def process_folder_with_position(input_folder, output_folder, width, height):
    """处理视频列表，并根据指定位置和大小裁剪视频
    
    Args:
        video_path_list: 输入视频路径列表
        output_path: 输出视频路径
        width: 裁剪宽度
        height: 裁剪高度
    """
     
    # 遍历输入文件夹中的所有视频文件
    gt_folder = '/apdcephfs_cq8/share_1367250/harlanhong/src/human_cmp/ted_test_set/ref'
    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4'):
            gt_path = os.path.join(gt_folder, video_file.replace('.mp4', '.jpg'))
            gt_frame = cv2.imread(gt_path)
            width, height = gt_frame.shape[1], gt_frame.shape[0]
            input_path = os.path.join(input_folder, video_file)
            input_vr = decord.VideoReader(input_path)
            input_frame = input_vr[0].asnumpy()
            input_width, input_height = input_frame.shape[1], input_frame.shape[0]
            pdb.set_trace()
            output_path = os.path.join(output_folder, video_file)
            x = input_width - 2*width
            y=0
            crop_video_with_position(input_path, output_path, x, y, width, height)

def process_folder(input_folder, output_folder, part_index, total_parts):
    """处理文件夹中的所有视频
    
    Args:
        input_folder: 输入视频文件夹路径
        output_folder: 输出视频文件夹路径 
        part_index: 需要截取的部分索引(从0开始)
        total_parts: 总共划分的份数
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有视频文件
    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4'):
            input_path = os.path.join(input_folder, video_file)
            output_path = os.path.join(output_folder, video_file)
            
            try:
                # 处理每个视频
                crop_video_horizontally_part(input_path, output_path, part_index, total_parts)
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {str(e)}")
                continue
                
    print(f"所有视频处理完成,已保存至: {output_folder}")



if __name__ == "__main__":
    # ablation
    # split_video_horizontally_part('ablation_result/ted_mref_concat/7.mp4', 'ablation_result/ted_mref_concat_split/7', 5)
    # split_video_horizontally_part('ablation_result/ted_mref_concat/36.mp4', 'ablation_result/ted_mref_concat_split/36', 5)
    split_video_horizontally_part('ablation_result/demo_all/000440-1.mp4', 'ablation_result/ted_mref_concat_split/000440-1', 5)
    
    # crop_video_horizontally_part('/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/user_test/rst.mp4', 'user_test/rst_crop.mp4', 4, 5)
    # process_folder('ablation_result/baseline_1ref', 'ablation_result/baseline_1ref_crop', 2, 4)
    # process_folder('ablation_result/baseline_2ref', 'ablation_result/baseline_2ref_crop', 3, 5)
    # process_folder_with_position('ablation_result/baseline_2ref_attn', 'ablation_result/baseline_2ref_attn_crop', 3, 5)
    # process_folder('ablation_result/baseline_2ref_attn', 'ablation_result/baseline_2ref_attn_crop', 3, 5)
    
    # process_folder('ablation_result/baseline_2ref_ted_full_10ref', 'ablation_result/baseline_2ref_ted_full_10ref_pose', 10, 13)
    # process_folder('ablation_result/baseline_2ref_ted_full', 'ablation_result/baseline_2ref_ted_full_crop', 2, 4)
    # process_folder('ablation_result/baseline_2ref_ted_full', 'ablation_result/baseline_2ref_ted_full_pose', 1, 4)
    # process_folder('ablation_result/baseline_dyhuman_full_1ref', 'ablation_result/baseline_dyhuman_full_1ref_pose', 1, 4)
    # process_folder('ablation_result/baseline_dyhuman_full_1ref', 'ablation_result/baseline_dyhuman_full_1ref_crop', 2, 4)
    # process_folder('ablation_result/baseline_dyhuman_full_2ref', 'ablation_result/baseline_dyhuman_full_2ref_pose', 2, 5)
    # process_folder('ablation_result/baseline_dyhuman_full_2ref', 'ablation_result/baseline_dyhuman_full_2ref_crop', 3, 5)
    # process_folder('ablation_result/baseline_dyhuman_full_5ref', 'ablation_result/baseline_dyhuman_full_5ref_pose', 5, 8)
    # process_folder('ablation_result/baseline_dyhuman_full_5ref', 'ablation_result/baseline_dyhuman_full_5ref_crop', 6, 8)
    # process_folder('ablation_result/baseline_dyhuman_full_10ref', 'ablation_result/baseline_dyhuman_full_10ref_pose', 10, 13)
    # process_folder('ablation_result/baseline_dyhuman_full_10ref', 'ablation_result/baseline_dyhuman_full_10ref_crop', 11, 13)
    # process_folder('ablation_result/baseline_2ref_ted_full_5ref', 'ablation_result/baseline_2ref_ted_full_5ref_crop', 6, 8)
    
    # process_folder('ablation_result/baseline_2ref_ted_full_3ref', 'ablation_result/baseline_2ref_ted_full_3ref_pose', 3, 6)
    # process_folder('ablation_result/baseline_2ref_ted_full_3ref', 'ablation_result/baseline_2ref_ted_full_3ref_crop', 4, 6)
    # process_folder('ablation_result/baseline_2ref_ted_full_4ref', 'ablation_result/baseline_2ref_ted_full_4ref_pose', 4, 7)
    # process_folder('ablation_result/baseline_2ref_ted_full_4ref', 'ablation_result/baseline_2ref_ted_full_4ref_crop', 5, 7)
    # process_folder('ablation_result/baseline_2ref_ted_full_2ref', 'ablation_result/baseline_2ref_ted_full_2ref_pose', 2, 5)
    # process_folder('ablation_result/baseline_2ref_ted_full_2ref', 'ablation_result/baseline_2ref_ted_full_2ref_crop', 3, 5)
    
    
    # process_folder('ablation_result/baseline_2ref_ted_full_5ref', 'ablation_result/baseline_2ref_ted_full_5ref_pose', 5, 8)
    # process_folder('ablation_result/baseline_2ref_ted_full_5ref', 'ablation_result/baseline_2ref_ted_full_5ref_crop', 6, 8)
    
    # process_folder('ablation_result/baseline_2ref_ted_full_10ref', 'ablation_result/baseline_2ref_ted_full_10ref_crop', 11, 13)

