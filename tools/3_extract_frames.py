import os
import subprocess

def extract_frames(input_folder, output_folder):
    """使用ffmpeg从文件夹中提取所有视频的帧
    
    Args:
        input_folder: 输入视频文件夹路径
        output_folder: 输出帧图像文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹中的所有视频文件
    for video_file in os.listdir(input_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(input_folder, video_file)
            
            # 为每个视频创建对应的输出子文件夹
            video_name = os.path.splitext(video_file)[0]
            frames_folder = os.path.join(output_folder, video_name)
            os.makedirs(frames_folder, exist_ok=True)
            
            try:
                # 使用ffmpeg提取帧
                output_pattern = os.path.join(frames_folder, 'frame_%06d.jpg')
                cmd = f'ffmpeg -i {video_path} -qscale:v 2 {output_pattern} -y'
                
                subprocess.call(cmd, shell=True)
                print(f'已提取视频帧: {video_file}')
                
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {str(e)}")
                continue
                
    print(f"所有视频帧提取完成,已保存至: {output_folder}")

if __name__ == "__main__":
    # input_folder = "ablation_result/baseline_2ref_ted_full_4ref_crop"
    # output_folder = "ablation_result/baseline_2ref_ted_full_4ref_crop_frames" 
    # extract_frames(input_folder, output_folder)
    input_folder = "ablation_result/baseline_2ref_ted_full_2ref_crop"
    output_folder = "ablation_result/baseline_2ref_ted_full_2ref_crop_frames" 
    # extract_frames(input_folder, output_folder)

    # input_folder = "ablation_result/baseline_dyhuman_full_10ref_crop"
    # output_folder = "ablation_result/baseline_dyhuman_full_10ref_crop_frames" 
    extract_frames(input_folder, output_folder)