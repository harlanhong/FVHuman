import os
import shutil

# 文件所在的目录
source_dir = "/mnt/localssd/final_ted_v4/_dwpose"
destination_dir = "/mnt/localssd/final_ted_dataset/dwpose/"

# 获取目录下的所有文件
files = os.listdir(source_dir)

for file_name in files:
    if 'face_pos' in file_name or 'hands_face' in file_name or 'hands_pos' in file_name or 'pose_id' in file_name or 'pose_pos' in file_name or 'pose_score' in file_name or 'weighted_heatmap' in file_name:
        continue
    # 提取 ID 号 (假设 ID 号在文件名的第一个下划线之前)
    id_number = file_name.split('_')[0]

    # 创建一个新的文件夹，用于存放相同 ID 的文件
    id_folder = os.path.join(destination_dir, id_number)
    if not os.path.exists(id_folder):
        os.makedirs(id_folder)

    # 移动文件到对应的 ID 文件夹中
    shutil.move(os.path.join(source_dir, file_name), os.path.join(id_folder, file_name))

print("Files have been organized by ID.")
