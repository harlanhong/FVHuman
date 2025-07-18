import os
import random
import shutil

def select_same_samples_across_modalities(parent_dir, dest_dir, num_samples=30):
    """
    从 final_ted_dataset 中的模态文件夹选取相同的样本，生成 mini set。
    
    :param parent_dir: final_ted_dataset 文件夹路径，包含多个模态文件夹
    :param dest_dir: mini set 目标文件夹
    :param num_samples: 选取的样本数量
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 获取 parent_dir 下的所有模态文件夹
    modality_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    # 从第一个模态文件夹中选取样本作为基准
    first_modality_dir = modality_dirs[0]
    all_samples = os.listdir(first_modality_dir)
    
    if len(all_samples) < num_samples:
        print(f"文件夹 {first_modality_dir} 中的样本数不足 {num_samples} 个")
        return
    
    # 随机选取指定数量的样本
    selected_samples = random.sample(all_samples, num_samples)
    
    # 为每个模态文件夹选取相同的样本
    for modality_dir in modality_dirs:
        modality_name = os.path.basename(modality_dir)
        modality_dest_dir = os.path.join(dest_dir, modality_name)
        
        if not os.path.exists(modality_dest_dir):
            os.makedirs(modality_dest_dir)
        
        # 将选取的样本从当前模态文件夹复制到目标文件夹
        for sample in selected_samples:
            source_path = os.path.join(modality_dir, sample)
            dest_path = os.path.join(modality_dest_dir, sample)
            if os.path.exists(source_path):
                os.system(f'cp -r {source_path} {modality_dest_dir}/')
            else:
                print(f"文件 {sample} 在模态 {modality_name} 中不存在")

    print(f"Mini set 已创建在: {dest_dir}")

# 示例用法
parent_dir = '/projects/p32321/fating/dataset/final_ted_dataset_v2'  # 这里是你的 final_ted_dataset 文件夹路径
dest_dir = '/projects/p32321/fating/dataset/final_ted_mini_set'

select_same_samples_across_modalities(parent_dir, dest_dir)
