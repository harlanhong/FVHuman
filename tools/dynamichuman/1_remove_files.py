#process part1
import glob
import numpy as np
import os
import shutil
# fp = '/projects/p32321/fating/dataset/PKU-DynamicaHuman/part1'
fp = '/projects/p32321/fating/dataset/PKU-DynamicaHuman/part2/amax/zxyun/FreeViewSynthesis/DyMulHumans/datasets/part2'
os.makedirs('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman',exist_ok=True)
folder_list = sorted(glob.glob(fp+'/*'))
for folder in folder_list:
    if 'Single' in folder:
        name = os.path.basename(folder)
        os.makedirs(f'/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/{name}',exist_ok=True)
        taget_folder = f'/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/{name}'
        os.system(f'mv {folder}/per_view/* {taget_folder}/')
        print(folder)