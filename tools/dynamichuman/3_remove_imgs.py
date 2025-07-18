import os
import glob
from tqdm import tqdm
   
delete_fps = glob.glob('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/*/*/images')
delete_fps+=glob.glob('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/*/*/com')
delete_fps+=glob.glob('/projects/p32321/fating/dataset/PKU-DynamicaHuman/dynamichuman/*/*/pha')

for fps in tqdm(delete_fps):
    print(f'rm -rf {fps}')
    os.system(f'rm -rf {fps}')