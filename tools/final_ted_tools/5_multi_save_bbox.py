import subprocess
import os
# 定义命令模板
command_template = 'bash -c "source /opt/venv/bin/activate && ' \
                   'cd /home/fhong/sensei-fs-link/src/pose2video/tools/ted_tools && ' \
                   'CUDA_VISIBLE_DEVICES={} python 5_save_bbox.py ' \
                   '--num_workers {} --job_idx {}"'

# 创建命令列表
workers = 32
commands = [command_template.format(i % 2, workers, i) for i in range(workers)]
# 创建进程列表并执行命令
processes = [subprocess.Popen(command, shell=True) for command in commands]

# 等待所有进程完成
for process in processes:
    process.wait()

os.system('flash_s3_upload --local-dir /home/fhong/final_ted_v2_bbox/ --s3-url s3://zhanxu-public/fating/')