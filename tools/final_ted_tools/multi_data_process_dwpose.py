import subprocess
import argparse
    
parser = argparse.ArgumentParser(
    description="Process videos to prepare data for training. Run this script twice with different GPU status parameters."
)

args = parser.parse_args()
# 定义命令模板
command_template = 'bash -c "source /opt/venv/bin/activate && ' \
                   'cd /sensei-fs/users/fhong/src/pose2video/ && ' \
                   'python tools/final_ted_tools/draw_dwpose_from_vid.py -p {} -r {}"'

# 创建命令列表

workers = 16
commands = [command_template.format(workers, i) for i in range(workers)]

# 创建进程列表并执行命令
processes = [subprocess.Popen(command, shell=True) for command in commands]

# 等待所有进程完成
for process in processes:
    process.wait()
