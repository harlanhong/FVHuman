import subprocess

# 定义命令模板
command_template = 'bash -c "source /opt/venv/bin/activate && ' \
                   'cd /home/fhong/sensei-fs-link/src/pose2video/tools/ted_tools && ' \
                   'CUDA_VISIBLE_DEVICES={} python extract_mask_from_vid.py -i /mnt/localssd/final_ted_dataset/videos -p {} -r {}"'

# 创建命令列表
workers = 128
commands = [command_template.format(i % 8, workers, i) for i in range(workers)]

# 创建进程列表并执行命令
processes = [subprocess.Popen(command, shell=True) for command in commands]

# 等待所有进程完成
for process in processes:
    process.wait()
