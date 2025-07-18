import subprocess

# 定义命令模板
command_template = 'bash -c "source /opt/conda/bin/activate py310 && ' \
                   'cd /sensei-fs/users/haiyangl/pose2video/ && ' \
                   'CUDA_VISIBLE_DEVICES={} python tools/extract_dwpose_from_vid.py ' \
                   '--num_workers {} --job_idx {}"'

# 创建命令列表
workers = 1
commands = [command_template.format(i % 1, workers, i) for i in range(workers)]

# 创建进程列表并执行命令
processes = [subprocess.Popen(command, shell=True) for command in commands]

# 等待所有进程完成
for process in processes:
    process.wait()