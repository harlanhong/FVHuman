import subprocess

# 定义命令模板
command_template = 'bash -c "source /opt/venv/bin/activate && ' \
                   'cd /sensei-fs/users/fhong/src/pose2video && ' \
                   'CUDA_VISIBLE_DEVICES={} python tools/extract_mask_from_vid.py -i /mnt/localssd/final_ted_v2 -p {} -r {} && flash_s3_upload --local_dir /mnt/localssd/final_ted_v2_mask/ --s3_url s3://zhanxu-public/fating/"'

# 创建命令列表
workers = 32
commands = [command_template.format(i % 8, workers, i) for i in range(workers)]

# 创建进程列表并执行命令
processes = [subprocess.Popen(command, shell=True) for command in commands]

# 等待所有进程完成
for process in processes:
    process.wait()
