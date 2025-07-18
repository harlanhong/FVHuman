import subprocess

command_template = 'bash -c "source /opt/conda/bin/activate sb && ' \
                   'cd /sensei-fs/users/haiyangl/pose2video/ && ' \
                   'CUDA_VISIBLE_DEVICES={} python tools/extract_dwpose_from_vid.py ' \
                   '--num_workers {} --job_idx {}"'

workers = 1
commands = [command_template.format(i % 1, workers, i) for i in range(workers)]
processes = [subprocess.Popen(command, shell=True) for command in commands]
for process in processes:
    process.wait()