import subprocess

command_template = 'bash -c "source /projects/p32296/miniconda3/bin/activate humancmp && ' \
                   'cd /projects/p32296/fating/src/novel-view-human-master && ' \
                   'CUDA_VISIBLE_DEVICES={} python tools/dynamichuman/tools/extract_dwpose_from_vid.py ' \
                   '--num_workers {} --job_idx {}"'

workers = 10
commands = [command_template.format(i % 1, workers, i) for i in range(workers)]
processes = [subprocess.Popen(command, shell=True) for command in commands]
for process in processes:
    process.wait()