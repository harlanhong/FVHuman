import time
import subprocess
import os
def task():
    print("Executing task...")
    # 在这里放置你想要执行的指令或代码
    # 比如调用某个函数或运行某个命令
    os.system('flash_s3_upload --local_dir /mnt/localssd/final_ted_dataset/masks/ --s3_url s3://zhanxu-public/fating/final_ted_dataset/')
    os.system('flash_s3_download --local_dir /mnt/localssd/final_ted_dataset/ --s3_url s3://zhanxu-public/fating/final_ted_dataset/masks/')
while True:
    task()  # 执行任务
    time.sleep(600)  # 休眠 600 秒 (即 10 分钟)
