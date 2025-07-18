import subprocess
import os
import signal
import argparse
import pdb
def kill_fuser_processes(device: str):
    try:
        # 获取 fuser 返回的进程信息
        result = subprocess.run(['fuser', '-v', '/dev/nvidia'+device], capture_output=True, text=True)

        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        # 解析输出，获取进程 ID
        # output = result.stdout.strip().split('\n')[1:]  # 忽略第一行（表头）
        pids =result.stdout.split(' ')[1:]  # 获取每行的第一个元素（PID）

        # 杀死所有进程
        for pid in pids:
            os.system(f'kill -9 {pid}')  # 或者使用 signal.SIGKILL 发送更强的终止信号

        print(f"Killed processes: {pids}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kill processes using a specified device.")
    parser.add_argument("device", type=str, help="The device path (e.g., /dev/nvidia3)")
    
    args = parser.parse_args()
    
    kill_fuser_processes(args.device)
