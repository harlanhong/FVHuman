import os
import tarfile

# 指定目标目录
directory = '/projects/p32321/fating/dataset/PKU-DynamicaHuman/part2'  # 将此路径替换为你自己的目录路径

def extract_and_remove_tar_files(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以.tar结尾
        if filename.endswith(".tar"):
            file_path = os.path.join(directory, filename)
            # 打开.tar文件并解压
            with tarfile.open(file_path) as tar:
                print(f"正在解压 {filename}...")
                tar.extractall(path=directory)
            # 删除.tar文件
            os.remove(file_path)
            print(f"已删除 {filename}")

if __name__ == "__main__":
    extract_and_remove_tar_files(directory)
