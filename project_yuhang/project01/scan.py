import os
import subprocess

def run_info():
    print("="*20 + " 磁盘挂载 " + "="*20)
    os.system("df -h | grep -v tmpfs")
    
    print("\n" + "="*20 + " 搜索 Qwen 模型 " + "="*20)
    # 暴力搜索几个核心路径
    os.system("find /workspace /data /mnt -maxdepth 3 -iname '*qwen*' 2>/dev/null")

if __name__ == "__main__":
    run_info()