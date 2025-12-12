import subprocess
import sys
import os

def main():
    # 目标脚本的绝对路径
    target_script = "/data/zhoucy/sim/cloth_simulation_newton/examples/example_fixed_triangle/run_fixed_triangel.py"
    
    # 检查目标脚本是否存在
    if not os.path.exists(target_script):
        print(f"Error: Target script not found at {target_script}")
        return

    # 构建 xvfb-run 命令
    # -a: 自动寻找可用的显示号
    # -s: 传递给 Xvfb 的参数
    cmd = [
        "xvfb-run",
        "-a",
        "-s", "-screen 0 1920x1080x24",
        sys.executable,  # 使用当前的 python 解释器
        target_script
    ]

    print("Starting Xvfb and running target script...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # 运行命令
        result = subprocess.run(cmd)
        print(f"Execution finished with return code: {result.returncode}")
    except FileNotFoundError:
        print("Error: xvfb-run not found. Please install xvfb.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
