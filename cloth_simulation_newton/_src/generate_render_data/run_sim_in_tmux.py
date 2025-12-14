import subprocess
import shlex

def run_in_tmux(session_name, command):
    # 检查会话是否存在
    result = subprocess.run(["tmux", "has-session", "-t", session_name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    # 如果不存在则创建
    if result.returncode != 0:
        print(f"Creating tmux session '{session_name}' ...")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name])
    else:
        print(f"Session '{session_name}' already exists.")

    # 将命令发送到会话
    print(f"Running command in tmux session '{session_name}' ...")
    subprocess.run(["tmux", "send-keys", "-t", session_name, command, "C-m"])

    print("Done. You can attach with:")
    print(f"  tmux attach -t {session_name}")


if __name__ == "__main__":
    session = "mysim"

    # 你要执行的命令（按需修改）
    command = "/data/zhoucy/sim/cloth_simulation_newton/run/generate_render_data/save_unit_test.py"

    run_in_tmux(session, command)
