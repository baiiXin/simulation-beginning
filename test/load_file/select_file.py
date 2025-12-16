import os



def select_file(folder, suffix=None):
    files = [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and (suffix is None or f.endswith(suffix))
    ]
    files.sort()

    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    idx = int(input("请选择文件编号: "))
    return os.path.join(folder, files[idx])

# 取自身目录并拼接
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MESH_DIR = os.path.join(BASE_DIR, "load_file")

load_file = select_file(MESH_DIR)

# 取文件名
name = os.path.splitext(os.path.basename(load_file))[0]
print(name)