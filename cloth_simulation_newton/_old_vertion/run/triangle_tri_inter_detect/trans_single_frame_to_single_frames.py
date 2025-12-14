from asyncore import poll3
import numpy as np

def load_and_report(path):
    try:
        arr = np.load(path)
        print("路径:", path)
        print("原始形状:", arr.shape)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 3:
            pos_numpy = arr
            pos_list = [pos_numpy]
            converted = pos_numpy[None, ...]
            print("转换为列表后: 长度", len(pos_list), "内部形状", pos_list[0].shape)
            print("转换为数组后形状:", converted.shape)
            np.save(path, pos_list)
            print("已保存回原路径，形状:", converted.shape)
            return converted
        else:
            print("无需转换，保持形状:", arr.shape)
        return arr
    except Exception as e:
        print("加载失败:", path, "| 错误:", e)
        return None

if __name__ == "__main__":
    p1 = r"c:\data\sim\simulation-beginning\cloth_simulation_newton\run\render\input\cloth_data_error_dump_1536.npy"
    #p2 = r"c:\data\sim\simulation-beginning\cloth_simulation_newton\run\render\input\cloth_topy_error_dump_1536.npy"
    #load_and_report(p1)
    #arr = np.load(p2)
    #pos_list = [arr[0]]
    #np.save(p2, pos_list)

    p3 = r"C:\data\sim\simulation-beginning\cloth_simulation_newton\run\render\input\cloth_topy_k80.npy"
    arr1 = np.load(p3)
    print("路径:", p3)
    print("原始形状:", arr1.shape)
