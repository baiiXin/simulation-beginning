import numpy as np

import sys
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from cloth import Mass, Spring

def main():
    # =================== 数据构造部分 ===================
    fixed_num = 0 # int((b-a)/h1+1)

    # 材料参数
    z=8.0
    mass_m = 1
    stiff_k = 8000

    # 阻尼参数
    dump = 0.999
    gravity = 9.8

    # simulation
    # 初始参数
    dt = 0.003
    N = 1
    ite_num = 1
    tolerance_newton = 1e-4

    # DeBUG 
    DeBUG = {
        'DeBUG': True,
        'record_hessian': True,
        'max_warning': True,
        'Spring': True,
        'Bending': False,
        'Contact': True,
        'Contact_EE': True,
        'Contact_VT': True,
        'Inertia_Hessian': True,
        'Eigen': True,
        'line_search_max_step': 1,
        'Damping': 0.0,
        'spring_type': 1
    }

    # 仿真计算
    Mass_X = np.array([[0.0,0.0,0.0], [3.0,0.0,0.0], [0.0,4.0,0.0], [1.0,1.0,1.0], [2.0,1.0,1.5], [1.0,2.0,1.5]])

    Mass_V = np.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])

    Mass_E = np.array([[0,1,2], [3,4,5]])

    Spring_ele = np.array([[0,1], [0,2], [1,2], [3,4], [3,5], [4,5]])

    Spring_len = np.array([np.linalg.norm(Mass_X[Spring_ele[i,0]] - Mass_X[Spring_ele[i,1]])*10 for i in range(Spring_ele.shape[0])])

    # 创建弹簧
    mySpring = Spring(
        num=Spring_ele.shape[0],
        ele=Spring_ele,
        rest_len=Spring_len,
        stiff_k=stiff_k
    )
    #print(Mass_X)
    # 创建质点
    myMass = Mass(
        num=Mass_X.shape[0],
        ele=Mass_E,
        pos_cur=Mass_X.copy(),
        vel_cur=Mass_V.copy(),
        pos_prev=Mass_X.copy(),
        vel_prev=Mass_V.copy(),
        mass=mass_m,
        damp=dump,
        gravity=gravity,
        Spring=mySpring,
        dt=dt,
        tolerance_newton=tolerance_newton,
        DeBUG=DeBUG,
    )

    # 储存结果
    cloth_data = [myMass.pos_cur.copy()]
    cloth_vel = [myMass.vel_cur.copy()]

    # 计算
    for i in range(N):
        print("\n\n=====Time step: ", i, "=====")
        [Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = myMass.Single_Newton_Method(mySpring, fixed_num, ite_num, time_step=i)
        cloth_data.append(myMass.pos_cur.copy())
        cloth_vel.append(myMass.vel_cur.copy())


    # =================== 保存最后一帧 ===================
    frame = -1
    verts = cloth_data[frame]
    vel = cloth_vel[frame]


    # =================== 保存 verts ===================
    save_dir = os.path.join(os.path.dirname(__file__), "../render/input")
    os.makedirs(save_dir, exist_ok=True)
    run_id = os.path.splitext(os.path.basename(__file__))[0].replace("save_", "")
    path_data = os.path.join(save_dir, f"cloth_data_{run_id}.npy")
    path_topy = os.path.join(save_dir, f"cloth_topy_{run_id}.npy")
    np.save(path_data, np.array(cloth_data))
    np.save(path_topy, Mass_E.astype(np.int32))



if __name__ == "__main__":
    main()

