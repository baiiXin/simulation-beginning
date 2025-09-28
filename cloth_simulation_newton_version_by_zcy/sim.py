import numpy as np
from generate_fun import generate_spring, generate_mass
from cloth import Mass, Spring

def generate_cloth(a, b, c, d, h1, h2, z, mass_m, stiff_k, dump, gravity, dt, N, ite_num, tolerance_newton, fixed_num):
    '''
    功能: 生成模拟信息, 生成收敛曲线信息
    input: 布料三角剖分参数; 质点参数; 弹簧参数; 时间参数; 迭代参数; 牛顿迭代参数; 固定质点数量
    output: 布料动画信息; 收敛曲线信息
            triangles: 三角剖分信息 (shape: (Spring_num 弹簧数量, 3 三角剖分顶点索引))
            cloth_data: 布料动画信息 (shape: (N 时间步数, Mass_num 质点数量, 3 空间维数))
    '''
    # Mass
    [Mass_num, Mass_X, Mass_E, Mass_V, Mass_m] = generate_mass(a, b, c, d, h1, h2, z, mass_m)
    print("Mass_num", Mass_num)
    # Spring
    [Spring_num, Spring_ele, Spring_len, Spring_stiff_k] = generate_spring(Mass_X, Mass_E, stiff_k)
    print("Spring_num", Spring_num)
    
    # 创建弹簧
    mySpring = Spring(
        num=Spring_num,
        ele=Spring_ele,
        rest_len=Spring_len,
        stiff_k=Spring_stiff_k
    )
    #print(Mass_X)
    # 创建质点
    myMass = Mass(
        num=Mass_num,
        ele=Mass_E,
        pos_cur=Mass_X.copy(),
        vel_cur=Mass_V.copy(),
        pos_prev=Mass_X.copy(),
        vel_prev=Mass_V.copy(),
        mass=Mass_m,
        dump=dump,
        gravity=gravity,
        Spring=mySpring,
        dt=dt,
        tolerance_newton=tolerance_newton
    )

    # 储存结果
    cloth_data = [myMass.pos_cur.copy()]

    # 计算
    for i in range(N):
        print("\n\n=====Time step: ", i, "=====")
        [Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm] = myMass.Single_Newton_Method(mySpring, fixed_num, ite_num)
        cloth_data.append(myMass.pos_cur.copy())

    triangles = Mass_E
    print(triangles.shape)
    print(mySpring.ele.shape)
    print(myMass.pos_cur.shape)
    print(cloth_data[0].shape)

    return triangles, cloth_data, Newton_steps, times_ms, Error_dx_norm, Residual_norm, Energy_norm

