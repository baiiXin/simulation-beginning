import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 控制点
P0 = np.array([0, 0, 0])
P1 = np.array([1, 2, 1])
P2 = np.array([2, 1, 2])
P3 = np.array([3, 3, 3])

# Bezier 曲线函数
def bezier(t, P0, P1, P2, P3):
    return (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3

# 生成曲线点
t_values = np.linspace(0, 1, 100)
curve_points = np.array([bezier(t, P0, P1, P2, P3) for t in t_values])

# 绘图
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 曲线
ax.plot(curve_points[:,0], curve_points[:,1], curve_points[:,2], color='blue', label='Bezier Curve')

# 控制点
ax.scatter([P0[0], P1[0], P2[0], P3[0]],
           [P0[1], P1[1], P2[1], P3[1]],
           [P0[2], P1[2], P2[2], P3[2]],
           color='red', label='Control Points')

# 控制多边形
ax.plot([P0[0], P1[0], P2[0], P3[0]],
        [P0[1], P1[1], P2[1], P3[1]],
        [P0[2], P1[2], P2[2], P3[2]], 'r--', alpha=0.3)

ax.set_title("3D Cubic Bezier Curve")
ax.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 控制点
P0 = np.array([0, 0, 0])
P1 = np.array([1, 2, 1])
P2 = np.array([2, 1, 2])
P3 = np.array([3, 3, 3])
control_points = [P0, P1, P2, P3]

# Bezier 曲线函数
def bezier(t, P):
    n = len(P)-1
    B = np.zeros(3)
    for i in range(n+1):
        B += comb(n,i) * ((1-t)**(n-i)) * (t**i) * P[i]
    return B

# 一阶导函数
def bezier_derivative(t, P):
    n = len(P)-1
    B_prime = np.zeros(3)
    for i in range(n):
        B_prime += n*(P[i+1]-P[i]) * comb(n-1,i) * ((1-t)**(n-1-i)) * (t**i)
    return B_prime

# 二阶导函数
def bezier_second_derivative(t, P):
    n = len(P)-1
    B_double_prime = np.zeros(3)
    for i in range(n-1):
        B_double_prime += n*(n-1)*(P[i+2]-2*P[i+1]+P[i]) * comb(n-2,i) * ((1-t)**(n-2-i)) * (t**i)
    return B_double_prime

# 组合数函数
from math import comb

# 生成曲线点
t_values = np.linspace(0, 1, 100)
curve_points = np.array([bezier(t, control_points) for t in t_values])
der_points = np.array([bezier_derivative(t, control_points) for t in t_values])
second_der_points = np.array([bezier_second_derivative(t, control_points) for t in t_values])

# 绘图
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲线
ax.plot(curve_points[:,0], curve_points[:,1], curve_points[:,2], color='blue', label='Bezier Curve')

# 绘制控制点
ax.scatter(*zip(*control_points), color='red', s=50, label='Control Points')
ax.plot(*zip(*control_points), 'r--', alpha=0.3)

# 绘制导向量（选取若干点画箭头）
for i in range(0, 100, 10):
    ax.quiver(*curve_points[i], *der_points[i], length=0.5, color='green', normalize=True)
    ax.quiver(*curve_points[i], *second_der_points[i], length=0.3, color='purple', normalize=True)

ax.set_title("3D Cubic Bezier Curve with Derivatives")
ax.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import comb

# 原始控制点（3次 Bezier）
P0 = np.array([0, 0, 0])
P1 = np.array([1, 2, 1])
P2 = np.array([2, 1, 2])
P3 = np.array([3, 3, 3])
P = [P0, P1, P2, P3]

# 升阶函数 (n -> n+1)
def degree_elevate(P):
    n = len(P)-1
    Q = []
    Q.append(P[0])
    for i in range(1, n+1):
        Q.append(i/(n+1)*P[i-1] + (1-i/(n+1))*P[i])
    Q.append(P[-1])
    return Q

# Bezier 曲线函数
def bezier_curve(t, P):
    n = len(P)-1
    B = np.zeros(3)
    for i in range(n+1):
        B += comb(n,i) * ((1-t)**(n-i)) * (t**i) * P[i]
    return B

# 生成曲线点
t_values = np.linspace(0, 1, 100)
curve_orig = np.array([bezier_curve(t, P) for t in t_values])

# 升阶后的控制点与曲线
P_elev = degree_elevate(P)
curve_elev = np.array([bezier_curve(t, P_elev) for t in t_values])

# 绘图
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# 原始曲线与控制点
ax.plot(curve_orig[:,0], curve_orig[:,1], curve_orig[:,2], 'b', label='Original 3rd-degree')
ax.scatter(*zip(*P), color='red', s=50)
ax.plot(*zip(*P), 'r--', alpha=0.3)

# 升阶曲线与控制点
ax.plot(curve_elev[:,0], curve_elev[:,1], curve_elev[:,2], 'g', linestyle='--', label='Elevated 4th-degree')
ax.scatter(*zip(*P_elev), color='orange', s=50)
ax.plot(*zip(*P_elev), 'orange', linestyle=':', alpha=0.3)

ax.set_title("3D Bezier Curve Degree Elevation")
ax.legend()
plt.show()
