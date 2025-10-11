### 环境

[Installation — Newton Physics](https://newton-physics.github.io/newton/guide/installation.html)

```
cd project
git clone https://github.com/baiiXin/simulation-beginning.git

cd newton
git clone https://github.com/baiiXin/newton.git

# install uv (windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# using uv install environment
uv run -m newton.examples basic_pendulum --viewer null
uv run --extra examples -m newton.examples robot_humanoid --num-envs 16
uv run --extra examples --extra torch-cu12 -m newton.examples robot_anymal_c_walk
uv run -m newton.examples
```



### 部分文件结构

cloth_simulation_by_zcy_first：The simplest model using only NumPy. 

cloth_simulation_by_zcy_new：Add self-collision using Torch, Warp, and Newton. 



| 文件夹/文件结构                             | 说明                                       |
| :------------------------------------------ | :----------------------------------------- |
| **cloth_simulation_by_zcy_first/** |                                            |
| - cloth.py                                  | 布料类，弹簧类，布料类仿真方法       |
| - generate_fun.py                           | 生成布料/三角网格，弹簧            |
| - sim.py                           | 仿真计算，结果储存             |
| run/                               | 可运行程序                     |
|                          |    |
| **cloth_simulation_by_zcy_new/**   |                                |
| torch_contact_computation/         | 碰撞相关计算                   |
|                                    |                                |
|  |  |
| **README.md**                         |                                            |



### 运动公式推导

见个人笔记: https://www.notion.so/Simulation-beginning-1f3af5a86ca2805caeb4c01b519e04a8?source=copy_link



### 来源/参考/将要参考

[1] Huamin Wang，https://games-cn.org/games103/，2021. 

[2] NVIDIA Warp Documentation，https://nvidia.github.io/warp/，2025. 

[3] Anka He Chen，[AnkaChan/Penetration-Free-Deformation-Project](https://github.com/AnkaChan/Penetration-Free-Deformation-Project)，2025. 

[4] Ladislav Kavan，https://youtu.be/sSKyQIxdhdA?si=p5EFise9r4GTpmh8，2017. 

[5] Chenfanfu Jiang et al.，[Preface - Physics-Based Simulation](https://phys-sim-book.github.io/)，2025. 

[6] DeepMind，https://sites.google.com/view/meshgraphnets，2021. 

[7] Newton Physics Documentation，[Newton Physics Documentation — Newton Physics](https://newton-physics.github.io/newton/guide/overview.html)，2025. 

