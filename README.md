### 环境

conda env create -f simulation25.yml --prefix D:\\data\envs\simulation25



### 部分文件结构

| 文件夹/文件结构                             | 说明                                       |
| :------------------------------------------ | :----------------------------------------- |
| **cloth_nv_warp（文件夹）**（待完成！！！）         | NVIDIA Warp Documentation 布料模拟示例程序 |
| - cloth_simulation.mp4                      | 运行’ read_usd.py‘ 的输出                  |
| - example_cloth.usd                         | 运行’ nv_warp_sim_cloth.py‘ 的输出         |
| - nv_warp_sim_cloth.py                      | **示例程序 -- 输出为USD文件**              |
| - read_usd.py                               | 读取USD文件（ai）                          |
|  |  |
| **cloth_simulation_by_zcy_first（文件夹）** |                                            |
| - cloth.py                                  | 布料类，弹簧类，布料类仿真方法       |
| - generate_fun.py                           | 生成布料/三角网格，弹簧            |
| - matplotlib_concurve_energy.py             | 绘制 energy **(还有问题)** （可直接运行！） |
| - matplotlib_concurve.py                    | 绘制 residual **(可能还有问题)** （可直接运行！） |
| - matplotlib_sim.py                         | 仿真动画生成（可直接运行！）   |
| - sim.py                                    | 仿真计算，结果储存                  |
|  |  |
| **ppt（文件夹）**                          |          |
| - 001-implicit_integration.png        | README插图，运动公式推导      |
| - 002-implicit_integration.png        |                        |
| - 003-newton_method.png               |                          |
| - 004-spring.png                      |                        |
| - 005-collision_impulse.png           |                        |
| - 006-collision_impulse.png           |                        |
| - GAMES-103-04_rigid_contact.pptx     | 刚体模拟ppt，含碰撞处理       |
| - GAMES-103-05_cloth.pptx             | 布料模拟ppt              |
| - GAMES-103-07_collision.pptx         | 碰撞处理ppt，自相交        |
| - homework_cloth.pdf                  | 布料模拟作业    |
|  |  |
| **README.md**                         |                                            |



### 运动公式推导

见个人笔记: https://www.notion.so/Simulation-beginning-1f3af5a86ca2805caeb4c01b519e04a8?source=copy_link



### 参考/将要参考

[1] Huamin Wang，https://games-cn.org/games103/，2021.

[2] NVIDIA Warp Documentation，https://nvidia.github.io/warp/，2025.

[3] Ladislav Kavan，https://youtu.be/sSKyQIxdhdA?si=p5EFise9r4GTpmh8，2017.

[4] Chenfanfu Jiang et al.，[Preface - Physics-Based Simulation](https://phys-sim-book.github.io/)，2025.

[5] DeepMind，https://sites.google.com/view/meshgraphnets，2021.