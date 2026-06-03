# UR5e Impedance Learning Force Tracking

本项目是一个基于 MuJoCo 的 UR5e 力跟踪与可变阻抗学习实验工程，核心目标是复现在未知或变化地形上，让机器人通过学习调节阻抗参数，在连续接触过程中稳定跟踪期望法向力。将力跟踪问题转化为未知环境下的阻抗参数调节问题，用深度强化学习提供前馈调参能力，再由可变阻抗反馈环保证接触稳定。本项目把这个框架落到 UR5e + MuJoCo 仿真中：PPO 策略不直接输出关节控制量，而是输出法向阻尼、法向刚度和 z 方向参考偏移，由阻抗/导纳回路、逆运动学和 MuJoCo 物理共同完成末端扫描与力跟踪。并设置了与传统阻抗控制和自适应阻抗控制的效果对比。

# 效果展示
<img width="800" height="432" alt="Image" src="https://github.com/user-attachments/assets/8e7001db-b83b-4d92-bef2-37068141e107" />

# 项目结构

```text
.
├── ac_impedence_control/public/training_modules/
│   ├── ur5e_force_impedance_env.py        # Gymnasium 环境：UR5e 力跟踪阻抗学习任务
│   ├── train_ppo_force_impedance.py       # scene-30.xml 上训练 PPO
│   └── train_ppo_force_impedance_wave.py  # scene_45.xml + 正弦波地形训练 PPO
├── model/universal_robots_ur5e/
│   ├── scene-30.xml                       # 固定地形/接触场景
│   ├── scene_45.xml                       # hfield 波浪地形场景
│   └── ur5e_2.xml                         # UR5e 机器人模型
├── src/
│   ├── pinocchio_kinematic.py             # Pinocchio + CasADi 逆运动学
│   ├── lowpass_filter.py                  # 末端速度低通滤波
│   └── matplot.py                         # 渲染时的实时曲线窗口
├── eval_force_impedance.py                # 固定导纳 vs PPO
├── eval_force_impedance_three_way.py      # 固定导纳 / PPO / adaptive_sigma_phi 三方对比
├── eval_force_impedance_three_way_wave.py # 波浪地形三方对比
├── eval_ppo_force_impedance_wave_only.py  # 单独评估波浪地形 PPO
├── analyze_ppo_wave_force_metrics.py      # 过冲与调节时间指标
└── terrain_force_metrics_table.py         # 多地形指标汇总
```

# 环境准备

建议使用 Python 3.9 或 3.10。核心依赖包括：

- `mujoco`
- `gymnasium`
- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `stable-baselines3` 或项目内 fork：`ac_impedence_control/public/stable-baselines3-acmpc`
- `pinocchio`
- `casadi`
- 可选渲染依赖：`pyqtgraph`, `PyQt5`

示例安装流程：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install mujoco gymnasium numpy scipy matplotlib torch tensorboard pyqtgraph PyQt5 casadi
pip install -e ac_impedence_control\public\stable-baselines3-acmpc
```

如果 `pinocchio` 在 Windows 上安装失败，建议使用 conda-forge：

```powershell
conda install -c conda-forge pinocchio casadi
```

# 训练

固定地形训练

```powershell
python ac_impedence_control\public\training_modules\train_ppo_force_impedance.py `
  --total_timesteps 200000 `
  --n_envs 8 `
  --save_dir runs\ppo_force_impedance
```

默认设置：

- 场景：`model/universal_robots_ur5e/scene-30.xml`
- 目标力：`-50 N`
- 扫描方向：沿 y 方向
- 动作：`[B_z_norm, K_z_norm, z_ref_delta_norm]`
- 输出：模型 checkpoint、最终模型、monitor CSV、TensorBoard 日志

波浪地形训练

```powershell
python ac_impedence_control\public\training_modules\train_ppo_force_impedance_wave.py `
  --total_timesteps 200000 `
  --n_envs 1 `
  --save_dir runs\ppo_force_impedance_wave
```

该脚本默认加载 `scene_45.xml`，并向 MuJoCo hfield 注入一维 y 方向正弦波地形。它对应论文中对斜面、曲面、正弦/高阶地形进行泛化验证的思路。

训练时可以打开 MuJoCo viewer 和实时力曲线：

```powershell
python ac_impedence_control\public\training_modules\train_ppo_force_impedance.py --render --n_envs 1
```

# 评估

固定导纳与 PPO 对比

```powershell
python eval_force_impedance.py `
  --model runs\ppo_force_impedance\ppo_force_impedance_final.zip `
  --episodes 3
```

只评估固定导纳基线：

```powershell
python eval_force_impedance.py --baseline_only --episodes 3
```

输出位于 `runs/force_impedance_eval/`，包括：

- `force_impedance_eval.csv`
- `force_tracking_episode0.png`

三方对比：固定导纳 / PPO / 自适应基线

```powershell
python eval_force_impedance_three_way.py `
  --model runs\ppo_force_impedance\ppo_force_impedance_final.zip `
  --methods fixed,ppo,adaptive `
  --episodes 3
```

其中：

- `fixed_admittance`：固定 `B_z/K_z` 的基线控制器。
- `ppo_impedance`：PPO 输出可变阻抗参数。
- `adaptive_sigma_phi`：基于力反馈的自适应 z 轴导纳基线，用于观察纯反馈方法的滞后与稳定性。

波浪地形三方对比

```powershell
python eval_force_impedance_three_way_wave.py `
  --model runs\ppo_force_impedance_wave\ppo_force_impedance_wave_final.zip `
  --methods fixed,ppo,adaptive `
  --episodes 3
```

控制流程

单个环境 step 的主要过程如下：

1. 读取 MuJoCo 中末端位姿、速度和力传感器数据。
2. PPO action 映射为实际阻抗参数 `B_z`, `K_z` 和参考偏移 `z_ref_delta`。
3. 根据目标力与测量力计算法向力误差。
4. 使用导纳/阻抗方程计算末端期望加速度，并得到期望位姿增量。
5. 末端沿 y 方向执行扫描，同时根据法向力反馈修正 z 方向参考。
6. Pinocchio + CasADi 求解逆运动学，得到关节目标。
7. MuJoCo 前进一步仿真。
8. 根据力误差、轨迹误差、动作平滑性、速度和扫描进度计算 reward。



# 参考论文

Yanghong Li, Li Zheng, Yahao Wang, Erbao Dong, and Shiwu Zhang, Impedance Learning-Based Adaptive Force Tracking for Robot on Unknown Terrains, IEEE Transactions on Robotics, 2025. DOI: `10.1109/TRO.2025.3530345`.


