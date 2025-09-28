# Visual Odometry System

这是一个基于RGB-D数据的视觉里程计系统，使用TUM数据集进行测试。

## 项目结构

```
solution/
├── __init__.py              # Python包初始化文件
├── main.py                  # 主程序入口
├── config.yaml              # 配置文件
├── requirements.txt         # Python依赖包
├── run_vo.sh               # 运行脚本
├── data_loader.py          # TUM数据集加载器
├── frame.py                # 帧和相机类
├── feature_matching.py     # 特征提取和匹配
├── visual_odometry.py      # 视觉里程计主类
└── visualizer.py           # 3D可视化模块
```

## 功能特点

1. **数据集处理**: 支持TUM RGB-D数据集格式
2. **特征提取**: 使用ORB或SIFT特征检测器
3. **位姿估计**: 基于PnP算法的相机位姿估计
4. **实时可视化**: 显示相机轨迹和当前图像
5. **性能监控**: 实时显示处理时间和匹配统计信息

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- opencv-python: 图像处理和计算机视觉
- numpy: 数值计算
- matplotlib: 图表绘制
- open3d: 3D可视化（可选）
- pyyaml: 配置文件解析

## 使用方法

### 基本运行

```bash
python3 main.py config.yaml
```

### 使用运行脚本

```bash
./run_vo.sh config.yaml
```

### 命令行参数

- `config_file`: 配置文件路径（必需）
- `--max_frames N`: 限制处理的最大帧数
- `--start_frame N`: 从第N帧开始处理
- `--save_trajectory FILE`: 保存轨迹到文件
- `--no_visualization`: 禁用可视化
- `--step_mode`: 单步模式（按键继续下一帧）

### 示例

```bash
# 处理前100帧
python3 main.py config.yaml --max_frames 100

# 从第50帧开始处理并保存轨迹
python3 main.py config.yaml --start_frame 50 --save_trajectory trajectory.txt

# 单步模式调试
python3 main.py config.yaml --step_mode --max_frames 10
```

## 配置文件

配置文件`config.yaml`包含以下主要参数：

```yaml
# 数据集路径
dataset_path: "../rgbd_dataset_freiburg1_xyz"

# 相机内参（TUM Freiburg1）
camera:
  fx: 517.3
  fy: 516.5
  cx: 318.6
  cy: 255.3
  depth_scale: 5000.0

# ORB特征参数
orb:
  n_features: 1000
  scale_factor: 1.2
  n_levels: 8

# 视觉里程计参数
vo:
  min_inliers: 20
  ransac_threshold: 1.0
  max_iterations: 1000
```

## 运行时控制

在可视化模式下，支持以下键盘控制：

- `q` 或 `ESC`: 退出程序
- `s`: 保存当前轨迹（需要指定--save_trajectory参数）
- 任意键: 单步模式下继续下一帧

## 输出信息

程序运行时会显示每帧的处理统计信息：

```
Frame   12: State=TRACKING    Matches= 85 Inliers= 67 Time=0.023s
Frame   13: State=TRACKING    Matches= 92 Inliers= 74 Time=0.025s
```

- **State**: VO系统状态（INITIALIZING/TRACKING/LOST）
- **Matches**: 特征匹配数量
- **Inliers**: RANSAC内点数量
- **Time**: 帧处理时间

## 算法流程

1. **初始化**: 使用前两帧建立初始参考
2. **特征提取**: 提取ORB特征点和描述子
3. **特征匹配**: 在连续帧间匹配特征点
4. **位姿估计**: 使用PnP算法估计相机位姿
5. **轨迹更新**: 更新相机轨迹和可视化

## 技术细节

- **坐标系**: 使用相机坐标系，Z轴向前
- **深度图**: TUM数据集深度值除以5000得到米单位
- **位姿表示**: 4x4变换矩阵表示相机位姿
- **特征匹配**: 使用Lowe's ratio test过滤匹配

## 故障排除

1. **ImportError**: 检查是否安装了所有依赖包
2. **FileNotFoundError**: 检查数据集路径和配置文件路径
3. **OpenCV错误**: 确保安装了正确版本的opencv-python
4. **内存不足**: 减少处理的帧数或降低特征点数量

## 扩展功能

系统设计为模块化，可以轻松扩展：

- 添加新的特征检测器（SIFT、SURF等）
- 实现回环检测和地图优化
- 支持其他数据集格式
- 添加更多可视化选项