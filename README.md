# 屏幕人物监测

这是一个 Windows 桌面工具，用 YOLO 检测屏幕区域中的人物，并可把鼠标/准星自动移动到目标人物中心附近。

## 功能

- 框选屏幕检测区域
- 设置目标标点
- 普通 YOLO 检测模式和姿态模型模式
- 单目标锁定，减少多目标跳动
- 自动鼠标移动，支持全局 `F8` 开关
- 模型下载列表和下载进度
- 鼠标倍率校准，用于适配不同 DPI、游戏灵敏度和系统环境

## 环境

建议使用 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果需要 NVIDIA GPU 加速，请按你的 CUDA/驱动版本安装对应的 PyTorch。安装后可用下面命令检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 启动

普通启动：

```powershell
.\.venv\Scripts\python.exe .\person_monitor.py
```

需要管理员权限时，可运行：

```powershell
.\launch_admin.bat
```

## 模型文件

模型权重文件（例如 `yolo26n.pt`、`yolov8x.pt`）不会提交到 GitHub。程序的“模型”页可以下载模型；也可以手动把 `.pt` 文件放到项目根目录。

## 注意

- `.venv/` 是本机虚拟环境，不建议提交到 GitHub。
- `person_monitor_settings.json` 是本机窗口位置、检测区域、鼠标参数等个人设置，不建议提交。
- 鼠标自动移动效果和游戏灵敏度、DPI、窗口权限有关，建议先用“校准鼠标倍率”。

