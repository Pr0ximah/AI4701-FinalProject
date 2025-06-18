# AI4701-FinalProject

SJTU 2024-2025 Spring AI4701 计算机视觉 期末大作业 **室内场景三维重建**

## 项目介绍

本项目实现了基于 2D 图像的室内场景三维重建。

<p align="center">
    <img src="doc/steps.png" alt="steps" width="75%" />
</p>

通过以上几个步骤，最终能够输出拍摄时的相机位姿和三维点云。

<p align="center">
    <img src="doc/pcd.png" alt="pcd" width="75%" />
</p>

## 项目结构

```
AI4701-FinalProject/
│
├── config/
│   ├── pcd_visualize_param_with_cameras.json        # Open3D可视化的视角设置
│   ├── pcd_visualize_param_without_cameras.json     # Open3D可视化的视角设置
│   └── config.yaml             # 配置文件，包含参数设置
│
├── data/
│   ├── images/                 # 存放输入的图像序列
│   │   └── ...
│   └── camera_intrinsic.txt    # 相机内参
│
├── doc/
│   ├── steps.png               # 项目流程图
│   └── pcd.png                 # 点云可视化示例图
│
├── output/
│   ├── recon_*/                # 重建结果目录，可能有多个
│   │   ├── images/             # 存放可视化结果图
│   │   │   ├── PCD/
│   │   │   │   └── ...         # 点云可视化结果
│   │   │   └── ...             # 其他可视化结果
│   │   ├── recon.yaml          # 本次重建的配置
│   │   ├── extrinsics.txt      # 重建后的相机外参
│   │   └── pcd.ply             # 重建生成的点云
│   └── ...
│
├── process/
│   ├── __init__.py
│   ├── bundle_adjustment.py    # 执行BA优化
│   ├── feature_extraction.py   # SIFT特征提取
│   ├── feature_matching.py     # 特征匹配
│   ├── initial_recon.py        # 初始化重建
│   ├── load_data.py            # 数据加载
│   └── recon.py                # 3D重建
│
├── util/
│   ├── cache/
│   │   └── ...                 # 中间步骤缓存文件
│   ├── __init__.py
│   ├── camera.py               # 相机模型
│   ├── data_cache.py           # 数据缓存管理
│   ├── point_cloud.py          # 点云处理与可视化
│   └── visualize_img.py        # 可视化cv图片
│
├── main.py                     # 主程序入口，调用处理流程
├── requirements.txt
└── README.md
```

## 运行方法

1. 安装依赖库

   ```bash
   pip install -r requirements.txt
   ```

2. 准备数据

   将输入的图像序列放入 `data/images/` 目录，并准备相机内参文件 `data/camera_intrinsic.txt`。

   相机内参文件格式为：

   ```
   fx  0   cx
   0   fy  cy
   0   0   1
   ```

3. 配置参数

   修改 `config/config.yaml` 文件，根据需要调整参数设置。

4. 运行主程序

   ```bash
   python main.py
   ```

**注：** 如果要调整 Open3D 可视化的视角，可以解注释`point_cloud.py`最后生成视角参数的代码，将`show_pcd`设置为`true`，运行`main.py`到出现点云可视化，手动旋转、缩放后关闭可视化窗口，再根据`camera_params.json`的内容，修改 `config/pcd_visualize_param_with_cameras.json` 或 `config/pcd_visualize_param_without_cameras.json` 文件中的参数设置，下一次运行的可视化就会使用新的视角。
