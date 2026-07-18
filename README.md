# ASRF Reproduction Experiment

本仓库是本人基于论文 **Alleviating Over-segmentation Errors by Detecting Action Boundaries** 官方代码整理的复现实验项目。项目主要用于科研实践考核，也用于记录和维护本人对 ASRF 方法的理解、代码排错、实验复现、结果分析和可视化工具。

原始论文：

- Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka, "Alleviating Over-segmentation Errors by Detecting Action Boundaries", WACV 2021.
- Paper: https://arxiv.org/abs/2007.06866
- Original code: https://github.com/yiskw713/asrf

## Project Purpose

本项目围绕 ASRF 在动作分割任务中的复现展开，重点包括：

- 阅读并理解 ASRF 的整体结构，包括 Action Segmentation Branch (ASB)、Boundary Regression Branch (BRB) 和基于边界的 refinement 流程。
- 使用预提取特征，在 GTEA 数据集的一个划分上完成训练和测试。
- 报告 Frame-wise Accuracy、Edit Score、F1@10、F1@25、F1@50 等指标。
- 对比未经过边界修正的 ASB 输出和经过 boundary refinement 的完整 ASRF 输出。
- 对至少两个视频样例进行时间轴可视化，分析 GT、修正前预测、修正后预测和边界预测。

## Main Modifications

相比原始代码，本仓库做了以下整理和修改：

1. 修复新版 pandas 下训练日志写入问题

   原代码使用 `pd.Series` 配合 `pd.concat` 追加日志，在新版 pandas 中会导致列错位，进而报错：

   ```text
   ValueError: Length of values does not match length of index
   ```

   当前版本改为按单行 `DataFrame` 追加日志，并兼容 resume 时读取旧日志列。

2. 改善训练标准输出

   在 `train.py` 中启用 stdout 行缓冲，使训练重定向到日志文件时，epoch 信息可以更及时写入。

3. 调整预测结果保存结构

   `save_pred.py` 现在将数组和图片分开保存：

   ```text
   result/.../prediction_arrays/   # gt / pred / refined_pred / boundary 的 .npy
   result/.../predictions/         # 最终 timeline 可视化图片
   result/.../boundary_plots/      # 单独 boundary 曲线图
   ```

4. 新增时间轴可视化工具

   新增：

   ```text
   utils/plot_timeline_predictions.py
   ```

   该脚本会将同一视频的 GT、Before refine、Refine 和 Boundary 画成对齐的时间轴横条，不同类别使用不同颜色，更适合实验报告和答辩展示。

5. 增加导出结果结构

   本地实验结果可整理到：

   ```text
   export/gtea_split1_asrf/
   ```

   其中包含配置、权重、指标、可视化图片、预测数组和实验记录。注意 `export/` 默认不提交到 Git。

## Repository Structure

```text
.
|-- csv/                         # 训练、验证、测试划分 csv
|-- imgs/                        # 可视化调色板图片
|-- libs/                        # 模型、数据集、loss、metric、postprocess 等核心代码
|   `-- models/tcn.py            # MS-TCN、ED-TCN、ASRF 主模型
|-- scripts/                     # 原始批量实验脚本
|-- utils/                       # 数据处理、配置生成和可视化工具
|   `-- plot_timeline_predictions.py
|-- train.py                     # 训练脚本
|-- evaluate.py                  # 测试评估脚本
|-- save_pred.py                 # 保存预测数组和 boundary 图
|-- requirements.txt             # 原始依赖记录
`-- README.md
```

以下目录为本地数据或实验产物，默认被 `.gitignore` 排除：

```text
dataset/
dataset_/
result/
export/
*.prm
*.pth
*.npy
*.log
```

## Environment

本项目原始依赖较旧，论文代码要求大致为：

```text
Python >= 3.7
PyTorch >= 1.0
torchvision
pandas
numpy
Pillow
PyYAML
```

本人当前实验环境为 Windows + Conda：

```text
conda env: asrf
Python: 3.10
PyTorch: CUDA 12.8 build
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

核心依赖安装示例：

```powershell
conda create -n asrf python=3.10 -y
conda activate asrf

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pandas PyYAML tqdm matplotlib opencv-python scikit-image scikit-learn scipy pillow
```

如果网络访问 PyPI 不稳定，可使用清华镜像：

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas PyYAML tqdm matplotlib opencv-python scikit-image scikit-learn scipy pillow
```

## Dataset Preparation

数据集和预提取特征不包含在本仓库中。可参考 MS-TCN 仓库下载 GTEA、50Salads、Breakfast 的预提取特征和标注：

```text
https://github.com/yabufarha/ms-tcn
```

数据目录期望结构大致为：

```text
dataset/
├── gtea/
│   ├── features/
│   ├── groundTruth/
│   ├── splits/
│   ├── mapping.txt
│   ├── gt_arr/
│   └── gt_boundary_arr/
```

如果只有原始 groundTruth，需要生成 numpy 标签数组和边界数组：

```powershell
python utils/generate_gt_array.py --dataset_dir ./dataset
python utils/generate_boundary_array.py --dataset_dir ./dataset
python utils/make_csv_files.py --dataset_dir ./dataset
```

## Running GTEA Split-1

### 1. Train

```powershell
python train.py ./result/gtea/dataset-gtea_split-1/config.yaml
```

如需同时保存训练输出：

```powershell
python train.py ./result/gtea/dataset-gtea_split-1/config.yaml 2>&1 | Tee-Object -FilePath train_gtea_split1.log
```

训练完成后会生成：

```text
result/gtea/dataset-gtea_split-1/final_model.prm
result/gtea/dataset-gtea_split-1/log.csv
```

### 2. Evaluate

```powershell
python evaluate.py ./result/gtea/dataset-gtea_split-1/config.yaml --refinement_method refinement_with_boundary
```

评估会同时输出：

- Before refinement：ASB 原始动作分割结果。
- Boundary scores：BRB 边界预测结果。
- After refinement：基于边界修正后的结果。

指标文件会保存在：

```text
result/gtea/dataset-gtea_split-1/
├── test_as_before_refine.csv
├── test_as_after_majority_vote.csv
├── test_br.csv
├── test_c_matrix_before_refinement.csv
└── test_c_matrix_after_majority_vote.csv
```

### 3. Save Predictions

```powershell
python save_pred.py ./result/gtea/dataset-gtea_split-1/config.yaml
```

输出结构：

```text
result/gtea/dataset-gtea_split-1/
├── prediction_arrays/
│   ├── *_gt.npy
│   ├── *_pred.npy
│   ├── *_refined_pred.npy
│   └── *_boundary.npy
├── boundary_plots/
│   └── *_boundary.png
```

### 4. Timeline Visualization

生成全部测试视频的时间轴可视化：

```powershell
python utils/plot_timeline_predictions.py ./result/gtea/dataset-gtea_split-1/prediction_arrays --dataset gtea
```

只生成指定视频：

```powershell
python utils/plot_timeline_predictions.py ./result/gtea/dataset-gtea_split-1/prediction_arrays --dataset gtea --names S1_Cheese_C1 S1_Coffee_C1
```

输出图片位于：

```text
result/gtea/dataset-gtea_split-1/predictions/
```

每张图包含四条横向时间轴：

```text
GT
Before refine
Refine
Boundary
```

## Current GTEA Split-1 Result

当前一次 GTEA split-1、20 epoch 实验结果如下：

| Method | Frame-wise Acc | Edit | F1@10 | F1@25 | F1@50 |
|---|---:|---:|---:|---:|---:|
| Before refinement | 71.28 | 83.83 | 86.99 | 84.76 | 72.12 |
| After boundary refinement | 68.72 | 76.57 | 84.90 | 81.63 | 71.84 |

Boundary branch:

| Boundary Acc | Precision | Recall | Boundary F1 |
|---:|---:|---:|---:|
| 97.64 | 50.00 | 60.47 | 54.74 |

本次实验中，boundary refinement 后指标下降。该现象可作为失败案例分析：当 BRB 边界预测不够稳定时，基于边界切段后的 majority vote 可能错误合并或覆盖 ASB 中原本正确的局部预测。

这与原论文中对 refinement 失败情况的讨论并不矛盾。论文也指出，即使使用真实边界，segment-level majority voting 仍可能在某些情况下使结果变差。

## Exported Artifacts

本地整理后的考核结果包位于：

```text
export/gtea_split1_asrf/
```

包含：

```text
config.yaml
README.md
weights/final_model.prm
metrics/*.csv
notes/experiment_note.md
predictions/*_timeline.png
prediction_arrays/*.npy
boundary_plots/*_boundary.png
```

该目录默认不提交到 GitHub。如果需要提交结果，请先确认文件大小和数据/权重共享要求。

## Git Workflow

当前远程仓库：

```text
https://github.com/Damer-F/asrf.git
```

由于网络环境原因，本地 remote 使用：

```text
https://gh-proxy.org/https://github.com/Damer-F/asrf.git
```

常用维护流程：

```powershell
git status
git add .
git commit -m "update message"
git push
```

## License and Citation

本项目基于 ASRF 官方实现整理，原始代码遵循 MIT License。

Citation:

```text
Yuchi Ishikawa, Seito Kasai, Yoshimitsu Aoki, Hirokatsu Kataoka,
"Alleviating Over-segmentation Errors by Detecting Action Boundaries",
WACV 2021.
```

References:

- Colin Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection", CVPR 2017.
- Yazan Abu Farha and Juergen Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", CVPR 2019.
