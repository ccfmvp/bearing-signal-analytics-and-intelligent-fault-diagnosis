# Cross-Domain Bearing Fault Diagnosis

> 轴承信号数据分析与智能故障诊断项目

## 1. Project Overview

本项目面向高速列车轴承故障诊断场景，构建了一套从源域振动信号处理、特征工程、源域分类建模到目标域迁移学习诊断的完整算法流程。
项目重点关注真实工业数据中常见的噪声干扰、样本不平衡、目标域标签缺失以及跨域分布偏移等问题，并结合可解释性分析提升模型结果的透明度与可信度。

项目核心思想如下：

- 以台架轴承振动数据作为**源域**（CWRU 轴承数据集）
- 以列车实际轴承振动样本作为**目标域**
- 先完成源域数据清洗、周期计算、滑动窗口切片与特征提取
- 再进行源域多模型分类训练与评估
- 最后通过领域自适应方法完成源域到目标域的迁移诊断

---

## 2. Project Objectives

本项目主要包含以下功能模块：

1. **源域振动信号预处理**
   - 基于轴承故障特征频率的周期计算
   - 数据集统计分析与概览可视化
   - 滑动窗口切片（固定窗口长度与重叠率）
   - 多粒度标签构建（23类 / 7类 / 4类）

2. **特征工程**
   - 时域特征提取（均值、方差、峰值、峭度、偏度等）
   - 频域特征提取（频谱均值、重心频率等）
   - 特征融合与标签汇总
   - 特征相关性分析与重要性筛选

3. **源域分类训练**
   - 5种深度学习模型：CNN、CNN-LSTM、CNN-BiLSTM、CNN-Transformer、ResNet-Transformer
   - 2种传统机器学习模型：XGBoost、LightGBM
   - 所有模型均采用 5 折交叉验证评估

4. **跨域迁移学习诊断**
   - MMD（最大均值差异）域适应
   - DANN（对抗域自适应神经网络）域适应
   - 源域到目标域的特征对齐与故障类别预测

5. **可解释性分析**
   - 特征重要性分析与排名
   - MMD 损失变化曲线分析
   - 模型权重热力图与 3D 可视化

---

## 3. Repository Structure

```text
cross_domain_bearing_fault_diagnosis/
├── README.md                  # 项目说明文档
├── requirements.txt           # Python 依赖
├── configs/                   # YAML 配置文件
│   ├── feature_pipeline.yaml          # 特征工程参数配置
│   ├── source_model_training.yaml     # 源域训练超参数配置
│   ├── domain_adaptation.yaml         # 域适应训练配置
│   └── explainability.yaml            # 可解释性分析配置
│
├── data/                      # 数据文件
│   ├── raw/                           # 原始 .mat 数据
│   │   ├── source_domain/             #   源域（CWRU 台架数据）
│   │   └── target_domain/             #   目标域（列车轴承数据）
│   ├── interim/                       # 中间处理结果
│   │   ├── source_windows/            #   源域滑动窗口 CSV
│   │   ├── target_windows/            #   目标域滑动窗口 CSV
│   │   ├── source_features/           #   源域提取特征 CSV
│   │   └── target_features/           #   目标域提取特征 CSV
│   ├── processed/                     # 最终结构化数据集
│   └── metadata/                      # 辅助元数据
│       ├── cycle_statistics.xlsx      #   周期统计表
│       ├── feature_mapping.csv        #   特征映射表
│       └── auxiliary_tables/          #   其他辅助表
│
├── src/                       # 核心源码
│   ├── data_processing/                # 数据预处理模块
│   │   ├── cycle_calculation.py               # 周期计算
│   │   ├── dataset_overview_visualization.py   # 数据集概览可视化
│   │   ├── source_sliding_window.py            # 源域滑动窗口
│   │   ├── target_sliding_window.py            # 目标域滑动窗口
│   │   ├── build_source_dataset_23cls.py       # 构建 23 类数据集
│   │   ├── build_source_dataset_7cls.py        # 构建 7 类数据集
│   │   ├── build_source_dataset_4cls.py        # 构建 4 类数据集
│   │   ├── normalization.py                    # 数据标准化
│   │   ├── target_dataset_builder.py           # 目标域数据集构建
│   │   └── __init__.py
│   │
│   ├── feature_engineering/            # 特征工程模块
│   │   ├── extract_source_features.py          # 源域特征提取
│   │   ├── extract_target_features.py          # 目标域特征提取
│   │   ├── merge_and_label_features.py         # 特征合并与标签
│   │   ├── feature_correlation_analysis.py     # 特征相关性分析
│   │   ├── feature_distribution_visualization.py# 特征分布可视化
│   │   ├── select_top_features.py              # Top-N 特征筛选
│   │   └── __init__.py
│   │
│   ├── models/                          # 模型定义
│   │   ├── cnn.py                              # CNN1D 模型
│   │   ├── cnn_lstm.py                         # CNN-LSTM 模型
│   │   ├── cnn_bilstm.py                       # CNN-BiLSTM 模型
│   │   ├── cnn_transformer.py                  # CNN-Transformer 模型
│   │   ├── resnet_transformer.py               # ResNet-Transformer 模型
│   │   ├── xgboost_model.py                    # XGBoost 工厂函数
│   │   ├── lightgbm_model.py                   # LightGBM 工厂函数
│   │   ├── components.py                       # LinearMapping / GRL 等组件
│   │   └── __init__.py
│   │
│   ├── training/                        # 源域分类训练
│   │   ├── train_cnn_23cls.py
│   │   ├── train_cnn_lstm_23cls.py
│   │   ├── train_cnn_bilstm_23cls.py
│   │   ├── train_cnn_transformer_23cls.py
│   │   ├── train_resnet_transformer_23cls.py
│   │   ├── train_xgboost_23cls.py
│   │   ├── train_lightgbm_23cls.py
│   │   └── __init__.py
│   │
│   ├── domain_adaptation/               # 域适应模块
│   │   ├── mmd_loss.py                         # MMD 损失函数
│   │   ├── dann_loss.py                        # DANN 损失函数（含 GRL）
│   │   ├── preprocess_transfer_data.py         # 迁移数据预处理
│   │   ├── transfer_normalization.py           # 迁移数据标准化
│   │   ├── train_cnn_bilstm_mmd.py             # CNN-BiLSTM + MMD
│   │   ├── train_cnn_bilstm_mmd_v1.py          # CNN-BiLSTM + MMD v1
│   │   ├── train_cnn_bilstm_dann.py            # CNN-BiLSTM + DANN
│   │   ├── train_resnet_transformer_mmd.py     # ResNet-Transformer + MMD
│   │   ├── train_xgboost_mmd.py                # XGBoost + MMD
│   │   └── __init__.py
│   │
│   ├── explainability/                  # 可解释性分析
│   │   ├── weight_3d_visualization.py          # 3D 权重可视化
│   │   ├── weight_heatmap_visualization.py     # 权重热力图
│   │   ├── feature_importance_analysis.py      # 特征重要性分析
│   │   ├── mmd_loss_visualization.py           # MMD 损失可视化
│   │   └── __init__.py
│   │
│   └── utils/                           # 通用工具
│       ├── metrics.py                          # 评估指标计算
│       ├── plotting.py                         # 绘图工具
│       ├── seed.py                             # 随机种子设置
│       ├── io_utils.py                         # 路径与目录工具
│       ├── count_total.py                      # 数据统计
│       ├── generate_mapping_feature.py         # 特征映射生成
│       └── __init__.py
│
├── scripts/                   # 流水线入口脚本
│   ├── run_source_feature_pipeline.py          # 问题1：特征工程全流程
│   ├── run_source_model_training.py            # 问题2：源域模型训练全流程
│   ├── run_domain_adaptation.py                # 问题3：域适应全流程
│   └── run_explainability_analysis.py          # 问题4：可解释性分析全流程
│
├── outputs/                   # 实验输出结果
│   ├── figures/                        # 可视化图表
│   │   ├── dataset_overview/
│   │   ├── bearing_geometry_analysis/
│   │   ├── feature_analysis/
│   │   ├── feature_selection/
│   │   ├── source_model_evaluation/
│   │   ├── domain_adaptation/
│   │   └── explainability/
│   ├── models/                         # 模型权重 (.pth)
│   │   ├── source_models/
│   │   └── transfer_models/
│   ├── evaluation/                     # 评估结果
│   │   ├── source_classification/
│   │   ├── target_predictions/
│   │   └── model_comparison/
│   └── tables/                         # 表格数据
│       ├── cross_validation/
│       └── feature_statistics/
│
├── docs/                      # 文档与报告素材
├── legacy/                    # 历史实验代码（仅作参考）
└── notebooks/                 # Jupyter 探索性笔记本
```

### Directory Description

#### `configs/`

存放 YAML 配置文件，集中管理特征工程、源域训练、迁移学习和可解释性分析的参数，便于统一调参与实验复现。

#### `data/`

统一管理原始数据、中间结果、处理后的结构化数据及辅助元数据。

- `data/raw/`：原始源域与目标域 `.mat` 振动数据
- `data/interim/`：滑动窗口切片结果、逐文件提取的特征等中间文件
- `data/processed/`：可直接用于训练或迁移学习的 CSV 数据集
- `data/metadata/`：周期统计表、特征映射表等辅助文件

#### `src/`

项目核心源码目录，按功能模块组织。

- `src/data_processing/`：数据预处理、周期计算、滑动窗口采样、数据标准化
- `src/feature_engineering/`：时频域特征提取、特征融合、相关性分析与筛选
- `src/models/`：所有模型类的纯定义，与训练逻辑解耦
- `src/training/`：源域分类训练脚本（含 5 折交叉验证、指标记录与可视化）
- `src/domain_adaptation/`：迁移学习方法实现，含 MMD/DANN 损失函数及多种迁移模型
- `src/explainability/`：特征重要性、MMD 损失曲线、权重热力图与 3D 可视化
- `src/utils/`：随机种子、路径管理、评估指标、绘图工具等通用函数

#### `scripts/`

用于统一调度各模块的入口脚本，按问题编号对应四个实验阶段，适合快速复现完整实验流程。

#### `outputs/`

统一存放模型权重、分类结果、评估表格和可视化图表，按类型分子目录管理。

#### `docs/`

存放赛题材料、方法说明文档和报告中使用的图表整理版本。

#### `legacy/`

保留历史版本代码和探索性实验（如 AutoDL 实验、KLD 损失实验等），不作为当前主流程的核心依赖，仅供追溯项目演化过程。

#### `notebooks/`

存放探索性分析 Jupyter Notebook。

---

## 4. Data Description

### 4.1 Source Domain Data

源域数据为 CWRU（Case Western Reserve University）轴承台架实验振动数据，包含不同采样频率与不同传感器位置的数据。

```text
data/raw/source_domain/
├── cwru_12khz_de/              # 12kHz 采样，驱动端（DE）传感器
├── cwru_12khz_fe/              # 12kHz 采样，风扇端（FE）传感器
├── cwru_48khz_de/              # 48kHz 采样，驱动端传感器
└── cwru_48khz_normal/          # 48kHz 采样，正常状态
```

源域数据涵盖多种故障类型，包括：

- **故障位置**：内圈（Inner Race）、外圈（Outer Race）、滚动体（Ball）
- **故障直径**：0.007"、0.014"、0.021"
- **负载**：0hp、1hp、2hp、3hp

源域数据主要用于：

- 周期计算与数据集统计分析
- 滑动窗口切片
- 时频域特征提取
- 源域多分类模型训练与交叉验证

### 4.2 Target Domain Data

目标域数据为列车实际轴承振动样本，采用 A–P 的文件命名方式。

```text
data/raw/target_domain/
└── train_bearing_a_to_p/       # 列车轴承 A 至 P 的振动数据
```

目标域数据主要用于：

- 目标域滑动窗口切片
- 目标域特征提取
- 迁移学习诊断
- 目标域故障类别预测

### 4.3 Processed Data

处理后的结构化数据位于 `data/processed/`，可直接用于模型训练：

```text
data/processed/
├── source_dataset_23cls.csv        # 源域 23 类完整数据集
├── source_dataset_7cls.csv         # 源域 7 类数据集（按故障直径合并）
├── source_dataset_4cls.csv         # 源域 4 类数据集（按故障位置合并）
├── source_selected_features.csv    # 特征筛选后的数据集
├── source_normalized.csv           # 标准化后的数据集
├── target_dataset.csv              # 目标域带标签数据集
├── transfer_source_input.csv       # 迁移学习源域输入
└── transfer_target_input.csv       # 迁移学习目标域输入
```

---

## 5. Environment Requirements

推荐环境如下：

| 包名 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 运行环境 |
| PyTorch | >= 2.0.0 | 深度学习框架 |
| NumPy | >= 1.24.0 | 数值计算 |
| Pandas | >= 2.0.0 | 数据处理 |
| SciPy | >= 1.10.0 | 科学计算与信号处理 |
| Scikit-learn | >= 1.2.0 | 机器学习工具 |
| XGBoost | >= 2.0.0 | 梯度提升树 |
| LightGBM | >= 4.0.0 | 基于直方图的梯度提升 |
| Matplotlib | >= 3.7.0 | 绑图 |
| Seaborn | >= 0.12.0 | 统计可视化 |
| tqdm | >= 4.65.0 | 进度条 |
| PyYAML | >= 6.0 | 配置文件解析 |
| h5py | >= 3.8.0 | MATLAB 数据读取 |

---

## 6. Installation

### 6.1 Clone Repository

```bash
git clone <your-repo-url>
cd cross_domain_bearing_fault_diagnosis
```

### 6.2 Install Dependencies

```bash
pip install -r requirements.txt
```

> 建议使用虚拟环境（conda / venv）进行隔离安装。

---

## 7. Quick Start

### 7.1 Prepare Raw Data

请先将原始数据放入以下目录：

- 源域数据 → `data/raw/source_domain/`
- 目标域数据 → `data/raw/target_domain/`

### 7.2 Run Source Feature Pipeline

执行源域数据预处理、周期计算、滑动窗口和特征提取流程（对应问题1）：

```bash
python scripts/run_source_feature_pipeline.py
```

该流程依次执行：周期计算 → 数据集概览可视化 → 滑动窗口采样 → 特征提取 → 特征合并与数据集构建。

### 7.3 Run Source Domain Model Training

执行源域多模型训练与交叉验证评估（对应问题2）：

```bash
python scripts/run_source_model_training.py
```

该流程依次执行：特征选择与归一化 → 深度学习模型训练（CNN / CNN-LSTM / CNN-BiLSTM / CNN-Transformer / ResNet-Transformer）→ 传统ML模型训练（XGBoost / LightGBM）。

### 7.4 Run Domain Adaptation

执行迁移学习训练与目标域故障预测（对应问题3）：

```bash
python scripts/run_domain_adaptation.py
```

该流程依次执行：目标域数据准备 → 迁移数据预处理 → MMD域适应训练（4种模型组合）→ DANN域适应训练 → 迁移模型结果比较。

### 7.5 Run Explainability Analysis

执行特征重要性分析、损失曲线分析及模型权重可视化（对应问题4）：

```bash
python scripts/run_explainability_analysis.py
```

---

## 8. Methodology

### 8.1 Data Processing

数据处理模块主要包括：

- **周期计算**：基于轴承故障特征频率（BPFI、BPFO、BSF、FTF）计算故障周期
- **滑动窗口采样**：固定窗口长度（1024点）与重叠率（50%）进行样本切片
- **样本标签构建**：支持 23 类（细粒度）、7 类（按直径合并）、4 类（按位置合并）三种粒度
- **数据标准化**：使用 `StandardScaler` 进行 Z-score 标准化

### 8.2 Feature Engineering

特征工程模块主要包括：

- **时域特征**：均值、标准差、峰值、峰峰值、均方根、峭度、偏度、波形因子、峰值因子、脉冲因子、裕度因子等
- **频域特征**：频谱均值、重心频率、中值频率、频率方差等
- **特征相关性分析**：基于皮尔逊相关系数的热力图可视化
- **特征重要性排名**：基于 XGBoost 的特征重要性评估
- **Top-N 特征筛选**：选取最优特征子集以降低维度、提升效率

### 8.3 Source Domain Classification Models

本项目实现并比较了 7 种源域分类模型，所有模型均采用 **5 折交叉验证**进行评估。

#### 深度学习模型

| 模型 | 架构 | 核心设计 |
|------|------|---------|
| **CNN1D** | 3层Conv1d + AdaptiveAvgPool1d | 经典一维卷积，全局平均池化 |
| **CNN-LSTM** | 3层Conv1d + 2层BiLSTM | CNN提取局部特征，LSTM捕获时序依赖 |
| **CNN-BiLSTM** | 3层Conv1d(单次池化) + 2层BiLSTM | 减少池化保留时序信息，BiLSTM双向建模 |
| **CNN-Transformer** | 3层Conv1d + 3层Transformer | CNN提取特征，Transformer捕获全局依赖 |
| **ResNet-Transformer** | 2个ResNetBlock + 2层Transformer | 残差连接缓解退化，Transformer全局建模 |

#### 传统机器学习模型

| 模型 | 配置 | 说明 |
|------|------|------|
| **XGBoost** | n_estimators=200, lr=0.1, max_depth=8 | 梯度提升树，支持多分类 |
| **LightGBM** | n_estimators=200, lr=0.05, max_depth=8 | 基于直方图的梯度提升，训练速度快 |

#### 评估指标

- Accuracy（准确率）
- Precision（精确率，加权平均）
- Recall（召回率，加权平均）
- F1-score（F1 分数，加权平均）
- Confusion Matrix（混淆矩阵）
- 每折训练/验证损失曲线

### 8.4 Domain Adaptation Methods

跨域迁移部分实现两种经典域适应策略：

#### MMD（Maximum Mean Discrepancy）

- **原理**：通过多核高斯核函数计算源域与目标域特征分布之间的距离，最小化该距离以实现特征对齐
- **实现**：`src/domain_adaptation/mmd_loss.py`
- **核函数配置**：kernel_mul=2.0, kernel_num=5
- **迁移组合**：
  - CNN-BiLSTM + MMD（含 LinearMapping 维度对齐）
  - CNN-BiLSTM + MMD v1（改进版本）
  - ResNet-Transformer + MMD
  - XGBoost + MMD（基于自定义目标函数的正则化）

#### DANN（Domain-Adversarial Neural Network）

- **原理**：通过梯度反转层（GRL）实现对抗训练，使特征提取器学习域不变表示
- **实现**：`src/domain_adaptation/dann_loss.py`
- **核心组件**：GradientReversalLayer（前向恒等、反向梯度取负并缩放）
- **alpha 调度**：支持 linear / step / constant 三种策略

### 8.5 Explainability Analysis

可解释性分析模块提供多维度模型解读能力：

- **特征重要性分析**：基于树模型 feature_importance 的排名可视化
- **特征相关性可视化**：皮尔逊相关系数热力图
- **MMD 损失曲线**：训练过程中 MMD 损失的变化趋势
- **权重热力图**：卷积核权重的二维热力图，用于分析模型关注区域
- **权重 3D 可视化**：模型参数的交互式三维可视化

---

## 9. Main Output Files

实验完成后，主要结果位于以下目录：

### 9.1 Figures

```text
outputs/figures/
├── dataset_overview/          # 数据集概览图
├── bearing_geometry_analysis/ # 轴承几何分析图
├── feature_analysis/          # 特征分布与分析图
├── feature_selection/         # 特征选择结果图
├── source_model_evaluation/   # 源域模型评估图（损失曲线、混淆矩阵）
├── domain_adaptation/         # 域适应结果图
└── explainability/            # 可解释性分析图
    ├── mmd_loss/              #   MMD 损失变化图
    ├── weight_3d/             #   权重 3D 可视化
    └── weight_heatmap/        #   权重热力图
```

### 9.2 Model Weights

```text
outputs/models/
├── source_models/             # 源域训练模型权重 (.pth)
│   ├── cnn_model.pth
│   ├── cnn_lstm_model.pth
│   ├── cnn_bilstm_model.pth
│   ├── cnn_transformer_model.pth
│   └── resnet_transformer_model.pth
└── transfer_models/           # 迁移学习模型权重 (.pth)
```

### 9.3 Evaluation Results

```text
outputs/evaluation/
├── source_classification/     # 源域分类结果
│   ├── CNN23/                 #   CNN 23类：分类报告、混淆矩阵
│   ├── CNNLSTM23/             #   CNN-LSTM 23类
│   ├── CNNBiLSTM23/           #   CNN-BiLSTM 23类
│   ├── CNNTransformer23/      #   CNN-Transformer 23类
│   ├── ResNetTransformer23/   #   ResNet-Transformer 23类
│   ├── XGBoost23/             #   XGBoost 23类
│   └── LightGBM23/            #   LightGBM 23类
├── target_predictions/        # 目标域预测结果
└── model_comparison/          # 模型对比结果
```

### 9.4 Tables

```text
outputs/tables/
├── cross_validation/          # 交叉验证统计表
│   ├── cross_validation_summary.csv
│   └── detailed_cv_results.csv
└── feature_statistics/        # 特征统计表
    └── all_features_importance.csv
```

---

## 10. Configuration Management

项目通过 `configs/` 下的 YAML 文件统一管理实验参数，主要配置项如下：

### feature_pipeline.yaml

```yaml
sliding_window:
  window_size: 1024
  overlap_ratio: 0.5
classification_labels:
  num_classes_23: 23
  num_classes_7: 7
  num_classes_4: 4
```

### source_model_training.yaml

```yaml
training:
  cv:
    n_splits: 5
    random_state: 42
  epochs: 50
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 1.0e-5
```

### domain_adaptation.yaml

```yaml
domain_adaptation:
  methods:
    cnn_bilstm_mmd:
      mmd_weight: 0.1
      kernel_mul: 2.0
      kernel_num: 5
    cnn_bilstm_dann:
      alpha_schedule: "linear"
```

---

## 11. Suggested Workflow

如果你希望完整复现本项目，建议按照以下顺序执行：

1. **准备数据** — 将原始源域和目标域 `.mat` 数据放置到 `data/raw/` 对应目录
2. **特征工程** — 运行 `python scripts/run_source_feature_pipeline.py`
3. **源域训练** — 运行 `python scripts/run_source_model_training.py`
4. **域适应** — 运行 `python scripts/run_domain_adaptation.py`
5. **可解释性** — 运行 `python scripts/run_explainability_analysis.py`
6. **查看结果** — 检查 `outputs/` 中的模型权重、图表与评估结果

---

## 12. Legacy Contents

`legacy/` 目录中保留了项目历史版本与探索性实验，包括但不限于：

- 比赛原始答题版本代码（answer1 ~ answer4）
- AutoDL 平台远程实验结果
- CNN + MMD 独立调参实验
- XGBoost + MMD 独立调参实验
- KLD 替代损失函数实验
- 概率矩阵热力图实验
- 更细粒度的权重可视化实验
- 早期特征工程探索代码

这些内容主要用于追溯项目演化过程，不建议直接作为当前主流程入口。

---

## 13. Notes

- 本项目为**故障诊断算法原型项目**，主要用于方法验证、实验复现与项目展示。
- 当前仓库重点体现算法流程与实验组织，不包含在线部署、接口服务或前后端系统。
- 训练脚本中 `src/models/` 已提取出纯模型定义，但训练脚本仍保留原始独立运行能力，两者并存以保持兼容。
- `configs/` 下的 YAML 配置文件定义了推荐的实验参数，实际训练脚本目前以硬编码参数为主，后续可逐步迁移为配置驱动。
- 若后续准备将项目包装为完整工程项目，可进一步补充：
  - 统一日志系统（logging / wandb）
  - 配置驱动训练（OmegaConf / hydra）
  - 命令行参数管理（argparse / click）
  - 模型推理接口与批量预测脚本
  - Web 可视化展示（Streamlit / Gradio）

---

## 14. Future Work

后续可扩展方向包括：

- 将所有训练脚本改造为配置驱动，实现统一参数管理
- 引入更完善的实验跟踪系统（如 MLflow / Weights & Biases）
- 增加模型推理脚本与批量预测接口
- 扩展为故障诊断演示平台（Streamlit / Gradio Web UI）
- 增加更多跨域适配方法对比实验（如 CORAL、JDA 等）
- 增加自动生成实验报告的脚本
- 支持更多公开轴承数据集（如 PU、MFPT、PHM 2012）

