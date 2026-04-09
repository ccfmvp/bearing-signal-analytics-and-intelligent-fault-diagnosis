# Cross-Domain Bearing Fault Diagnosis

> 面向高速列车轴承场景的跨域智能故障诊断原型项目

## 1. Project Overview

本项目面向高速列车轴承故障诊断场景，构建了一套从源域振动信号处理、特征工程、源域分类建模到目标域迁移学习诊断的完整算法流程。  
项目重点关注真实工业数据中常见的噪声干扰、样本不平衡、目标域标签缺失以及跨域分布偏移等问题，并结合可解释性分析提升模型结果的透明度与可信度。

项目核心思想如下：

- 以台架轴承振动数据作为**源域**
- 以列车实际轴承振动样本作为**目标域**
- 先完成源域数据清洗、周期计算、滑动窗口切片与特征提取
- 再进行源域多模型分类训练与评估
- 最后通过领域自适应方法完成源域到目标域的迁移诊断

---

## 2. Project Objectives

本项目主要包含以下功能模块：

1. **源域振动信号预处理**
   - 周期计算
   - 数据集统计分析
   - 滑动窗口切片
   - 标签构建

2. **特征工程**
   - 时域特征提取
   - 频域特征提取
   - 特征融合与标签汇总
   - 特征相关性分析与重要性筛选

3. **源域分类训练**
   - CNN
   - CNN-LSTM
   - CNN-BiLSTM
   - CNN-Transformer
   - ResNet-Transformer
   - XGBoost
   - LightGBM

4. **跨域迁移学习诊断**
   - MMD-based domain adaptation
   - DANN-based domain adaptation
   - 源域到目标域的特征对齐与故障类别预测

5. **可解释性分析**
   - 特征重要性分析
   - MMD 损失变化分析
   - 模型权重热力图与 3D 可视化

---

## 3. Repository Structure

```text
cross_domain_bearing_fault_diagnosis/
├── README.md
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── metadata/
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── models/
│   ├── training/
│   ├── domain_adaptation/
│   ├── explainability/
│   └── utils/
├── scripts/
├── outputs/
│   ├── figures/
│   ├── models/
│   ├── evaluation/
│   └── tables/
├── docs/
├── legacy/
└── notebooks/
```

### Directory Description

#### `configs/`
存放配置文件，包括特征工程、源域训练、迁移学习和可解释性分析的参数配置。

#### `data/`
统一管理原始数据、中间结果、处理后的结构化数据及辅助元数据。

- `data/raw/`：原始源域与目标域 `.mat` 数据
- `data/interim/`：滑动窗口结果、原始特征等中间文件
- `data/processed/`：可直接用于训练或迁移学习的数据集
- `data/metadata/`：周期统计表、特征映射表等辅助文件

#### `src/`
项目核心源码目录。

- `src/data_processing/`：数据预处理、周期计算、滑动窗口
- `src/feature_engineering/`：特征提取、融合、筛选与可视化
- `src/models/`：模型定义
- `src/training/`：源域分类训练与交叉验证
- `src/domain_adaptation/`：迁移学习训练与对比实验
- `src/explainability/`：可解释性分析
- `src/utils/`：通用工具函数

#### `scripts/`
用于统一调度各模块的入口脚本，适合快速复现实验流程。

#### `outputs/`
统一存放模型权重、分类结果、评估表格和可视化图表。

#### `docs/`
存放赛题材料、方法说明文档和报告中使用的图表整理版本。

#### `legacy/`
保留历史版本代码和探索性实验，不作为当前主流程的核心依赖。

#### `notebooks/`
存放探索性分析 notebook。

---

## 4. Data Description

## 4.1 Source Domain Data

源域数据为轴承台架实验振动数据，包含不同采样频率与不同传感器位置的数据。  
默认目录如下：

```text
data/raw/source_domain/
├── cwru_12khz_de/
├── cwru_12khz_fe/
├── cwru_48khz_de/
└── cwru_48khz_normal/
```

源域数据主要用于：

- 周期计算
- 滑动窗口切片
- 特征提取
- 分类模型训练与验证

---

## 4.2 Target Domain Data

目标域数据为列车轴承振动样本，采用 A–P 的文件命名方式。  
默认目录如下：

```text
data/raw/target_domain/
└── train_bearing_a_to_p/
```

目标域数据主要用于：

- 目标域滑动窗口切片
- 目标域特征提取
- 迁移学习诊断
- 目标域故障类别预测

---

## 4.3 Processed Data

处理后的数据通常位于：

```text
data/processed/
├── source_dataset_23cls.csv
├── source_dataset_7cls.csv
├── source_dataset_4cls.csv
├── source_selected_features.csv
├── source_normalized.csv
├── target_dataset.csv
├── transfer_source_input.csv
└── transfer_target_input.csv
```

> 如果原始数据不随仓库分发，请手动将数据放置到 `data/raw/` 对应目录下。

---

## 5. Environment Requirements

推荐环境如下：

- Python 3.10+
- NumPy
- Pandas
- SciPy
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- PyTorch

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

---

## 7. Quick Start

## 7.1 Prepare Raw Data

请先将原始数据放入以下目录：

- `data/raw/source_domain/`
- `data/raw/target_domain/`

---

## 7.2 Run Source Feature Pipeline

执行源域数据预处理、周期计算、滑动窗口和特征提取流程：

```bash
python scripts/run_source_feature_pipeline.py
```

---

## 7.3 Run Source Domain Model Training

执行源域多模型训练与交叉验证评估：

```bash
python scripts/run_source_model_training.py
```

---

## 7.4 Run Domain Adaptation

执行迁移学习训练与目标域故障预测：

```bash
python scripts/run_domain_adaptation.py
```

---

## 7.5 Run Explainability Analysis

执行特征重要性分析、损失曲线分析及模型权重可视化：

```bash
python scripts/run_explainability_analysis.py
```

---

## 8. Methodology

## 8.1 Data Processing

数据处理模块主要包括：

- 基于轴承故障机理的周期计算
- 源域和目标域滑动窗口采样
- 样本标签构建
- 数据标准化与结构化输出

---

## 8.2 Feature Engineering

特征工程模块主要包括：

- 时域特征提取
- 频域特征提取
- 特征融合
- 特征相关性热力图
- 特征重要性排名
- Top-N 特征筛选

---

## 8.3 Source Domain Classification Models

本项目实现并比较了以下源域分类模型：

- CNN
- CNN-LSTM
- CNN-BiLSTM
- CNN-Transformer
- ResNet-Transformer
- XGBoost
- LightGBM

评估方式包括：

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- K-fold Cross Validation

---

## 8.4 Domain Adaptation Methods

跨域迁移部分主要包括：

- **MMD**：通过最小化源域与目标域特征分布差异实现对齐
- **DANN**：通过对抗训练学习域不变特征表示

当前迁移实验主要围绕以下模型展开：

- CNN-BiLSTM + MMD
- CNN-BiLSTM + DANN
- ResNet-Transformer + MMD
- XGBoost + MMD

---

## 8.5 Explainability Analysis

可解释性分析主要包括：

- 特征重要性分析
- 特征相关性可视化
- MMD 损失曲线分析
- 权重热力图可视化
- 权重 3D 可视化

---

## 9. Main Output Files

实验完成后，主要结果位于以下目录：

### 9.1 Figures

```text
outputs/figures/
├── dataset_overview/
├── bearing_geometry_analysis/
├── feature_analysis/
├── feature_selection/
├── source_model_evaluation/
├── domain_adaptation/
└── explainability/
```

### 9.2 Model Weights

```text
outputs/models/
├── source_models/
└── transfer_models/
```

### 9.3 Evaluation Results

```text
outputs/evaluation/
├── source_classification/
├── target_predictions/
└── model_comparison/
```

### 9.4 Tables

```text
outputs/tables/
├── cross_validation/
└── feature_statistics/
```

---

## 10. Suggested Workflow

如果你希望完整复现本项目，建议按照以下顺序执行：

1. 准备原始源域和目标域数据
2. 运行源域特征工程流程
3. 运行源域模型训练与交叉验证
4. 运行迁移学习训练
5. 运行可解释性分析
6. 查看 `outputs/` 中的模型、图表与结果文件

---

## 11. Legacy Contents

`legacy/` 目录中保留了项目历史版本与探索性实验，包括但不限于：

- 比赛原始答题版本代码
- AutoDL 平台实验结果
- CNN + MMD 独立实验
- XGBoost + MMD 独立实验
- KLD 替代损失函数实验
- 概率矩阵热力图实验
- 更细粒度的权重可视化实验

这些内容主要用于追溯项目演化过程，不建议直接作为当前主流程入口。

---

## 12. Notes

- 本项目为**故障诊断算法原型项目**，主要用于方法验证、实验复现与项目展示。
- 当前仓库重点体现算法流程与实验组织，不包含在线部署、接口服务或前后端系统。
- 若你后续准备将项目继续包装为工程项目，可进一步补充：
  - 统一日志系统
  - 配置管理
  - 命令行参数管理
  - 模型推理接口
  - Web 可视化展示

---

## 13. Future Work

后续可扩展方向包括：

- 增加统一实验配置管理
- 引入更完善的日志与实验跟踪系统
- 增加模型推理脚本与批量预测接口
- 扩展为故障诊断演示平台
- 增加更多跨域适配方法对比实验
- 增加自动生成实验报告的脚本