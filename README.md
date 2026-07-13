# NIPT Data Analysis — 2025 CUMCM Problem C

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUMCM](https://img.shields.io/badge/CUMCM-2025%20Problem%20C-red.svg)](https://www.mcm.edu.cn/)

> **2025 高教社杯全国大学生数学建模竞赛 (CUMCM) C 题：NIPT 数据分析与优化**
>
> 基于 Ridge 回归、风险优化、聚类分析和集成学习的无创产前检测数据分析全流程。

## 📋 项目概述

本项目针对 NIPT（Non-Invasive Prenatal Testing，无创产前检测）数据，从四个递进维度进行建模分析：

| 问题 | 目标 | 核心方法 | 关键技术 |
|:---:|------|---------|---------|
| **一** | Y 染色体浓度预测 | Ridge 回归 + 特征工程 | 交互项构造、Spearman 相关性、残差诊断 |
| **二** | 最优检测时点确定 | 线性回归 + 风险函数最小化 | scipy.optimize、正态性检验、3σ 异常值处理 |
| **三** | 个性化检测策略 | 聚类分析 + 差分进化 | KMeans/GMM/凝聚聚类、敏感性分析、并行优化 |
| **四** | 女胎异常检测 | 集成学习分类 | Random Forest / XGBoost / SVM、GridSearchCV |

## 🗂 项目结构

```
nipt-cumcm2025/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                        # 数据目录（需自行放置附件.xlsx）
│   └── README.md
├── src/                         # 源代码
│   ├── problem1_ridge_regression.py      # 问题一：Ridge 回归
│   ├── problem2_optimal_timing.py        # 问题二：最优检测时点
│   ├── problem3_clustering_optimization.py # 问题三：聚类与优化
│   └── problem4_female_anomaly_detection.py # 问题四：异常检测
├── outputs/                     # 输出目录（图表、CSV、报告）
├── paper/                       # 最终论文
│   └── final_paper.pdf
└── docs/                        # 补充文档
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- 依赖安装：

```bash
pip install -r requirements.txt
```

### 数据准备

将竞赛提供的 `附件.xlsx` 放置于项目根目录下：

```
nipt-cumcm2025/
├── 附件.xlsx          # <-- 放置于此
├── src/
├── ...
```

### 按顺序运行

```bash
# 问题一：Y 染色体浓度 Ridge 回归预测
python src/problem1_ridge_regression.py

# 问题二：最优检测时点优化
python src/problem2_optimal_timing.py

# 问题三：聚类分析与个性化策略
python src/problem3_clustering_optimization.py

# 问题四：女胎异常检测模型
python src/problem4_female_anomaly_detection.py
```

每个脚本会在项目根目录下生成对应的 `问题X结果/` 文件夹，包含全部图表和 CSV 数据。

## 📊 方法详情

### 问题一：Ridge 回归预测模型

- **数据预处理**：孕周字符串解析 (`11w+6` → 83天)、缺失值时序填充、年龄过滤 (<20岁)、BMI 异常值 IQR 截尾
- **特征工程**：构造交互项（BMI×孕周、年龄×BMI 等）和高阶项（二次项）
- **模型**：RidgeCV 超参数优化 + 标准化 + 5 折交叉验证
- **诊断**：残差分析（QQ图、Shapiro-Wilk 检验）、Spearman 相关系数热力图、预测精度评估

### 问题二：最优检测时点

- **数据筛选**：男胎（Y 浓度 ≥ 4%）首次达标记录
- **统计检验**：Shapiro-Wilk / D'Agostino-Pearson 正态性检验
- **BMI 分组**：5 组 [20-28, 28-32, 32-36, 36-40, 40+)
- **优化模型**：自定义风险函数 = 时间风险 + 浓度风险 + 模型不确定性 + R² 偏差，使用 `scipy.optimize.minimize` (L-BFGS-B)
- **误差量化**：最小二乘法参数标准误差、t 分布预测区间

### 问题三：聚类与高级优化

- **多方法聚类**：KMeans + AgglomerativeClustering + GaussianMixture
- **多尺度标准化**：StandardScaler / RobustScaler 对比
- **聚类评估**：Silhouette + Calinski-Harabasz + Davies-Bouldin
- **高级优化**：L-BFGS-B + 差分进化 + basinhopping 多起点
- **敏感性分析**：Bootstrap + 蒙特卡洛模拟

### 问题四：女胎异常检测

- **特征处理**：Z-score 标准化、孕周格式解析
- **多模型对比**：Random Forest / GradientBoosting / XGBoost / LogisticRegression / SVM
- **超参数搜索**：GridSearchCV
- **评估指标**：ROC-AUC、Precision-Recall、混淆矩阵、特征重要性

## 📈 关键发现

1. **孕周是 Y 染色体浓度的最强预测因子**，Ridge 回归 R² 表现稳健
2. **BMI 显著影响检测达标时间**——高 BMI 组别需要延迟 2-4 周检测
3. **聚类分析揭示了 3-4 个不同的孕妇亚群**，支持个性化检测策略
4. **XGBoost 在女胎异常检测上表现最优**，特征重要性分析为临床决策提供可解释性

## 📄 论文

最终论文见 [paper/final_paper.pdf](paper/final_paper.pdf)

## ⚠️ 数据说明

- 本项目使用 2025 年 CUMCM C 题官方提供的数据集
- 原始数据文件 `附件.xlsx` 未包含在仓库中（竞赛数据版权）
- 使用时请将数据文件放置于项目根目录

## 📝 License

MIT License — 详见 [LICENSE](LICENSE)

---

*CUMCM 2025 · Problem C · NIPT Data Analysis*
