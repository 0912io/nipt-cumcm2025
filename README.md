# NIPT 时点选择与胎儿异常判定

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUMCM](https://img.shields.io/badge/CUMCM-2025%20Problem%20C-red.svg)](https://www.mcm.edu.cn/)

本仓库包含两部分，不要混用它们的数字。

| 部分 | 是什么 | 从哪里看 |
| --- | --- | --- |
| 历史参赛稿 | 2025 年 C 题提交时的论文和对应脚本 | `paper/final_paper.pdf`，`src/problem1_ridge_regression.py` 至 `src/problem4_female_anomaly_detection.py` |
| 赛后修订研究 | 公开仓库上的方法修订，不是重新提交的竞赛论文 | `paper/revised_study.md`，`src/revised/` |

历史脚本保持原有计算，只在文件开头注明了它们的位置。修订研究不把 Y 染色体 Z 值、非整倍体或出生结局用来解释 Y 浓度，不删除从未达标的孕妇，也不用 \(Y\le 0.08\) 定义女胎。修订论文里的表来自本地官方附件上的一次运行，原始工作簿不在仓库中。

## 数据

官方工作簿有两张表：`男胎检测数据` 与 `女胎检测数据`。男胎和女胎按表名区分。把 `附件.xlsx` 放在仓库根目录后再运行；该文件匹配 `.gitignore` 中的 `*.xlsx`，不要提交。说明见 [data/README.md](data/README.md)。

## 运行修订分析

```bash
pip install -r requirements.txt
python -m src.revised.run_analysis
```

缺少 `附件.xlsx` 时，程序会说明文件应放在本地且不能入库，并以退出码 2 结束，不会填写编造的结果。小表写到 `paper/tables/`。同一份运行副本在 `results/revised/`，该目录被忽略。

不读取工作簿的检查：

```bash
python -m pytest -q tests
python -m compileall -q src
```

推送到 `main` 或打开 Pull Request 时，GitHub Actions 在 Ubuntu 24.04 上用 Python 3.9–3.12 安装依赖、编译 `src/` 并运行 `tests/`。

## 历史脚本

四个历史脚本仍从当前工作目录读取 `附件.xlsx`，并把图和表写到仓库根目录的 `问题一结果/` 至 `问题四结果/`。这些目录已被忽略。它们复现的是参赛稿里的流程，不是 `paper/revised_study.md` 的流程。

```bash
python src/problem1_ridge_regression.py
python src/problem2_optimal_timing.py
python src/problem3_clustering_optimization.py
python src/problem4_female_anomaly_detection.py
```

## 目录

```
src/problem1_ridge_regression.py          # 历史：问题一
src/problem2_optimal_timing.py            # 历史：问题二
src/problem3_clustering_optimization.py   # 历史：问题三
src/problem4_female_anomaly_detection.py  # 历史：问题四
src/revised/                              # 赛后修订
paper/final_paper.pdf                     # 历史参赛稿，保持原文件
paper/revised_study.md                    # 赛后修订论文
paper/tables/                             # 修订分析的小表
```

## 许可

MIT License，见 [LICENSE](LICENSE)。竞赛附件另有版权，不因本仓库的代码许可而可以再分发。
