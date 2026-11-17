# Data Directory

## Required File

The four analysis scripts read the competition workbook from the **repository root**, not from this folder. Place `附件.xlsx` at the repository root and run the scripts from there:

```
附件.xlsx
```

Each script opens that relative path (`pd.read_excel("附件.xlsx")` or the same filename passed into its loader). `data/附件.xlsx` is not read.

## Data Source

- **Competition**: 2025 高教社杯全国大学生数学建模竞赛 (CUMCM)
- **Problem**: C 题 — NIPT 数据分析
- **Official Site**: https://www.mcm.edu.cn/

## ⚠️ Note

The dataset is copyrighted competition material and is **not included** in this repository. You must obtain it from official CUMCM channels. This directory only holds this note.
