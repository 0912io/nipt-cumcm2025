"""预先登记的常数。修改这些值会改变修订研究的主分析，不应事后为了贴近某一周数而调整。"""

from __future__ import annotations

Y_THRESHOLD = 0.04
PRIMARY_SLOPES = (1.0, 3.0, 6.0)
PRIMARY_LAMBDA = 1.0
T_MIN = 10.0
T_MAX = 25.0
T_STEP = 0.1
RISK_NORM_LOW = 10.0
RISK_NORM_HIGH = 40.0
N_MIN_GROUP = 20
K_CANDIDATES = (2, 3, 4, 5)
RANDOM_SEED = 2025
N_BOOT_TIME = 200
N_NOISE = 100
MAX_BMI_CUTS = 36
OFFICIAL_BMI_CUTS = (28.0, 32.0, 36.0, 40.0)

# 问题一主模型禁止使用的列：它们不是检测前用来解释 Y 浓度的协变量。
BANNED_Y_PREDICTORS = ("zy", "x_frac", "aneuploidy", "birth_healthy")
