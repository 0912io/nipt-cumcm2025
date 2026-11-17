# 历史参赛脚本：对应 paper/final_paper.pdf，仅供对照复现。
# 赛后修订研究不修改本文件的计算行为；新的分析在 src/revised/。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings
import os
import logging

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STSong']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
rizhiqri = logging.getLogger(__name__)

class Peizhi:
    TARGET_VARIABLE = 'Y染色体浓度'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    ALPHA_RANGE_MIN = -3
    ALPHA_RANGE_MAX = 3
    ALPHA_RANGE_POINTS = 50
    OUTPUT_DIR = "问题一结果"
    IMAGE_DPI = 300
    CSV_ENCODING = 'utf-8-sig'
    
    CORE_PREDICTORS = [
        '孕妇BMI', '检测时怀孕天数', '年龄', '体重', '身高',
        'X染色体浓度', '18号染色体的Z值', 'Y染色体的Z值', 
        '13号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
        'GC含量', '末次月经距检测日天数',
        'IVF妊娠_编码', '怀孕次数_编码', '胎儿健康_编码',
        'T13_异常', 'T18_异常', 'T21_异常'
    ]

def yanzheng_shujuji(dataset: pd.DataFrame) -> None:
    """数据验证"""
    if dataset.empty:
        raise ValueError("❌ 数据集为空")
    
    if Peizhi.TARGET_VARIABLE not in dataset.columns:
        raise KeyError(f"❌ 目标变量 '{Peizhi.TARGET_VARIABLE}' 不存在")
    
    target_series = dataset[Peizhi.TARGET_VARIABLE]
    if target_series.isna().all():
        raise ValueError("❌ 目标变量全部为缺失值")
    
    available_predictors = [col for col in Peizhi.CORE_PREDICTORS if col in dataset.columns]
    if len(available_predictors) < 5:
        raise ValueError(f"❌ 可用预测变量数量不足({len(available_predictors)})")
    
    rizhiqri.info(f"✅ 数据验证通过 - 样本数: {len(dataset)}, 可用预测变量: {len(available_predictors)}")

def jiance_queshi_zhi(dataset: pd.DataFrame) -> pd.DataFrame:
    """检测缺失值"""
    missing_analysis = pd.DataFrame({
        '变量名': dataset.columns,
        '缺失数量': dataset.isnull().sum(),
        '缺失比例(%)': (dataset.isnull().sum() / len(dataset) * 100).round(2),
        '数据类型': dataset.dtypes.values
    })
    
    missing_analysis = missing_analysis[missing_analysis['缺失数量'] > 0].sort_values('缺失比例(%)', ascending=False)
    
    if len(missing_analysis) > 0:
        rizhiqri.info(f"发现 {len(missing_analysis)} 个变量存在缺失值")
    else:
        rizhiqri.info("✅ 未发现缺失值")
    
    return missing_analysis

def jiexi_yunzhou(gestational_str):
    """解析孕周字符串格式 (如 '11w+6' -> 11*7+6 = 83天)"""
    if pd.isna(gestational_str) or gestational_str == '':
        return np.nan
    
    try:
        # 处理字符串格式，如 '11w+6', '15w+0', '20w'
        gestational_str = str(gestational_str).strip()
        
        if 'w' in gestational_str:
            parts = gestational_str.split('w')
            weeks = int(parts[0])
            
            if len(parts) > 1 and '+' in parts[1]:
                days = int(parts[1].replace('+', ''))
            else:
                days = 0
                
            # 转换为总天数
            total_days = weeks * 7 + days
            return total_days
        else:
            # 如果是纯数字，假设是周数
            return float(gestational_str) * 7
    except:
        return np.nan

def tianbu_yunzhou(dataset: pd.DataFrame) -> pd.DataFrame:
    """孕周时序填充"""
    dataset_filled = dataset.copy()
    week_columns = ['检测孕周', '孕周', '检测时怀孕天数', '妊娠周数']
    target_col = None
    
    for col in week_columns:
        if col in dataset_filled.columns:
            target_col = col
            break
    
    if target_col is None:
        return dataset_filled
    
    # 解析已在main函数中完成，这里不需要再次解析
    
    missing_before = dataset_filled[target_col].isnull().sum()
    
    if missing_before > 0:
        if '检测日期' in dataset_filled.columns:
            dataset_filled = dataset_filled.sort_values('检测日期')
        
        dataset_filled[target_col] = dataset_filled[target_col].interpolate(method='linear')
        dataset_filled[target_col] = dataset_filled[target_col].bfill().ffill()
        
        missing_after = dataset_filled[target_col].isnull().sum()
        filled_count = missing_before - missing_after
        rizhiqri.info(f"✅ {target_col} 时序填充完成: 填充了 {filled_count} 个缺失值")
    
    return dataset_filled

def guolv_muqin_nianling(dataset: pd.DataFrame) -> pd.DataFrame:
    """过滤生育年龄"""
    dataset_filtered = dataset.copy()
    age_columns = ['年龄', '母亲年龄', '孕妇年龄']
    age_col = None
    
    for col in age_columns:
        if col in dataset_filtered.columns:
            age_col = col
            break
    
    if age_col is None:
        return dataset_filtered
    
    original_count = len(dataset_filtered)
    dataset_filtered = dataset_filtered[dataset_filtered[age_col] >= 20]
    filtered_count = original_count - len(dataset_filtered)
    
    if filtered_count > 0:
        rizhiqri.info(f"✅ 年龄过滤完成: 删除了 {filtered_count} 条年龄<20岁的记录")
    
    return dataset_filtered

def jiance_yichang_zhi(dataset: pd.DataFrame, column: str, method: str = 'both') -> dict:
    """异常值检测"""
    if column not in dataset.columns:
        return {'outliers_iqr': [], 'outliers_zscore': [], 'summary': {}}
    
    data_series = dataset[column].dropna()
    if len(data_series) == 0:
        return {'outliers_iqr': [], 'outliers_zscore': [], 'summary': {}}
    
    jieguo = {
        'outliers_iqr': [],
        'outliers_zscore': [],
        'summary': {
            'total_count': len(data_series),
            'mean': data_series.mean(),
            'std': data_series.std(),
            'q1': data_series.quantile(0.25),
            'q3': data_series.quantile(0.75)
        }
    }
    
    if method in ['iqr', 'both']:
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
        jieguo['outliers_iqr'] = outliers_iqr.index.tolist()
        jieguo['summary']['iqr_bounds'] = (lower_bound, upper_bound)
    
    if method in ['zscore', 'both']:
        z_scores = np.abs(stats.zscore(data_series))
        outliers_zscore = data_series[z_scores > 3].index.tolist()
        jieguo['outliers_zscore'] = outliers_zscore
    
    return jieguo

def chuli_yichang_zhi(dataset: pd.DataFrame, column: str, method: str = 'cap') -> pd.DataFrame:
    """处理异常值"""
    if column not in dataset.columns:
        return dataset
    
    dataset_processed = dataset.copy()
    data_series = dataset_processed[column]
    
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_mask = (data_series < lower_bound) | (data_series > upper_bound)
    outliers_count = outliers_mask.sum()
    
    if outliers_count == 0:
        return dataset_processed
    
    if method == 'cap':
        dataset_processed.loc[data_series < lower_bound, column] = lower_bound
        dataset_processed.loc[data_series > upper_bound, column] = upper_bound
        rizhiqri.info(f"✅ {column} 异常值截尾处理完成: 处理了 {outliers_count} 个异常值")
    elif method == 'remove':
        dataset_processed = dataset_processed[~outliers_mask]
        rizhiqri.info(f"✅ {column} 异常值删除完成: 删除了 {outliers_count} 条记录")
    
    return dataset_processed

def zonghe_shuju_yuchuli(dataset: pd.DataFrame) -> tuple:
    """综合数据预处理"""
    rizhiqri.info("开始综合数据预处理")
    
    processing_report = {
        'original_shape': dataset.shape,
        'missing_values': {},
        'outliers': {},
        'filtering': {},
        'final_shape': None
    }
    
    # 缺失值分析
    missing_analysis = jiance_queshi_zhi(dataset)
    processing_report['missing_values'] = missing_analysis.to_dict('records') if len(missing_analysis) > 0 else []
    
    # 孕周时序填充
    dataset_processed = tianbu_yunzhou(dataset)
    
    # 年龄过滤
    original_count_before_age = len(dataset_processed)
    dataset_processed = guolv_muqin_nianling(dataset_processed)
    age_filtered_count = original_count_before_age - len(dataset_processed)
    processing_report['filtering']['age_filtered'] = age_filtered_count
    
    # Y染色体浓度异常值检测
    y_outliers = jiance_yichang_zhi(dataset_processed, Peizhi.TARGET_VARIABLE, 'both')
    processing_report['outliers']['y_chromosome'] = {
        'iqr_outliers': len(y_outliers['outliers_iqr']),
        'zscore_outliers': len(y_outliers['outliers_zscore']),
        'summary': y_outliers['summary']
    }
    
    # BMI异常值处理
    bmi_columns = ['BMI', '孕妇BMI', 'bmi']
    bmi_col = None
    for col in bmi_columns:
        if col in dataset_processed.columns:
            bmi_col = col
            break
    
    if bmi_col:
        bmi_outliers = jiance_yichang_zhi(dataset_processed, bmi_col, 'iqr')
        dataset_processed = chuli_yichang_zhi(dataset_processed, bmi_col, 'cap')
        processing_report['outliers']['bmi'] = {
            'outliers_detected': len(bmi_outliers['outliers_iqr']),
            'processing_method': 'capping'
        }
    
    processing_report['final_shape'] = dataset_processed.shape
    
    rizhiqri.info(f"数据预处理完成: {processing_report['original_shape']} → {processing_report['final_shape']}")
    
    return dataset_processed, processing_report

def baocun_chuli_shuju(dataset: pd.DataFrame, processing_report: dict, output_dir: str) -> None:
    """保存处理后数据"""
    processed_data_file = os.path.join(output_dir, "处理后数据.xlsx")
    dataset.to_excel(processed_data_file, index=False)
    
    report_file = os.path.join(output_dir, "数据预处理报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("数据预处理详细报告\n")
        f.write("="*50 + "\n\n")
        f.write(f"1. 数据规模变化:\n")
        f.write(f"   原始数据: {processing_report['original_shape'][0]} 行 × {processing_report['original_shape'][1]} 列\n")
        f.write(f"   最终数据: {processing_report['final_shape'][0]} 行 × {processing_report['final_shape'][1]} 列\n\n")
        f.write(f"2. 缺失值情况:\n")
        if processing_report['missing_values']:
            for missing in processing_report['missing_values']:
                f.write(f"   {missing['变量名']}: {missing['缺失数量']} 个 ({missing['缺失比例(%)']:.1f}%)\n")
        else:
            f.write("   无缺失值\n")
        f.write("\n")
        f.write(f"3. 异常值处理:\n")
        if 'y_chromosome' in processing_report['outliers']:
            y_info = processing_report['outliers']['y_chromosome']
            f.write(f"   Y染色体浓度异常值: IQR法检测{y_info['iqr_outliers']}个, Z-score检测{y_info['zscore_outliers']}个\n")
        if 'bmi' in processing_report['outliers']:
            bmi_info = processing_report['outliers']['bmi']
            f.write(f"   BMI异常值: 检测到{bmi_info['outliers_detected']}个\n")
        f.write(f"4. 数据过滤:\n")
        f.write(f"   年龄过滤 (<20岁): 删除 {processing_report['filtering']['age_filtered']} 条记录\n")
    
    rizhiqri.info(f"✅ 处理后数据已保存到: {processed_data_file}")

def zhunbei_jianmo_shuju(dataset: pd.DataFrame) -> tuple:
    """准备建模数据"""
    valid_predictors = [var for var in Peizhi.CORE_PREDICTORS if var in dataset.columns]
    modeling_columns = [Peizhi.TARGET_VARIABLE] + valid_predictors
    modeling_data = dataset[modeling_columns].dropna()
    
    if len(modeling_data) == 0:
        raise ValueError("❌ 移除缺失值后无剩余数据")
    
    X = modeling_data[valid_predictors]
    y = modeling_data[Peizhi.TARGET_VARIABLE]
    
    rizhiqri.info(f"✅ 建模数据准备完成 - 有效样本: {len(modeling_data)}/{len(dataset)}")
    return X, y

def tezheng_gongcheng(X: pd.DataFrame) -> tuple:
    """特征工程"""
    enhanced_features = X.copy()
    interaction_terms = []
    polynomial_terms = []
    
    # 交互项
    interaction_configs = [
        ('孕妇BMI', '检测时怀孕天数', 'BMI与孕周交互'),
        ('年龄', '孕妇BMI', '年龄BMI交互'),
        ('Y染色体的Z值', 'X染色体浓度', 'Y_Z与X浓度交互')
    ]
    
    for var1, var2, interaction_name in interaction_configs:
        if var1 in X.columns and var2 in X.columns:
            enhanced_features[interaction_name] = X[var1] * X[var2]
            interaction_terms.append(interaction_name)
    
    # 高阶项
    key_vars = ['孕妇BMI', '检测时怀孕天数', '年龄']
    for var in key_vars:
        if var in X.columns:
            poly_name = f'{var}_二次项'
            enhanced_features[poly_name] = X[var] ** 2
            polynomial_terms.append(poly_name)
    
    rizhiqri.info(f"✅ 特征工程完成 - 交互项: {len(interaction_terms)}, 高阶项: {len(polynomial_terms)}")
    
    return enhanced_features, interaction_terms, polynomial_terms

def fenxi_tezheng_guanlian(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """分析特征关联度"""
    association_results = []
    for tezheng in X.columns:
        try:
            correlation_coef, p_val = stats.spearmanr(X[tezheng], y)
            association_results.append({
                '变量名': tezheng,
                '关联系数': correlation_coef,
                'P值': p_val,
                '显著性标记': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.',
                '关联强度': abs(correlation_coef)
            })
        except Exception as e:
            rizhiqri.warning(f"特征 {tezheng} 关联度计算失败: {str(e)}")
            continue
    
    if not association_results:
        raise ValueError("❌ 无法计算任何特征的关联度")
    
    association_df = pd.DataFrame(association_results).sort_values('关联强度', ascending=False)
    rizhiqri.info(f"✅ 关联度分析完成，共分析 {len(association_df)} 个特征")
    
    return association_df

def chuangjian_xiangguanxishu_relitu(X: pd.DataFrame, y: pd.Series, output_dir: str) -> None:
    """创建斯皮尔曼相关系数热力图"""
    data_for_corr = X.copy()
    data_for_corr[Peizhi.TARGET_VARIABLE] = y
    spearman_corr = data_for_corr.corr(method='spearman')
    
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(spearman_corr, mask=mask, annot=True, cmap=cmap, center=0,
                fmt='.3f', square=True, cbar_kws={"shrink": .8},
                annot_kws={'size': 8, 'weight': 'bold'}, linewidths=0.5)
    
    plt.title('特征间斯皮尔曼相关系数热力图', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('特征变量', fontsize=12)
    plt.ylabel('特征变量', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 统计信息
    n_features = len(spearman_corr.columns)
    avg_corr = spearman_corr.abs().mean().mean()
    target_corr = spearman_corr[Peizhi.TARGET_VARIABLE].drop(Peizhi.TARGET_VARIABLE)
    top_corr_feature = target_corr.abs().idxmax()
    top_corr_value = target_corr[top_corr_feature]
    
    stats_text = f'统计信息:\n特征数量: {n_features}\n平均绝对相关系数: {avg_corr:.3f}\n与目标变量最强相关: {top_corr_feature}\n相关系数值: {top_corr_value:.3f}'
    plt.text(1.02, 0.5, stats_text, transform=plt.gca().transAxes,
            verticalalignment='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_斯皮尔曼相关系数热力图.png'), 
               dpi=Peizhi.IMAGE_DPI, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # 保存相关系数矩阵
    corr_matrix_file = os.path.join(output_dir, "斯皮尔曼相关系数矩阵.csv")
    spearman_corr.to_csv(corr_matrix_file, encoding=Peizhi.CSV_ENCODING)
    
    rizhiqri.info(f"✅ 斯皮尔曼相关系数热力图创建完成")

def youhua_ridge_moxing(X_train: np.ndarray, y_train: pd.Series) -> tuple:
    """Ridge回归超参数优化"""
    alpha_range = np.logspace(Peizhi.ALPHA_RANGE_MIN, Peizhi.ALPHA_RANGE_MAX, Peizhi.ALPHA_RANGE_POINTS)
    
    ridge_cv = RidgeCV(alphas=alpha_range, cv=Peizhi.CV_FOLDS, scoring='r2')
    ridge_cv.fit(X_train, y_train)
    best_alpha = ridge_cv.alpha_
    
    train_scores, validation_scores = validation_curve(
        Ridge(), X_train, y_train, 
        param_name='alpha', param_range=alpha_range, 
        cv=Peizhi.CV_FOLDS, scoring='r2', n_jobs=-1
    )
    
    rizhiqri.info(f"✅ 超参数优化完成 - 最优α: {best_alpha:.6f}")
    
    return best_alpha, train_scores, validation_scores

def fenlei_tezheng(feature_name: str) -> str:
    """特征分类"""
    if any(keyword in feature_name for keyword in ['交互', '与']):
        return '交互特征'
    elif any(keyword in feature_name for keyword in ['二次项', '平方']):
        return '高阶特征'
    elif feature_name in ['IVF妊娠_编码', '怀孕次数_编码', '胎儿健康_编码', 'T13_异常', 'T18_异常', 'T21_异常']:
        return '分类特征'
    else:
        return '连续特征'

def xunlian_zuizhong_moxing(X_train: np.ndarray, X_test: np.ndarray, 
                     y_train: pd.Series, y_test: pd.Series, 
                     best_alpha: float, feature_names: list) -> dict:
    """训练最终模型"""
    zuizhong_moxing = Ridge(alpha=best_alpha)
    zuizhong_moxing.fit(X_train, y_train)
    
    y_train_pred = zuizhong_moxing.predict(X_train)
    y_test_pred = zuizhong_moxing.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    
    xishu = zuizhong_moxing.coef_
    coefficient_analysis = pd.DataFrame({
        '特征名称': feature_names,
        '回归系数': xishu,
        '系数绝对值': np.abs(xishu)
    }).sort_values('系数绝对值', ascending=False)
    
    coefficient_analysis['特征分类'] = coefficient_analysis['特征名称'].apply(fenlei_tezheng)
    
    rizhiqri.info(f"✅ 模型训练完成 - 测试集R²: {test_r2:.4f}, RMSE: {rmse:.6f}")
    
    return {
        'model': zuizhong_moxing,
        'predictions': {
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'residuals': y_test - y_test_pred
        },
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae
        },
        'coefficient_analysis': coefficient_analysis
    }

def chuangjian_tubiao_jichu(figsize=(10, 8)):
    """创建基础图表设置"""
    plt.figure(figsize=figsize)
    return plt.gca()

def baocun_tubiao(filename: str, output_dir: str):
    """保存图表"""
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), 
               dpi=Peizhi.IMAGE_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

def chuangjian_chaocanshu_youhua_tubiao(alpha_range: np.ndarray, train_scores: np.ndarray, 
                                           validation_scores: np.ndarray, best_alpha: float,
                                           output_dir: str) -> None:
    """创建超参数优化图表"""
    chuangjian_tubiao_jichu()
    
    plt.semilogx(alpha_range, np.mean(train_scores, axis=1), 'b-', label='训练R²', linewidth=2.5)
    plt.semilogx(alpha_range, np.mean(validation_scores, axis=1), 'r-', label='验证R²', linewidth=2.5)
    plt.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2, 
               label=f'最优α={best_alpha:.4f}')
    
    plt.fill_between(alpha_range, 
                    np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                    np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                    alpha=0.2, color='blue')
    plt.fill_between(alpha_range, 
                    np.mean(validation_scores, axis=1) - np.std(validation_scores, axis=1),
                    np.mean(validation_scores, axis=1) + np.std(validation_scores, axis=1),
                    alpha=0.2, color='red')
    
    plt.xlabel('正则化强度 α', fontsize=12)
    plt.ylabel('决定系数 R²', fontsize=12)
    plt.title('Ridge回归超参数寻优过程', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    baocun_tubiao('1_Ridge回归超参数优化过程.png', output_dir)

def chuangjian_tezheng_xishu_tubiao(coefficient_analysis: pd.DataFrame, output_dir: str) -> None:
    """创建特征系数图表"""
    chuangjian_tubiao_jichu((12, 8))
    
    top_15_features = coefficient_analysis.head(15)
    coef_colors = ['#e74c3c' if coef < 0 else '#3498db' for coef in top_15_features['回归系数']]
    
    bars = plt.barh(range(len(top_15_features)), top_15_features['回归系数'], 
                   color=coef_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.yticks(range(len(top_15_features)), top_15_features['特征名称'], fontsize=10)
    plt.xlabel('标准化回归系数', fontsize=12)
    plt.title('前15位特征系数重要性排序', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, (bar, coef_value) in enumerate(zip(bars, top_15_features['回归系数'])):
        plt.text(coef_value + (0.0005 if coef_value > 0 else -0.0005), i, 
                f'{coef_value:.4f}', va='center', 
                ha='left' if coef_value > 0 else 'right', fontsize=9, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', alpha=0.8, label='正向影响'),
                      Patch(facecolor='#e74c3c', alpha=0.8, label='负向影响')]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    baocun_tubiao('2_Ridge回归特征系数重要性.png', output_dir)

def chuangjian_tezheng_gongxian_tubiao(coefficient_analysis: pd.DataFrame, output_dir: str) -> None:
    """创建特征贡献占比图表"""
    chuangjian_tubiao_jichu()
    
    type_contribution = coefficient_analysis.groupby('特征分类')['系数绝对值'].sum()
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(type_contribution)]
    
    wedges, texts, autotexts = plt.pie(type_contribution.values, 
                                      labels=type_contribution.index, 
                                      autopct='%1.1f%%', 
                                      startangle=90,
                                      colors=colors,
                                      explode=[0.05] * len(type_contribution),
                                      shadow=True)
    
    plt.title('各类特征预测贡献占比', fontsize=14, fontweight='bold', pad=20)
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    baocun_tubiao('3_Ridge回归特征类型贡献占比.png', output_dir)

def chuangjian_yuce_jingdu_tubiao(y_test: pd.Series, y_test_pred: np.ndarray,
                                   test_r2: float, rmse: float, output_dir: str) -> None:
    """创建预测精度评估图表"""
    chuangjian_tubiao_jichu()
    
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='#3498db', s=50, 
               edgecolor='darkblue', linewidth=0.5, label='预测点')
    
    min_value = min(y_test.min(), y_test_pred.min())
    max_value = max(y_test.max(), y_test_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], 
            'r--', linewidth=3, label='理想预测线', alpha=0.8)
    
    residuals = y_test - y_test_pred
    std_residual = np.std(residuals)
    plt.plot([min_value, max_value], [min_value + std_residual, max_value + std_residual], 
            'g--', linewidth=2, alpha=0.6, label=f'±1σ预测区间')
    plt.plot([min_value, max_value], [min_value - std_residual, max_value - std_residual], 
            'g--', linewidth=2, alpha=0.6)
    
    plt.xlabel('真实Y染色体浓度', fontsize=12)
    plt.ylabel('预测Y染色体浓度', fontsize=12)
    plt.title(f'模型预测精度评估', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / y_test)) * 100
    
    metrics_text = f'性能指标:\nR² = {test_r2:.4f}\nRMSE = {rmse:.6f}\nMAE = {mae:.6f}\nMAPE = {mape:.2f}%\n样本数 = {len(y_test)}'
    
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    baocun_tubiao('4_Ridge回归预测精度评估.png', output_dir)

def chuangjian_cancha_tubiao(y_test_pred: np.ndarray, residuals: np.ndarray, output_dir: str) -> None:
    """创建残差分析图表"""
    # 残差vs预测值
    chuangjian_tubiao_jichu()
    plt.scatter(y_test_pred, residuals, alpha=0.6, color='#3498db', s=50,
               edgecolor='darkblue', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.xlabel('模型预测值', fontsize=12)
    plt.ylabel('预测残差', fontsize=12)
    plt.title('残差vs预测值分布图', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    baocun_tubiao('5_残差vs预测值分布图.png', output_dir)
    
    # 残差分布直方图
    chuangjian_tubiao_jichu()
    n, bins, patches = plt.hist(residuals, bins=25, alpha=0.7, color='skyblue', 
                               edgecolor='black', linewidth=1)
    
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y_normal = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
               np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
    y_normal = y_normal * len(residuals) * (bins[1] - bins[0])
    
    plt.plot(x, y_normal, 'r-', linewidth=2, label='理论正态分布')
    plt.axvline(mu, color='red', linestyle='--', linewidth=2, label=f'均值: {mu:.6f}')
    plt.xlabel('预测残差', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.title('残差分布直方图', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    baocun_tubiao('6_残差分布直方图.png', output_dir)
    
    # QQ图
    chuangjian_tubiao_jichu()
    res = stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('残差正态性QQ图检验', fontsize=14, fontweight='bold')
    plt.xlabel('理论分位数', fontsize=12)
    plt.ylabel('样本分位数', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    test_results = f'正态性检验:\nShapiro-Wilk:\n  统计量: {shapiro_stat:.4f}\n  P值: {shapiro_p:.6f}'
    plt.text(0.05, 0.95, test_results, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    baocun_tubiao('7_残差正态性QQ图检验.png', output_dir)

def chuangjian_Y_ransetiludufenbu_yunzhou_sandianntu(dataset: pd.DataFrame, output_dir: str) -> None:
    """创建Y染色体浓度与孕周二元分布散点图"""
    y_col = 'Y染色体浓度'
    # 尝试多个可能的孕周列名（注意：数据预处理时已将"检测孕周"重命名为"检测时怀孕天数"）
    gestational_cols = ['检测时怀孕天数', '检测孕周', '孕周', '妊娠周数']
    gestational_col = None
    
    for col in gestational_cols:
        if col in dataset.columns:
            gestational_col = col
            break
    
    if y_col not in dataset.columns or gestational_col is None:
        rizhiqri.warning(f"缺少必要的列，跳过Y染色体浓度与孕周散点图")
        return
    
    valid_data = dataset[[y_col, gestational_col]].dropna()
    if len(valid_data) < 10:
        rizhiqri.warning("有效数据点过少，跳过Y染色体浓度与孕周散点图")
        return
    
    y_values = valid_data[y_col].values
    gestational_values = valid_data[gestational_col].values
    
    chuangjian_tubiao_jichu((12, 8))
    
    # 创建散点图
    scatter = plt.scatter(gestational_values, y_values, alpha=0.6, c=gestational_values, 
                         s=60, cmap='viridis', edgecolor='black', linewidth=0.5)
    
    # 添加回归线
    lr_model = LinearRegression()
    gestational_reshaped = gestational_values.reshape(-1, 1)
    lr_model.fit(gestational_reshaped, y_values)
    y_fitted = lr_model.predict(gestational_reshaped)
    
    sort_idx = np.argsort(gestational_values)
    plt.plot(gestational_values[sort_idx], y_fitted[sort_idx], 'r-', linewidth=3, 
            label=f'线性回归线 (R²={lr_model.score(gestational_reshaped, y_values):.4f})')
    
    # 计算相关系数
    correlation_coef, p_val = stats.spearmanr(gestational_values, y_values)
    
    plt.xlabel(f'{gestational_col}{"(天)" if "天数" in gestational_col else "(周)"}', fontsize=12)
    plt.ylabel('Y染色体浓度', fontsize=12)
    plt.title('Y染色体浓度与孕周二元分布散点图', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label=f'{gestational_col}{"(天)" if "天数" in gestational_col else "(周)"}')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f'统计信息:\n样本数量: {len(valid_data)}\n相关系数: {correlation_coef:.4f}\nP值: {p_val:.6f}\n显著性: {"***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black'))
    
    baocun_tubiao('8_Y染色体浓度与孕周二元分布散点图.png', output_dir)
    
    rizhiqri.info(f"✅ Y染色体浓度与孕周散点图创建完成")

def chuangjian_Y_ransetiludufenbu_BMI_sandianntu(dataset: pd.DataFrame, output_dir: str) -> None:
    """创建Y染色体浓度与BMI二元分布散点图"""
    y_col = 'Y染色体浓度'
    # 尝试多个可能的BMI列名
    bmi_cols = ['孕妇BMI', 'BMI', 'bmi']
    bmi_col = None
    
    for col in bmi_cols:
        if col in dataset.columns:
            bmi_col = col
            break
    
    if y_col not in dataset.columns or bmi_col is None:
        rizhiqri.warning(f"缺少必要的列，跳过Y染色体浓度与BMI散点图")
        return
    
    valid_data = dataset[[y_col, bmi_col]].dropna()
    if len(valid_data) < 10:
        rizhiqri.warning("有效数据点过少，跳过Y染色体浓度与BMI散点图")
        return
    
    y_values = valid_data[y_col].values
    bmi_values = valid_data[bmi_col].values
    
    chuangjian_tubiao_jichu((12, 8))
    
    # 创建散点图，根据BMI值进行颜色映射
    scatter = plt.scatter(bmi_values, y_values, alpha=0.6, c=bmi_values, 
                         s=60, cmap='plasma', edgecolor='black', linewidth=0.5)
    
    # 添加回归线
    lr_model = LinearRegression()
    bmi_reshaped = bmi_values.reshape(-1, 1)
    lr_model.fit(bmi_reshaped, y_values)
    y_fitted = lr_model.predict(bmi_reshaped)
    
    sort_idx = np.argsort(bmi_values)
    plt.plot(bmi_values[sort_idx], y_fitted[sort_idx], 'r-', linewidth=3, 
            label=f'线性回归线 (R²={lr_model.score(bmi_reshaped, y_values):.4f})')
    
    # 计算相关系数
    correlation_coef, p_val = stats.spearmanr(bmi_values, y_values)
    
    # 添加BMI分类区间线
    plt.axvline(x=18.5, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='BMI分类线')
    plt.axvline(x=25, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=30, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('孕妇BMI (kg/m²)', fontsize=12)
    plt.ylabel('Y染色体浓度', fontsize=12)
    plt.title('Y染色体浓度与BMI二元分布散点图', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='BMI值 (kg/m²)')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息和BMI分类标注
    stats_text = f'统计信息:\n样本数量: {len(valid_data)}\n相关系数: {correlation_coef:.4f}\nP值: {p_val:.6f}\n显著性: {"***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='black'))
    
    # BMI分类标注
    bmi_categories = f'BMI分类:\n<18.5 偏瘦\n18.5-25 正常\n25-30 超重\n≥30 肥胖'
    plt.text(0.98, 0.02, bmi_categories, transform=plt.gca().transAxes,
            verticalalignment='bottom', horizontalalignment='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange'))
    
    baocun_tubiao('9_Y染色体浓度与BMI二元分布散点图.png', output_dir)
    
    rizhiqri.info(f"✅ Y染色体浓度与BMI散点图创建完成")

def chuangjian_Y_ransetiludu_GC_hanliang_cancha_tubiao(dataset: pd.DataFrame, output_dir: str) -> None:
    """创建Y染色体浓度与GC含量残差拟合图表"""
    y_col = 'Y染色体浓度'
    gc_col = 'GC含量'
    
    if y_col not in dataset.columns or gc_col not in dataset.columns:
        rizhiqri.warning(f"缺少必要的列，跳过Y染色体-GC残差拟合图表")
        return
    
    valid_data = dataset[[y_col, gc_col]].dropna()
    if len(valid_data) < 10:
        rizhiqri.warning("有效数据点过少，跳过Y染色体-GC残差拟合图表")
        return
    
    y_values = valid_data[y_col].values
    gc_values = valid_data[gc_col].values
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 散点图和拟合线
    ax1.scatter(gc_values, y_values, alpha=0.6, color='#3498db', s=50, 
               edgecolor='darkblue', linewidth=0.5, label='观测值')
    
    lr_model = LinearRegression()
    gc_reshaped = gc_values.reshape(-1, 1)
    lr_model.fit(gc_reshaped, y_values)
    y_fitted = lr_model.predict(gc_reshaped)
    
    sort_idx = np.argsort(gc_values)
    ax1.plot(gc_values[sort_idx], y_fitted[sort_idx], 'r-', linewidth=2.5, 
            label=f'线性拟合 (R²={lr_model.score(gc_reshaped, y_values):.4f})')
    
    ax1.set_xlabel('GC含量', fontsize=12)
    ax1.set_ylabel('Y染色体浓度', fontsize=12)
    ax1.set_title('Y染色体浓度 vs GC含量拟合关系', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 残差图
    residuals = y_values - y_fitted
    ax2.scatter(gc_values, residuals, alpha=0.6, color='#e74c3c', s=50,
               edgecolor='darkred', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('GC含量', fontsize=12)
    ax2.set_ylabel('拟合残差', fontsize=12)
    ax2.set_title('残差 vs GC含量', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 残差分布
    ax3.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    y_normal = ((1 / (np.sqrt(2 * np.pi) * sigma)) * 
               np.exp(-0.5 * (1 / sigma * (x - mu)) ** 2))
    ax3.plot(x, y_normal, 'r-', linewidth=2, label='理论正态分布')
    ax3.set_xlabel('残差值', fontsize=12)
    ax3.set_ylabel('密度', fontsize=12)
    ax3.set_title('残差分布直方图', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 拟合值vs残差
    ax4.scatter(y_fitted, residuals, alpha=0.6, color='#9b59b6', s=50,
               edgecolor='darkviolet', linewidth=0.5)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    ax4.set_xlabel('拟合值', fontsize=12)
    ax4.set_ylabel('残差', fontsize=12)
    ax4.set_title('拟合值 vs 残差', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'Y染色体浓度与GC含量关系分析', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, '10_Y染色体浓度与GC含量残差拟合分析.png'), 
               dpi=Peizhi.IMAGE_DPI, bbox_inches='tight', facecolor='white')
    plt.show()

def baocun_fenxi_jieguo(coefficient_analysis: pd.DataFrame, y_test: pd.Series, 
                         y_test_pred: np.ndarray, residuals: np.ndarray,
                         association_df: pd.DataFrame, output_dir: str) -> None:
    """保存分析结果"""
    # 回归系数分析
    coeff_file = os.path.join(output_dir, "Ridge回归系数深度分析.csv")
    coefficient_analysis.to_csv(coeff_file, index=False, encoding=Peizhi.CSV_ENCODING)
    
    # 预测结果分析
    prediction_analysis = pd.DataFrame({
        '样本编号': range(len(y_test)),
        '真实Y染色体浓度': y_test.values,
        '预测Y染色体浓度': y_test_pred,
        '预测残差': residuals,
        '绝对预测误差': np.abs(residuals),
        '相对预测误差(%)': np.abs(residuals) / y_test.values * 100,
        '误差等级': pd.cut(np.abs(residuals), bins=3, labels=['低误差', '中误差', '高误差'])
    })
    
    pred_file = os.path.join(output_dir, "Ridge回归预测结果详细分析.csv")
    prediction_analysis.to_csv(pred_file, index=False, encoding=Peizhi.CSV_ENCODING)
    
    # 特征关联度分析
    assoc_file = os.path.join(output_dir, "特征关联度深度分析.csv")
    association_df.to_csv(assoc_file, index=False, encoding=Peizhi.CSV_ENCODING)
    
    rizhiqri.info("✅ 分析结果保存完成")

def zhixing_ridge_fenxi(dataset: pd.DataFrame) -> dict:
    """执行完整的Ridge回归分析流程"""
    rizhiqri.info("="*80)
    rizhiqri.info("Y染色体浓度预测模型：Ridge回归深度建模分析")
    rizhiqri.info("="*80)
    
    # 数据验证
    yanzheng_shujuji(dataset)
    
    # 数据准备
    X, y = zhunbei_jianmo_shuju(dataset)
    
    # 特征工程
    X_enhanced, interaction_terms, polynomial_terms = tezheng_gongcheng(X)
    
    # 特征关联度分析
    association_df = fenxi_tezheng_guanlian(X_enhanced, y)
    
    # 创建相关系数热力图
    chuangjian_xiangguanxishu_relitu(X_enhanced, y, Peizhi.OUTPUT_DIR)
    
    # 数据集划分与标准化
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=Peizhi.TEST_SIZE, random_state=Peizhi.RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 超参数优化
    alpha_range = np.logspace(Peizhi.ALPHA_RANGE_MIN, Peizhi.ALPHA_RANGE_MAX, Peizhi.ALPHA_RANGE_POINTS)
    best_alpha, train_scores, validation_scores = youhua_ridge_moxing(X_train_scaled, y_train)
    
    # 最终模型训练
    feature_names = list(X_enhanced.columns)
    moxing_jieguo = xunlian_zuizhong_moxing(X_train_scaled, X_test_scaled, y_train, y_test, 
                                    best_alpha, feature_names)
    
    # 可视化分析
    rizhiqri.info("开始创建可视化图表...")
    
    chuangjian_chaocanshu_youhua_tubiao(alpha_range, train_scores, validation_scores, 
                                           best_alpha, Peizhi.OUTPUT_DIR)
    chuangjian_tezheng_xishu_tubiao(moxing_jieguo['coefficient_analysis'], Peizhi.OUTPUT_DIR)
    chuangjian_tezheng_gongxian_tubiao(moxing_jieguo['coefficient_analysis'], Peizhi.OUTPUT_DIR)
    chuangjian_yuce_jingdu_tubiao(y_test, moxing_jieguo['predictions']['y_test_pred'],
                                   moxing_jieguo['metrics']['test_r2'], 
                                   moxing_jieguo['metrics']['rmse'], Peizhi.OUTPUT_DIR)
    
    # 二元分布散点图分析
    rizhiqri.info("开始创建二元分布散点图...")
    chuangjian_Y_ransetiludufenbu_yunzhou_sandianntu(dataset, Peizhi.OUTPUT_DIR)
    chuangjian_Y_ransetiludufenbu_BMI_sandianntu(dataset, Peizhi.OUTPUT_DIR)
    
    # 残差诊断分析
    rizhiqri.info("开始创建残差诊断图表...")
    residuals = moxing_jieguo['predictions']['residuals']
    y_test_pred = moxing_jieguo['predictions']['y_test_pred']
    
    chuangjian_cancha_tubiao(y_test_pred, residuals, Peizhi.OUTPUT_DIR)
    
    # Y染色体浓度与GC含量关系分析
    chuangjian_Y_ransetiludu_GC_hanliang_cancha_tubiao(dataset, Peizhi.OUTPUT_DIR)
    
    # 保存结果
    baocun_fenxi_jieguo(moxing_jieguo['coefficient_analysis'], y_test,
                        moxing_jieguo['predictions']['y_test_pred'],
                        moxing_jieguo['predictions']['residuals'],
                        association_df, Peizhi.OUTPUT_DIR)
    
    rizhiqri.info("="*80)
    rizhiqri.info("🎉 Ridge回归深度分析完成！")
    rizhiqri.info("="*80)
    
    return {
        'model': moxing_jieguo['model'],
        'scaler': scaler,
        'metrics': moxing_jieguo['metrics'],
        'coefficient_analysis': moxing_jieguo['coefficient_analysis'],
        'association_analysis': association_df,
        'feature_names': feature_names
    }

def zhuchengxu():
    """主程序入口"""
    try:
        rizhiqri.info("开始加载数据文件...")
        shujuji = pd.read_excel("附件.xlsx")
        
        # 创建输出目录
        if not os.path.exists(Peizhi.OUTPUT_DIR):
            os.makedirs(Peizhi.OUTPUT_DIR)
            rizhiqri.info(f"已创建输出目录: {Peizhi.OUTPUT_DIR}")
        
        # 基础数据预处理
        rizhiqri.info("执行基础数据预处理...")
        
        # 先解析孕周字符串格式
        if '检测孕周' in shujuji.columns:
            shujuji['检测孕周'] = shujuji['检测孕周'].apply(jiexi_yunzhou)
            rizhiqri.info("✅ 检测孕周字符串格式解析完成")
        
        column_mapping = {'BMI': '孕妇BMI', '检测孕周': '检测时怀孕天数'}
        shujuji = shujuji.rename(columns=column_mapping)
        
        # 分类变量编码
        if 'IVF妊娠' in shujuji.columns:
            shujuji['IVF妊娠_编码'] = shujuji['IVF妊娠'].map({'自然受孕': 0, 'IVF': 1}).fillna(0)
        
        if '怀孕次数' in shujuji.columns:
            shujuji['怀孕次数_编码'] = shujuji['怀孕次数'].map({'1': 1, '2': 2, '≥3': 3}).fillna(1)
        
        if '胎儿是否健康' in shujuji.columns:
            shujuji['胎儿健康_编码'] = shujuji['胎儿是否健康'].map({'是': 1, '否': 0}).fillna(0)
        
        # 异常情况编码
        if '染色体的非整倍体' in shujuji.columns:
            shujuji['T13_异常'] = shujuji['染色体的非整倍体'].str.contains('T13', na=False).astype(int)
            shujuji['T18_异常'] = shujuji['染色体的非整倍体'].str.contains('T18', na=False).astype(int)
            shujuji['T21_异常'] = shujuji['染色体的非整倍体'].str.contains('T21', na=False).astype(int)
        
        # 计算时间间隔特征
        if '末次月经距检测日天数' not in shujuji.columns:
            if '末次月经' in shujuji.columns and '检测日期' in shujuji.columns:
                try:
                    shujuji['末次月经'] = pd.to_datetime(shujuji['末次月经'], errors='coerce')
                    shujuji['检测日期'] = pd.to_datetime(shujuji['检测日期'].astype(str), 
                                                    format='%Y%m%d', errors='coerce')
                    shujuji['末次月经距检测日天数'] = (shujuji['检测日期'] - shujuji['末次月经']).dt.days
                except Exception as e:
                    rizhiqri.warning(f"日期处理警告: {e}")
                    if '检测时怀孕天数' in shujuji.columns:
                        shujuji['末次月经距检测日天数'] = shujuji['检测时怀孕天数'] * 7
        
        rizhiqri.info(f"基础数据预处理完成 - 数据维度: {shujuji.shape}")
        
        # 综合数据预处理
        dataset_processed, preprocessing_report = zonghe_shuju_yuchuli(shujuji)
        
        # 保存处理后的数据
        baocun_chuli_shuju(dataset_processed, preprocessing_report, Peizhi.OUTPUT_DIR)
        
        # 执行Ridge回归建模分析
        jieguo = zhixing_ridge_fenxi(dataset_processed)
        
        rizhiqri.info("="*80)
        rizhiqri.info("🎉 完整分析流程执行完成！")
        rizhiqri.info("="*80)
        rizhiqri.info("生成的文件:")
        rizhiqri.info("  📄 处理后数据.xlsx - 清洗后的数据集")
        rizhiqri.info("  📄 数据预处理报告.txt - 数据预处理详细报告")
        rizhiqri.info("  📊 1_Ridge回归超参数优化过程.png - 超参数寻优可视化")
        rizhiqri.info("  📊 2_Ridge回归特征系数重要性.png - 特征系数排序图")
        rizhiqri.info("  📊 3_Ridge回归特征类型贡献占比.png - 特征类型贡献图")
        rizhiqri.info("  📊 4_Ridge回归预测精度评估.png - 预测效果评估图")
        rizhiqri.info("  📊 5_残差vs预测值分布图.png - 残差分布散点图")
        rizhiqri.info("  📊 6_残差分布直方图.png - 残差分布统计图")
        rizhiqri.info("  📊 7_残差正态性QQ图检验.png - 正态性检验图")
        rizhiqri.info("  📊 8_Y染色体浓度与孕周二元分布散点图.png - Y染色体浓度与孕周关系图")
        rizhiqri.info("  📊 9_Y染色体浓度与BMI二元分布散点图.png - Y染色体浓度与BMI关系图")
        rizhiqri.info("  📊 10_Y染色体浓度与GC含量残差拟合分析.png - Y染色体-GC关系分析图")
        rizhiqri.info("  📊 11_斯皮尔曼相关系数热力图.png - 特征间相关性热力图")
        rizhiqri.info("  📄 Ridge回归系数深度分析.csv - 模型系数分析")
        rizhiqri.info("  📄 Ridge回归预测结果详细分析.csv - 预测结果分析")
        rizhiqri.info("  📄 特征关联度深度分析.csv - 特征关联度分析")
        rizhiqri.info("  📄 斯皮尔曼相关系数矩阵.csv - 特征相关系数矩阵")
        rizhiqri.info(f"所有文件已保存到 '{Peizhi.OUTPUT_DIR}' 文件夹")
        
    except FileNotFoundError as e:
        rizhiqri.error(f"❌ 数据文件未找到: {e}")
    except Exception as e:
        rizhiqri.error(f"❌ 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    zhuchengxu()
