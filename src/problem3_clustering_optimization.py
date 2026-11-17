# 历史参赛脚本：对应 paper/final_paper.pdf，仅供对照复现。
# 赛后修订研究不修改本文件的计算行为；新的分析在 src/revised/。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.stats import norm, gaussian_kde
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
import logging
import time
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from dataclasses import dataclass
from pathlib import Path
import joblib
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AdvancedOptimizationConfig:
    """高级优化配置类"""
    # 聚类配置
    max_clusters: int = 8
    min_cluster_size: int = 10
    clustering_methods: List[str] = None
    scaling_methods: List[str] = None
    
    # 优化配置
    detection_error_std: float = 0.5
    n_simulations: int = 1000
    optimization_methods: List[str] = None
    n_optimization_trials: int = 3
    
    # 风险权重（自适应调整）
    early_risk_weight: float = 0.25
    success_rate_weight: float = 0.15
    weighted_risk_weight: float = 0.25
    error_risk_weight: float = 0.20
    delay_risk_weight: float = 0.10
    consistency_risk_weight: float = 0.05
    
    # 计算配置
    n_jobs: int = -1
    cache_size: int = 256
    use_gpu: bool = False
    
    # 验证配置
    cross_validation_folds: int = 5
    bootstrap_samples: int = 100
    
    # 输出配置
    output_dir: str = "问题三结果"
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.clustering_methods is None:
            self.clustering_methods = ['kmeans', 'agglomerative', 'gmm']
        if self.scaling_methods is None:
            self.scaling_methods = ['standard', 'robust']
        if self.optimization_methods is None:
            self.optimization_methods = ['L-BFGS-B', 'differential_evolution']
        if self.n_jobs == -1:
            self.n_jobs = min(8, mp.cpu_count())

class AdvancedMultiFactorNIPTOptimization:
    """
    高级多因素NIPT时点优化器
    
    实现了更先进的算法和优化策略，包括：
    - 多种聚类算法比较
    - 自适应参数优化
    - 并行计算加速
    - 交叉验证和bootstrap验证
    - 数值稳定性改进
    """
    
    def __init__(self, data_file: str, config: Optional[AdvancedOptimizationConfig] = None):
        """
        初始化高级多因素优化器
        
        Args:
            data_file: 数据文件路径
            config: 高级优化配置
        """
        self.data_file = data_file
        self.config = config or AdvancedOptimizationConfig()
        
        # 数据属性
        self.data: Optional[pd.DataFrame] = None
        self.qualified_data: Optional[pd.DataFrame] = None
        self.feature_data: Optional[pd.DataFrame] = None
        
        # 分析结果
        self.clustering_results: Dict[str, Any] = {}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.optimal_clusters: Optional[int] = None
        self.optimal_timings: Optional[List[float]] = None
        self.final_results: Optional[List[Dict[str, Any]]] = None
        self.validation_results: Dict[str, Any] = {}
        
        # 创建输出目录
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        logger.info(f"初始化高级多因素NIPT优化器")
        logger.info(f"使用 {self.config.n_jobs} 个并行进程")
    
    def _auto_adjust_config_for_data_size(self, data_size: int):
        """根据数据量自动调整配置，更宽松的参数设置"""
        logger.info(f"🔧 根据数据量({data_size})自动调整配置...")
        
        original_min_cluster_size = self.config.min_cluster_size
        original_max_clusters = self.config.max_clusters
        
        # 根据数据量调整参数（更宽松的设置）
        if data_size < 20:
            self.config.min_cluster_size = max(2, data_size // 8)  # 进一步降低最小聚类大小
            self.config.max_clusters = min(3, max(2, data_size // 6))
            self.config.bootstrap_samples = 20
            self.config.n_simulations = 200
            logger.warning("数据量很小，已调整为极小数据集配置")
        elif data_size < 50:
            self.config.min_cluster_size = max(3, data_size // 12)  # 降低最小聚类大小
            self.config.max_clusters = min(4, data_size // 8)
            self.config.bootstrap_samples = 25
            self.config.n_simulations = 250
            logger.warning("数据量较小，已调整为适合小数据集的配置")
        elif data_size < 100:
            self.config.min_cluster_size = max(4, data_size // 18)  # 降低最小聚类大小
            self.config.max_clusters = min(5, data_size // 12)
            self.config.bootstrap_samples = 40
            self.config.n_simulations = 400
        elif data_size < 200:
            self.config.min_cluster_size = max(5, data_size // 25)  # 降低最小聚类大小
            self.config.max_clusters = min(6, data_size // 18)
        
        # 确保至少有2个聚类，最小聚类大小不超过数据的40%
        self.config.max_clusters = max(2, self.config.max_clusters)
        self.config.min_cluster_size = min(self.config.min_cluster_size, max(2, data_size // 4))
        
        if (self.config.min_cluster_size != original_min_cluster_size or 
            self.config.max_clusters != original_max_clusters):
            logger.info(f"配置已调整: min_cluster_size {original_min_cluster_size}→{self.config.min_cluster_size}, "
                       f"max_clusters {original_max_clusters}→{self.config.max_clusters}")
            logger.info(f"调整后最大允许数据保留率: {(1 - self.config.min_cluster_size/data_size)*100:.1f}%")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """高级数据加载和预处理"""
        logger.info("📊 开始高级数据预处理...")
        start_time = time.time()
        
        try:
            # 数据加载
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
                
            self.data = pd.read_excel(self.data_file)
            logger.info(f"成功加载数据，共 {len(self.data)} 行")
            
            # 数据质量检查
            self._validate_data_quality()
            
            # 男胎筛选（包含边界值，减少数据丢失）
            male_data = self.data[self.data['Y染色体浓度'] >= 0].copy()
            logger.info(f"男胎总数（包含边界值）: {len(male_data)}")
            
            if len(male_data) == 0:
                raise ValueError("没有找到男胎数据")
            
            # 高级孕周解析
            male_data['检测孕周_数值'] = self._advanced_parse_gestational_weeks(male_data['检测孕周'])
            
            # 并行处理达标记录
            qualified_records = self._parallel_process_qualified_records(male_data)
            
            if qualified_records:
                self.qualified_data = pd.DataFrame(qualified_records).reset_index(drop=True)
                self._advanced_feature_engineering()
                
                processing_time = time.time() - start_time
                logger.info(f"高级数据预处理完成，耗时: {processing_time:.2f}秒")
                logger.info(f"符合条件的孕妇数: {len(self.qualified_data)}")
            else:
                self.qualified_data = pd.DataFrame()
                logger.warning("没有符合条件的数据")
                
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            raise
            
        return self.qualified_data
    
    def _validate_data_quality(self):
        """验证数据质量"""
        logger.info("🔍 验证数据质量...")
        
        # 检查必需列
        required_cols = ['Y染色体浓度', '检测孕周', '孕妇代码']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
        
        # 数据统计
        total_rows = len(self.data)
        missing_y_chrom = self.data['Y染色体浓度'].isna().sum()
        missing_weeks = self.data['检测孕周'].isna().sum()
        
        logger.info(f"数据质量报告:")
        logger.info(f"  总行数: {total_rows}")
        logger.info(f"  Y染色体浓度缺失: {missing_y_chrom} ({missing_y_chrom/total_rows*100:.1f}%)")
        logger.info(f"  检测孕周缺失: {missing_weeks} ({missing_weeks/total_rows*100:.1f}%)")
        
        # 异常值检测
        y_chrom_outliers = ((self.data['Y染色体浓度'] < 0) | (self.data['Y染色体浓度'] > 1)).sum()
        if y_chrom_outliers > 0:
            logger.warning(f"发现 {y_chrom_outliers} 个Y染色体浓度异常值")
    
    def _advanced_parse_gestational_weeks(self, week_series: pd.Series) -> pd.Series:
        """高级孕周解析，支持更多格式，减少数据丢失"""
        def advanced_parse(week_str):
            if pd.isna(week_str):
                return None
            if isinstance(week_str, (int, float)):
                return float(week_str)
            
            try:
                week_str = str(week_str).strip().lower()
                
                # 处理各种格式
                if 'w' in week_str:
                    if '+' in week_str:
                        # 格式: "12w+3"
                        parts = week_str.replace('w', '+').split('+')
                        weeks = float(parts[0])
                        days = float(parts[1]) if len(parts) > 1 else 0
                        return weeks + days / 7
                    else:
                        # 格式: "12w"
                        return float(week_str.replace('w', ''))
                elif '周' in week_str:
                    # 中文格式
                    week_str = week_str.replace('周', '').replace('孕', '')
                    return float(week_str)
                else:
                    # 纯数字
                    return float(week_str)
                    
            except Exception:
                logger.debug(f"无法解析孕周: {week_str}")
                return None
        
        # 先解析所有值
        parsed_series = week_series.apply(advanced_parse)
        
        # 对解析失败的数据使用均值填充，减少数据丢失
        valid_weeks = parsed_series.dropna()
        if len(valid_weeks) > 0:
            mean_week = valid_weeks.mean()
            parsed_series = parsed_series.fillna(mean_week)
            logger.info(f"孕周解析完成，使用均值 {mean_week:.1f} 填充 {parsed_series.isna().sum()} 个缺失值")
        else:
            # 如果没有任何有效值，使用合理的默认值
            parsed_series = parsed_series.fillna(20.0)  # 20周作为合理默认值
            logger.warning("没有有效孕周数据，使用默认值20周填充")
        
        return parsed_series
    
    def _parallel_process_qualified_records(self, male_data: pd.DataFrame) -> List[Dict]:
        """并行处理达标记录"""
        logger.info("🔄 并行处理达标记录...")
        
        # 按孕妇代码分组
        groups = [(code, group) for code, group in male_data.groupby('孕妇代码')]
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = [executor.submit(self._process_single_patient, code, group) 
                      for code, group in groups]
            
            qualified_records = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理患者数据"):
                try:
                    record = future.result()
                    if record is not None:
                        qualified_records.append(record)
                except Exception as e:
                    logger.warning(f"处理患者数据时出错: {e}")
        
        return qualified_records
    
    @staticmethod
    def _process_single_patient(code: str, group: pd.DataFrame) -> Optional[Dict]:
        """处理单个患者数据（静态方法用于多进程），减少数据丢失"""
        try:
            # 按检测孕周排序
            patient_data = group.sort_values('检测孕周_数值')
            # 不再严格要求检测孕周数值，因为已经在解析时填充了
            
            if len(patient_data) == 0:
                return None
            
            # 计算达标统计
            total_tests = len(patient_data)
            qualified_mask = patient_data['Y染色体浓度'] >= 0.04
            qualified_tests = qualified_mask.sum()
            达标_ratio = qualified_tests / total_tests if total_tests > 0 else 0
            
            # 优先找到第一个达标时间，如果没有达标记录则使用第一个记录
            qualified_data = patient_data[qualified_mask]
            
            if not qualified_data.empty:
                # 有达标记录，使用第一个达标记录
                first_qualified = qualified_data.iloc[0]
                record = first_qualified.to_dict()
                record['达标时间'] = first_qualified['检测孕周_数值']
                record['达标比例'] = 达标_ratio
                record['总检测次数'] = total_tests
                record['达标检测次数'] = qualified_tests
                record['数据类型'] = '达标记录'
                return record
            else:
                # 没有达标记录，但仍保留数据用于分析，使用最后一次检测
                last_record = patient_data.iloc[-1]
                record = last_record.to_dict()
                record['达标时间'] = last_record['检测孕周_数值']
                record['达标比例'] = 达标_ratio  # 为0
                record['总检测次数'] = total_tests
                record['达标检测次数'] = qualified_tests  # 为0
                record['数据类型'] = '未达标记录'
                logger.debug(f"患者 {code} 无达标记录，但保留数据用于分析")
                return record
            
        except Exception as e:
            logger.debug(f"处理患者 {code} 数据时出错: {e}")
            # 即使出错也尝试返回基础信息
            try:
                if len(group) > 0:
                    first_record = group.iloc[0]
                    record = first_record.to_dict()
                    record['达标时间'] = record.get('检测孕周_数值', 20.0)  # 使用默认值
                    record['达标比例'] = 0.0
                    record['总检测次数'] = len(group)
                    record['达标检测次数'] = 0
                    record['数据类型'] = '异常记录'
                    return record
            except:
                pass
        
        return None
    
    def _advanced_feature_engineering(self):
        """高级特征工程"""
        logger.info("🔧 进行高级特征工程...")
        
        # 基础特征
        feature_columns = ['孕妇年龄', '孕妇身高', '孕妇体重', '孕妇BMI', 
                          'GC含量', '怀孕次数', '生产次数', '达标比例']
        
        # 重命名映射
        column_mapping = {
            '孕妇BMI': 'BMI',
            'GC含量': 'GC_content',
            '怀孕次数': 'pregnancy_count',
            '生产次数': 'birth_count'
        }
        
        self.qualified_data = self.qualified_data.rename(columns=column_mapping)
        
        # 提取可用特征
        available_features = []
        for col in feature_columns:
            mapped_col = column_mapping.get(col, col)
            if mapped_col in self.qualified_data.columns:
                available_features.append(mapped_col)
        
        # 创建派生特征
        self._create_derived_features(available_features)
        
        # 特征选择和清洗
        self.feature_data = self.qualified_data[available_features + ['达标时间']].copy()
        
        # 数值化和异常值处理
        self._clean_and_normalize_features(available_features)
        
        logger.info(f"特征工程完成，最终特征: {list(self.feature_data.columns)}")
    
    def _create_derived_features(self, available_features: List[str]):
        """创建衍生特征 - 扩展版本"""
        logger.info("🔧 创建衍生特征...")
        
        # 基础特征映射
        if '孕妇身高' in self.qualified_data.columns:
            self.qualified_data['身高'] = self.qualified_data['孕妇身高']
        if '孕妇体重' in self.qualified_data.columns:
            self.qualified_data['体重'] = self.qualified_data['孕妇体重']
        if '孕妇年龄' in self.qualified_data.columns:
            self.qualified_data['年龄'] = self.qualified_data['孕妇年龄']
        if 'BMI' in self.qualified_data.columns:
            self.qualified_data['孕妇BMI'] = self.qualified_data['BMI']
        
        # 1. BMI相关特征
        if 'BMI' in available_features:
            self.qualified_data['BMI_category'] = pd.cut(
                self.qualified_data['BMI'],
                bins=[0, 18.5, 24, 28, float('inf')],
                labels=[0, 1, 2, 3]  # 数值化分类
            ).astype(float)
            
            self.qualified_data['BMI_squared'] = self.qualified_data['BMI'] ** 2
            self.qualified_data['BMI_log'] = np.log1p(self.qualified_data['BMI'])
            self.qualified_data['BMI_normalized'] = (self.qualified_data['BMI'] - self.qualified_data['BMI'].mean()) / self.qualified_data['BMI'].std()
        
        # 2. 年龄相关特征
        if '年龄' in self.qualified_data.columns:
            self.qualified_data['age_group'] = pd.cut(
                self.qualified_data['年龄'],
                bins=[0, 25, 30, 35, float('inf')],
                labels=[0, 1, 2, 3]  # 数值化分类
            ).astype(float)
            
            self.qualified_data['age_squared'] = self.qualified_data['年龄'] ** 2
            self.qualified_data['age_normalized'] = (self.qualified_data['年龄'] - self.qualified_data['年龄'].mean()) / self.qualified_data['年龄'].std()
        
        # 3. 身高体重相关特征
        if all(col in self.qualified_data.columns for col in ['身高', '体重']):
            self.qualified_data['height_weight_ratio'] = self.qualified_data['身高'] / self.qualified_data['体重']
            self.qualified_data['body_surface_area'] = np.sqrt((self.qualified_data['身高'] * self.qualified_data['体重']) / 3600)
        
        # 4. 时间相关特征
        self.qualified_data['当前检测达标_std'] = self.qualified_data.groupby('孕妇代码')['达标时间'].transform('std').fillna(0)
        self.qualified_data['检测时怀孕天数'] = self.qualified_data['达标时间'] * 7
        self.qualified_data['达标时间_log'] = np.log1p(self.qualified_data['达标时间'])
        self.qualified_data['达标时间_squared'] = self.qualified_data['达标时间'] ** 2
        self.qualified_data['达标时间_normalized'] = (self.qualified_data['达标时间'] - self.qualified_data['达标时间'].mean()) / self.qualified_data['达标时间'].std()
        
        # 5. 生育相关特征
        if all(col in available_features for col in ['pregnancy_count', 'birth_count']):
            # 首先进行数据类型转换，处理字符串数据
            self.qualified_data['pregnancy_count'] = pd.to_numeric(self.qualified_data['pregnancy_count'], errors='coerce').fillna(0)
            self.qualified_data['birth_count'] = pd.to_numeric(self.qualified_data['birth_count'], errors='coerce').fillna(0)
            
            self.qualified_data['birth_experience'] = (self.qualified_data['birth_count'] > 0).astype(int)
            self.qualified_data['pregnancy_birth_ratio'] = np.where(
                self.qualified_data['pregnancy_count'] > 0,
                self.qualified_data['birth_count'] / self.qualified_data['pregnancy_count'],
                0
            )
            self.qualified_data['total_reproduction'] = self.qualified_data['pregnancy_count'] + self.qualified_data['birth_count']
        
        # 6. 达标相关特征
        self.qualified_data['达标率'] = self.qualified_data['达标比例']
        if '总检测次数' in self.qualified_data.columns:
            self.qualified_data['detection_efficiency'] = (
                self.qualified_data['达标检测次数'] / self.qualified_data['总检测次数']
            )
            self.qualified_data['detection_intensity'] = self.qualified_data['总检测次数'] / self.qualified_data['达标时间']
        
        # 7. GC含量相关特征
        if 'GC_content' in available_features:
            # 数据类型转换
            self.qualified_data['GC_content'] = pd.to_numeric(self.qualified_data['GC_content'], errors='coerce').fillna(self.qualified_data['GC_content'].median() if pd.api.types.is_numeric_dtype(self.qualified_data['GC_content']) else 0)
            
            self.qualified_data['GC_content_log'] = np.log1p(self.qualified_data['GC_content'])
            self.qualified_data['GC_content_squared'] = self.qualified_data['GC_content'] ** 2
            gc_mean = self.qualified_data['GC_content'].mean()
            gc_std = self.qualified_data['GC_content'].std()
            if gc_std > 0:
                self.qualified_data['GC_content_normalized'] = (self.qualified_data['GC_content'] - gc_mean) / gc_std
            else:
                self.qualified_data['GC_content_normalized'] = 0
        
        # 8. 交互特征
        if all(col in self.qualified_data.columns for col in ['BMI', '年龄']):
            self.qualified_data['BMI_age_interaction'] = self.qualified_data['BMI'] * self.qualified_data['年龄']
            self.qualified_data['BMI_age_ratio'] = self.qualified_data['BMI'] / self.qualified_data['年龄']
        
        if all(col in self.qualified_data.columns for col in ['达标率', 'BMI']):
            self.qualified_data['qualification_BMI_interaction'] = self.qualified_data['达标率'] * self.qualified_data['BMI']
        
        # 9. 统计特征（基于组内统计）
        if 'BMI' in self.qualified_data.columns:
            self.qualified_data['BMI_rank'] = self.qualified_data['BMI'].rank(pct=True)
            self.qualified_data['BMI_percentile'] = pd.qcut(self.qualified_data['BMI'], 10, labels=False, duplicates='drop').astype(float)
        
        if '达标时间' in self.qualified_data.columns:
            self.qualified_data['qualification_time_rank'] = self.qualified_data['达标时间'].rank(pct=True)
            self.qualified_data['qualification_time_deviation'] = abs(
                self.qualified_data['达标时间'] - self.qualified_data['达标时间'].median()
            )
        
        # 10. 风险评估特征
        if all(col in self.qualified_data.columns for col in ['年龄', 'BMI']):
            # 高龄高BMI风险
            self.qualified_data['high_risk_age'] = (self.qualified_data['年龄'] >= 35).astype(int)
            self.qualified_data['high_risk_BMI'] = (self.qualified_data['BMI'] >= 28).astype(int)
            self.qualified_data['combined_risk_score'] = (
                self.qualified_data['high_risk_age'] + self.qualified_data['high_risk_BMI']
            )
        
        # 记录创建的特征数量
        current_features = len(self.qualified_data.columns)
        logger.info(f"特征工程完成，当前特征数量: {current_features}")
        logger.info("创建的衍生特征包括: BMI类别、年龄组、身高体重比、体表面积、交互特征、统计特征、风险评估特征等")
    
    def _clean_and_normalize_features(self, available_features: List[str]):
        """清洗和标准化特征，减少数据丢失"""
        # 数值化处理
        for col in available_features:
            if col in self.feature_data.columns:
                self.feature_data[col] = pd.to_numeric(self.feature_data[col], errors='coerce')
        
        # 缺失值填充（使用中位数填充，减少数据丢失）
        original_len = len(self.feature_data)
        missing_info = []
        
        for col in available_features:
            if col in self.feature_data.columns:
                missing_count = self.feature_data[col].isna().sum()
                if missing_count > 0:
                    # 使用中位数填充数值型特征
                    if self.feature_data[col].dtype in ['float64', 'int64']:
                        fill_value = self.feature_data[col].median()
                        if pd.isna(fill_value):  # 如果中位数也是NaN，使用均值
                            fill_value = self.feature_data[col].mean()
                            if pd.isna(fill_value):  # 如果均值也是NaN，使用默认值
                                if col == 'BMI':
                                    fill_value = 22.0  # BMI正常值
                                elif col == '孕妇年龄':
                                    fill_value = 28.0  # 平均年龄
                                elif col == '达标比例':
                                    fill_value = 0.5   # 50%达标率
                                else:
                                    fill_value = 0.0
                    else:
                        # 分类特征使用众数
                        fill_value = self.feature_data[col].mode().iloc[0] if len(self.feature_data[col].mode()) > 0 else 0
                    
                    self.feature_data[col] = self.feature_data[col].fillna(fill_value)
                    missing_info.append(f"{col}: {missing_count}个缺失值用{fill_value}填充")
        
        if missing_info:
            logger.info(f"缺失值填充完成: {', '.join(missing_info)}")
        
        # 异常值处理（使用IQR方法，但更宽松）
        for col in available_features:
            if col in self.feature_data.columns and self.feature_data[col].dtype in ['float64', 'int64']:
                Q1 = self.feature_data[col].quantile(0.25)
                Q3 = self.feature_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # 使用更宽松的异常值检测（2.5倍IQR而不是1.5倍）
                lower_bound = Q1 - 2.5 * IQR
                upper_bound = Q3 + 2.5 * IQR
                
                # 记录异常值数量
                outliers = ((self.feature_data[col] < lower_bound) | 
                           (self.feature_data[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"特征 {col} 发现 {outliers} 个异常值（宽松检测）")
                
                # 将异常值设为边界值
                self.feature_data[col] = self.feature_data[col].clip(lower_bound, upper_bound)
        
        # 最终检查，如果还有极少数缺失值则删除
        remaining_missing = self.feature_data.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"仍有 {remaining_missing} 个缺失值，删除对应行")
            self.feature_data = self.feature_data.dropna()
            self.qualified_data = self.qualified_data.loc[self.feature_data.index].reset_index(drop=True)
            self.feature_data = self.feature_data.reset_index(drop=True)
        
        final_len = len(self.feature_data)
        logger.info(f"数据清洗完成，保留样本: {final_len}/{original_len} ({final_len/original_len*100:.1f}%)")
    
    def advanced_feature_importance_analysis(self) -> pd.DataFrame:
        """高级特征重要性分析"""
        logger.info("🔍 进行高级特征重要性分析...")
        
        if len(self.feature_data) == 0:
            return pd.DataFrame()
        
        feature_cols = [col for col in self.feature_data.columns if col != '达标时间']
        X = self.feature_data[feature_cols]
        y = self.feature_data['达标时间']
        
        # 多种方法计算特征重要性
        importance_methods = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=self.config.n_jobs),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        importance_results = {}
        
        for method_name, estimator in importance_methods.items():
            logger.info(f"使用 {method_name} 分析特征重要性...")
            
            # 交叉验证评估
            cv_scores = cross_val_score(estimator, X, y, cv=self.config.cross_validation_folds, 
                                       scoring='neg_mean_squared_error', n_jobs=self.config.n_jobs)
            
            # 训练模型
            estimator.fit(X, y)
            
            # 记录结果
            importance_results[method_name] = {
                'importance': estimator.feature_importances_,
                'cv_score': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{method_name} CV RMSE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 综合特征重要性
        combined_importance = np.mean([
            importance_results['random_forest']['importance'],
            importance_results['gradient_boosting']['importance']
        ], axis=0)
        
        # 创建结果DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'rf_importance': importance_results['random_forest']['importance'],
            'gb_importance': importance_results['gradient_boosting']['importance'],
            'combined_importance': combined_importance
        }).sort_values('combined_importance', ascending=False)
        
        self.feature_importance = feature_importance
        
        # 可视化
        self._visualize_advanced_feature_importance(feature_importance, importance_results)
        
        return feature_importance
    
    def _visualize_advanced_feature_importance(self, feature_importance: pd.DataFrame, 
                                             importance_results: Dict[str, Any]):
        """可视化高级特征重要性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('高级特征重要性分析', fontsize=16, fontweight='bold')
        
        # 1. 综合特征重要性
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(feature_importance)), feature_importance['combined_importance'])
        ax1.set_xlabel('特征')
        ax1.set_ylabel('综合重要性')
        ax1.set_title('多算法综合特征重要性')
        ax1.set_xticks(range(len(feature_importance)))
        ax1.set_xticklabels(feature_importance['feature'], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, importance in zip(bars, feature_importance['combined_importance']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{importance:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 不同算法比较
        ax2 = axes[0, 1]
        x = np.arange(len(feature_importance))
        width = 0.35
        
        ax2.bar(x - width/2, feature_importance['rf_importance'], width, 
               label='Random Forest', alpha=0.8)
        ax2.bar(x + width/2, feature_importance['gb_importance'], width, 
               label='Gradient Boosting', alpha=0.8)
        
        ax2.set_xlabel('特征')
        ax2.set_ylabel('特征重要性')
        ax2.set_title('不同算法特征重要性对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(feature_importance['feature'], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 特征相关性热力图
        ax3 = axes[1, 0]
        feature_cols = [col for col in self.feature_data.columns if col != '达标时间']
        correlation_matrix = self.feature_data[feature_cols].corr()
        
        im = ax3.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(correlation_matrix.columns)))
        ax3.set_yticks(range(len(correlation_matrix.columns)))
        ax3.set_xticklabels(correlation_matrix.columns, rotation=45)
        ax3.set_yticklabels(correlation_matrix.columns)
        ax3.set_title('特征相关性矩阵')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4. 模型性能对比
        ax4 = axes[1, 1]
        methods = list(importance_results.keys())
        cv_scores = [importance_results[method]['cv_score'] for method in methods]
        cv_stds = [importance_results[method]['cv_std'] for method in methods]
        
        bars = ax4.bar(methods, cv_scores, yerr=cv_stds, capsize=5)
        ax4.set_ylabel('RMSE')
        ax4.set_title('模型交叉验证性能')
        ax4.grid(True, alpha=0.3)
        
        for bar, score, std in zip(bars, cv_scores, cv_stds):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{score:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, '高级特征重要性分析.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def advanced_clustering_analysis(self) -> Tuple[int, np.ndarray]:
        """高级聚类分析"""
        logger.info("🔍 进行高级聚类分析...")
        
        # 准备特征数据
        important_features = self._select_clustering_features()
        cluster_data = self.qualified_data[important_features].values
        
        # 尝试不同的标准化方法和聚类算法
        best_score = -np.inf
        best_method = None
        best_labels = None
        best_k = None
        
        clustering_results = {}
        
        for scaler_name in self.config.scaling_methods:
            for cluster_method in self.config.clustering_methods:
                logger.info(f"测试 {scaler_name} + {cluster_method}...")
                
                # 标准化数据
                if scaler_name == 'standard':
                    scaler = StandardScaler()
                elif scaler_name == 'robust':
                    scaler = RobustScaler()
                
                scaled_data = scaler.fit_transform(cluster_data)
                
                # 聚类分析
                method_results = self._evaluate_clustering_method(scaled_data, cluster_method)
                clustering_results[f"{scaler_name}_{cluster_method}"] = method_results
                
                # 选择最佳方法
                if method_results['best_silhouette_score'] > best_score:
                    best_score = method_results['best_silhouette_score']
                    best_method = f"{scaler_name}_{cluster_method}"
                    best_labels = method_results['best_labels']
                    best_k = method_results['best_k']
        
        logger.info(f"最佳聚类方法: {best_method}, k={best_k}, 轮廓系数={best_score:.4f}")
        
        # 保存结果
        self.clustering_results = clustering_results
        self.optimal_clusters = best_k
        self.qualified_data['cluster'] = self._sort_cluster_labels(best_labels, best_k)
        
        # 可视化聚类结果
        self._visualize_clustering_analysis(clustering_results, best_method)
        
        # 生成聚类过程综合分析可视化
        self._generate_clustering_process_visualization()
        
        # 生成BMI分组可视化
        self._generate_bmi_grouping_visualization()
        
        return best_k, best_labels
    
    def _select_clustering_features(self) -> List[str]:
        """选择聚类特征"""
        # 首先检查数据中有哪些列可用
        available_columns = self.qualified_data.columns.tolist()
        logger.info(f"qualified_data中可用列: {available_columns}")
        
        # 用户指定的聚类特征
        target_features = ['身高', '体重', '检测时怀孕天数', '年龄', '当前检测达标_std', '孕妇BMI', '达标率']
        
        # 检查哪些特征在数据中可用
        available_clustering_features = []
        for feature in target_features:
            if feature in available_columns:
                available_clustering_features.append(feature)
            else:
                logger.warning(f"聚类特征 '{feature}' 在数据中不可用")
        
        # 如果没有可用的指定特征，则使用默认特征
        if not available_clustering_features:
            logger.warning("没有找到指定的聚类特征，使用默认特征")
            # 尝试使用一些常见的特征名
            fallback_features = ['BMI', '孕妇BMI', 'GC_content', 'pregnancy_count', 'birth_count', '达标比例']
            
            for feature in fallback_features:
                if feature in available_columns and feature not in available_clustering_features:
                    available_clustering_features.append(feature)
            
            # 如果还是没有，使用特征重要性
            if self.feature_importance is not None and not available_clustering_features:
                top_features = self.feature_importance.head(5)['feature'].tolist()
                for feature in top_features:
                    if feature in available_columns and feature != '达标时间':
                        available_clustering_features.append(feature)
        
        # 确保至少有一个特征
        if not available_clustering_features:
            # 最后的回退：使用数值型列（排除达标时间）
            numeric_columns = self.qualified_data.select_dtypes(include=[np.number]).columns.tolist()
            available_clustering_features = [col for col in numeric_columns if col != '达标时间'][:5]
        
        logger.info(f"聚类使用特征: {available_clustering_features}")
        logger.info(f"聚类数据维度: ({len(self.qualified_data)}, {len(available_clustering_features)})")
        
        return available_clustering_features
    
    def _evaluate_clustering_method(self, scaled_data: np.ndarray, method: str) -> Dict[str, Any]:
        """评估聚类方法"""
        if method == 'kmeans':
            return self._evaluate_kmeans(scaled_data)
        elif method == 'agglomerative':
            return self._evaluate_agglomerative(scaled_data)
        elif method == 'gmm':
            return self._evaluate_gmm(scaled_data)
        elif method == 'dbscan':
            return self._evaluate_dbscan(scaled_data)
        else:
            raise ValueError(f"未知的聚类方法: {method}")
    
    def _evaluate_kmeans(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """评估K-means聚类"""
        k_range = range(2, self.config.max_clusters + 1)
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        valid_k_range = []
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                labels = kmeans.fit_predict(scaled_data)
                
                # 检查聚类结果（更宽松的检查）
                unique_labels, counts = np.unique(labels, return_counts=True)
                # 只要有足够的簇且最小簇不少于2个样本就接受
                min_acceptable_size = max(2, self.config.min_cluster_size // 2)
                if len(unique_labels) < k or np.min(counts) < min_acceptable_size:
                    logger.debug(f"K-means跳过k={k}: 簇太小({np.min(counts)}<{min_acceptable_size})或簇数不足({len(unique_labels)}<{k})")
                    continue
                
                # 计算评估指标
                sil_score = silhouette_score(scaled_data, labels)
                cal_score = calinski_harabasz_score(scaled_data, labels)
                db_score = davies_bouldin_score(scaled_data, labels)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
                davies_bouldin_scores.append(db_score)
                valid_k_range.append(k)
                
            except Exception as e:
                logger.debug(f"k={k}聚类失败: {e}")
                continue
        
        # 选择最佳k
        if silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            best_k = valid_k_range[best_idx]
            
            # 重新训练最佳模型
            best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
            best_labels = best_kmeans.fit_predict(scaled_data)
            
            return {
                'best_k': best_k,
                'best_labels': best_labels,
                'best_silhouette_score': silhouette_scores[best_idx],
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores,
                'k_range': valid_k_range
            }
        else:
            return {'best_silhouette_score': -1, 'k_range': [], 'silhouette_scores': []}
    
    def _evaluate_agglomerative(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """评估层次聚类"""
        k_range = range(2, self.config.max_clusters + 1)
        silhouette_scores = []
        valid_k_range = []
        
        for k in k_range:
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = agg.fit_predict(scaled_data)
                
                # 检查聚类结果（更宽松的检查）
                unique_labels, counts = np.unique(labels, return_counts=True)
                # 只要有足够的簇且最小簇不少于2个样本就接受
                min_acceptable_size = max(2, self.config.min_cluster_size // 2)
                if len(unique_labels) < k or np.min(counts) < min_acceptable_size:
                    logger.debug(f"Agglomerative跳过k={k}: 簇太小({np.min(counts)}<{min_acceptable_size})或簇数不足({len(unique_labels)}<{k})")
                    continue
                
                sil_score = silhouette_score(scaled_data, labels)
                silhouette_scores.append(sil_score)
                valid_k_range.append(k)
                
            except Exception as e:
                logger.debug(f"k={k}层次聚类失败: {e}")
                continue
        
        if silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            best_k = valid_k_range[best_idx]
            
            best_agg = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
            best_labels = best_agg.fit_predict(scaled_data)
            
            return {
                'best_k': best_k,
                'best_labels': best_labels,
                'best_silhouette_score': silhouette_scores[best_idx],
                'silhouette_scores': silhouette_scores,
                'k_range': valid_k_range
            }
        else:
            return {'best_silhouette_score': -1, 'k_range': [], 'silhouette_scores': []}
    
    def _evaluate_dbscan(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """评估DBSCAN聚类"""
        eps_range = np.arange(0.1, 2.0, 0.1)
        min_samples_range = [5, 10, 15, 20]
        
        best_score = -1
        best_params = None
        best_labels = None
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(scaled_data)
                
                # 检查聚类结果
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                
                if n_clusters < 2 or n_clusters > self.config.max_clusters:
                    continue
                
                # 检查噪声点比例（更宽松的噪声容忍度）
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                if noise_ratio > 0.2:  # 噪声点不超过20%（从10%放宽到20%）
                    continue
                
                try:
                    silhouette_avg = silhouette_score(scaled_data, labels)
                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_labels = labels
                except:
                    continue
        
        if best_params:
            return {
                'best_k': len(np.unique(best_labels)) - (1 if -1 in best_labels else 0),
                'best_labels': best_labels,
                'best_silhouette_score': best_score,
                'best_params': best_params
            }
        else:
            return {'best_silhouette_score': -1}
    
    def _evaluate_gmm(self, scaled_data: np.ndarray) -> Dict[str, Any]:
        """评估高斯混合模型聚类"""
        k_range = range(2, self.config.max_clusters + 1)
        bic_scores = []
        aic_scores = []
        silhouette_scores = []
        valid_k_range = []
        
        for k in k_range:
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
                labels = gmm.fit_predict(scaled_data)
                
                # 检查聚类结果（更宽松的检查）
                unique_labels, counts = np.unique(labels, return_counts=True)
                # 只要有足够的簇且最小簇不少于2个样本就接受
                min_acceptable_size = max(2, self.config.min_cluster_size // 2)
                if len(unique_labels) < k or np.min(counts) < min_acceptable_size:
                    logger.debug(f"GMM跳过k={k}: 簇太小({np.min(counts)}<{min_acceptable_size})或簇数不足({len(unique_labels)}<{k})")
                    continue
                
                # 计算评估指标
                bic_score = gmm.bic(scaled_data)
                aic_score = gmm.aic(scaled_data)
                sil_score = silhouette_score(scaled_data, labels)
                
                bic_scores.append(bic_score)
                aic_scores.append(aic_score)
                silhouette_scores.append(sil_score)
                valid_k_range.append(k)
                
            except Exception as e:
                logger.debug(f"k={k} GMM失败: {e}")
                continue
        
        # 选择最佳k（基于轮廓系数）
        if silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            best_k = valid_k_range[best_idx]
            
            # 重新训练最佳模型
            best_gmm = GaussianMixture(n_components=best_k, random_state=42, covariance_type='full')
            best_labels = best_gmm.fit_predict(scaled_data)
            
            return {
                'best_k': best_k,
                'best_labels': best_labels,
                'best_silhouette_score': silhouette_scores[best_idx],
                'best_bic': bic_scores[best_idx],
                'best_aic': aic_scores[best_idx],
                'bic_scores': bic_scores,
                'aic_scores': aic_scores,
                'silhouette_scores': silhouette_scores,
                'k_range': valid_k_range
            }
        else:
            return {'best_silhouette_score': -1, 'k_range': [], 'silhouette_scores': []}
    
    def _sort_cluster_labels(self, cluster_labels: np.ndarray, n_clusters: int) -> np.ndarray:
        """根据BMI均值排序簇标签并确保无重叠区间"""
        cluster_info = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.sum(mask) > 0:
                cluster_bmi_data = self.qualified_data[mask]['BMI']
                cluster_bmi_mean = cluster_bmi_data.mean()
                cluster_bmi_min = cluster_bmi_data.min()
                cluster_bmi_max = cluster_bmi_data.max()
                cluster_info.append((i, cluster_bmi_mean, cluster_bmi_min, cluster_bmi_max, np.sum(mask)))
        
        # 按BMI均值排序
        cluster_info.sort(key=lambda x: x[1])
        
        # 检查并调整重叠区间
        adjusted_cluster_info = self._adjust_overlapping_bmi_ranges(cluster_info)
        
        # 创建标签映射
        label_mapping = {old_label: new_label for new_label, (old_label, _, _, _, _) in enumerate(adjusted_cluster_info)}
        return np.array([label_mapping.get(label, -1) for label in cluster_labels])
    
    def _adjust_overlapping_bmi_ranges(self, cluster_info: List[tuple]) -> List[tuple]:
        """调整重叠的BMI区间，确保各组区间不重叠"""
        if len(cluster_info) <= 1:
            return cluster_info
        
        adjusted_info = []
        
        for i, (cluster_id, mean_bmi, min_bmi, max_bmi, size) in enumerate(cluster_info):
            if i == 0:
                # 第一组保持原样
                adjusted_info.append((cluster_id, mean_bmi, min_bmi, max_bmi, size))
            else:
                # 检查与前一组的重叠
                prev_max = adjusted_info[-1][3]  # 前一组的最大值
                
                if min_bmi <= prev_max:
                    # 存在重叠，调整边界
                    # 使用前一组最大值和当前组最大值的中点作为分界点
                    boundary = (prev_max + max_bmi) / 2
                    
                    # 调整前一组的最大值
                    prev_cluster = adjusted_info[-1]
                    adjusted_info[-1] = (prev_cluster[0], prev_cluster[1], prev_cluster[2], boundary, prev_cluster[4])
                    
                    # 调整当前组的最小值
                    adjusted_min = boundary + 0.01  # 添加小间隔避免边界重叠
                    adjusted_info.append((cluster_id, mean_bmi, adjusted_min, max_bmi, size))
                    
                    logger.info(f"调整BMI区间重叠: 组{len(adjusted_info)-1}最大值调整为{boundary:.2f}, "
                               f"组{len(adjusted_info)}最小值调整为{adjusted_min:.2f}")
                else:
                    # 无重叠，保持原样
                    adjusted_info.append((cluster_id, mean_bmi, min_bmi, max_bmi, size))
        
        return adjusted_info
    
    def _visualize_clustering_analysis(self, clustering_results: Dict[str, Any], best_method: str):
        """可视化聚类分析结果"""
        n_methods = len(clustering_results)
        if n_methods == 0:
            logger.warning("没有聚类结果可视化")
            return
        
        # 计算子图布局
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # 处理单个子图的情况
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_methods > 1 else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('高级聚类分析结果', fontsize=16, fontweight='bold')
        
        for i, (method_name, results) in enumerate(clustering_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 检查是否有有效的聚类结果
            if ('silhouette_scores' in results and 
                'k_range' in results and 
                results['silhouette_scores'] and 
                results['k_range'] and
                len(results['silhouette_scores']) == len(results['k_range'])):
                
                k_range = results['k_range']
                silhouette_scores = results['silhouette_scores']
                
                # 绘制轮廓系数
                ax.plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color='blue', alpha=0.7)
                
                # 标记最佳点
                best_k = results.get('best_k')
                if best_k and best_k in k_range:
                    best_idx = k_range.index(best_k)
                    best_score = silhouette_scores[best_idx]
                    ax.plot(best_k, best_score, 'ro', markersize=12, markerfacecolor='red', markeredgecolor='darkred')
                    ax.text(best_k, best_score, f'  k={best_k}\n  score={best_score:.3f}', 
                           va='center', fontsize=10, fontweight='bold', color='red')
                
                ax.set_xlabel('聚类数 (k)')
                ax.set_ylabel('轮廓系数')
                ax.set_title(f'{method_name}\n{"★最佳" if method_name == best_method else ""}', 
                           fontweight='bold' if method_name == best_method else 'normal')
                ax.grid(True, alpha=0.3)
                
                # 设置x轴刻度
                if k_range:
                    ax.set_xticks(k_range)
                
                # 设置y轴范围
                if silhouette_scores:
                    y_min, y_max = min(silhouette_scores), max(silhouette_scores)
                    y_padding = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - y_padding, y_max + y_padding)
                
            else:
                # 没有有效结果的情况
                ax.text(0.5, 0.5, f'{method_name}\n无有效结果\n(数据量可能不足)', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=12, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 隐藏多余的子图
        for i in range(len(clustering_results), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        
        try:
            plt.savefig(os.path.join(self.config.output_dir, '高级聚类分析.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info("聚类分析图表已保存")
        except Exception as e:
            logger.warning(f"保存聚类分析图表失败: {e}")
        
        plt.show()
    
    def _generate_clustering_process_visualization(self):
        """生成聚类过程综合分析可视化"""
        logger.info("🎨 生成聚类过程综合分析可视化...")
        
        if not self.clustering_results:
            logger.warning("没有聚类结果可可视化")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 创建聚类方法对比表
        comparison_data = []
        for method_name, results in self.clustering_results.items():
            if 'best_k' in results and results['best_silhouette_score'] > 0:
                method_display = method_name.replace('_', ' ').title()
                if 'kmeans' in method_name.lower():
                    method_display = 'K-Means'
                elif 'agglomerative' in method_name.lower():
                    method_display = 'Hierarchical'
                elif 'gmm' in method_name.lower():
                    method_display = 'GMM'
                
                comparison_data.append({
                    'Method': method_display,
                    'Clusters': results['best_k'],
                    'Silhouette Score': f"{results['best_silhouette_score']:.3f}"
                })
        
        # 主标题
        fig.suptitle('聚类过程综合分析可视化', fontsize=18, fontweight='bold', y=0.95)
        
        # 1. 聚类方法对比表
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('tight')
        ax1.axis('off')
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            table = ax1.table(cellText=df_comparison.values,
                            colLabels=df_comparison.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.8)
            
            # 设置表格样式
            for i in range(len(df_comparison.columns)):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('聚类方法对比', fontsize=14, fontweight='bold', pad=20)
        
        # 2. 各方法轮廓系数对比
        ax2 = fig.add_subplot(gs[0, 2:])
        methods = []
        scores = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for i, (method_name, results) in enumerate(self.clustering_results.items()):
            if 'best_silhouette_score' in results and results['best_silhouette_score'] > 0:
                method_display = method_name.replace('standard_', '').replace('robust_', '').upper()
                methods.append(method_display)
                scores.append(results['best_silhouette_score'])
        
        if methods:
            bars = ax2.bar(methods, scores, color=colors[:len(methods)], alpha=0.8)
            ax2.set_ylabel('轮廓系数')
            ax2.set_title('各聚类方法轮廓系数对比')
            ax2.grid(True, alpha=0.3)
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3-4. 各方法详细结果
        plot_idx = 1
        for method_name, results in list(self.clustering_results.items())[:2]:
            if ('silhouette_scores' in results and 'k_range' in results and 
                results['silhouette_scores'] and results['k_range']):
                
                ax = fig.add_subplot(gs[plot_idx, :2])
                k_range = results['k_range']
                silhouette_scores = results['silhouette_scores']
                
                ax.plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, 
                       color=colors[plot_idx-1], alpha=0.8)
                
                best_k = results.get('best_k')
                if best_k and best_k in k_range:
                    best_idx = k_range.index(best_k)
                    best_score = silhouette_scores[best_idx]
                    ax.plot(best_k, best_score, 'ro', markersize=12, 
                           markerfacecolor='red', markeredgecolor='darkred')
                    ax.text(best_k, best_score, f'  最佳: k={best_k}', 
                           va='center', fontsize=10, fontweight='bold', color='red')
                
                ax.set_xlabel('聚类数 (k)')
                ax.set_ylabel('轮廓系数')
                ax.set_title(f'{method_name.upper()} 聚类评估')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(k_range)
                
                plot_idx += 1
        
        # 5. GMM BIC/AIC 分析（如果有）
        gmm_results = None
        for method_name, results in self.clustering_results.items():
            if 'gmm' in method_name and 'bic_scores' in results:
                gmm_results = results
                break
        
        if gmm_results and plot_idx <= 2:
            ax = fig.add_subplot(gs[plot_idx, 2:])
            k_range = gmm_results['k_range']
            bic_scores = gmm_results['bic_scores']
            
            ax.plot(k_range, bic_scores, 'o-', linewidth=2, markersize=8, 
                   color='purple', alpha=0.8, label='BIC Score')
            
            best_k = gmm_results.get('best_k')
            if best_k and best_k in k_range:
                best_idx = k_range.index(best_k)
                best_bic = bic_scores[best_idx]
                ax.plot(best_k, best_bic, 'go', markersize=12, 
                       markerfacecolor='green', markeredgecolor='darkgreen')
                ax.text(best_k, best_bic, f'  最佳: k={best_k}\n  BIC={best_bic:.1f}', 
                       va='center', fontsize=10, fontweight='bold', color='green')
            
            ax.set_xlabel('聚类数 (k)')
            ax.set_ylabel('BIC Score')
            ax.set_title('GMM BIC 评估')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(k_range)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config.output_dir, '聚类过程综合分析可视化.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"聚类过程分析图表已保存为：{save_path}")
        plt.show()
    
    def _generate_bmi_grouping_visualization(self):
        """生成BMI分组可视化"""
        logger.info("🎨 生成BMI分组结果可视化...")
        
        if self.optimal_clusters is None or len(self.qualified_data) == 0:
            logger.warning("没有聚类结果或数据为空")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('BMI分组结果可视化（无重叠区间）', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#A8E6CF']
        
        # 计算调整后的无重叠BMI区间
        bmi_ranges = self._get_adjusted_bmi_ranges()
        
        logger.info("修正后的无重叠BMI分组区间：")
        for group_id, bmi_min, bmi_max, size in bmi_ranges:
            logger.info(f"组{group_id}：[{bmi_min:.2f}, {bmi_max:.2f}]，样本数：{size}")
        
        # 1. BMI分布散点图
        ax1 = axes[0, 0]
        for cluster_id in range(self.optimal_clusters):
            cluster_data = self.qualified_data[self.qualified_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                ax1.scatter(cluster_data['孕妇BMI'], cluster_data['达标时间'], 
                           alpha=0.7, s=100, color=colors[cluster_id % len(colors)],
                           label=f'组{cluster_id + 1} (n={len(cluster_data)})')
        
        # 添加区间分界线
        for i, (_, bmi_min, bmi_max, _) in enumerate(bmi_ranges[:-1]):  # 除了最后一组
            next_min = bmi_ranges[i+1][1]
            boundary = (bmi_max + next_min) / 2
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(boundary, ax1.get_ylim()[1]*0.9, f'边界{i+1}-{i+2}', 
                    rotation=90, ha='center', va='top', color='red', fontweight='bold')
        
        ax1.set_xlabel('BMI')
        ax1.set_ylabel('达标时间 (周)')
        ax1.set_title('BMI与达标时间分布（含分组边界）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. BMI区间柱状图
        ax2 = axes[0, 1]
        if bmi_ranges:
            group_ids = [f'组{r[0]}' for r in bmi_ranges]
            group_sizes = [r[3] for r in bmi_ranges]
            
            bars = ax2.bar(group_ids, group_sizes, color=colors[:len(group_ids)], alpha=0.8)
            ax2.set_ylabel('样本数量')
            ax2.set_title('各组样本数量分布')
            ax2.grid(True, alpha=0.3)
            
            for bar, size in zip(bars, group_sizes):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        str(size), ha='center', va='bottom', fontweight='bold')
        
        # 3. BMI无重叠区间范围图
        ax3 = axes[1, 0]
        if bmi_ranges:
            y_pos = np.arange(len(bmi_ranges))
            bmi_mins = [r[1] for r in bmi_ranges]
            bmi_maxs = [r[2] for r in bmi_ranges]
            ranges = [bmi_maxs[i] - bmi_mins[i] for i in range(len(bmi_ranges))]
            
            bars = ax3.barh(y_pos, ranges, left=bmi_mins, 
                           color=colors[:len(bmi_ranges)], alpha=0.7)
            
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f'组{r[0]}' for r in bmi_ranges])
            ax3.set_xlabel('BMI')
            ax3.set_title('各组BMI无重叠区间范围')
            ax3.grid(True, alpha=0.3)
            
            # 添加区间标注
            for i, (group_id, bmi_min, bmi_max, size) in enumerate(bmi_ranges):
                ax3.text(bmi_min + (bmi_max - bmi_min) / 2, i,
                        f'[{bmi_min:.1f}, {bmi_max:.1f}]',
                        ha='center', va='center', fontweight='bold', color='white')
                
                # 验证无重叠
                if i < len(bmi_ranges) - 1:
                    next_min = bmi_ranges[i+1][1]
                    if bmi_max >= next_min:
                        ax3.text(bmi_max, i, '⚠️重叠', ha='left', va='center', 
                                color='red', fontweight='bold')
        
        # 4. 各组统计表（包含调整信息）
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        if bmi_ranges:
            stats_data = []
            for i, (group_id, adjusted_min, adjusted_max, size) in enumerate(bmi_ranges):
                cluster_data = self.qualified_data[self.qualified_data['cluster'] == i]
                if len(cluster_data) > 0:
                    original_min = cluster_data["孕妇BMI"].min()
                    original_max = cluster_data["孕妇BMI"].max()
                    
                    # 检查是否有调整
                    adjustment_note = ""
                    if abs(adjusted_min - original_min) > 0.01 or abs(adjusted_max - original_max) > 0.01:
                        adjustment_note = " *调整"
                    
                    stats_data.append([
                        f'组{group_id}{adjustment_note}',
                        f'{size}',
                        f'[{adjusted_min:.1f}, {adjusted_max:.1f}]',
                        f'{cluster_data["孕妇BMI"].mean():.1f}±{cluster_data["孕妇BMI"].std():.1f}',
                        f'{cluster_data["达标时间"].mean():.1f}±{cluster_data["达标时间"].std():.1f}'
                    ])
            
            if stats_data:
                columns = ['组别', '样本数', '调整后区间', 'BMI均值±标准差', '达标时间均值±标准差']
                table = ax4.table(cellText=stats_data,
                                colLabels=columns,
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.8)
                
                # 设置表格样式
                for i in range(len(columns)):
                    table[(0, i)].set_facecolor('#4ECDC4')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                # 标记调整的行
                for i, row_data in enumerate(stats_data):
                    if '*调整' in row_data[0]:
                        for j in range(len(columns)):
                            table[(i+1, j)].set_facecolor('#FFE5E5')  # 浅红色背景
        
        ax4.set_title('各组统计信息（*表示区间已调整）', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config.output_dir, 'BMI分组结果可视化.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"BMI分组结果图表已保存为：{save_path}")
        plt.show()
    
    def _get_adjusted_bmi_ranges(self) -> List[tuple]:
        """获取调整后的无重叠BMI区间"""
        bmi_ranges = []
        
        # 收集原始聚类的BMI信息
        cluster_info = []
        for cluster_id in range(self.optimal_clusters):
            cluster_data = self.qualified_data[self.qualified_data['cluster'] == cluster_id]
            if len(cluster_data) > 0:
                cluster_bmi_data = cluster_data['孕妇BMI']
                cluster_bmi_mean = cluster_bmi_data.mean()
                cluster_bmi_min = cluster_bmi_data.min()
                cluster_bmi_max = cluster_bmi_data.max()
                cluster_info.append((cluster_id, cluster_bmi_mean, cluster_bmi_min, cluster_bmi_max, len(cluster_data)))
        
        # 按BMI均值排序
        cluster_info.sort(key=lambda x: x[1])
        
        # 调整重叠区间
        adjusted_info = self._adjust_overlapping_bmi_ranges(cluster_info)
        
        # 转换为返回格式
        for cluster_id, mean_bmi, min_bmi, max_bmi, size in adjusted_info:
            bmi_ranges.append((cluster_id + 1, min_bmi, max_bmi, size))
        
        return bmi_ranges
    
    def advanced_optimization(self) -> List[Dict[str, Any]]:
        """高级时点优化"""
        logger.info("🎯 开始高级时点优化...")
        
        results = []
        optimal_timings = []
        
        with ProcessPoolExecutor(max_workers=min(self.config.n_jobs, self.optimal_clusters)) as executor:
            futures = []
            
            for cluster_id in range(self.optimal_clusters):
                group_data = self.qualified_data[self.qualified_data['cluster'] == cluster_id]
                
                if len(group_data) < self.config.min_cluster_size:
                    continue
                
                future = executor.submit(self._optimize_single_group_advanced, cluster_id, group_data)
                futures.append((cluster_id, future))
            
            for cluster_id, future in tqdm(futures, desc="优化各组"):
                try:
                    result = future.result()
                    results.append(result)
                    optimal_timings.append(result['optimal_timing'])
                except Exception as e:
                    logger.error(f"优化第 {cluster_id + 1} 组时出错: {e}")
        
        self.optimal_timings = optimal_timings
        self.final_results = results
        
        # 验证结果
        self._validate_optimization_results()
        
        return results
    
    def _optimize_single_group_advanced(self, cluster_id: int, group_data: pd.DataFrame) -> Dict[str, Any]:
        """高级单组优化（支持多进程）"""
        target_times = group_data['达标时间'].values
        success_ratios = group_data['达标比例'].values
        
        # 多种优化方法
        optimization_results = {}
        
        for method in self.config.optimization_methods:
            try:
                if method == 'L-BFGS-B':
                    optimal_timing = self._optimize_with_lbfgs(target_times, success_ratios)
                elif method == 'differential_evolution':
                    optimal_timing = self._optimize_with_differential_evolution(target_times, success_ratios)
                else:
                    optimal_timing = np.average(target_times, weights=success_ratios)
                
                optimization_results[method] = optimal_timing
            except Exception as e:
                logger.warning(f"优化方法 {method} 失败: {e}")
                optimization_results[method] = np.average(target_times, weights=success_ratios)
        
        # 选择最佳结果（使用投票机制）
        optimal_timing = np.median(list(optimization_results.values()))
        
        # 计算详细统计
        result = self._calculate_advanced_group_statistics(cluster_id, group_data, optimal_timing)
        result['optimization_methods'] = optimization_results
        
        return result
    
    def _optimize_with_lbfgs(self, target_times: np.ndarray, success_ratios: np.ndarray) -> float:
        """使用L-BFGS-B优化"""
        def objective(timing):
            return self._calculate_advanced_objective(timing[0], target_times, success_ratios)
        
        avg_target = np.average(target_times, weights=success_ratios)
        # 用户指定的搜索范围：10-30周
        bounds = [(10.0, 30.0)]
        
        # 确保初始值在边界内
        initial_guess = np.clip(avg_target, 10.0, 30.0)
        
        result = minimize(objective, [initial_guess], bounds=bounds, method='L-BFGS-B')
        return result.x[0] if result.success else initial_guess
    
    def _optimize_with_differential_evolution(self, target_times: np.ndarray, success_ratios: np.ndarray) -> float:
        """使用差分进化算法优化"""
        def objective(timing):
            return self._calculate_advanced_objective(timing[0], target_times, success_ratios)
        
        avg_target = np.average(target_times, weights=success_ratios)
        # 用户指定的搜索范围：10-30周
        bounds = [(10.0, 30.0)]
        
        # 确保回退值在边界内
        fallback_value = np.clip(avg_target, 10.0, 30.0)
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        return result.x[0] if result.success else fallback_value
    
    def _calculate_advanced_objective(self, timing: float, target_times: np.ndarray, success_ratios: np.ndarray) -> float:
        """
        计算高级目标函数
        基于用户要求的优化策略：
        - 目标函数：最小化[时间风险+失败风险]
        - 时间风险：12周内风险1.0，27周内风险2.0，27周后风险4.0  
        - 失败风险：(1-成功率)×3.0
        """
        
        # 1. 时间风险计算（分段函数）
        if timing <= 12:
            time_risk = 1.0
        elif timing <= 27:
            time_risk = 2.0
        else:
            time_risk = 4.0
        
        # 2. 成功率计算（基于达标时间）
        success_count = np.sum(target_times <= timing)
        total_count = len(target_times)
        success_rate = success_count / total_count if total_count > 0 else 0
        
        # 加权成功率（考虑达标比例）
        if np.sum(success_ratios) > 0:
            weights = success_ratios / np.sum(success_ratios)
            weighted_success_rate = np.sum((target_times <= timing) * weights)
        else:
            weighted_success_rate = success_rate
        
        # 3. 失败风险计算
        failure_risk = (1 - weighted_success_rate) * 3.0
        
        # 4. 总风险（时间风险 + 失败风险）
        total_risk = time_risk + failure_risk
        
        return total_risk
    
    def _calculate_adaptive_weights(self, target_times: np.ndarray, success_ratios: np.ndarray) -> Dict[str, float]:
        """计算自适应权重"""
        # 基于数据特征调整权重
        time_variance = np.var(target_times)
        ratio_variance = np.var(success_ratios)
        
        # 基础权重
        base_weights = {
            'early': self.config.early_risk_weight,
            'success': self.config.success_rate_weight,
            'weighted': self.config.weighted_risk_weight,
            'error': self.config.error_risk_weight,
            'delay': self.config.delay_risk_weight,
            'consistency': self.config.consistency_risk_weight,
            'distribution': 0.05  # 新增的分布风险权重
        }
        
        # 自适应调整
        if time_variance > 2.0:  # 时间分散度高
            base_weights['error'] *= 1.5
            base_weights['distribution'] *= 1.5
        
        if ratio_variance > 0.1:  # 成功率分散度高
            base_weights['consistency'] *= 2.0
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        return {k: v / total_weight for k, v in base_weights.items()}
    
    def _calculate_distribution_risk(self, timing: float, target_times: np.ndarray, success_ratios: np.ndarray) -> float:
        """计算分布风险"""
        try:
            # 使用核密度估计
            kde = gaussian_kde(target_times, weights=success_ratios)
            
            # 在优化时点处的密度
            timing_density = kde(timing)[0]
            
            # 最大密度点
            time_range = np.linspace(target_times.min(), target_times.max(), 100)
            densities = kde(time_range)
            max_density = np.max(densities)
            
            # 分布风险：偏离密度最大点的程度
            distribution_risk = 1 - (timing_density / max_density) if max_density > 0 else 0
            
            return max(0, min(1, distribution_risk))
            
        except Exception:
            return 0
    
    def _calculate_advanced_group_statistics(self, cluster_id: int, group_data: pd.DataFrame, optimal_timing: float) -> Dict[str, Any]:
        """计算高级组统计信息"""
        # 基础统计
        bmi_stats = {
            'bmi_min': group_data['BMI'].min(),
            'bmi_max': group_data['BMI'].max(),
            'bmi_mean': group_data['BMI'].mean(),
            'bmi_std': group_data['BMI'].std()
        }
        
        # 其他特征统计
        feature_stats = {}
        for col, stat_name in [('孕妇年龄', 'age'), ('孕妇身高', 'height'), ('孕妇体重', 'weight')]:
            if col in group_data.columns:
                feature_stats.update({
                    f'{stat_name}_mean': group_data[col].mean(),
                    f'{stat_name}_std': group_data[col].std()
                })
        
        # 性能指标
        target_times = group_data['达标时间'].values
        success_ratios = group_data['达标比例'].values
        
        performance_metrics = self._calculate_advanced_performance_metrics(target_times, success_ratios, optimal_timing)
        
        # 风险评估
        risk_assessment = self._calculate_risk_assessment(target_times, success_ratios, optimal_timing)
        
        return {
            'group': cluster_id + 1,
            'bmi_range': f"[{bmi_stats['bmi_min']:.1f}, {bmi_stats['bmi_max']:.1f}]",
            'size': len(group_data),
            'optimal_timing': optimal_timing,
            **bmi_stats,
            **feature_stats,
            **performance_metrics,
            **risk_assessment
        }
    
    def _calculate_advanced_performance_metrics(self, target_times: np.ndarray, success_ratios: np.ndarray, optimal_timing: float) -> Dict[str, float]:
        """计算高级性能指标"""
        metrics = {}
        
        # 基础指标
        metrics['avg_target_time'] = np.average(target_times, weights=success_ratios)
        metrics['std_target_time'] = np.std(target_times)
        metrics['avg_success_ratio'] = np.mean(success_ratios)
        metrics['std_success_ratio'] = np.std(success_ratios)
        
        # 成功率指标
        metrics['success_rate'] = np.mean(target_times <= optimal_timing)
        metrics['weighted_success_rate'] = np.average(target_times <= optimal_timing, weights=success_ratios)
        
        # 鲁棒性指标
        metrics['robust_success_rate'] = self._calculate_robust_success_rate_advanced(target_times, success_ratios, optimal_timing)
        
        # 风险指标
        metrics['early_risk'] = np.mean(target_times > optimal_timing)
        metrics['error_impact'] = metrics['weighted_success_rate'] - metrics['robust_success_rate']
        
        # 置信区间
        metrics['success_rate_ci_lower'], metrics['success_rate_ci_upper'] = self._calculate_confidence_interval(target_times, optimal_timing)
        
        return metrics
    
    def _calculate_robust_success_rate_advanced(self, target_times: np.ndarray, success_ratios: np.ndarray, detection_time: float) -> float:
        """计算高级鲁棒成功率"""
        # 使用bootstrap采样提高估计准确性
        bootstrap_results = []
        
        np.random.seed(42)
        for _ in range(self.config.bootstrap_samples):
            # 重采样
            indices = np.random.choice(len(target_times), size=len(target_times), replace=True)
            bootstrap_target_times = target_times[indices]
            bootstrap_success_ratios = success_ratios[indices]
            
            # 添加检测误差
            actual_detection_time = np.random.normal(detection_time, self.config.detection_error_std)
            simulated_target_times = np.random.normal(bootstrap_target_times, 0.15)
            
            # 计算成功率
            individual_success = simulated_target_times <= actual_detection_time
            weighted_success_rate = np.average(individual_success, weights=bootstrap_success_ratios)
            bootstrap_results.append(weighted_success_rate)
        
        return np.mean(bootstrap_results)
    
    def _calculate_confidence_interval(self, target_times: np.ndarray, optimal_timing: float) -> Tuple[float, float]:
        """计算置信区间"""
        # 使用bootstrap估计置信区间
        bootstrap_success_rates = []
        
        np.random.seed(42)
        for _ in range(self.config.bootstrap_samples):
            indices = np.random.choice(len(target_times), size=len(target_times), replace=True)
            bootstrap_target_times = target_times[indices]
            success_rate = np.mean(bootstrap_target_times <= optimal_timing)
            bootstrap_success_rates.append(success_rate)
        
        return np.percentile(bootstrap_success_rates, [2.5, 97.5])
    
    def _calculate_risk_assessment(self, target_times: np.ndarray, success_ratios: np.ndarray, optimal_timing: float) -> Dict[str, float]:
        """计算风险评估"""
        risk_assessment = {}
        
        # 时间偏差风险
        avg_target_time = np.average(target_times, weights=success_ratios)
        risk_assessment['timing_deviation_risk'] = abs(optimal_timing - avg_target_time) / avg_target_time
        
        # 分布风险
        risk_assessment['distribution_risk'] = self._calculate_distribution_risk(optimal_timing, target_times, success_ratios)
        
        # 综合风险评分
        early_risk = np.mean(target_times > optimal_timing)
        weighted_success_rate = np.average(target_times <= optimal_timing, weights=success_ratios)
        robust_success_rate = self._calculate_robust_success_rate_advanced(target_times, success_ratios, optimal_timing)
        
        risk_assessment['comprehensive_risk'] = (
            early_risk * 0.3 +
            (1 - weighted_success_rate) * 0.3 +
            (weighted_success_rate - robust_success_rate) * 0.2 +
            risk_assessment['timing_deviation_risk'] * 0.1 +
            risk_assessment['distribution_risk'] * 0.1
        )
        
        return risk_assessment
    
    def _validate_optimization_results(self):
        """验证优化结果"""
        logger.info("🔍 验证优化结果...")
        
        validation_results = {}
        
        # 交叉验证
        validation_results['cross_validation'] = self._perform_cross_validation()
        
        # 敏感性分析
        validation_results['sensitivity_analysis'] = self._perform_sensitivity_analysis()
        
        # 稳定性测试
        validation_results['stability_test'] = self._perform_stability_test()
        
        self.validation_results = validation_results
        
        logger.info("验证完成")
    
    def _perform_cross_validation(self) -> Dict[str, Any]:
        """执行交叉验证"""
        # 简化的交叉验证实现
        cv_results = {'mean_error': 0, 'std_error': 0, 'consistency_score': 0}
        
        try:
            # 计算预测误差
            errors = []
            for result in self.final_results:
                predicted = result['optimal_timing']
                actual = result['avg_target_time']
                error = abs(predicted - actual) / actual
                errors.append(error)
            
            cv_results['mean_error'] = np.mean(errors)
            cv_results['std_error'] = np.std(errors)
            cv_results['consistency_score'] = 1 / (1 + cv_results['std_error'])  # 一致性评分
            
        except Exception as e:
            logger.warning(f"交叉验证失败: {e}")
        
        return cv_results
    
    def _perform_sensitivity_analysis(self) -> Dict[str, Any]:
        """执行敏感性分析"""
        sensitivity_results = {}
        
        # 参数敏感性
        parameter_variations = [0.8, 0.9, 1.1, 1.2]  # 参数变化范围
        
        for param_name in ['detection_error_std', 'n_simulations']:
            param_sensitivity = []
            original_value = getattr(self.config, param_name)
            
            for variation in parameter_variations:
                # 临时修改参数
                setattr(self.config, param_name, original_value * variation)
                
                # 重新计算部分结果（简化）
                avg_change = 0
                for result in self.final_results:
                    # 简化的敏感性测试
                    target_times = np.array([result['avg_target_time']])
                    success_ratios = np.array([result['avg_success_ratio']])
                    new_timing = self._optimize_with_lbfgs(target_times, success_ratios)
                    change = abs(new_timing - result['optimal_timing']) / result['optimal_timing']
                    avg_change += change
                
                param_sensitivity.append(avg_change / len(self.final_results))
                
                # 恢复原参数
                setattr(self.config, param_name, original_value)
            
            sensitivity_results[param_name] = {
                'variations': parameter_variations,
                'sensitivity': param_sensitivity,
                'max_sensitivity': max(param_sensitivity)
            }
        
        return sensitivity_results
    
    def _perform_stability_test(self) -> Dict[str, Any]:
        """执行稳定性测试"""
        stability_results = {}
        
        # 多次运行测试
        n_runs = 5
        timing_variations = []
        
        for run in range(n_runs):
            run_timings = []
            
            for cluster_id in range(self.optimal_clusters):
                group_data = self.qualified_data[self.qualified_data['cluster'] == cluster_id]
                
                if len(group_data) > 0:
                    target_times = group_data['达标时间'].values
                    success_ratios = group_data['达标比例'].values
                    
                    # 添加随机扰动
                    np.random.seed(42 + run)
                    perturbed_target_times = target_times + np.random.normal(0, 0.1, len(target_times))
                    
                    # 重新优化
                    timing = self._optimize_with_lbfgs(perturbed_target_times, success_ratios)
                    run_timings.append(timing)
            
            timing_variations.append(run_timings)
        
        # 计算稳定性指标
        timing_variations = np.array(timing_variations)
        stability_results['timing_std'] = np.std(timing_variations, axis=0).tolist()
        stability_results['avg_stability'] = np.mean(np.std(timing_variations, axis=0))
        stability_results['stability_score'] = 1 / (1 + stability_results['avg_stability'])
        
        return stability_results
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        logger.info("📊 生成综合分析报告...")
        
        if not self.final_results:
            logger.warning("没有结果可生成报告")
            return
        
        report = []
        report.append("="*80)
        report.append("高级多因素NIPT时点优化分析报告")
        report.append("="*80)
        report.append("")
        
        # 数据概况
        report.append("📊 数据概况")
        report.append("-" * 40)
        report.append(f"总样本数: {len(self.qualified_data)}")
        report.append(f"最优分组数: {self.optimal_clusters}")
        report.append(f"使用聚类方法: {', '.join(self.config.clustering_methods)}")
        report.append(f"使用优化算法: {', '.join(self.config.optimization_methods)}")
        report.append("")
        
        # 特征重要性
        if self.feature_importance is not None:
            report.append("🔍 特征重要性分析")
            report.append("-" * 40)
            for _, row in self.feature_importance.head(5).iterrows():
                report.append(f"  {row['feature']}: {row['combined_importance']:.4f}")
            report.append("")
        
        # 聚类分析结果
        if self.clustering_results:
            report.append("🎯 聚类分析结果")
            report.append("-" * 40)
            for method, result in self.clustering_results.items():
                if 'best_silhouette_score' in result and result['best_silhouette_score'] > 0:
                    method_display = method.replace('standard_', '').replace('robust_', '').upper()
                    if 'best_bic' in result:  # GMM结果
                        report.append(f"  {method_display}: k={result.get('best_k', 'N/A')}, 轮廓系数={result['best_silhouette_score']:.4f}, BIC={result['best_bic']:.1f}")
                    else:
                        report.append(f"  {method_display}: k={result.get('best_k', 'N/A')}, 轮廓系数={result['best_silhouette_score']:.4f}")
            report.append("")
        
        # 优化结果（使用调整后的无重叠区间）
        report.append("🎯 各组优化结果（无重叠BMI区间）")
        report.append("-" * 40)
        
        # 获取调整后的BMI区间
        adjusted_bmi_ranges = self._get_adjusted_bmi_ranges()
        
        for i, result in enumerate(self.final_results):
            # 使用调整后的BMI区间
            adjusted_range = "[N/A, N/A]"
            if i < len(adjusted_bmi_ranges):
                _, adj_min, adj_max, _ = adjusted_bmi_ranges[i]
                adjusted_range = f"[{adj_min:.1f}, {adj_max:.1f}]"
                
                # 检查是否有调整
                original_range = result['bmi_range']
                is_adjusted = adjusted_range != original_range
                adjustment_note = " *已调整" if is_adjusted else ""
            else:
                adjustment_note = ""
            
            report.append(f"\n组 {result['group']}: BMI {adjusted_range}{adjustment_note}")
            if adjustment_note:
                report.append(f"  原始区间: {result['bmi_range']} → 调整区间: {adjusted_range}")
            
            report.append(f"  样本数量: {result['size']}")
            report.append(f"  BMI统计: {result['bmi_mean']:.1f} ± {result.get('bmi_std', 0):.1f}")
            
            # 添加其他特征统计
            for key in ['age_mean', 'height_mean', 'weight_mean']:
                if key in result:
                    feature_name = key.replace('_mean', '').replace('age', '年龄').replace('height', '身高').replace('weight', '体重')
                    unit = ' 岁' if 'age' in key else (' cm' if 'height' in key else ' kg')
                    std_key = key.replace('_mean', '_std')
                    std_val = result.get(std_key, 0)
                    report.append(f"  {feature_name}: {result[key]:.1f} ± {std_val:.1f}{unit}")
            
            report.append(f"  🎯 最佳检测时点: {result['optimal_timing']:.1f} 周")
            report.append(f"  平均达标时间: {result['avg_target_time']:.1f} ± {result['std_target_time']:.1f} 周")
            report.append(f"  理论成功率: {result['success_rate']:.3f} ({result['success_rate']*100:.1f}%)")
            report.append(f"  加权成功率: {result['weighted_success_rate']:.3f} ({result['weighted_success_rate']*100:.1f}%)")
            report.append(f"  鲁棒成功率: {result['robust_success_rate']:.3f} ({result['robust_success_rate']*100:.1f}%)")
            
            # 置信区间
            if 'success_rate_ci_lower' in result and 'success_rate_ci_upper' in result:
                report.append(f"  95%置信区间: [{result['success_rate_ci_lower']:.3f}, {result['success_rate_ci_upper']:.3f}]")
            
            # 风险评估
            if 'comprehensive_risk' in result:
                report.append(f"  综合风险评分: {result['comprehensive_risk']:.3f}")
        
        # 验证结果
        if self.validation_results:
            report.append("\n🔍 验证结果")
            report.append("-" * 40)
            
            if 'cross_validation' in self.validation_results:
                cv = self.validation_results['cross_validation']
                report.append(f"交叉验证: 平均误差={cv['mean_error']:.3f}, 一致性评分={cv['consistency_score']:.3f}")
            
            if 'stability_test' in self.validation_results:
                st = self.validation_results['stability_test']
                report.append(f"稳定性测试: 稳定性评分={st['stability_score']:.3f}")
        
        # BMI区间无重叠验证
        report.append(f"\n🔍 BMI区间无重叠验证")
        report.append("-" * 40)
        overlap_validation = self._validate_no_overlap()
        if overlap_validation['has_overlap']:
            report.append("❌ 检测到区间重叠")
            for overlap in overlap_validation['overlaps']:
                report.append(f"  组{overlap['group1']} 和 组{overlap['group2']}: 重叠区域 {overlap['overlap_range']}")
        else:
            report.append("✅ 所有BMI区间均无重叠")
            for i, (group_id, min_bmi, max_bmi, size) in enumerate(adjusted_bmi_ranges):
                if i < len(adjusted_bmi_ranges) - 1:
                    next_group_id, next_min, _, _ = adjusted_bmi_ranges[i + 1]
                    gap = next_min - max_bmi
                    report.append(f"  组{group_id} 和 组{next_group_id} 间隔: {gap:.2f}")
        
        # 整体效果
        total_weight = sum(r['size'] for r in self.final_results)
        overall_success = sum(r['robust_success_rate'] * r['size'] for r in self.final_results) / total_weight
        overall_risk = sum(r.get('comprehensive_risk', 0) * r['size'] for r in self.final_results) / total_weight
        
        report.append(f"\n📈 整体效果评估")
        report.append("-" * 40)
        report.append(f"整体鲁棒成功率: {overall_success:.3f} ({overall_success*100:.1f}%)")
        report.append(f"整体综合风险: {overall_risk:.3f}")
        
        # 保存报告
        report_text = "\n".join(report)
        report_path = os.path.join(self.config.output_dir, '高级优化分析报告.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(report_text)
        logger.info(f"综合报告已保存到: {report_path}")
    
    def _validate_no_overlap(self) -> Dict[str, Any]:
        """验证BMI区间无重叠"""
        adjusted_bmi_ranges = self._get_adjusted_bmi_ranges()
        validation_result = {
            'has_overlap': False,
            'overlaps': [],
            'gaps': []
        }
        
        for i in range(len(adjusted_bmi_ranges) - 1):
            current_group_id, current_min, current_max, current_size = adjusted_bmi_ranges[i]
            next_group_id, next_min, next_max, next_size = adjusted_bmi_ranges[i + 1]
            
            if current_max > next_min:
                # 发现重叠
                validation_result['has_overlap'] = True
                overlap_start = max(current_min, next_min)
                overlap_end = min(current_max, next_max)
                validation_result['overlaps'].append({
                    'group1': current_group_id,
                    'group2': next_group_id,
                    'overlap_range': f'[{overlap_start:.2f}, {overlap_end:.2f}]'
                })
            else:
                # 记录间隔
                gap = next_min - current_max
                validation_result['gaps'].append({
                    'group1': current_group_id,
                    'group2': next_group_id,
                    'gap': gap
                })
        
        return validation_result
    
    def save_all_results(self):
        """保存所有结果"""
        logger.info("💾 保存所有分析结果...")
        
        # 保存主要结果
        if self.final_results:
            results_df = pd.DataFrame(self.final_results)
            results_df.to_csv(os.path.join(self.config.output_dir, '高级优化结果.csv'), 
                            index=False, encoding='utf-8-sig')
        
        # 保存特征重要性
        if self.feature_importance is not None:
            self.feature_importance.to_csv(os.path.join(self.config.output_dir, '高级特征重要性.csv'), 
                                         index=False, encoding='utf-8-sig')
        
        # 保存聚类结果
        if self.clustering_results:
            clustering_summary = []
            for method, result in self.clustering_results.items():
                if 'best_k' in result:
                    clustering_summary.append({
                        'method': method,
                        'best_k': result['best_k'],
                        'silhouette_score': result.get('best_silhouette_score', -1)
                    })
            
            if clustering_summary:
                clustering_df = pd.DataFrame(clustering_summary)
                clustering_df.to_csv(os.path.join(self.config.output_dir, '聚类分析结果.csv'), 
                                   index=False, encoding='utf-8-sig')
        
        # 保存验证结果
        if self.validation_results:
            import json
            validation_path = os.path.join(self.config.output_dir, '验证结果.json')
            with open(validation_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
        
        # 保存详细数据
        if self.qualified_data is not None:
            self.qualified_data.to_csv(os.path.join(self.config.output_dir, '高级处理数据.csv'), 
                                     index=False, encoding='utf-8-sig')
        
        logger.info(f"所有结果已保存到 {self.config.output_dir} 目录")
    
    def comprehensive_sensitivity_analysis(self):
        """综合敏感性分析"""
        logger.info("🔍 开始综合敏感性分析...")
        
        if not self.final_results or self.optimal_clusters is None:
            logger.warning("没有优化结果，跳过敏感性分析")
            return
        
        sensitivity_results = {}
        
        # 1. 参数敏感性分析
        sensitivity_results['parameter_sensitivity'] = self._parameter_sensitivity_analysis()
        
        # 2. 特征敏感性分析
        sensitivity_results['feature_sensitivity'] = self._feature_sensitivity_analysis()
        
        # 3. 数据扰动敏感性分析
        sensitivity_results['data_perturbation'] = self._data_perturbation_sensitivity()
        
        # 4. 目标函数敏感性分析
        sensitivity_results['objective_function'] = self._objective_function_sensitivity()
        
        # 保存敏感性分析结果
        self.sensitivity_results = sensitivity_results
        
        # 可视化敏感性分析结果
        self._visualize_sensitivity_analysis()
        
        logger.info("敏感性分析完成")
    
    def _parameter_sensitivity_analysis(self) -> Dict[str, Any]:
        """参数敏感性分析"""
        logger.info("进行参数敏感性分析...")
        
        # 测试参数及其变化范围
        test_parameters = {
            'detection_error_std': [0.3, 0.4, 0.5, 0.6, 0.7],
            'n_simulations': [250, 500, 750, 1000, 1250],
            'bootstrap_samples': [25, 50, 75, 100, 125]
        }
        
        parameter_sensitivity = {}
        original_values = {}
        
        for param_name, test_values in test_parameters.items():
            # 保存原始值
            original_values[param_name] = getattr(self.config, param_name)
            
            sensitivity_data = {
                'parameter_values': test_values,
                'timing_changes': [],
                'success_rate_changes': [],
                'risk_changes': []
            }
            
            for test_value in test_values:
                # 临时设置参数值
                setattr(self.config, param_name, test_value)
                
                # 重新计算优化结果
                avg_timing_change = 0
                avg_success_change = 0
                avg_risk_change = 0
                
                for i, result in enumerate(self.final_results):
                    cluster_data = self.qualified_data[self.qualified_data['cluster'] == i]
                    if len(cluster_data) > 0:
                        target_times = cluster_data['达标时间'].values
                        success_ratios = cluster_data['达标比例'].values
                        
                        # 重新优化
                        new_timing = self._optimize_with_lbfgs(target_times, success_ratios)
                        
                        # 计算变化
                        timing_change = abs(new_timing - result['optimal_timing']) / result['optimal_timing']
                        
                        # 重新计算成功率
                        new_success_rate = np.mean(target_times <= new_timing)
                        success_change = abs(new_success_rate - result['success_rate']) / result['success_rate']
                        
                        # 计算风险变化
                        new_risk = self._calculate_advanced_objective(new_timing, target_times, success_ratios)
                        original_risk = self._calculate_advanced_objective(result['optimal_timing'], target_times, success_ratios)
                        risk_change = abs(new_risk - original_risk) / original_risk if original_risk > 0 else 0
                        
                        avg_timing_change += timing_change
                        avg_success_change += success_change
                        avg_risk_change += risk_change
                
                n_results = len(self.final_results)
                sensitivity_data['timing_changes'].append(avg_timing_change / n_results)
                sensitivity_data['success_rate_changes'].append(avg_success_change / n_results)
                sensitivity_data['risk_changes'].append(avg_risk_change / n_results)
            
            parameter_sensitivity[param_name] = sensitivity_data
            
            # 恢复原始值
            setattr(self.config, param_name, original_values[param_name])
        
        return parameter_sensitivity
    
    def _feature_sensitivity_analysis(self) -> Dict[str, Any]:
        """特征敏感性分析"""
        logger.info("进行特征敏感性分析...")
        
        if self.feature_importance is None:
            return {}
        
        feature_sensitivity = {}
        
        # 获取重要特征
        top_features = self.feature_importance.head(5)['feature'].tolist()
        
        for feature in top_features:
            if feature in self.qualified_data.columns and feature != '达标时间':
                sensitivity_data = {
                    'feature_name': feature,
                    'noise_levels': [0.05, 0.1, 0.15, 0.2, 0.25],
                    'clustering_stability': [],
                    'optimization_stability': []
                }
                
                original_values = self.qualified_data[feature].copy()
                
                for noise_level in sensitivity_data['noise_levels']:
                    # 添加噪声
                    noise = np.random.normal(0, noise_level * original_values.std(), len(original_values))
                    self.qualified_data[feature] = original_values + noise
                    
                    try:
                        # 重新聚类
                        important_features = [f for f in [feature, 'BMI', 'GC_content'] if f in self.qualified_data.columns]
                        if len(important_features) > 0:
                            cluster_data = self.qualified_data[important_features].values
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(cluster_data)
                            
                            kmeans = KMeans(n_clusters=self.optimal_clusters, random_state=42)
                            new_labels = kmeans.fit_predict(scaled_data)
                            
                            # 计算聚类稳定性（调整兰德指数）
                            original_labels = self.qualified_data['cluster'].values
                            from sklearn.metrics import adjusted_rand_score
                            clustering_stability = adjusted_rand_score(original_labels, new_labels)
                            sensitivity_data['clustering_stability'].append(clustering_stability)
                            
                            # 计算优化稳定性
                            optimization_stability = 0
                            for cluster_id in range(self.optimal_clusters):
                                cluster_mask = new_labels == cluster_id
                                if np.sum(cluster_mask) > 0:
                                    cluster_target_times = self.qualified_data[cluster_mask]['达标时间'].values
                                    cluster_success_ratios = self.qualified_data[cluster_mask]['达标比例'].values
                                    
                                    if len(cluster_target_times) > 0:
                                        new_timing = self._optimize_with_lbfgs(cluster_target_times, cluster_success_ratios)
                                        original_timing = self.final_results[min(cluster_id, len(self.final_results)-1)]['optimal_timing']
                                        stability = 1 - abs(new_timing - original_timing) / original_timing
                                        optimization_stability += stability
                            
                            optimization_stability /= self.optimal_clusters
                            sensitivity_data['optimization_stability'].append(optimization_stability)
                        else:
                            sensitivity_data['clustering_stability'].append(0)
                            sensitivity_data['optimization_stability'].append(0)
                            
                    except Exception as e:
                        logger.debug(f"特征敏感性分析出错: {e}")
                        sensitivity_data['clustering_stability'].append(0)
                        sensitivity_data['optimization_stability'].append(0)
                
                # 恢复原始值
                self.qualified_data[feature] = original_values
                
                feature_sensitivity[feature] = sensitivity_data
        
        return feature_sensitivity
    
    def _data_perturbation_sensitivity(self) -> Dict[str, Any]:
        """数据扰动敏感性分析"""
        logger.info("进行数据扰动敏感性分析...")
        
        perturbation_results = {
            'sample_size_sensitivity': [],
            'outlier_sensitivity': [],
            'missing_data_sensitivity': []
        }
        
        # 1. 样本量敏感性
        sample_ratios = [0.7, 0.8, 0.9, 1.0, 1.1]  # 最后一个是bootstrap
        for ratio in sample_ratios:
            if ratio <= 1.0:
                # 减少样本
                n_samples = int(len(self.qualified_data) * ratio)
                sample_indices = np.random.choice(len(self.qualified_data), n_samples, replace=False)
                sample_data = self.qualified_data.iloc[sample_indices]
            else:
                # Bootstrap采样
                n_samples = int(len(self.qualified_data) * ratio)
                sample_indices = np.random.choice(len(self.qualified_data), n_samples, replace=True)
                sample_data = self.qualified_data.iloc[sample_indices]
            
            # 重新计算优化结果
            stability_score = self._calculate_result_stability(sample_data)
            perturbation_results['sample_size_sensitivity'].append({
                'sample_ratio': ratio,
                'stability_score': stability_score
            })
        
        # 2. 异常值敏感性
        outlier_ratios = [0.0, 0.05, 0.1, 0.15, 0.2]
        for outlier_ratio in outlier_ratios:
            # 添加异常值
            data_with_outliers = self.qualified_data.copy()
            n_outliers = int(len(data_with_outliers) * outlier_ratio)
            
            if n_outliers > 0:
                outlier_indices = np.random.choice(len(data_with_outliers), n_outliers, replace=False)
                # 在关键特征中添加异常值
                key_features = ['BMI', '达标时间', '达标比例']
                for feature in key_features:
                    if feature in data_with_outliers.columns:
                        feature_std = data_with_outliers[feature].std()
                        outlier_values = np.random.normal(0, 3 * feature_std, n_outliers)
                        data_with_outliers.loc[data_with_outliers.index[outlier_indices], feature] += outlier_values
            
            stability_score = self._calculate_result_stability(data_with_outliers)
            perturbation_results['outlier_sensitivity'].append({
                'outlier_ratio': outlier_ratio,
                'stability_score': stability_score
            })
        
        return perturbation_results
    
    def _objective_function_sensitivity(self) -> Dict[str, Any]:
        """目标函数敏感性分析"""
        logger.info("进行目标函数敏感性分析...")
        
        # 测试不同的风险权重组合
        weight_variations = [
            {'time_weight': 0.8, 'failure_weight': 3.2},  # 更重视时间风险
            {'time_weight': 1.2, 'failure_weight': 2.8},  # 更重视失败风险
            {'time_weight': 0.5, 'failure_weight': 3.5},  # 极端权重1
            {'time_weight': 1.5, 'failure_weight': 2.5},  # 极端权重2
        ]
        
        objective_sensitivity = {
            'weight_variations': weight_variations,
            'timing_changes': [],
            'success_rate_changes': []
        }
        
        for weights in weight_variations:
            # 使用不同权重重新优化
            avg_timing_change = 0
            avg_success_change = 0
            
            for i, result in enumerate(self.final_results):
                cluster_data = self.qualified_data[self.qualified_data['cluster'] == i]
                if len(cluster_data) > 0:
                    target_times = cluster_data['达标时间'].values
                    success_ratios = cluster_data['达标比例'].values
                    
                    # 使用修改后的目标函数
                    def modified_objective(timing):
                        timing_val = timing[0]
                        
                        # 修改后的时间风险
                        if timing_val <= 12:
                            time_risk = weights['time_weight'] * 1.0
                        elif timing_val <= 27:
                            time_risk = weights['time_weight'] * 2.0
                        else:
                            time_risk = weights['time_weight'] * 4.0
                        
                        # 修改后的失败风险
                        success_count = np.sum(target_times <= timing_val)
                        total_count = len(target_times)
                        success_rate = success_count / total_count if total_count > 0 else 0
                        
                        if np.sum(success_ratios) > 0:
                            weights_norm = success_ratios / np.sum(success_ratios)
                            weighted_success_rate = np.sum((target_times <= timing_val) * weights_norm)
                        else:
                            weighted_success_rate = success_rate
                        
                        failure_risk = (1 - weighted_success_rate) * weights['failure_weight']
                        
                        return time_risk + failure_risk
                    
                    # 优化
                    bounds = [(10.0, 30.0)]
                    avg_target = np.average(target_times, weights=success_ratios)
                    initial_guess = np.clip(avg_target, 10.0, 30.0)
                    
                    result_opt = minimize(modified_objective, [initial_guess], bounds=bounds, method='L-BFGS-B')
                    new_timing = result_opt.x[0] if result_opt.success else initial_guess
                    
                    # 计算变化
                    timing_change = abs(new_timing - result['optimal_timing']) / result['optimal_timing']
                    
                    new_success_rate = np.mean(target_times <= new_timing)
                    success_change = abs(new_success_rate - result['success_rate']) / result['success_rate']
                    
                    avg_timing_change += timing_change
                    avg_success_change += success_change
            
            n_results = len(self.final_results)
            objective_sensitivity['timing_changes'].append(avg_timing_change / n_results)
            objective_sensitivity['success_rate_changes'].append(avg_success_change / n_results)
        
        return objective_sensitivity
    
    def _calculate_result_stability(self, sample_data: pd.DataFrame) -> float:
        """计算结果稳定性"""
        try:
            # 简化的稳定性计算
            if len(sample_data) < 10:
                return 0.0
            
            # 重新进行基本的聚类和优化
            key_features = ['BMI', 'GC_content', 'pregnancy_count', 'birth_count']
            available_features = [f for f in key_features if f in sample_data.columns]
            
            if len(available_features) < 2:
                return 0.0
            
            cluster_data = sample_data[available_features].values
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            kmeans = KMeans(n_clusters=min(self.optimal_clusters, len(sample_data)//5), random_state=42)
            labels = kmeans.fit_predict(scaled_data)
            
            # 计算平均优化结果的稳定性
            stability_scores = []
            for cluster_id in range(max(labels) + 1):
                cluster_mask = labels == cluster_id
                if np.sum(cluster_mask) > 2:
                    cluster_target_times = sample_data[cluster_mask]['达标时间'].values
                    cluster_success_ratios = sample_data[cluster_mask]['达标比例'].values
                    
                    if len(cluster_target_times) > 0:
                        new_timing = self._optimize_with_lbfgs(cluster_target_times, cluster_success_ratios)
                        
                        # 与原始结果比较
                        if cluster_id < len(self.final_results):
                            original_timing = self.final_results[cluster_id]['optimal_timing']
                            stability = 1 - abs(new_timing - original_timing) / original_timing
                            stability_scores.append(max(0, stability))
            
            return np.mean(stability_scores) if stability_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _visualize_sensitivity_analysis(self):
        """可视化敏感性分析结果"""
        logger.info("🎨 生成敏感性分析可视化...")
        
        if not hasattr(self, 'sensitivity_results'):
            return
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('敏感性分析综合结果', fontsize=18, fontweight='bold')
        
        # 1. 参数敏感性分析
        if 'parameter_sensitivity' in self.sensitivity_results:
            param_sens = self.sensitivity_results['parameter_sensitivity']
            
            row = 0
            for i, (param_name, data) in enumerate(param_sens.items()):
                if i >= 3:  # 最多显示3个参数
                    break
                
                ax = fig.add_subplot(gs[row, i])
                
                param_values = data['parameter_values']
                timing_changes = data['timing_changes']
                success_changes = data['success_rate_changes']
                
                ax2 = ax.twinx()
                
                line1 = ax.plot(param_values, timing_changes, 'o-', color='blue', 
                               label='时点变化', linewidth=2, markersize=6)
                line2 = ax2.plot(param_values, success_changes, 's-', color='red', 
                                label='成功率变化', linewidth=2, markersize=6)
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('时点变化率', color='blue')
                ax2.set_ylabel('成功率变化率', color='red')
                ax.set_title(f'{param_name} 敏感性')
                
                # 合并图例
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper left')
                
                ax.grid(True, alpha=0.3)
        
        # 2. 特征敏感性分析
        if 'feature_sensitivity' in self.sensitivity_results:
            feat_sens = self.sensitivity_results['feature_sensitivity']
            
            ax = fig.add_subplot(gs[1, :])
            
            features = list(feat_sens.keys())
            if features:
                x = np.arange(len(features))
                
                # 计算平均稳定性
                clustering_stability = []
                optimization_stability = []
                
                for feature in features:
                    data = feat_sens[feature]
                    clustering_stability.append(np.mean(data['clustering_stability']))
                    optimization_stability.append(np.mean(data['optimization_stability']))
                
                width = 0.35
                ax.bar(x - width/2, clustering_stability, width, label='聚类稳定性', alpha=0.8)
                ax.bar(x + width/2, optimization_stability, width, label='优化稳定性', alpha=0.8)
                
                ax.set_xlabel('特征')
                ax.set_ylabel('平均稳定性')
                ax.set_title('特征扰动敏感性分析')
                ax.set_xticks(x)
                ax.set_xticklabels(features, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 3. 数据扰动敏感性
        if 'data_perturbation' in self.sensitivity_results:
            pert_sens = self.sensitivity_results['data_perturbation']
            
            # 样本量敏感性
            ax = fig.add_subplot(gs[2, 0])
            sample_data = pert_sens['sample_size_sensitivity']
            if sample_data:
                ratios = [d['sample_ratio'] for d in sample_data]
                scores = [d['stability_score'] for d in sample_data]
                
                ax.plot(ratios, scores, 'o-', linewidth=2, markersize=8, color='green')
                ax.set_xlabel('样本量比例')
                ax.set_ylabel('稳定性评分')
                ax.set_title('样本量敏感性')
                ax.grid(True, alpha=0.3)
            
            # 异常值敏感性
            ax = fig.add_subplot(gs[2, 1])
            outlier_data = pert_sens['outlier_sensitivity']
            if outlier_data:
                ratios = [d['outlier_ratio'] for d in outlier_data]
                scores = [d['stability_score'] for d in outlier_data]
                
                ax.plot(ratios, scores, 'o-', linewidth=2, markersize=8, color='orange')
                ax.set_xlabel('异常值比例')
                ax.set_ylabel('稳定性评分')
                ax.set_title('异常值敏感性')
                ax.grid(True, alpha=0.3)
        
        # 4. 目标函数敏感性
        if 'objective_function' in self.sensitivity_results:
            obj_sens = self.sensitivity_results['objective_function']
            
            ax = fig.add_subplot(gs[2, 2])
            
            if obj_sens['timing_changes'] and obj_sens['success_rate_changes']:
                x = range(len(obj_sens['timing_changes']))
                width = 0.35
                
                ax.bar([i - width/2 for i in x], obj_sens['timing_changes'], 
                      width, label='时点变化', alpha=0.8)
                ax.bar([i + width/2 for i in x], obj_sens['success_rate_changes'], 
                      width, label='成功率变化', alpha=0.8)
                
                ax.set_xlabel('权重组合')
                ax.set_ylabel('变化率')
                ax.set_title('目标函数权重敏感性')
                ax.set_xticks(x)
                ax.set_xticklabels([f'组合{i+1}' for i in x])
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 5. 敏感性总结表
        ax = fig.add_subplot(gs[3, :])
        ax.axis('tight')
        ax.axis('off')
        
        # 创建敏感性总结
        summary_data = []
        if 'parameter_sensitivity' in self.sensitivity_results:
            for param_name, data in self.sensitivity_results['parameter_sensitivity'].items():
                max_change = max(max(data['timing_changes']), max(data['success_rate_changes']))
                sensitivity_level = "高" if max_change > 0.1 else ("中" if max_change > 0.05 else "低")
                summary_data.append([param_name, f"{max_change:.3f}", sensitivity_level])
        
        if summary_data:
            columns = ['参数/特征', '最大变化率', '敏感性等级']
            table = ax.table(cellText=summary_data,
                           colLabels=columns,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # 设置表格样式
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4ECDC4')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('敏感性分析总结', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.config.output_dir, '敏感性分析可视化.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"敏感性分析图表已保存为：{save_path}")
        plt.show()

def main():
    """
    主函数 - 优化版本
    
    数据保留优化措施：
    1. 男胎筛选：使用 >= 0 替代 > 0，包含边界值
    2. 孕周解析：失败时使用均值填充，避免数据丢失
    3. 患者数据处理：保留未达标记录用于分析
    4. 特征清洗：使用中位数/均值填充替代删除缺失值
    5. 聚类检查：更宽松的最小簇大小检查（min_size//2）
    6. 异常值处理：使用2.5倍IQR替代1.5倍IQR
    7. 配置参数：降低min_cluster_size，提高数据保留率
    """
    print("🔬 问题3: 多因素影响下的NIPT时点优化（高级优化版）")
    print("=" * 80)
    print("📈 数据保留优化措施：")
    print("   ✓ 宽松男胎筛选（包含边界值）")
    print("   ✓ 智能孕周填充（避免解析失败丢失）")
    print("   ✓ 保留未达标记录用于分析")
    print("   ✓ 缺失值填充替代删除")
    print("   ✓ 宽松聚类检查和异常值处理")
    print("🔧 BMI分组区间优化：")
    print("   ✓ 智能检测和调整重叠区间")
    print("   ✓ 确保各组BMI区间严格无重叠")
    print("   ✓ 自动生成边界分界线")
    print("   ✓ 可视化显示调整过程")
    print("=" * 80)
    
    # 显示分析阶段概览
    print("\n开始运行问题三综合深度分析..")
    print("\n分析阶段概览：")
    print("1. 数据预处理和特征工程")
    print("2. 特征重要性分析+可视化")
    print("3. 多因素聚类分组+可视化")
    print("4. 检测时点优化+可视化")
    print("5. 敏感性分析+可视化")
    print("6. 结果保存与综合汇总")
    print()
    
    start_time = time.time()
    
    try:
        # 高级配置（优化版，更宽松的参数设置以减少数据丢失）
        config = AdvancedOptimizationConfig(
            max_clusters=5,  # 适中的最大聚类数
            min_cluster_size=3,  # 进一步减小最小簇大小，降低数据丢失
            clustering_methods=['kmeans', 'agglomerative', 'gmm'],  # 保持多种聚类方法
            scaling_methods=['standard'],  # 使用StandardScaler进行Z-score标准化
            optimization_methods=['L-BFGS-B', 'differential_evolution'],
            n_optimization_trials=3,
            detection_error_std=0.5,
            n_simulations=400,  # 适当减少模拟次数以加快速度
            cross_validation_folds=3,  # 减少交叉验证折数以适应小数据集
            bootstrap_samples=30,  # 适当减少bootstrap样本数
            n_jobs=4,
            output_dir="问题三结果"
        )
        
        # 初始化优化器
        optimizer = AdvancedMultiFactorNIPTOptimization('附件.xlsx', config)
        
        # 阶段1：数据加载与预处理
        print("=" * 50)
        print("阶段1：数据预处理和特征工程")
        print("=" * 50)
        
        data = optimizer.load_and_preprocess_data()
        
        if len(data) == 0:
            logger.error("❌ 没有符合条件的数据")
            return
        
        # 显示详细的数据保留统计
        original_shape = optimizer.data.shape if optimizer.data is not None else (0, 0)
        male_count = len(optimizer.data[optimizer.data['Y染色体浓度'] >= 0]) if optimizer.data is not None else 0
        aggregated_shape = (len(optimizer.qualified_data), len(optimizer.qualified_data.columns)) if optimizer.qualified_data is not None else (0, 0)
        
        print(f"\n📊 数据保留优化统计报告:")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📈 原始数据：{original_shape[0]:,} 行 × {original_shape[1]} 列")
        print(f"🚹 男胎筛选（优化后）：{male_count:,} 行 (保留率: {male_count/original_shape[0]*100:.1f}%)")
        
        if hasattr(optimizer, 'qualified_data') and optimizer.qualified_data is not None:
            达标_records = len(optimizer.qualified_data[optimizer.qualified_data.get('数据类型', '') == '达标记录']) if '数据类型' in optimizer.qualified_data.columns else len(optimizer.qualified_data)
            未达标_records = len(optimizer.qualified_data[optimizer.qualified_data.get('数据类型', '') == '未达标记录']) if '数据类型' in optimizer.qualified_data.columns else 0
            异常_records = len(optimizer.qualified_data[optimizer.qualified_data.get('数据类型', '') == '异常记录']) if '数据类型' in optimizer.qualified_data.columns else 0
            
            print(f"📊 聚合后数据：{aggregated_shape[0]:,} 行 × {aggregated_shape[1]} 列")
            print(f"   ├─ ✅ 达标记录：{达标_records:,} 个")
            if 未达标_records > 0:
                print(f"   ├─ ⚠️  未达标记录（保留）：{未达标_records:,} 个")
            if 异常_records > 0:
                print(f"   └─ 🔄 异常记录（恢复）：{异常_records:,} 个")
            
            final_retention = aggregated_shape[0] / original_shape[0] * 100
            print(f"🎯 总体数据保留率：{final_retention:.1f}% ({aggregated_shape[0]:,}/{original_shape[0]:,})")
            
            if final_retention >= 80:
                print("   🟢 数据保留率优秀 (≥80%)")
            elif final_retention >= 60:
                print("   🟡 数据保留率良好 (≥60%)")
            elif final_retention >= 40:
                print("   🟠 数据保留率一般 (≥40%)")
            else:
                print("   🔴 数据保留率偏低 (<40%)")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # 根据数据量自动调整配置
        optimizer._auto_adjust_config_for_data_size(len(data))
        
        # 阶段2：特征重要性分析+可视化
        print("\n" + "=" * 50)
        print("阶段2：特征重要性分析+可视化")
        print("=" * 50)
        feature_importance = optimizer.advanced_feature_importance_analysis()
        
        # 阶段3：多因素聚类分组+可视化
        print("\n" + "=" * 50)
        print("阶段3：多因素聚类分组+可视化")
        print("=" * 50)
        optimal_k, cluster_labels = optimizer.advanced_clustering_analysis()
        
        # 阶段4：检测时点优化+可视化
        print("\n" + "=" * 50)
        print("阶段4：检测时点优化+可视化")
        print("=" * 50)
        results = optimizer.advanced_optimization()
        
        # 阶段5：敏感性分析+可视化
        print("\n" + "=" * 50)
        print("阶段5：敏感性分析+可视化")
        print("=" * 50)
        optimizer.comprehensive_sensitivity_analysis()
        
        # 阶段6：结果保存与综合汇总
        print("\n" + "=" * 50)
        print("阶段6：结果保存与综合汇总")
        print("=" * 50)
        
        # 生成综合报告
        optimizer.generate_comprehensive_report()
        
        # 保存所有结果
        optimizer.save_all_results()
        
        total_time = time.time() - start_time
        logger.info(f"✅ 高级多因素NIPT时点优化分析完成!")
        logger.info(f"📊 总耗时: {total_time:.2f}秒")
        logger.info(f"📁 结果保存在: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
