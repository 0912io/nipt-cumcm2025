import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
 
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
 
# 创建输出文件夹
output_dir = "问题四结果"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class NvTaiYiChangJianCe:
    def __init__(self, shu_ju_wen_jian):
        """初始化女胎异常检测器"""
        self.shu_ju_wen_jian = shu_ju_wen_jian
        self.shu_ju = None
        self.nv_tai_shu_ju = None
        self.te_zheng_shu_ju = None
        self.te_zheng_ju_zhen = None
        self.mu_biao_bian_liang = None
        self.yuan_shi_te_zheng = None  # 存储原始特征数据
        self.biao_zhun_hua_te_zheng = None    # 存储Z-score标准化后的数据
        self.mo_xing_zu = {}
        self.te_zheng_zhong_yao_xing = None
        self.zui_jia_mo_xing = None
        self.biao_zhun_hua_qi = StandardScaler()
         
    def jia_zai_yu_chu_li_shu_ju(self):
        """加载和预处理女胎数据"""
        print("📊 加载女胎数据...")
        self.shu_ju = pd.read_excel(self.shu_ju_wen_jian)
         
        # 筛选女胎数据（Y染色体浓度较低，通常<=0.08表示女胎）
        nv_tai_yan_ma = self.shu_ju['Y染色体浓度'] <= 0.08
        self.nv_tai_shu_ju = self.shu_ju[nv_tai_yan_ma].copy()
         
        print(f"女胎总数: {len(self.nv_tai_shu_ju)}")
         
        # 处理孕周格式
        def jie_xi_yun_zhou(yun_zhou_str):
            if pd.isna(yun_zhou_str):
                return None
            if isinstance(yun_zhou_str, (int, float)):
                return float(yun_zhou_str)
            try:
                yun_zhou_str = str(yun_zhou_str)
                if 'w' in yun_zhou_str:
                    bu_fen = yun_zhou_str.split('w')
                    zhou_shu = int(bu_fen[0])
                    if '+' in bu_fen[1]:
                        tian_shu = int(bu_fen[1].replace('+', ''))
                        return zhou_shu + tian_shu / 7
                    else:
                        return zhou_shu
                else:
                    return float(yun_zhou_str)
            except:
                return None
         
        # 转换孕周为数值
        self.nv_tai_shu_ju['检测孕周_数值'] = self.nv_tai_shu_ju['检测孕周'].apply(jie_xi_yun_zhou)
         
        # 处理目标变量：染色体非整倍体
        self.nv_tai_shu_ju['异常标记'] = self.nv_tai_shu_ju['染色体的非整倍体'].notna()
        self.nv_tai_shu_ju['异常标记'] = self.nv_tai_shu_ju['异常标记'].astype(int)
         
        print(f"异常样本数: {self.nv_tai_shu_ju['异常标记'].sum()}")
        print(f"正常样本数: {len(self.nv_tai_shu_ju) - self.nv_tai_shu_ju['异常标记'].sum()}")
        print(f"异常比例: {self.nv_tai_shu_ju['异常标记'].mean():.3f}")
         
        return self.nv_tai_shu_ju
     
    def ti_qu_te_zheng(self):
        """提取特征变量"""
        print("\n🔍 提取特征变量...")
         
        # 定义特征列
        te_zheng_lie = [
            # 
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
            # GC含量特征
            'GC含量', '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
            # 读段数相关特征
            '原始读段数', '唯一比对的读段数', '在参考基因组上比对的比例', '重复读段的比例', '被过滤掉读段数的比例',
            # X染色体浓度
            'X染色体浓度',
            # 孕妇基本信息
            '孕妇BMI', '年龄', '身高', '体重', '检测孕周_数值',
            # 妊娠相关
            '怀孕次数', '生产次数'
        ]
         
        # 检查可用特征
        ke_yong_te_zheng = []
        for lie_ming in te_zheng_lie:
            if lie_ming in self.nv_tai_shu_ju.columns:
                ke_yong_te_zheng.append(lie_ming)
            else:
                print(f"警告: 特征 '{lie_ming}' 不存在")
         
        print(f"可用特征数量: {len(ke_yong_te_zheng)}")
         
        # 提取特征数据
        self.te_zheng_shu_ju = self.nv_tai_shu_ju[ke_yong_te_zheng + ['异常标记']].copy()
         
        # 处理缺失值
        print(f"处理前样本数: {len(self.te_zheng_shu_ju)}")
         
        # 数值化处理
        for lie_ming in ke_yong_te_zheng:
            self.te_zheng_shu_ju[lie_ming] = pd.to_numeric(self.te_zheng_shu_ju[lie_ming], errors='coerce')
         
        # 删除包含过多缺失值的行
        yu_zhi = len(ke_yong_te_zheng) * 0.7  # 至少70%的特征不能缺失
        self.te_zheng_shu_ju = self.te_zheng_shu_ju.dropna(thresh=yu_zhi)
         
        # 填充剩余缺失值
        for lie_ming in ke_yong_te_zheng:
            if self.te_zheng_shu_ju[lie_ming].isna().any():
                zhong_wei_shu = self.te_zheng_shu_ju[lie_ming].median()
                self.te_zheng_shu_ju[lie_ming].fillna(zhong_wei_shu, inplace=True)
         
        print(f"处理后样本数: {len(self.te_zheng_shu_ju)}")
         
        # 准备机器学习数据
        self.te_zheng_ju_zhen = self.te_zheng_shu_ju[ke_yong_te_zheng]
        self.mu_biao_bian_liang = self.te_zheng_shu_ju['异常标记']
         
        print(f"特征矩阵形状: {self.te_zheng_ju_zhen.shape}")
        print(f"目标变量分布: 正常={len(self.mu_biao_bian_liang) - self.mu_biao_bian_liang.sum()}, 异常={self.mu_biao_bian_liang.sum()}")
         
        return ke_yong_te_zheng
    
    def z_fen_shu_biao_zhun_hua(self):
        """Z-分数标准化处理特征向量"""
        print("\n📐 Z-分数标准化处理...")
        
        # 保存原始数据用于对比
        self.yuan_shi_te_zheng = self.te_zheng_ju_zhen.copy()
        
        # 应用Z-分数标准化
        self.biao_zhun_hua_te_zheng = self.biao_zhun_hua_qi.fit_transform(self.te_zheng_ju_zhen)
        self.biao_zhun_hua_te_zheng = pd.DataFrame(self.biao_zhun_hua_te_zheng, columns=self.te_zheng_ju_zhen.columns, index=self.te_zheng_ju_zhen.index)
        
        # 输出标准化统计信息
        print("📊 Z-分数标准化统计信息:")
        print(f"  原始数据特征均值范围: [{self.te_zheng_ju_zhen.mean().min():.4f}, {self.te_zheng_ju_zhen.mean().max():.4f}]")
        print(f"  原始数据特征标准差范围: [{self.te_zheng_ju_zhen.std().min():.4f}, {self.te_zheng_ju_zhen.std().max():.4f}]")
        print(f"  标准化后特征均值范围: [{self.biao_zhun_hua_te_zheng.mean().min():.4f}, {self.biao_zhun_hua_te_zheng.mean().max():.4f}]")
        print(f"  标准化后特征标准差范围: [{self.biao_zhun_hua_te_zheng.std().min():.4f}, {self.biao_zhun_hua_te_zheng.std().max():.4f}]")
        
        # 可视化标准化效果
        self.ke_shi_hua_biao_zhun_hua_xiao_guo()
        
        return self.biao_zhun_hua_te_zheng
    
    def ke_shi_hua_biao_zhun_hua_xiao_guo(self):
        """可视化Z-分数标准化效果"""
        # 选择几个重要特征进行可视化对比
        zhong_yao_te_zheng = ['18号染色体的Z值', '21号染色体的Z值', '13号染色体的Z值', 'X染色体浓度']
        ke_yong_zhong_yao = [f for f in zhong_yao_te_zheng if f in self.te_zheng_ju_zhen.columns][:4]
        
        if len(ke_yong_zhong_yao) < 2:
            ke_yong_zhong_yao = self.te_zheng_ju_zhen.columns[:4].tolist()
        
        fig, axes = plt.subplots(2, len(ke_yong_zhong_yao), figsize=(5*len(ke_yong_zhong_yao), 10))
        fig.suptitle('Z-score标准化效果对比', fontsize=16, fontweight='bold')
        
        for i, te_zheng in enumerate(ke_yong_zhong_yao):
            # 原始数据分布
            ax1 = axes[0, i] if len(ke_yong_zhong_yao) > 1 else axes[0]
            ax1.hist(self.te_zheng_ju_zhen[te_zheng], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'原始数据: {te_zheng}')
            ax1.set_xlabel('数值')
            ax1.set_ylabel('频数')
            ax1.grid(True, alpha=0.3)
            
            # 添加统计信息
            jun_zhi = self.te_zheng_ju_zhen[te_zheng].mean()
            biao_zhun_cha = self.te_zheng_ju_zhen[te_zheng].std()
            ax1.axvline(jun_zhi, color='red', linestyle='--', alpha=0.8, label=f'均值: {jun_zhi:.3f}')
            ax1.axvline(jun_zhi + biao_zhun_cha, color='orange', linestyle='--', alpha=0.8)
            ax1.axvline(jun_zhi - biao_zhun_cha, color='orange', linestyle='--', alpha=0.8, label=f'±1σ: {biao_zhun_cha:.3f}')
            ax1.legend()
            
            # 标准化后数据分布
            ax2 = axes[1, i] if len(ke_yong_zhong_yao) > 1 else axes[1]
            ax2.hist(self.biao_zhun_hua_te_zheng[te_zheng], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title(f'Z-分数标准化后: {te_zheng}')
            ax2.set_xlabel('标准化数值')
            ax2.set_ylabel('频数')
            ax2.grid(True, alpha=0.3)
            
            # 添加标准正态分布参考线
            biao_zhun_hua_jun_zhi = self.biao_zhun_hua_te_zheng[te_zheng].mean()
            biao_zhun_hua_biao_cha = self.biao_zhun_hua_te_zheng[te_zheng].std()
            ax2.axvline(biao_zhun_hua_jun_zhi, color='red', linestyle='--', alpha=0.8, label=f'均值: {biao_zhun_hua_jun_zhi:.3f}')
            ax2.axvline(biao_zhun_hua_jun_zhi + biao_zhun_hua_biao_cha, color='orange', linestyle='--', alpha=0.8)
            ax2.axvline(biao_zhun_hua_jun_zhi - biao_zhun_hua_biao_cha, color='orange', linestyle='--', alpha=0.8, label=f'±1σ: {biao_zhun_hua_biao_cha:.3f}')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig('问题四结果/Z-score标准化效果对比.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出详细的标准化对比表
        dui_bi_biao = pd.DataFrame({
            '特征': self.te_zheng_ju_zhen.columns,
            '原始均值': self.te_zheng_ju_zhen.mean(),
            '原始标准差': self.te_zheng_ju_zhen.std(),
            '标准化均值': self.biao_zhun_hua_te_zheng.mean(),
            '标准化标准差': self.biao_zhun_hua_te_zheng.std(),
            '标准化效果': ['良好' if abs(m) < 0.01 and abs(s - 1) < 0.01 else '需检查' 
                      for m, s in zip(self.biao_zhun_hua_te_zheng.mean(), self.biao_zhun_hua_te_zheng.std())]
        })
        
        print("\n📋 详细标准化对比表:")
        print(dui_bi_biao.round(4).to_string(index=False))
     
    def fen_xi_te_zheng_zhong_yao_xing(self):
        """分析特征重要性（使用Z-分数标准化数据）"""
        print("\n🔍 分析特征重要性...")
        
        # 确保已经进行了Z-分数标准化
        if not hasattr(self, 'biao_zhun_hua_te_zheng'):
            print("⚠️ 未找到标准化数据，使用原始数据进行分析...")
            fen_xi_shu_ju = self.te_zheng_ju_zhen
        else:
            print("✅ 使用Z-分数标准化数据进行特征重要性分析")
            fen_xi_shu_ju = self.biao_zhun_hua_te_zheng
         
        # 使用随机森林分析特征重要性
        sui_ji_sen_lin = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        sui_ji_sen_lin.fit(fen_xi_shu_ju, self.mu_biao_bian_liang)
         
        # 获取特征重要性
        te_zheng_zhong_yao_xing = pd.DataFrame({
            'feature': fen_xi_shu_ju.columns,
            'importance': sui_ji_sen_lin.feature_importances_
        }).sort_values('importance', ascending=False)
         
        self.te_zheng_zhong_yao_xing = te_zheng_zhong_yao_xing
         
        print("特征重要性排序 (Top 10):")
        for idx, hang in te_zheng_zhong_yao_xing.head(10).iterrows():
            print(f"  {hang['feature']}: {hang['importance']:.4f}")
         
        # 可视化特征重要性
        plt.figure(figsize=(20, 16))
         
        # 特征重要性条形图
        plt.subplot(3, 3, 1)
        ding_ji_te_zheng = te_zheng_zhong_yao_xing.head(15)
        plt.barh(range(len(ding_ji_te_zheng)), ding_ji_te_zheng['importance'])
        plt.yticks(range(len(ding_ji_te_zheng)), ding_ji_te_zheng['feature'])
        plt.xlabel('重要性')
        plt.title('Top 15 特征重要性分析\n(基于Z-分数标准化数据)')
        plt.gca().invert_yaxis()
         
        # 异常样本vs正常样本的特征分布对比（选择重要特征，使用标准化数据）
        ding_ji_te_zheng_ming = te_zheng_zhong_yao_xing.head(8)['feature'].tolist()
         
        for i, te_zheng in enumerate(ding_ji_te_zheng_ming[:6]):
            plt.subplot(3, 3, i+2)
             
            zheng_chang_shu_ju = fen_xi_shu_ju[self.mu_biao_bian_liang == 0][te_zheng]
            yi_chang_shu_ju = fen_xi_shu_ju[self.mu_biao_bian_liang == 1][te_zheng]
             
            plt.hist(zheng_chang_shu_ju, alpha=0.7, label='正常', bins=20, color='skyblue')
            plt.hist(yi_chang_shu_ju, alpha=0.7, label='异常', bins=20, color='salmon')
             
            plt.xlabel(f'{te_zheng}\n(Z-分数标准化)')
            plt.ylabel('频数')
            plt.title(f'{te_zheng} 分布对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        # 添加标准化前后对比图
        plt.subplot(3, 3, 8)
        if hasattr(self, 'biao_zhun_hua_te_zheng'):
            dui_bi_te_zheng = ding_ji_te_zheng_ming[0]
            
            # 原始数据
            plt.hist(self.te_zheng_ju_zhen[dui_bi_te_zheng], alpha=0.6, label='原始数据', bins=20, color='lightblue')
            # 标准化数据
            plt.hist(self.biao_zhun_hua_te_zheng[dui_bi_te_zheng], alpha=0.6, label='Z-分数标准化', bins=20, color='lightcoral')
            
            plt.xlabel(f'{dui_bi_te_zheng}')
            plt.ylabel('频数')
            plt.title('标准化前后对比\n(最重要特征)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 特征重要性累积贡献图
        plt.subplot(3, 3, 9)
        lei_ji_zhong_yao_xing = te_zheng_zhong_yao_xing['importance'].cumsum()
        plt.plot(range(1, len(lei_ji_zhong_yao_xing)+1), lei_ji_zhong_yao_xing, 'b-', linewidth=2)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%贡献线')
        plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90%贡献线')
        plt.xlabel('特征数量')
        plt.ylabel('累积重要性')
        plt.title('特征重要性累积贡献')
        plt.legend()
        plt.grid(True, alpha=0.3)
         
        plt.tight_layout()
        plt.savefig('问题四结果/女胎异常检测特征重要性分析.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出标准化对特征重要性的影响分析
        if hasattr(self, 'biao_zhun_hua_te_zheng'):
            print("\n📊 Z-分数标准化对特征重要性的影响:")
            print(f"  使用标准化数据分析特征重要性")
            print(f"  最重要的3个特征:")
            for i, (_, hang) in enumerate(te_zheng_zhong_yao_xing.head(3).iterrows()):
                print(f"    {i+1}. {hang['feature']}: {hang['importance']:.4f}")
        
        return te_zheng_zhong_yao_xing
     
    def gou_jian_mo_xing(self):
        """构建多种分类模型（使用Z-分数标准化数据）"""
        print("\n🤖 构建分类模型...")
        
        # 使用已标准化的数据
        if hasattr(self, 'biao_zhun_hua_te_zheng'):
            print("✅ 使用Z-分数标准化数据训练模型")
            biao_zhun_hua_shu_ju = self.biao_zhun_hua_te_zheng.values
        else:
            print("⚠️ 未找到标准化数据，重新进行标准化...")
            biao_zhun_hua_shu_ju = self.biao_zhun_hua_qi.fit_transform(self.te_zheng_ju_zhen)
         
        # 分割训练测试集
        X_xun_lian, X_ce_shi, y_xun_lian, y_ce_shi = train_test_split(
            biao_zhun_hua_shu_ju, self.mu_biao_bian_liang, test_size=0.3, random_state=42, stratify=self.mu_biao_bian_liang
        )
         
        # 定义模型
        mo_xing_zi_dian = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
        }
         
        # 训练和评估模型
        jie_guo_zi_dian = {}
         
        for ming_cheng, mo_xing in mo_xing_zi_dian.items():
            print(f"\n训练 {ming_cheng}...")
             
            # 交叉验证
            jiao_cha_fen_shu = cross_val_score(mo_xing, X_xun_lian, y_xun_lian, cv=5, scoring='roc_auc')
             
            # 训练模型
            mo_xing.fit(X_xun_lian, y_xun_lian)
             
            # 预测
            y_yu_ce = mo_xing.predict(X_ce_shi)
            y_yu_ce_gai_lu = mo_xing.predict_proba(X_ce_shi)[:, 1]
             
            # 计算指标
            auc_de_fen = roc_auc_score(y_ce_shi, y_yu_ce_gai_lu)
             
            jie_guo_zi_dian[ming_cheng] = {
                'model': mo_xing,
                'cv_scores': jiao_cha_fen_shu,
                'cv_mean': jiao_cha_fen_shu.mean(),
                'cv_std': jiao_cha_fen_shu.std(),
                'auc_score': auc_de_fen,
                'y_test': y_ce_shi,
                'y_pred': y_yu_ce,
                'y_pred_proba': y_yu_ce_gai_lu
            }
             
            print(f"  交叉验证 AUC: {jiao_cha_fen_shu.mean():.4f} ± {jiao_cha_fen_shu.std():.4f}")
            print(f"  测试集 AUC: {auc_de_fen:.4f}")
         
        self.mo_xing_zu = jie_guo_zi_dian
         
        # 选择最佳模型
        zui_jia_mo_xing_ming = max(jie_guo_zi_dian.keys(), key=lambda x: jie_guo_zi_dian[x]['auc_score'])
        self.zui_jia_mo_xing = jie_guo_zi_dian[zui_jia_mo_xing_ming]['model']
         
        print(f"\n🏆 最佳模型: {zui_jia_mo_xing_ming}")
        print(f"最佳模型 AUC: {jie_guo_zi_dian[zui_jia_mo_xing_ming]['auc_score']:.4f}")
         
        return jie_guo_zi_dian
     
    def ping_gu_mo_xing(self):
        """评估模型性能"""
        print("\n📊 模型性能评估...")
         
        if not self.mo_xing_zu:
            print("请先构建模型!")
            return
         
        # 创建评估图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('女胎异常检测模型性能评估', fontsize=16, fontweight='bold')
         
        # 1. AUC比较
        ax1 = axes[0, 0]
        mo_xing_ming_cheng = list(self.mo_xing_zu.keys())
        auc_de_fen_zu = [self.mo_xing_zu[ming]['auc_score'] for ming in mo_xing_ming_cheng]
         
        tiao_xing = ax1.bar(mo_xing_ming_cheng, auc_de_fen_zu, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax1.set_ylabel('AUC Score')
        ax1.set_title('模型AUC性能比较')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45)
         
        # 添加数值标签
        for tiao, de_fen in zip(tiao_xing, auc_de_fen_zu):
            gao_du = tiao.get_height()
            ax1.text(tiao.get_x() + tiao.get_width()/2., gao_du + 0.01,
                    f'{de_fen:.3f}', ha='center', va='bottom', fontweight='bold')
         
        # 2. ROC曲线
        ax2 = axes[0, 1]
        yan_se = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
         
        for i, (ming_cheng, jie_guo) in enumerate(self.mo_xing_zu.items()):
            jia_yang_xing_lv, zhen_yang_xing_lv, _ = roc_curve(jie_guo['y_test'], jie_guo['y_pred_proba'])
            auc = jie_guo['auc_score']
            ax2.plot(jia_yang_xing_lv, zhen_yang_xing_lv, color=yan_se[i], linewidth=2,
                    label=f'{ming_cheng} (AUC = {auc:.3f})')
         
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax2.set_xlabel('假阳性率')
        ax2.set_ylabel('真阳性率')
        ax2.set_title('ROC曲线比较')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
         
        # 3. Precision-Recall曲线
        ax3 = axes[0, 2]
         
        for i, (ming_cheng, jie_guo) in enumerate(self.mo_xing_zu.items()):
            jing_que_lv, zhao_hui_lv, _ = precision_recall_curve(jie_guo['y_test'], jie_guo['y_pred_proba'])
            ping_jun_jing_que_lv = average_precision_score(jie_guo['y_test'], jie_guo['y_pred_proba'])
            ax3.plot(zhao_hui_lv, jing_que_lv, color=yan_se[i], linewidth=2,
                    label=f'{ming_cheng} (AP = {ping_jun_jing_que_lv:.3f})')
         
        ax3.set_xlabel('召回率')
        ax3.set_ylabel('精确率')
        ax3.set_title('Precision-Recall曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
         
        # 4. 最佳模型混淆矩阵
        ax4 = axes[1, 0]
        zui_jia_mo_xing_ming = max(self.mo_xing_zu.keys(), key=lambda x: self.mo_xing_zu[x]['auc_score'])
        zui_jia_jie_guo = self.mo_xing_zu[zui_jia_mo_xing_ming]
         
        hun_yao_ju_zhen = confusion_matrix(zui_jia_jie_guo['y_test'], zui_jia_jie_guo['y_pred'])
        sns.heatmap(hun_yao_ju_zhen, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_xlabel('预测标签')
        ax4.set_ylabel('真实标签')
        ax4.set_title(f'最佳模型 ({zui_jia_mo_xing_ming}) 混淆矩阵')
         
        # 5. 交叉验证性能
        ax5 = axes[1, 1]
        jiao_cha_jun_zhi = [self.mo_xing_zu[ming]['cv_mean'] for ming in mo_xing_ming_cheng]
        jiao_cha_biao_cha = [self.mo_xing_zu[ming]['cv_std'] for ming in mo_xing_ming_cheng]
         
        ax5.bar(mo_xing_ming_cheng, jiao_cha_jun_zhi, yerr=jiao_cha_biao_cha, capsize=5,
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax5.set_ylabel('交叉验证 AUC')
        ax5.set_title('5折交叉验证性能')
        plt.setp(ax5.get_xticklabels(), rotation=45)
         
        # 6. 特征重要性（使用最佳模型）
        ax6 = axes[1, 2]
        if hasattr(zui_jia_jie_guo['model'], 'feature_importances_'):
            zhong_yao_xing_zhi = zui_jia_jie_guo['model'].feature_importances_
            te_zheng_ming_cheng = self.te_zheng_ju_zhen.columns
             
            # 选择top 10特征
            suo_yin = np.argsort(zhong_yao_xing_zhi)[::-1][:10]
            ding_ji_zhong_yao_xing = zhong_yao_xing_zhi[suo_yin]
            ding_ji_te_zheng = te_zheng_ming_cheng[suo_yin]
             
            ax6.barh(range(len(ding_ji_zhong_yao_xing)), ding_ji_zhong_yao_xing)
            ax6.set_yticks(range(len(ding_ji_zhong_yao_xing)))
            ax6.set_yticklabels(ding_ji_te_zheng)
            ax6.set_xlabel('重要性')
            ax6.set_title(f'{zui_jia_mo_xing_ming} 特征重要性')
            ax6.invert_yaxis()
         
        plt.tight_layout()
        plt.savefig('问题四结果/女胎异常检测模型性能评估.png', dpi=300, bbox_inches='tight')
        plt.show()
         
        # 打印详细分类报告
        print(f"\n📋 最佳模型 ({zui_jia_mo_xing_ming}) 详细性能:")
        print(classification_report(zui_jia_jie_guo['y_test'], zui_jia_jie_guo['y_pred'],
                                  target_names=['正常', '异常']))
     
    def lin_chuang_jue_ce_fen_xi(self):
        """临床决策分析"""
        print("\n🏥 临床决策分析...")
         
        if not self.zui_jia_mo_xing:
            print("请先构建模型!")
            return
         
        # 使用最佳模型进行预测
        if hasattr(self, 'biao_zhun_hua_te_zheng'):
            print("✅ 使用Z-分数标准化数据进行临床决策分析")
            biao_zhun_hua_shu_ju = self.biao_zhun_hua_te_zheng.values
        else:
            print("⚠️ 未找到标准化数据，重新进行标准化...")
            biao_zhun_hua_shu_ju = self.biao_zhun_hua_qi.transform(self.te_zheng_ju_zhen)
        yu_ce_gai_lu = self.zui_jia_mo_xing.predict_proba(biao_zhun_hua_shu_ju)[:, 1]
         
        # 不同阈值下的性能分析
        yu_zhi_zu = np.arange(0.1, 0.9, 0.05)
        zhi_biao_lie_biao = []
         
        for yu_zhi in yu_zhi_zu:
            yu_ce_jie_guo = (yu_ce_gai_lu >= yu_zhi).astype(int)
             
            # 计算混淆矩阵元素
            zhen_yin = np.sum((self.mu_biao_bian_liang == 0) & (yu_ce_jie_guo == 0))
            jia_yang = np.sum((self.mu_biao_bian_liang == 0) & (yu_ce_jie_guo == 1))
            jia_yin = np.sum((self.mu_biao_bian_liang == 1) & (yu_ce_jie_guo == 0))
            zhen_yang = np.sum((self.mu_biao_bian_liang == 1) & (yu_ce_jie_guo == 1))
             
            # 计算指标
            min_gan_xing = zhen_yang / (zhen_yang + jia_yin) if (zhen_yang + jia_yin) > 0 else 0  # 敏感性
            te_yi_xing = zhen_yin / (zhen_yin + jia_yang) if (zhen_yin + jia_yang) > 0 else 0  # 特异性
            yang_xing_yu_ce_zhi = zhen_yang / (zhen_yang + jia_yang) if (zhen_yang + jia_yang) > 0 else 0  # 阳性预测值
            yin_xing_yu_ce_zhi = zhen_yin / (zhen_yin + jia_yin) if (zhen_yin + jia_yin) > 0 else 0  # 阴性预测值
             
            zhi_biao_lie_biao.append({
                'threshold': yu_zhi,
                'sensitivity': min_gan_xing,
                'specificity': te_yi_xing,
                'ppv': yang_xing_yu_ce_zhi,
                'npv': yin_xing_yu_ce_zhi,
                'tp': zhen_yang, 'tn': zhen_yin, 'fp': jia_yang, 'fn': jia_yin
            })
         
        zhi_biao_biao = pd.DataFrame(zhi_biao_lie_biao)
         
        # 可视化阈值分析
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('女胎异常检测临床决策分析', fontsize=16, fontweight='bold')
         
        # 1. 敏感性和特异性曲线
        ax1 = axes[0, 0]
        ax1.plot(zhi_biao_biao['threshold'], zhi_biao_biao['sensitivity'], 'b-', linewidth=2, label='敏感性')
        ax1.plot(zhi_biao_biao['threshold'], zhi_biao_biao['specificity'], 'r-', linewidth=2, label='特异性')
        ax1.set_xlabel('判定阈值')
        ax1.set_ylabel('性能指标')
        ax1.set_title('敏感性与特异性曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
         
        # 2. 预测值曲线
        ax2 = axes[0, 1]
        ax2.plot(zhi_biao_biao['threshold'], zhi_biao_biao['ppv'], 'g-', linewidth=2, label='阳性预测值')
        ax2.plot(zhi_biao_biao['threshold'], zhi_biao_biao['npv'], 'm-', linewidth=2, label='阴性预测值')
        ax2.set_xlabel('判定阈值')
        ax2.set_ylabel('预测值')
        ax2.set_title('阳性/阴性预测值曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
         
        # 3. 推荐阈值分析
        ax3 = axes[1, 0]
         
        # 计算Youden指数 (敏感性 + 特异性 - 1)
        youden_zhi_shu = zhi_biao_biao['sensitivity'] + zhi_biao_biao['specificity'] - 1
        zui_jia_youden_suo_yin = youden_zhi_shu.idxmax()
        zui_jia_yu_zhi = zhi_biao_biao.loc[zui_jia_youden_suo_yin, 'threshold']
         
        ax3.plot(zhi_biao_biao['threshold'], youden_zhi_shu, 'purple', linewidth=2)
        ax3.axvline(x=zui_jia_yu_zhi, color='red', linestyle='--', alpha=0.7)
        ax3.plot(zui_jia_yu_zhi, youden_zhi_shu.iloc[zui_jia_youden_suo_yin], 'ro', markersize=10)
        ax3.text(zui_jia_yu_zhi, youden_zhi_shu.iloc[zui_jia_youden_suo_yin], 
                f'  最佳阈值: {zui_jia_yu_zhi:.3f}', va='center', fontweight='bold')
        ax3.set_xlabel('判定阈值')
        ax3.set_ylabel('Youden指数')
        ax3.set_title('最佳判定阈值选择 (Youden指数)')
        ax3.grid(True, alpha=0.3)
         
        # 4. 风险分层
        ax4 = axes[1, 1]
         
        # 将预测概率分为风险等级
        di_feng_xian = np.sum(yu_ce_gai_lu < 0.3)
        zhong_feng_xian = np.sum((yu_ce_gai_lu >= 0.3) & (yu_ce_gai_lu < 0.7))
        gao_feng_xian = np.sum(yu_ce_gai_lu >= 0.7)
         
        feng_xian_shu_liang = [di_feng_xian, zhong_feng_xian, gao_feng_xian]
        feng_xian_biao_qian = ['低风险\n(<30%)', '中风险\n(30%-70%)', '高风险\n(≥70%)']
        yan_se_zu = ['#4ECDC4', '#FECA57', '#FF6B6B']
         
        zhu_zhuang_tu = ax4.bar(feng_xian_biao_qian, feng_xian_shu_liang, color=yan_se_zu)
        ax4.set_ylabel('样本数量')
        ax4.set_title('风险分层分布')
         
        # 添加百分比标签
        zong_yang_ben_shu = len(yu_ce_gai_lu)
        for tiao, shu_liang in zip(zhu_zhuang_tu, feng_xian_shu_liang):
            gao_du = tiao.get_height()
            bai_fen_bi = shu_liang / zong_yang_ben_shu * 100
            ax4.text(tiao.get_x() + tiao.get_width()/2., gao_du + 0.5,
                    f'{shu_liang}\n({bai_fen_bi:.1f}%)', ha='center', va='bottom', fontweight='bold')
         
        plt.tight_layout()
        plt.savefig('问题四结果/女胎异常检测临床决策分析.png', dpi=300, bbox_inches='tight')
        plt.show()
         
        # 输出推荐阈值信息
        zui_jia_zhi_biao = zhi_biao_biao.iloc[zui_jia_youden_suo_yin]
        print(f"\n🎯 推荐判定阈值: {zui_jia_yu_zhi:.3f}")
        print(f"该阈值下性能指标:")
        print(f"  敏感性: {zui_jia_zhi_biao['sensitivity']:.3f} ({zui_jia_zhi_biao['sensitivity']*100:.1f}%)")
        print(f"  特异性: {zui_jia_zhi_biao['specificity']:.3f} ({zui_jia_zhi_biao['specificity']*100:.1f}%)")
        print(f"  阳性预测值: {zui_jia_zhi_biao['ppv']:.3f} ({zui_jia_zhi_biao['ppv']*100:.1f}%)")
        print(f"  阴性预测值: {zui_jia_zhi_biao['npv']:.3f} ({zui_jia_zhi_biao['npv']*100:.1f}%)")
        print(f"  Youden指数: {youden_zhi_shu.iloc[zui_jia_youden_suo_yin]:.3f}")
         
        return zui_jia_yu_zhi, zhi_biao_biao
     
    def sheng_cheng_yu_ce_mo_xing(self):
        """生成最终预测模型"""
        print("\n🔮 生成女胎异常判定模型...")
         
        if not self.zui_jia_mo_xing:
            print("请先构建模型!")
            return
         
        # 获取特征重要性
        if hasattr(self.zui_jia_mo_xing, 'feature_importances_'):
            te_zheng_zhong_yao_xing = pd.DataFrame({
                'feature': self.te_zheng_ju_zhen.columns,
                'importance': self.zui_jia_mo_xing.feature_importances_
            }).sort_values('importance', ascending=False)
             
            print("\n📊 最终模型特征重要性 (Top 10):")
            for idx, hang in te_zheng_zhong_yao_xing.head(10).iterrows():
                print(f"  {hang['feature']}: {hang['importance']:.4f}")
         
        # 模型参数
        print(f"\n🔧 最终模型参数:")
        print(f"模型类型: {type(self.zui_jia_mo_xing).__name__}")
        if hasattr(self.zui_jia_mo_xing, 'get_params'):
            can_shu = self.zui_jia_mo_xing.get_params()
            guan_jian_can_shu = ['n_estimators', 'max_depth', 'learning_rate', 'C', 'gamma']
            for can_shu_ming in guan_jian_can_shu:
                if can_shu_ming in can_shu:
                    print(f"  {can_shu_ming}: {can_shu[can_shu_ming]}")
         
        return self.zui_jia_mo_xing
     
    def da_yin_zong_he_jie_guo(self):
        """打印综合结果"""
        print("\n" + "="*70)
        print("女胎异常判定方法 - 综合结果")
        print("="*70)
         
        print(f"\n📊 数据概况:")
        print(f"- 女胎总样本数: {len(self.nv_tai_shu_ju)}")
        print(f"- 可用分析样本数: {len(self.te_zheng_ju_zhen)}")
        print(f"- 异常样本数: {self.mu_biao_bian_liang.sum()}")
        print(f"- 正常样本数: {len(self.mu_biao_bian_liang) - self.mu_biao_bian_liang.sum()}")
        print(f"- 异常比例: {self.mu_biao_bian_liang.mean():.3f} ({self.mu_biao_bian_liang.mean()*100:.1f}%)")
         
        print(f"\n🔍 特征工程:")
        print(f"- 特征数量: {len(self.te_zheng_ju_zhen.columns)}")
        print(f"- 主要特征类别: Z值、GC含量、读段数、X染色体浓度、BMI等")
        if hasattr(self, 'biao_zhun_hua_te_zheng') and self.biao_zhun_hua_te_zheng is not None:
            print(f"- Z-分数标准化: ✅ 已完成")
            print(f"- 标准化后特征均值范围: [{self.biao_zhun_hua_te_zheng.mean().min():.4f}, {self.biao_zhun_hua_te_zheng.mean().max():.4f}]")
            print(f"- 标准化后特征标准差范围: [{self.biao_zhun_hua_te_zheng.std().min():.4f}, {self.biao_zhun_hua_te_zheng.std().max():.4f}]")
        else:
            print(f"- Z-分数标准化: ❌ 未完成")
         
        if self.te_zheng_zhong_yao_xing is not None:
            print(f"\n🏆 Top 5 重要特征:")
            for idx, hang in self.te_zheng_zhong_yao_xing.head(5).iterrows():
                print(f"  {hang['feature']}: {hang['importance']:.4f}")
         
        if self.mo_xing_zu:
            print(f"\n🤖 模型性能:")
            zui_jia_mo_xing_ming = max(self.mo_xing_zu.keys(), key=lambda x: self.mo_xing_zu[x]['auc_score'])
            zui_jia_jie_guo = self.mo_xing_zu[zui_jia_mo_xing_ming]
             
            print(f"- 最佳模型: {zui_jia_mo_xing_ming}")
            print(f"- 最佳AUC: {zui_jia_jie_guo['auc_score']:.4f}")
            print(f"- 交叉验证AUC: {zui_jia_jie_guo['cv_mean']:.4f} ± {zui_jia_jie_guo['cv_std']:.4f}")
             
            # 模型排名
            print(f"\n📈 模型性能排名:")
            pai_xu_mo_xing = sorted(self.mo_xing_zu.items(), 
                                 key=lambda x: x[1]['auc_score'], reverse=True)
            for i, (ming_cheng, jie_guo) in enumerate(pai_xu_mo_xing, 1):
                print(f"  {i}. {ming_cheng}: AUC = {jie_guo['auc_score']:.4f}")
 
def main():
    """主函数"""
    print("🔬 问题4: 女胎异常判定方法")
    print("=" * 50)
     
    # 初始化检测器
    jian_ce_qi = NvTaiYiChangJianCe('附件.xlsx')
     
    # 加载数据
    shu_ju = jian_ce_qi.jia_zai_yu_chu_li_shu_ju()
     
    if len(shu_ju) == 0:
        print("❌ 没有符合条件的数据")
        return
     
    # 提取特征
    te_zheng = jian_ce_qi.ti_qu_te_zheng()
     
    if len(jian_ce_qi.te_zheng_ju_zhen) == 0:
        print("❌ 特征提取失败")
        return
    
    # Z-分数标准化处理
    biao_zhun_hua_te_zheng = jian_ce_qi.z_fen_shu_biao_zhun_hua()
     
    # 分析特征重要性
    te_zheng_zhong_yao_xing = jian_ce_qi.fen_xi_te_zheng_zhong_yao_xing()
     
    # 构建模型
    jie_guo = jian_ce_qi.gou_jian_mo_xing()
     
    # 评估模型
    jian_ce_qi.ping_gu_mo_xing()
     
    # 临床决策分析
    zui_jia_yu_zhi, zhi_biao_biao = jian_ce_qi.lin_chuang_jue_ce_fen_xi()
     
    # 生成最终模型
    zui_zhong_mo_xing = jian_ce_qi.sheng_cheng_yu_ce_mo_xing()
     
    # 打印综合结果
    jian_ce_qi.da_yin_zong_he_jie_guo()
     
    print("\n✅ 女胎异常判定方法构建完成!")
    print("📊 结果图表已保存")
    print(f"🎯 推荐判定阈值: {zui_jia_yu_zhi:.3f}")
 
if __name__ == "__main__":
    main()