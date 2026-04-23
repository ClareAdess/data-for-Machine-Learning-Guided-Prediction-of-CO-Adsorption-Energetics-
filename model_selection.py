"""
机器学习模型对比与SHAP分析脚本
用于材料性质预测的多个回归模型性能对比

文件结构说明:
============================================================
项目根目录/
│
├── model_comparison.py          # 主程序脚本
│
├── data/                        # 输入数据文件夹（需手动创建）
│   └── dataset.csv              # 原始数据文件
│
└── results/                     # 输出结果文件夹（自动创建）
    ├── models/                  # 保存训练好的模型 (.pkl)
    ├── predictions/             # 保存预测结果 (.csv)
    ├── shap_figures/            # SHAP分析图 (.png)
    └── performance/             # 性能对比结果 (.csv)
============================================================
"""

import os
import re
import random
import warnings
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# ==================== 全局配置 ====================
# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 简化的路径配置 ====================
# 获取当前脚本所在目录
CURRENT_DIR = Path(__file__).parent.absolute()

# 定义清晰的文件夹结构
DATA_DIR = CURRENT_DIR / 'data'  # 输入数据
RESULTS_DIR = CURRENT_DIR / 'results'  # 输出根目录
MODEL_DIR = RESULTS_DIR / 'models'  # 保存模型
PRED_DIR = RESULTS_DIR / 'predictions'  # 保存预测结果
SHAP_DIR = RESULTS_DIR / 'shap_figures'  # 保存SHAP图
PERF_DIR = RESULTS_DIR / 'performance'  # 保存性能对比

# 创建所有必要的目录
for dir_path in [RESULTS_DIR, MODEL_DIR, PRED_DIR, SHAP_DIR, PERF_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 随机种子配置（保持原代码不变）
RANDOM_SEEDS = {
    'xgboost': 3,
    'random_forest': 36,
    'gradient_boost': 3,
    'svr': 909,
    'ridge': 687,
    'adaboost': 36
}


# ==================== SHAP 分析函数 ====================
def shap_analysis(X: pd.DataFrame, y: pd.Series, save_dir: Path = None) -> None:
    """
    计算并保存SHAP分析图
    """
    if save_dir is None:
        save_dir = SHAP_DIR

    print("开始SHAP分析...")

    # 训练 XGBoost 模型（保持原参数）
    model = xgb.XGBRegressor(n_estimators=200, n_jobs=1, random_state=54)
    model.fit(X, y)

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer(X)

    # 计算特征重要性
    shap_importance = np.abs(shap_values.values).mean(axis=0)

    # ===================== 1. SHAP 点阵图（可单独微调） =====================
    # 大小
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)

    # 字体 + 样式
    plt.xticks(fontsize=11)  # X 轴刻度字体
    plt.yticks(fontsize=11)  # Y 轴刻度字体
    plt.xlabel("SHAP Value", fontsize=13, fontweight='bold')  # 轴标签
    plt.title("SHAP Summary Dot Plot", fontsize=14, pad=15)  # 标题
    plt.gca().spines[['top', 'right']].set_visible(False)  # 隐藏边框
    plt.tight_layout()

    plt.savefig(save_dir / 'shap_summary_dot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===================== 2. SHAP 柱状图（可单独微调） =====================
    # 大小
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)

    # 字体 + 样式
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel("Mean |SHAP Value|", fontsize=13, fontweight='bold')
    plt.title("SHAP Feature Importance", fontsize=14, pad=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    plt.savefig(save_dir / 'shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===================== 3. 依赖图（可单独微调） =====================
    top_feature_idx = np.abs(shap_values.values).mean(axis=0).argmax()
    top_feature = X.columns[top_feature_idx]

    # 大小
    plt.figure(figsize=(9, 5))
    shap.dependence_plot(
        top_feature, shap_values.values, X,
        interaction_index='auto', show=False
    )

    # 字体 + 样式 + 颜色
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.xlabel(top_feature, fontsize=13, fontweight='bold')
    plt.ylabel(f"SHAP Value", fontsize=13, fontweight='bold')
    plt.title(f"SHAP Dependence: {top_feature}", fontsize=14, pad=15)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    plt.savefig(save_dir / 'shap_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP分析完成，图像已保存至 {save_dir}")


# ==================== 预测结果保存函数 ====================
def save_predictions(
        X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series,
        model_name: str, model: object
) -> None:
    """
    保存模型的预测结果到CSV文件（保持原格式）
    """
    y_train_pred = pd.DataFrame(model.predict(X_train)).squeeze()
    y_test_pred = pd.DataFrame(model.predict(X_test))

    results = pd.concat([
        y_train.reset_index(drop=True),
        y_train_pred,
        y_test.reset_index(drop=True),
        y_test_pred
    ], axis=1)

    save_path = PRED_DIR / f"{model_name}_train_trainprediction_test_testprediction.csv"
    results.to_csv(save_path, header=False, index=False)
    print(f"  预测结果已保存至 {save_path}")


# ==================== 模型训练与评估函数 ====================
def train_and_evaluate_models(
        X: pd.DataFrame,
        y: pd.Series,
        models: dict,
        test_size: float = 0.15
) -> pd.DataFrame:
    """
    训练多个模型并评估性能（保持原逻辑不变）
    新增：训练后预测全量 X，合并 真实Y、预测Y、所有X 并保存
    """
    results = []
    scoring_metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']

    for name, model in models.items():
        print(f"\n  > 训练模型: {name}")

        # 获取对应的随机种子
        seed_key = name.lower().replace(' ', '_')
        random_state = RANDOM_SEEDS.get(seed_key, 42)

        # 配置测试集比例（保持原逻辑）
        current_test_size = test_size
        if name in ['Adaboost', 'SVR', 'Ridge']:
            current_test_size = 0.1

        # K折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring=scoring_metrics[2])

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=current_test_size, random_state=random_state
        )

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # ===================== 【新增：全量预测 + 保存结果】 =====================
        # 1. 对整个 X 做预测
        y_all_pred = model.predict(X)

        # 2. 合并：真实值 + 预测值 + 所有特征 X
        final_result = pd.DataFrame({
            '真实_Y': y.values,
            '预测_Y': y_all_pred
        })
        # 把所有特征 X 拼在后面
        final_result = pd.concat([final_result, X.reset_index(drop=True)], axis=1)

        # 3. 保存 CSV（每个模型单独保存）
        pred_save_path = MODEL_DIR / f"{name}_全量预测结果.csv"
        final_result.to_csv(pred_save_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ 全量预测结果已保存：{pred_save_path}")
        # ======================================================================

        # 计算评估指标
        results.append({
            'Model': name,
            'R2_train': r2_score(y_train, y_train_pred),
            'R2_test': r2_score(y_test, y_test_pred),
            'MSE_train': mean_squared_error(y_train, y_train_pred),
            'MSE_test': mean_squared_error(y_test, y_test_pred),
            'MAE_train': mean_absolute_error(y_train, y_train_pred),
            'MAE_test': mean_absolute_error(y_test, y_test_pred),
            'CV_MAE_mean': cv_scores.mean(),
            'CV_MAE_std': cv_scores.std()
        })

        # 保存模型
        model_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"  模型已保存至 {model_path}")

        # 保存预测结果（你原来的函数，保留不动）
        save_predictions(X_train, X_test, y_train, y_test, name, model)

    return pd.DataFrame(results)



# ==================== 主程序 ====================
def main():
    """主函数：执行完整的模型对比流程"""

    print("=" * 60)
    print("材料性质预测模型对比分析")
    print("=" * 60)

    # 显示当前的目录结构
    print(f"\n[目录结构]")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {RESULTS_DIR}")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  预测目录: {PRED_DIR}")
    print(f"  SHAP目录: {SHAP_DIR}")
    print(f"  性能目录: {PERF_DIR}")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    data_path = DATA_DIR / 'dataset.xlsx'

    # 检查数据文件是否存在
    if not data_path.exists():
        print(f"  错误: 数据文件不存在！")
        print(f"  请确保数据文件位于: {data_path}")
        print(f"  提示: 请在项目根目录下创建 'data' 文件夹，并将 output.csv 放入其中")
        return None

    data_last = pd.read_excel(data_path)
    print(f"  数据形状: {data_last.shape}")

    # 分离特征和目标变量
    X = data_last.iloc[:, 1:]  # 特征
    y = data_last.iloc[:, 0]  # 目标变量
    print(f"  特征数量: {X.shape[1]}")
    print(f"  样本数量: {X.shape[0]}")

    # 2. SHAP分析（默认注释，与原代码一致）
    print("\n[2] 执行SHAP特征重要性分析...")
    #shap_analysis(X, y)  # 默认注释，需要时取消注释

    # 3. 定义模型（保持原参数不变）
    print("\n[3] 定义对比模型...")
    baselines = {
        'XGBoost': xgb.XGBRegressor(n_estimators=200, n_jobs=1, random_state=3),
        'RandomForest': RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=36),
        'GradientBoost': GradientBoostingRegressor(n_estimators=200, random_state=3),
        'SVR': SVR(),
        'Ridge': Ridge(alpha=1.0),
        'Adaboost': AdaBoostRegressor(n_estimators=200, random_state=36),
        # ================= 新增的4个模型 =================
        'DecisionTree': DecisionTreeRegressor(random_state=3),
        'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=3),
        'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'LinearRegression': LinearRegression(n_jobs=-1)
    }

    # 4. 训练并评估模型
    print("\n[4] 训练并评估模型...")
    results_df = train_and_evaluate_models(X, y, baselines)

    # 5. 输出结果
    print("\n[5] 模型性能对比结果:")
    print("-" * 80)
    results_df_sorted = results_df.sort_values('R2_test', ascending=False)
    print(results_df_sorted.to_string(index=False))

    # 6. 保存结果到性能目录
    output_path = PERF_DIR / 'bar_对比.csv'
    results_df_sorted.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[6] 结果已保存至 {output_path}")

    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"所有输出文件已保存至: {RESULTS_DIR}")
    print("=" * 60)

    return results_df_sorted


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体，避免显示乱码（可根据系统字体调整）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

def bar_chart_from_csv(csv_path, value_col, output_path=None, figsize=(10, 6),
                       title="柱状图", xlabel="model", ylabel="数值",
                       legend_title="图例（首列）", cmap='YlGn_r'):
    """
    读取 CSV 文件，对指定列绘制柱状图，图例使用 CSV 第一列的值。
    配色：从绿到黄的渐变（科研常见风格），默认使用 'YlGn_r' colormap。
    参数 cmap 可换成其他科研配色如 'viridis', 'plasma' 等。
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV 文件为空")

    legend_labels = df.iloc[:, 0].astype(str).tolist()

    if isinstance(value_col, str):
        if value_col not in df.columns:
            raise KeyError(f"列名 '{value_col}' 不存在，可用的列名: {list(df.columns)}")
        values = df[value_col]
    else:
        if value_col >= df.shape[1]:
            raise IndexError(f"列索引 {value_col} 超出范围，最大索引为 {df.shape[1]-1}")
        values = df.iloc[:, value_col]

    if not pd.api.types.is_numeric_dtype(values):
        try:
            values = pd.to_numeric(values)
        except ValueError:
            raise ValueError(f"指定列 '{value_col}' 无法转换为数值类型")

    n = len(values)
    fig, ax = plt.subplots(figsize=figsize)

    # 使用 colormap 生成从绿到黄的颜色（根据柱子顺序均匀采样）
    # 注意：YlGn_r 中数值小（左侧柱子）为深绿，数值大（右侧）为浅黄，符合视觉习惯
    cmap_obj = plt.cm.get_cmap(cmap)
    # 为每个柱子分配一个在 [0,1] 区间均匀分布的颜色索引
    color_indices = np.linspace(0, 1, n)
    colors = [cmap_obj(i) for i in color_indices]

    bars = ax.bar(range(n), values, color=colors, alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.legend(bars, legend_labels, title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
    ax.set_xticks([])   # 隐藏 X 轴刻度

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output_path}")
    else:
        plt.show()
    plt.close(fig)

# ==================== 脚本入口 ====================
if __name__ == "__main__":
    col = 'MSE_test'
    bar_chart_from_csv(
        csv_path=r"D:\wechat\WeChat Files\wxid_h7uop5ppyhvz22\FileStorage\File\2026-04\第一次预测整理\results\performance\bar_对比.csv",
        value_col=col,  # 也可使用索引 1
        output_path=rf"D:\wechat\WeChat Files\wxid_h7uop5ppyhvz22\FileStorage\File\2026-04\第一次预测整理\results\performance\{col}_无ridge_chart.png",  # 保存为图片，若不需要保存可设为 None
        title=f"各模型{col}对比",
        legend_title="model"
    )
'''
    # 忽略警告信息（可选）
    warnings.filterwarnings('ignore')

    # 设置随机种子以确保可重复性
    np.random.seed(42)
    random.seed(42)

    # 运行主程序
    results = main()
    
'''