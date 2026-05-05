import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 解决 Windows 环境下 Matplotlib 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ── 路径配置（相对路径，打包后无需修改）──────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = r"D:\HELLO\huggingface\strawberry_recognition_system\bishe\data\Straw6D_Raw\boxes"  # 原始 CSV 目录
CACHE_PATH    = os.path.join(BASE_DIR, "data", "preprocessed.parquet")  # 预处理缓存文件
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── 数据读取与缓存 ────────────────────────────────────────────────────────────
def load_data():
    """
    优先读取预处理缓存（data/preprocessed.parquet）。
    若缓存不存在，则遍历原始 CSV 目录合并数据，处理完后保存缓存供下次直接使用。
    """
    if os.path.exists(CACHE_PATH):
        print(f"检测到预处理缓存，直接加载: {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        print(f"缓存加载完成，样本总数: {len(df)}")
        return df

    # 缓存不存在 → 执行原始预处理
    print(f"未检测到缓存，正在从原始 CSV 目录加载: {RAW_DATA_PATH}")
    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"原始数据目录 {RAW_DATA_PATH} 下未找到任何 CSV 文件！\n"
            f"请将 Straw6D 原始数据放入该目录后重新运行。"
        )

    print(f"检测到 {len(csv_files)} 个标注文件，正在加载并合并...")
    all_dfs = []
    for file in tqdm(csv_files, desc="Reading CSVs"):
        df = pd.read_csv(file)
        if not df.empty:
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"数据合并完成，样本总数: {len(combined_df)}")

    # 保存缓存，下次直接读取
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    combined_df.to_parquet(CACHE_PATH, index=False)
    print(f"预处理缓存已保存至: {CACHE_PATH}")

    return combined_df


# ── 特征工程 ──────────────────────────────────────────────────────────────────
def feature_engineering(df):
    """
    特征工程：
    - 几何特征 : w, h, l（宽、高、长）
    - 姿态特征 : roll, pitch, yaw（欧拉角）
    - 衍生特征 : volume（体积）、wl_ratio、wh_ratio
    - 目标标签 : label
    """
    print("开始特征工程...")
    epsilon = 1e-6
    df['volume']   = df['w'] * df['h'] * df['l']
    df['wl_ratio'] = df['w'] / (df['l'] + epsilon)
    df['wh_ratio'] = df['w'] / (df['h'] + epsilon)

    features = ['w', 'h', 'l', 'roll', 'pitch', 'yaw', 'volume', 'wl_ratio', 'wh_ratio']
    target   = 'label'

    X = df[features].fillna(0)
    y = df[target]
    return X, y


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    # 1. 数据加载（有缓存直接用，没有则预处理并缓存）
    raw_df = load_data()

    # 2. 特征工程
    X, y = feature_engineering(raw_df)

    # 3. 划分数据集（70% 训练，30% 测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 5. PCA 降维（仅用于可视化，不影响分类器输入）
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    print(f"PCA 累计解释方差比: {pca.explained_variance_ratio_.cumsum()[-1]:.4f}")

    # 6. 随机森林训练（在完整特征空间上）
    print("训练随机森林分类模型...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, oob_score=True)
    rf_model.fit(X_train_scaled, y_train)
    print(f"OOB Score: {rf_model.oob_score_:.4f}")

    # 7. 模型评估
    y_pred = rf_model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)

    print("\n--- [统计机器学习结课报告] 模型评估指标 ---")
    print(f"总样本数: {len(raw_df)} | 测试集准确率: {acc:.4f}")
    print("\n分类详细报告:")
    print(classification_report(y_test, y_pred))

    # 8. 可视化
    # (a) PCA 散点图
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=X_train_pca[:, 0], y=X_train_pca[:, 1],
        hue=y_train, palette='viridis', alpha=0.6
    )
    plt.title(f'PCA 特征降维可视化（样本量: {len(X_train)}）')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.savefig(os.path.join(RESULTS_DIR, 'pca_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # (b) 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('预测结果混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # (c) 特征重要性柱状图
    feat_names  = ['w', 'h', 'l', 'roll', 'pitch', 'yaw', 'volume', 'wl_ratio', 'wh_ratio']
    importances = rf_model.feature_importances_
    feat_df     = pd.DataFrame({'特征': feat_names, '重要性': importances})
    feat_df     = feat_df.sort_values('重要性', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='重要性', y='特征', palette='Blues_r')
    plt.title('随机森林特征重要性（Gini Importance）')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n所有图表已保存至: {RESULTS_DIR}")
    print("任务执行完毕！")


if __name__ == "__main__":
    main()
