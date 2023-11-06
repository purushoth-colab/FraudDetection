import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# 设置全局随机数种子
np.random.seed(42)

# 1. 读取CSV文件
data_train = pd.read_csv('/Users/shenyuanzhe/Desktop/resampled_data_CNN.csv')
data_test = pd.read_csv('/Users/shenyuanzhe/Desktop/test_data.csv')

# 3. 选择需要的列
X_train = data_train.iloc[:, 0:29]
X_test = data_test.iloc[:, 1:30]

y_train = data_train['Class']
y_train = np.array(y_train)

y_test = data_test['Class']
y_test = np.array(y_test)

# 4. 使用PCA降至16维度
pca = PCA(n_components=16)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model = RandomForestClassifier()
model_name = 'RandomForest'

print(f"Training {model_name} model...")
model.fit(X_train_pca, y_train)

# 进行预测并计算F1分数
y_pred = model.predict(X_test_pca)

# 计算精确率和召回率
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印性能指标
print(f"Model: {model_name}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 计算ROC AUC（如果模型支持）
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_test_pca)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {roc_auc:.4f}")

print("\n")


