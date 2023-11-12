import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.combine import SMOTEENN

# 设置全局随机数种子
np.random.seed(42)

# 1. 读取CSV文件
data_train = pd.read_csv('train_data.csv')
data_test = pd.read_csv('test_data.csv')

# 3. 选择需要的列
X_test = data_test.iloc[:, 1:30]
y_test = data_test['Class']
y_test = np.array(y_test)
# 通过过采样的方式均衡样本

X_train = data_train.iloc[:, 1:30]  # 选择V1到V28列
y_train = data_train['Class']       # 选择Class列作为目标分类
oversampler = SMOTE(random_state=0)
X_train, y_train = oversampler.fit_resample(X_train, y_train)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
y_train = np.array(y_train)
# 定义要尝试的模型列表
models = {
    'XGBoost': xgb.XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'LogisticRegression': LogisticRegression(),
    'KNeighbors': KNeighborsClassifier()
}

# 循环迭代每个模型并评估性能
# model = xgb.XGBClassifier()
# model_name = 'XGBoost'
for model_name, model in models.items():
    print(f"Training {model_name} model...")
    model.fit(X_train, y_train)

    # 进行预测并计算F1分数
    y_pred = model.predict(X_test)

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
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {roc_auc:.4f}")

    print("\n")