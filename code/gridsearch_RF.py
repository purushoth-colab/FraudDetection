import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import time
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


model = RandomForestClassifier(random_state=42);
model_name = 'RandomForest'


# 定义参数网格进行网格搜索
param_grid = {
    'n_estimators': [50, 100, 200],  # 可选的树的数量
    'max_depth': [None, 10, 20, 30],  # 最大树深度
    'min_samples_split': [2, 5, 10],  # 内部节点再划分所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点最少样本数
    'max_features': ['sqrt', 'log2'],  # 不同的特征选择策略
    'class_weight': [None, 'balanced', "balanced_subsample"],  # 处理不平衡数据集的类别权重
    'random_state': [42]  # 随机种子
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)

print(f"Training {model_name} model with parameter grid search...")

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# 输出最佳参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳参数的模型进行训练
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# 进行预测并计算F1分数
start_inference_time = time.time()
y_pred = best_model.predict(X_test)
end_inference_time = time.time()

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
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC: {roc_auc:.4f}")

# 输出训练和推理时间
training_time = end_time - start_time
inference_time = end_inference_time - start_inference_time
print(f"Training Time: {training_time:.2f} seconds")
print(f"Inference Time: {inference_time:.2f} seconds")
print("\n")
