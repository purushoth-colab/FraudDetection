import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import xgboost as xgb

# 设置全局随机数种子
np.random.seed(42)

# 1. 读取CSV文件
data_train = pd.read_csv('/Users/shenyuanzhe/Desktop/train_data.csv')
data_test = pd.read_csv('/Users/shenyuanzhe/Desktop/test_data.csv')

# 3. 选择需要的列
X_train = data_train.iloc[:, 1:30]
X_test = data_test.iloc[:, 1:30]

y_train = data_train['Class']
y_train = np.array(y_train)

y_test = data_test['Class']
y_test = np.array(y_test)


# 1. 读取CSV文件
data = pd.read_csv('/Users/shenyuanzhe/Desktop/test_data.csv')

# 3. 选择需要的列
X = data.iloc[:, 1:30]  # 选择V1到V212列

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(29, 22),
            nn.ReLU(),
            nn.Linear(22, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 22),
            nn.ReLU(),
            nn.Linear(22, 29),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder()
criterion = nn.MSELoss()

# 11. 加载模型并进行推理
model.load_state_dict(torch.load('best_autoencoder_model_22_16.pth'))
model.eval()

with torch.no_grad():
    X_train = torch.Tensor(X_train.values)
    X_train = model.encoder(X_train)
    X_test = torch.Tensor(X_test.values)
    X_test = model.encoder(X_test)
    


model = xgb.XGBClassifier()
model_name = 'XGBoost'

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


