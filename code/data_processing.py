import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score  # 导入roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 1. 读取CSV文件
data = pd.read_csv('/Users/shenyuanzhe/Desktop/creditcard.csv')

# 2. 删除包含缺失值的行
data.dropna(inplace=True)
print(data.columns)

# 2. 选择需要的列
X = data.iloc[:, 0:30]  # 选择V1到V28列
y = data['Class']       # 选择Class列作为目标分类

# 3. 标准化Amount列
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

# 4. 将所有列映射到-1到1
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
X = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)

# 将X_train和y_train合并为一个DataFrame，并保存指定的列
selected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

train_data = pd.DataFrame(data=X, columns=selected_columns)
train_data['Class'] = y

# 保存训练集和测试集为CSV文件
train_data.to_csv('train_data.csv', index=False)
test_data = pd.DataFrame(data=X, columns=selected_columns)
test_data['Class'] = y
test_data.to_csv('test_data.csv', index=False)

