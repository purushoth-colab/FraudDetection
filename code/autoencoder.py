import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. 读取CSV文件
data = pd.read_csv('/Users/shenyuanzhe/Desktop/train_data.csv')

# 3. 选择需要的列
X = data.iloc[:, 1:30]  # 选择V1到V28列

# 5. 自定义数据集类
class CreditCardDataset(Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 6. 创建数据加载器
batch_size = 9192
train_dataset = CreditCardDataset(X)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 7. 创建Autoencoder模型

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

# 8. 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 9. 训练Autoencoder
num_epochs = 100

best_model = None
best_loss = float('inf')

for epoch in range(num_epochs):
    train_losses = []
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # 输出每个epoch的平均损失
    print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {np.mean(train_losses):.6f}')
    
    # 如果当前模型性能更好，则保存该模型
    if np.mean(train_losses) < best_loss:
        best_loss = np.mean(train_losses)
        best_model = model.state_dict()

# 10. 保存最佳的模型
torch.save(best_model, 'best_autoencoder_model_22_16.pth')
