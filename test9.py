import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pickle

class MyModel(nn.Module):
    def __init__(self, value_type_shape, equilibrium_shape, smiles_shape):
        super(MyModel, self).__init__()
        self.dense1 = nn.Linear(value_type_shape, 32)
        self.dense2 = nn.Linear(equilibrium_shape, 32)
        self.dense3 = nn.Linear(smiles_shape, 64)
        self.concat_dense = nn.Linear(128, 64)
        self.output_dense = nn.Linear(64, 1)
    
    def forward(self, inputs):
        value_type_input, equilibrium_input, smiles_input = inputs[:, :value_type_shape], inputs[:, value_type_shape:value_type_shape+equilibrium_shape], inputs[:, -2048:]
        x1 = torch.relu(self.dense1(value_type_input))
        x2 = torch.relu(self.dense2(equilibrium_input))
        x3 = torch.relu(self.dense3(smiles_input))
        x = torch.cat((x1, x2, x3), dim=1)
        x = torch.relu(self.concat_dense(x))
        output = self.output_dense(x)
        return output


print("Starting evaluation")

# 假设你已经知道输入层的维度
value_type_shape = 3  # 举例，实际数值应与训练模型时相同
equilibrium_shape = 684  # 举例
smiles_shape = 2048    # 化学指纹长度

model = MyModel(value_type_shape, equilibrium_shape, smiles_shape)
model.load_state_dict(torch.load('final_model_state_dict.pth'))
model.eval()  # 设置为评估模式

try:
    with open('data_prepared.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
except Exception as e:
    print("Failed to write pickle file:", e)
else:
    print("Pickle file written successfully")


X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
with torch.no_grad():  # 关闭梯度计算
    predictions = model(X_test_tensor)
    predictions = predictions.squeeze()  # 移除额外的维度

# 计算均方误差
mse = mean_squared_error(y_test_tensor.numpy(), predictions.numpy())
print(f"Mean Squared Error: {mse}")
