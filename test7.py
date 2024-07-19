import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm

def process_data():
    # 初始化變數
    if not os.path.exists('data_prepared.pkl'):
        df = pd.read_csv('NIST_database_cleaned_with_SMILES_filtered.csv')
        df['Value'] = df['Value'].str.replace(r'\(', '', regex=True).str.replace(r'\)', '', regex=True).astype(float)

        value_type_encoder = OneHotEncoder()
        encoded_value_types = value_type_encoder.fit_transform(df[['Value type']]).toarray()

        equilibrium_encoder = OneHotEncoder()
        encoded_equilibrium = equilibrium_encoder.fit_transform(df[['Equilibrium']]).toarray()

        scalers = {}
        processed_values = []
        for value_type in ['Log K', 'DH (kJ/mol)', 'DS (J/mol.K)']:
            scaler = StandardScaler()
            values = scaler.fit_transform(df[df['Value type'] == value_type]['Value'].values.reshape(-1, 1)).flatten()
            scalers[value_type] = scaler
            df.loc[df['Value type'] == value_type, 'Value'] = values
            processed_values.append(values)

        def smiles_to_vector(smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                    fp = mfpgen.GetFingerprint(mol)
                    return list(fp)
                else:
                    return [0] * 2048
            except Exception as e:
                print(f"Error processing SMILES: {smiles} with error {e}")
                return [0] * 2048


        smiles_vectors = np.array(df['SMILES'].apply(smiles_to_vector).tolist())

        X = np.hstack((encoded_value_types, encoded_equilibrium, smiles_vectors))
        y = df['Value'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with open('data_prepared.pkl', 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers), f)
        print("Data saved to file")

    else:
        with open('data_prepared.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers = pickle.load(f)
        print("Data loaded from file")

    return X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder

X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder = process_data()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
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

value_type_shape = value_type_encoder.categories_[0].size
equilibrium_shape = equilibrium_encoder.categories_[0].size
smiles_shape = 2048

model = MyModel(value_type_shape, equilibrium_shape, smiles_shape)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
print("Model created")

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)


def batch_convert_to_tensor(numpy_array, batch_size=10000):
    tensor_list = []
    for i in range(0, numpy_array.shape[0], batch_size):
        batch_tensor = torch.tensor(numpy_array[i:i + batch_size], dtype=torch.float32)
        tensor_list.append(batch_tensor)
    return torch.cat(tensor_list)


try:
    X_train_tensor = batch_convert_to_tensor(X_train)
    y_train_tensor = batch_convert_to_tensor(y_train)

except Exception as e:
    print(f"Error converting to Tensor: {e}")
    exit(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
print(X_test_tensor.shape, y_test_tensor.shape)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("Start training")
best_loss = float('inf')
for epoch in range(50):
    with tqdm.tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")

            X_batch, y_batch = batch
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=loss.item())
            # 可以选择在这里加入模型保存的逻辑，比如保存最佳模型
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'best_model_state_dict.pth')

print("Training finished")
# 训练完成后保存模型
torch.save(model.state_dict(), 'final_model_state_dict.pth')
print("Model saved")
