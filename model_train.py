import logging
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
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO)

class MyModel(nn.Module):
    def __init__(self, value_type_shape, equilibrium_shape, smiles_shape):
        super(MyModel, self).__init__()
        
        self.value_type_branch = nn.Sequential(
            nn.Linear(value_type_shape, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.equilibrium_branch = nn.Sequential(
            nn.Linear(equilibrium_shape, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.smiles_branch = nn.Sequential(
            nn.Linear(smiles_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.combined_layers = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(32, 1)
        
    def forward(self, inputs):
        value_type_input = inputs[:, :self.value_type_branch[0].in_features]
        equilibrium_input = inputs[:, self.value_type_branch[0].in_features:self.value_type_branch[0].in_features+self.equilibrium_branch[0].in_features]
        smiles_input = inputs[:, -self.smiles_branch[0].in_features:]
        
        x1 = self.value_type_branch(value_type_input)
        x2 = self.equilibrium_branch(equilibrium_input)
        x3 = self.smiles_branch(smiles_input)
        
        combined = torch.cat((x1, x2, x3), dim=1)
        x = self.combined_layers(combined)
        
        return self.output_layer(x)

def process_data():
    if not os.path.exists('data_prepared.pkl'):
        try:
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
            logging.info("Data saved to file successfully")
        except Exception as e:
            logging.error(f"Error during data processing and saving: {e}")
            raise
    else:
        try:
            with open('data_prepared.pkl', 'rb') as f:
                data = pickle.load(f)
            if len(data) != 7:
                raise ValueError("Loaded data does not have the expected number of elements")
            X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers = data
            logging.info("Data loaded from file successfully")
        except EOFError:
            logging.error("The data file is empty or corrupted. Will recreate the file.")
            os.remove('data_prepared.pkl')
            return process_data()
        except Exception as e:
            logging.error(f"Error loading data from file: {e}")
            raise

    return X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers

def analyze_feature_importance(X_train, y_train, sample_size=10000):
    print("Starting feature importance analysis...")
    
    # 減少樣本數量
    if X_train.shape[0] > sample_size:
        indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
        X_sample, y_sample = X_train[indices], y_train[indices]
    else:
        X_sample, y_sample = X_train, y_train
    
    print(f"Using {X_sample.shape[0]} samples for analysis")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_sample, y_sample)
    
    print("Calculating feature importances...")
    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({'feature': range(X_sample.shape[1]), 'importance': importances})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Calculating permutation importance...")
    perm_importance = permutation_importance(rf_model, X_sample, y_sample, n_repeats=10, random_state=42, n_jobs=-1)
    perm_importance_df = pd.DataFrame({
        'feature': range(X_sample.shape[1]),
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    plt.bar(perm_importance_df['feature'], perm_importance_df['importance'])
    plt.title('Top 20 Permutation Importances')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('permutation_importance.png')
    plt.close()

    print("Feature importance analysis completed.")

def train_model():
    X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers = process_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 分析特徵重要性
    analyze_feature_importance(X_train, y_train, sample_size=10000)

    value_type_shape = value_type_encoder.categories_[0].size
    equilibrium_shape = equilibrium_encoder.categories_[0].size
    smiles_shape = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MyModel(value_type_shape, equilibrium_shape, smiles_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = nn.MSELoss()

    print("Model created")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    patience, min_delta = 10, 0.001
    print("Start training")
    best_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(100):
        model.train()
        train_losses = []
        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            for X_batch, y_batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                val_loss = loss_fn(preds, y_batch.unsqueeze(1))
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_state_dict.pth')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        scheduler.step(avg_val_loss)

    print("Training finished")

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.close()

if __name__ == '__main__':
    train_model()