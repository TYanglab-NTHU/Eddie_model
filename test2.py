import pandas as pd
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote
import time

def get_smiles(name, row_number):
    try:
        # 预处理名称
        print(f"Process name: {name}")
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles'
        smi = urlopen(url).read().decode('utf8')
        print(smi)
        return smi
    except Exception as e:
        print(f"Failed to get SMILES for row {row_number} (Ligand: {name})")
        failed_names.append(name)  # 将失败的名称添加到列表中
        return None

# 读取CSV文件
df = pd.read_csv('unique_ligands_processed.csv')

# 创建两个新的列表用于存储SMILES和失败的名称
smiles_list = []
failed_names = []

for index, row in df.iterrows():
    smiles = get_smiles(row['Processed Name'], index)
    smiles_list.append(smiles)
    if index % 10 == 0:  # 每处理10行输出一次进度
        print(f"Processed {index + 1} lines")
    time.sleep(0.2)  # 添加一个小延时以防止服务器过载

df['SMILES'] = smiles_list

# 将结果保存到一个新的CSV文件
df.to_csv('NIST_database_with_SMILES.csv', index=False)

# 如果有失败的名称，将它们保存到另一个CSV文件
if failed_names:
    df_failed = pd.DataFrame(failed_names, columns=['Failed Names'])
    df_failed.to_csv('failed_names.csv', index=False)
    print(f"Failed names saved to 'failed_names.csv'.")

# 显示前几行结果
print(df.head())
