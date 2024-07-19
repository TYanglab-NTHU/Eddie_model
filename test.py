import pandas as pd
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote
import time
import re

def preprocess_name(name):
    # 使用正则表达式查找括号内的内容
    match = re.search(r'\(([^)]+)\)', name)
    if match:
        # 如果找到括号内的内容，返回括号内的部分
        return match.group(1)
    else:
        # 如果没有括号，返回整个名称
        return name

def get_smiles(name, row_number):
    try:
        # 预处理名称
        processed_name = preprocess_name(name)
        print(f"Process name: {processed_name}")
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(processed_name) + '/smiles'
        smi = urlopen(url).read().decode('utf8')
        return smi
    except Exception as e:
        print(f"Failed to get SMILES for row {row_number} (Ligand: {name})")
        return None

# 读取CSV文件
df = pd.read_csv('unique_ligands_processed.csv')

# 创建一个新的列用于存储SMILES
smiles_list = []
for index, row in df.iterrows():
    smiles = get_smiles(row['Processed Name'], index)
    print(smiles)
    smiles_list.append(smiles)
    if index % 10 == 0:  # 每处理10行输出一次进度
        print(f"Processed {index + 1} lines")
    time.sleep(0.1)  # 添加一个小延时以防止服务器过载

df['SMILES'] = smiles_list

# 将结果保存到一个新的CSV文件
df.to_csv('NIST_database_with_SMILES.csv', index=False)

# 显示前几行结果
print(df.head())
