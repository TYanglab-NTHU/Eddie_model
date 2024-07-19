import pandas as pd
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_smiles(name, row_number):
    try:
        processed_name = name
        print(f"Process name: {processed_name}")
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(processed_name) + '/smiles'
        smi = urlopen(url).read().decode('utf8')
        print(smi)
        return smi
    except Exception as e:
        print(f"Failed to get SMILES for row {row_number} (Ligand: {name})")
        return None

def fetch_smiles(data):
    index, row = data
    return get_smiles(row['Processed Name'], index), index

# 读取CSV文件
df = pd.read_csv('unique_ligands_processed.csv')

# 使用 ThreadPoolExecutor 并行获取 SMILES
smiles_results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_smiles, item) for item in df.iterrows()]
    for future in as_completed(futures):
        smiles, index = future.result()
        smiles_results.append((index, smiles))
        if index % 10 == 0:
            print(f"Processed {index + 1} lines")

# 排序并保存结果
smiles_results.sort()
df['SMILES'] = [smiles for _, smiles in smiles_results]

# 将结果保存到一个新的CSV文件
df.to_csv('NIST_database_with_SMILES.csv', index=False)

# 显示前几行结果
print(df.head())
