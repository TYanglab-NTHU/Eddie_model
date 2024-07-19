import pandas as pd
from rdkit import Chem
from urllib.request import urlopen
from urllib.parse import quote
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_smiles(name, row_number):
    try:
        print(f"Process name: {name}")
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles'
        smi = urlopen(url).read().decode('utf8')
        print(smi)
        time.sleep(0.5)  # 每个请求后添加小延时
        return smi
    except Exception as e:
        print(f"Failed to get SMILES for row {row_number} (Ligand: {name})")
        return None

def fetch_smiles(data):
    index, row = data
    return get_smiles(row['Processed Name'], index), index

# 读取CSV文件
df = pd.read_csv('unique_ligands_processed.csv', nrows=50)

# 创建两个新的列表用于存储SMILES和失败的名称
smiles_results = []
failed_names = []

# 使用 ThreadPoolExecutor 并行获取 SMILES
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(fetch_smiles, (index, row)) for index, row in df.iterrows()]
    for future in as_completed(futures):
        smiles, index = future.result()
        if smiles is None:
            failed_names.append(df.iloc[index]['Processed Name'])
        smiles_results.append((index, smiles))
        if index % 10 == 0:
            print(f"Processed {index + 1} lines")

# 排序并保存结果
smiles_results.sort()
df['SMILES'] = [smiles for _, smiles in smiles_results]

# 将结果保存到一个新的CSV文件
df.to_csv('NIST_database_with_SMILES.csv', index=False)

# 如果有失败的名称，将它们保存到另一个CSV文件
if failed_names:
    df_failed = pd.DataFrame(failed_names, columns=['Failed Names'])
    df_failed.to_csv('failed_names.csv', index=False)
    print(f"Failed names saved to 'failed_names.csv'.")

# 显示前几行结果
print(df.head())
