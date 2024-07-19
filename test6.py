import pandas as pd
import re
from mendeleev import element
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 導入tqdm

df = pd.read_csv('NIST_database_normalized.csv', dtype={5: str, 6: str})

# 新增兩個欄位用於儲存Ionization Energies和Ionic Radius
df['Ionization Energies'] = None
df['Ionic Radius'] = None

def greek_to_arabic(greek_num):
    greek_arabic_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8}
    return greek_arabic_map.get(greek_num, None)

def process_row(index, row):
    metal_ion = row['Metal ion']
    match = re.match(r'([A-Za-z]+)([\d\+\-²³⁴]*)', metal_ion)
    if match:
        element_symbol = match.group(1)
        charge = match.group(2).replace('⁺', '').replace('²', '2').replace('³', '3').replace('⁴', '4')
        if charge == '':
            charge = '1'
        
        try:
            el = element(element_symbol)
            charge_arabic = int(charge)
            
            selected_radius = None
            for radius in el.ionic_radii:
                if radius.charge == charge_arabic and greek_to_arabic(radius.coordination) == charge_arabic:
                    selected_radius = radius.ionic_radius
                    break
            
            return index, str(el.ionenergies), selected_radius
        except ValueError:
            return index, None, None
    return index, None, None

def update_df(results):
    for result in results:
        index, ionization_energies, ionic_radius = result
        df.at[index, 'Ionization Energies'] = ionization_energies
        df.at[index, 'Ionic Radius'] = ionic_radius

# 使用ThreadPoolExecutor來平行化處理
with ThreadPoolExecutor(max_workers=10) as executor:
    # 使用tqdm來顯示提交任務的進度條
    futures = [executor.submit(process_row, index, row) for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Submitting tasks")]
    # 使用tqdm來顯示任務完成的進度條
    results = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks")]
    update_df(results)

# 儲存修改後的DataFrame到新的CSV文件
df.to_csv('NIST_database_normalized_updated2.csv', index=False)