import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle
import os
import re
import json

# Suspension Strut and Coil Spring Assembly
ITEM_CLASS_PATTERN = r'The following is product information about (.+?). Your task is to extract'
ITEM_INFOMATION_PATTERN = r'likely that the total number is multiple, mark the field as 2.\nIf it is impossible to determine the total number, mark the field as \'unclear\'.\n Here is the product information and title([\s\S]*)Take a deep breath and analyze this problem step-by-step. Please explain every step'





def load_annotated_data() -> Dict[str, int]:
    res_d = dict()
    df_1 = pd.read_excel('./data/减震器总成_pcs_1231.xlsx', sheet_name='new_model')
    # df_2 = pd.read_excel('./data/减震器总成_pcs_1231.xlsx', sheet_name='new_model_2')
    for idx, row in tqdm(df_1.iterrows(), total=len(df_1)):
        anno_label = row['rightness']
        # import pdb; pdb.set_trace()
        if pd.isna(anno_label):
            continue
        sku = row['item_number']
        if anno_label == '1' or anno_label == 1:
            res_d[sku] = int(row['notes_pcs'])
        elif anno_label == '0' or anno_label == 0:
            truth = row['truth']
            if not pd.isna(truth):
                res_d[sku] = int(truth)
        else:
            pass
    
    df_2 = pd.read_excel('./data/减震器总成_pcs_1231.xlsx', sheet_name='baseline')
    for idx, row in tqdm(df_2.iterrows(), total=len(df_2)):
        anno_label = row['rightness']
        # import pdb; pdb.set_trace()
        if pd.isna(anno_label):
            continue
        sku = row['item_number']
        if anno_label == '1' or anno_label == 1:
            res_d[sku] = int(row['notes_pcs'])
        elif anno_label == '0' or anno_label == 0:
            truth = row['truth']
            if not pd.isna(truth):
                res_d[sku] = int(truth)
        else:
            pass

    return res_d


def load_sku_2_info(sku_2_pcs: Dict[str, int]) -> Dict[str, Tuple[str, str]]:
    '''
    能够 load 出 sku 相关的 pcs 的 info
    '''
    skus = set(sku_2_pcs.keys())
    res_d = dict()
    df = pd.read_csv('./data/减震器总成_pcs_all.csv', header=0)
    # import pdb; pdb.set_trace()
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sku = row['sku_id']
        if sku in skus:
            prompt = row['prompt']
            # import pdb; pdb.set_trace()
            item_class_match = re.findall(ITEM_CLASS_PATTERN, prompt)
            if item_class_match:
                item_class = item_class_match[0]
            else:
                item_class = 'unknown'
            item_info_match = re.findall(ITEM_INFOMATION_PATTERN, prompt, re.DOTALL)
            if item_info_match:
                item_info = item_info_match[0]
            else:
                item_info = 'unknown'
            # import pdb; pdb.set_trace()
            res_d[sku] = (item_class, item_info, sku_2_pcs[sku])

        # import pdb; pdb.set_trace()
    return res_d


if __name__ == '__main__':
    # res_d = load_annotated_data()
    with open('./temp.pkl', 'rb') as f:
        res_d = pickle.load(f)
    print(len(res_d))
    res_d_l = list(res_d.keys())
    # import pdb; pdb.set_trace()
    res_info_d = load_sku_2_info(res_d)
    res_info_d_l = list(res_info_d.keys())
    print(len(res_info_d))
    print(res_info_d[res_info_d_l[0]])
    with open('./data/pcs_info.json', 'w') as f:
        json.dump(res_info_d, f, indent=4)
    
    import pdb; pdb.set_trace()
