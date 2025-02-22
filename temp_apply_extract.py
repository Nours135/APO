import pandas as pd
import pickle
from tqdm import tqdm
import os
import json


from PromptTemplates.prompt_templates_attrExtract import PromptTemplate_attrExtract_allinone
from eval_attr_extract import prompt_temp_18, chatgpt_api, parse_attr_extract_res_allinone
from utils.debug import *




def load_data_set(flag):
    if flag == 'hair extensions':
        data_df_path = '/autodl-fs/data/yuchen/attr_extract/manual_input/hair_extensions_sample4000_0217.csv'
    elif flag == 'wigs':
        data_df_path = '/autodl-fs/data/yuchen/attr_extract/manual_input/hair_extension_wigs_sample4000_0217.csv'
    data_df = pd.read_csv(data_df_path, header=0)
    sku_ids = data_df['sku_id'].tolist()[:1000]
    with open('data/jiafa/global_infor_str_stored_4_jiafa.pkl', 'rb') as fp:
        total_sku_2_infor_string = pickle.load(fp)
    res = {
        sku_id: total_sku_2_infor_string[sku_id] for sku_id in sku_ids
    }
    # import pdb; pdb.set_trace()
    return res


if __name__ == '__main__':
    prompt_template = PromptTemplate_attrExtract_allinone(template_str=prompt_temp_18)
    # output the result
    if not os.path.exists('data/jiafa/temp_output'):
        os.makedirs('data/jiafa/temp_output')

    for task in ['hair extensions', 'wigs']:
        total_sku_2_infor_string = load_data_set(task)
        total_res = []
        # c = 0
        for sku_id, infor_string in tqdm(total_sku_2_infor_string.items(), total=len(total_sku_2_infor_string), desc=f'Process {task}'):
            prompt = prompt_template(information={
                'sku_id': sku_id,
                'item_class': task,
                'infomation': infor_string
            })
            res = chatgpt_api(prompt, model_assign='gpt-4', base='self')
            parsed_res = parse_attr_extract_res_allinone(res)
            parsed_res['sku_id'] = sku_id
            # import pdb; pdb.set_trace()
            total_res.append(parsed_res)
            # c += 1
            # if c > 50:
            #     break   # DEBUG
        df = pd.DataFrame(total_res)
        task_name = task.replace(' ', '_')
        df.to_csv(f'data/jiafa/temp_output/{task_name}_res.csv', index=False)
