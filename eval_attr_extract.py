# 评估属性抽取

import re
import json
from functools import partial
from typing import Tuple, Union, List, Dict, Callable
from tqdm import tqdm
import Levenshtein
import pandas as pd
import argparse
import pickle

from utils.chatgpt_package import chatgpt_api
from prompEval.Estimator import BaseEstimator, UCBEstimator
from gradient import ProTeGi
from PromptTemplates.prompt_templates_attrExtract import PromptTemplate_attrExtract_allinone, PromptTemplate_attrExtract_single, all_in_one_prompt_template, all_in_one_prompt_template_cot

from utils.debug import *
from utils.logger import return_logger


# logger 打印每一次迭代时获取到的 gradient
gradient_logger = return_logger('attrExtract_gradient', 'attrExtract_gradient.log', False, 'info', overwrite=True)
# logger 打印每一次迭代时 prompt 的 scores
scores_logger = return_logger('attrExtract_scores', 'attrExtract_scores.log', False, 'info', overwrite=True)


def property_post_process(res: Dict[str, str]) -> Dict[str, str]:
    processed_res = dict()
    for key, value in res.items():
        value = value.strip().lower()
        if 'inches' not in value:
            if 'inch' in value:
                value = value.replace('inch', 'inches')
            if '"' in value:
                value = value.replace('"', 'inches')
            processed_res[key] = value
        elif 'grams' in value:
            processed_res[key] = value.replace('grams', 'g')
        elif 'gram' in value:
            processed_res[key] = value.replace('gram', 'g')
        
            
    return processed_res

def load_test_dataset() -> List[Tuple[Dict[str, object], Dict[str, str]]]:
    data_path = './data/jiafa/假发属性抽取数据打标.xlsx'
    with open('./data/jiafa/global_infor_str_stored_4_jiafa.pkl', 'rb') as f:
        loaded_all_info_str = pickle.load(f)
    # import pdb; pdb.set_trace()
    # 只打标了前100条数据
    df1 = pd.read_excel(data_path, sheet_name='接发_attrs_gpt_res_0217', dtype=str)
    df1 = df1.iloc[:100, :]
    df2 = pd.read_excel(data_path, sheet_name='头套_user_attr_gpt_res_0218', dtype=str)
    df2 = df2.iloc[:100, :]
    # df = pd.concat([df1, df2], ignore_index=True)
    total_res = []
    # import pdb; pdb.set_trace()
    for _, row in df1.iterrows():
        tmp = dict()
        sku_id = row['sku_id']
        if sku_id not in loaded_all_info_str:
            continue
        info_str = loaded_all_info_str[sku_id]
        for col in row.index:
            if col in ['sku_id', 'url', 'Brand', 'brand']:
                continue
            if 'Unnamed' in col:
                continue
            if pd.isna(row[col]):
                tmp[col] = 'unclear'
            else:
                tmp[col] = row[col]
        total_res.append(({'sku_id': sku_id, 'item_class': 'hair extensions', 'infomation': info_str}, property_post_process(tmp)))
    for _, row in df2.iterrows():
        tmp = dict()
        sku_id = row['sku_id']
        if sku_id not in loaded_all_info_str:
            continue
        info_str = loaded_all_info_str[sku_id]
        for col in row.index:
            if col in ['sku_id', 'url', 'Brand', 'brand']:
                continue
            if 'Unnamed' in col:
                continue
            if pd.isna(row[col]):
                tmp[col] = 'unclear'
            else:
                tmp[col] = row[col]
        total_res.append(({'sku_id': sku_id, 'item_class': 'wigs', 'infomation': info_str}, property_post_process(tmp)))
    return total_res


def edit_distance(text1: str, text2: str) -> float:
    edit_dist = Levenshtein.distance(text1.lower(), text2.lower())
    max_len = max(len(text1), len(text2))
    return edit_dist / max_len if max_len > 0 else 0.0

def safe_json_loads(text: str) -> Dict[str, str]:
    try:
        return json.loads(text)
    except:
        try:
            return json.loads(text.replace("'", '"'))
        except:
            try:
                return eval(text)
            except:
                return dict()

def estimate_func(gt: Dict[str, str], res: Union[str, Dict[str, str]], result_parse_func: Callable[[Union[str, List[str]]], Dict[str, str]]) -> float:
    '''
    暂时是写了一个等权的，编辑距离的 distance 计算，可以作为训练，但是评估的时候这个数值和准确率召回率不一致
    需要重新写过和准召回一样的评估函数
    '''
    if not isinstance(res, dict):
        res = result_parse_func(res)
        # import pdb; pdb.set_trace()  # DEBUG function
        # res = safe_json_loads(res)
        if len(res) == 0:
            return 0

    total_edit_dist = 0
    for key, value in gt.items():
        if key not in res:
            if value.lower() == 'unclear':  # 如果是 unclear，允许json不抽取出来
                total_edit_dist += 0
            else:
                total_edit_dist += 1
        elif str(res[key]).lower() == 'unclear' and str(value).lower() == 'unclear':
            total_edit_dist += 0    # 两个都是 unclear
        elif str(res[key]).lower() == 'unclear' or str(value).lower() == 'unclear':
            total_edit_dist += 1    # 一个是不清楚，一个不是不清楚，则编辑距离为 1
        else:
            total_edit_dist += edit_distance(str(res[key]), str(value))
    # import pdb; pdb.set_trace()  # DEBUG function
    return 1 - (total_edit_dist / len(gt))


def parse_attr_extract_res_single(gpt_res_list: List[str]) -> Dict[str, str]:
    res = dict()
    for gpt_res in gpt_res_list:
        parsed_single_res = gpt_res.split('|')
        if len(parsed_single_res) != 2:
            continue
        key = parsed_single_res[0].strip()
        value = parsed_single_res[1].strip()
        res[key] = value
    return res


def parse_attr_extract_res_allinone(gpt_res: str) -> Dict[str, str]:
    '''解析 all in one 类型的 gpt 抽取结果
    从 str 中解析 json 字符串
    '''
    # json_pattern = r'\{\s*(?:"[^"]*"\s*:\s*"[^"]*"\s*(?:,\s*)?)*\s*\}'
    json_pattern = r'\{.*\}'
    # 查找字符串中的JSON部分
    match = re.search(json_pattern, gpt_res, re.DOTALL)
    # import pdb; pdb.set_trace()  # DEBUG function
    if match:
        json_str = match.group(0)
        return safe_json_loads(json_str)
    else:
        return dict()


def gpt_func_single_property(prompts: List[str]) -> List[str]:
    '''如果是单个属性构造prompt的话
    templates 的 __call__ 方法应该返回的是一个 List[str]
    使用这个 gpt_func

    SOS，我感觉自己的代码也要开始屎山起来了，主要是一开始定的数据类型可扩展性不够强，其实如果能对齐template中需要自定义的callable的返回值类型，
    那么也可以避免这种问题。
    '''
    res = []
    for prompt in prompts:
        res.append(chatgpt_api(prompt, model_assign='gpt-4', base='self'))
    return res

gpt_func_total_property = partial(chatgpt_api, model_assign='gpt-4', base='self')

prompt_temp_11 = '''As a highly skilled expert in analyzing and extracting product attributes from text, your role is to identify and provide accurate attribute values based on provided definitions and product details.

Your Key Objectives:
1. **Comprehensive Analysis**: Thoroughly review and understand all product information supplied.
2. **Precise Matching**: Accurately align product information with the defined attributes by leveraging detailed context and examples provided.

Provided Information:
- **Item Class**: __item_class__
- **Property Definitions**: __property_def__
- **Product Information**: __info_str__

**Output Requirements**:
- Construct a JSON output that contains attribute-value pairs only if they demonstrate a clear alignment with definitions in the property list.

Example JSON Syntax:
```json
{
"attribute_1": "value_1",
"attribute_2": "value_2",
"attribute_3": "value_3"
}
```

**Guidelines**:
- **Contextual Sensitivity**: Be attentive to the contextual nuances of attribute definitions, especially when similar terms appear within product information. Ensure extracted attributes align unambiguously with the intended definitions, reflecting expert discernment in ambiguous scenarios.
- **Detail Orientation**: Provide a thorough yet concise extraction, considering potential synonyms or style variants as defined within property examples, ensuring the selected values precisely fit the attribute context.
- Unit should be included in the value.

*Note: This structured output should be created with careful attention to the alignment with provided definitions, avoiding assumptions not directly supported by data in the item description.*'''


prompt_temp_18 = '''As an AI specializing in extracting product attributes, your task is to extract attribute-value pairs that align directly with the provided attribute definitions and product descriptions.

Your Key Objectives:
1. **Data Extraction**: Scrutinize the provided descriptions to identify data that directly corresponds to the attribute definitions.
2. **Alignment Accuracy**: Ensure the extracted data matches the examples provided in the property definitions, prioritizing precision over inference.

Provided Information:
- **Item Class**: __item_class__
- **Property Definitions**: __property_def__
- **Product Information**: __info_str__

**Output Requirements**:
Your output should be a JSON format object representing the attribute-value pairs from the product information that explicitly match the attribute definitions.

Example JSON Syntax:
```json
{
"attribute_1": "value_1",
"attribute_2": "value_2",
"attribute_3": "value_3"
}
```

**Guidelines**:
- **Literal Extraction**: Focus on literal matches to property definitions. Use product information that directly aligns with provided examples or clearly fits the descriptions without assumptions or broad interpretations.
- **Synonyms and Variants**: Consider only synonyms and style variants explicitly covered by examples in the property definitions.
- **Avoid Overinterpretation**: Do not infer information not explicitly stated in the product details or supported directly by the provided definitions.
- Unit should be included in the value.

*This task requires adherence to attribute definitions without adding interpretations or assumptions that cannot be substantiated by the product text.*"'''

def main_run_exp(args):
    if args.allinone_prompt:
        gpt_func = gpt_func_total_property
        estimate_func_final = partial(estimate_func, result_parse_func=parse_attr_extract_res_allinone)
        PromptTemplate = PromptTemplate_attrExtract_allinone
        prompt_templates = {
            'all_in_one': PromptTemplate(template_str=all_in_one_prompt_template),
            'all_in_one_cot': PromptTemplate(template_str=all_in_one_prompt_template_cot),
            'prompt_11': PromptTemplate(template_str=prompt_temp_11),
            'prompt_18': PromptTemplate(template_str=prompt_temp_18)
        }
        
    else:
        gpt_func = gpt_func_single_property
        estimate_func_final = partial(estimate_func, result_parse_func=parse_attr_extract_res_single)
        PromptTemplate = PromptTemplate_attrExtract_single
        prompt_templates = {
            'single_property': None  # TODO
        }

    test_data = load_test_dataset()
    # import pdb; pdb.set_trace()
    # classes = set([item['item_class'] for item, gt in test_data])
    # print(len(test_data))
    # import pdb; pdb.set_trace()

    apo_args = {
        'errors_per_gradient': 4,
        'gradients_per_error': 1,
        'n_gradients': 4, # 运行几次获取 gradients 的代码
        'steps_per_gradient': 1,
    }

    estimator = UCBEstimator(test_data, estimate_func_final, gpt_func)
    optimizer = ProTeGi(apo_args, estimate_func_final, gradient_logger)
    estimator.add_prompts(prompt_templates)

    global_added_prompt_idx = 0
    for epoch in range(10):
        estimator.evaluate_prompts(max_workers=args.max_workers, max_steps=200)
        estimator.read_out(scores_logger)
        best_prompt_name = estimator.get_best_prompt()
        best_prompt_template = str(estimator.prompt_templates[best_prompt_name])
        best_prompt, best_res, best_gt = estimator.load_prompt_results(best_prompt_name)
        gradients = optimizer.get_gradients(best_prompt_template, best_prompt, best_gt, best_res)    # 得到 gradiant
        # import pdb; pdb.set_trace()
        new_task_sections = []
        for feedback, error_string in tqdm(gradients, desc='applying gradients'):
            tmp = optimizer.apply_gradient(
                best_prompt_template, error_string, feedback, optimizer.opt['steps_per_gradient'])
            new_task_sections += tmp
        for prompt_template in new_task_sections:
            estimator.add_prompt(f'_added_{global_added_prompt_idx}', PromptTemplate(prompt_template))
            global_added_prompt_idx += 1

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--allinone_prompt', action='store_true', default=False)
    parser.add_argument('--max_workers', type=int, default=7)
    args = parser.parse_args()
    main_run_exp(args)



    # python eval_attr_extract.py --allinone_prompt --max_workers 7



