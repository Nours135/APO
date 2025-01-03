import re
import json
from functools import partial
from typing import Tuple, Union
from tqdm import tqdm

from utils.chatgpt_package import chatgpt_api
from prompEval.Estimator import BaseEstimator, UCBEstimator
from gradient import ProTeGi
from PromptTemplates.prompt_templates_pcs import prompt_template_cot, prompt_template_noncot, PromptTemplate_pcs

from utils.debug import *
from utils.logger import return_logger

# logger 打印每一次迭代时获取到的 gradient
gradient_logger = return_logger('pcs_gradient', 'pcs_gradient.log', False, 'info', overwrite=True)
# logger 打印每一次迭代时 prompt 的 scores
scores_logger = return_logger('pcs_scores', 'pcs_scores.log', False, 'info', overwrite=True)

def parse_pcs_res(gpt_res) -> str:
    if not isinstance(gpt_res, str):
        return 'not_predicted'
    raw_res = gpt_res.split('|')[-1]
    if 'unclear' in raw_res.lower():
        return 'unclear'
    match = re.search(r'\d+', raw_res)
    return match.group() if match else 'unclear'


def load_test_dataset():
    data_path = './data/gpt4_res_4_pcs_fintune_standard.json'
    with open(data_path, 'r') as f:
        data = json.load(f)

    parse_pattern = "'unclear.' Product Information:"
    pattern = r'The following is some product information about (.*?)\. Please extract'
    
    test_dataset = []
    for item in data:
        res = item['instruction'].split(parse_pattern)[-1]
        gt = item['output']
        match_res = re.match(pattern, item['instruction'])
        if match_res is None:
            # import pdb; pdb.set_trace()
            continue
        item_class = match_res.group(1)
        # import pdb; pdb.set_trace()
        if len(res) == 0:
            # import pdb; pdb.set_trace()
            continue
        test_dataset.append(({
            'item_class': item_class,
            'infomation': res
        }, gt))
    
    return test_dataset

def load_test_dataset_anno():
    data_path = './data/pcs_info.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    processed_data = []
    # import pdb; pdb.set_trace()
    for sku_id, item in data.items():
        item_class = item[0]
        infomation = item[1]
        gt = item[2]
        processed_data.append(({
            'item_class': item_class,
            'infomation': infomation
        }, gt))
    return processed_data

def estimate_func(res: Union[str, int], gt: Union[str, int]) -> int:
    parsed_res = parse_pcs_res(res)
    if isinstance(gt, str):
        parsed_gt = parse_pcs_res(gt)
    else:
        parsed_gt = str(gt)
    # import pdb; pdb.set_trace()
    if parsed_res == 'not_predicted':
        return 0
    if parsed_res == 'unclear':
        return 0
        # if parsed_gt == 'unclear' or parsed_gt == '1':
        #     return 1
        # else:
        #     return 0
    if parsed_gt == 'unclear':
        if parsed_res == '1':
            return 1
        else:
            return 0
    return int(parsed_gt == parsed_res)



def main_run_exp():

    gpt_func = partial(chatgpt_api, model_assign='gpt-4', base='self')
    test_data = load_test_dataset_anno()
    # classes = set([item['item_class'] for item, gt in test_data])
    # import pdb; pdb.set_trace()
    
    prompt_templates = {
        'prompt_template_cot': prompt_template_cot,
        'prompt_template_noncot': prompt_template_noncot
    }

    args = {
        'errors_per_gradient': 4,
        'gradients_per_error': 1,
        'n_gradients': 4, # 运行几次获取 gradients 的代码
        'steps_per_gradient': 1,
    }

    estimator = UCBEstimator(test_data, estimate_func, gpt_func)
    optimizer = ProTeGi(args, estimate_func, gradient_logger)
    estimator.add_prompts(prompt_templates)

    global_added_prompt_idx = 0
    for epoch in range(10):
        estimator.evaluate_prompts(max_workers=7, max_steps=700)
        estimator.read_out(scores_logger)
        best_prompt_name = estimator.get_best_prompt()
        best_prompt_template = str(estimator.prompt_templates[best_prompt_name])
        best_prompt, best_res, best_gt, best_evaluate_res = estimator.load_prompt_results(best_prompt_name)
        gradients = optimizer.get_gradients(best_prompt_template, best_prompt, best_gt, best_res)    # 得到 gradiant
        # import pdb; pdb.set_trace()
        new_task_sections = []
        for feedback, error_string in tqdm(gradients, desc='applying gradients'):
            tmp = optimizer.apply_gradient(
                best_prompt_template, error_string, feedback, optimizer.opt['steps_per_gradient'])
            new_task_sections += tmp
        for prompt_template in new_task_sections:
            estimator.add_prompt(f'_added_{global_added_prompt_idx}', PromptTemplate_pcs(prompt_template))
            global_added_prompt_idx += 1

        
if __name__ == "__main__":

    main_run_exp()