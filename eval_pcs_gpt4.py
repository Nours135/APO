import re
import json
from functools import partial


from utils.chatgpt_package import chatgpt_api
from prompEval.BaseEstimator import BaseEstimator
from utils.debug import *



def parse_pcs_res(gpt_res) -> str:
    if not isinstance(gpt_res, str):
        return 'not_predicted'
    raw_res = gpt_res.split('|')[-1]
    if 'unclear' in raw_res:
        return 'unclear'
    return re.sub(r'[^0-9]', '', raw_res) 


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
            import pdb; pdb.set_trace()
            continue
        test_dataset.append(({
            'item_class': item_class,
            'infomation': res
        }, gt))
    
    return test_dataset

def estimate_func(res: str, gt: str) -> int:
    parsed_res = parse_pcs_res(res)
    parsed_gt = parse_pcs_res(gt)

    if parsed_res == 'not_predicted':
        return 0
    if parsed_res == 'unclear':
        if parsed_gt == 'unclear' or parsed_gt == '1':
            return 1
        else:
            return 0
    if parsed_gt == 'unclear':
        if parsed_res == '1':
            return 1
        else:
            return 0
    return int(parsed_gt == parsed_res)



def main_run_exp():

    gpt_func = partial(chatgpt_api, model_assign='gpt-4', base='self')
    test_data = load_test_dataset()
    # classes = set([item['item_class'] for item, gt in test_data])
    # import pdb; pdb.set_trace()

    def prompt_template_1(information: object):
        item_class = information['item_class']
        info_str = information['infomation']
        prompt = f"The following is some product information about {item_class}. Please extract the number of products from this information. If the product information does not clearly specify the number of products, please mark the corresponding field as 'unclear.' Here is the product information and title: {info_str}.\n"
        prompt += (
        "\nTake a deep breath and analyze this problem step-by-step. "
        "Please explain every step of your reasoning process, focusing only on the most critical decision points. "
        "Finally, output the result in the format 'Number of Products|xxx,' including only the number. "
        "For example: 'Number of Products|2'."
        )
        return prompt
    
    prompt_templates = {
        'prompt_template_1': prompt_template_1
    }

    estimator = BaseEstimator(test_data, estimate_func, gpt_func)
    estimator.evaluate_prompts(prompt_templates)




if __name__ == "__main__":
    main_run_exp()