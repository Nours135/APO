# 评估属性抽取

import re
import json
from functools import partial
from typing import Tuple, Union
from tqdm import tqdm

from utils.chatgpt_package import chatgpt_api
from prompEval.Estimator import BaseEstimator, UCBEstimator
from gradient import ProTeGi
from PromptTemplates.prompt_templates_pcs import prompt_template_cot, prompt_template_noncot, prompt_template_opti_res, PromptTemplate_pcs

from utils.debug import *
from utils.logger import return_logger


# logger 打印每一次迭代时获取到的 gradient
gradient_logger = return_logger('attrExtract_gradient', 'attrExtract_gradient.log', False, 'info', overwrite=True)
# logger 打印每一次迭代时 prompt 的 scores
scores_logger = return_logger('attrExtract_scores', 'attrExtract_scores.log', False, 'info', overwrite=True)



def parse_attr_extract_res(gpt_res) -> str:
    pass


def load_test_dataset():
    pass


def main():
    pass



def estimate_func(res: Union[str, str], gt: Union[str, str]) -> float:
    pass


def main_run_exp():
    gpt_func = partial(chatgpt_api, model_assign='gpt-4', base='self')
    test_data = None
    # classes = set([item['item_class'] for item, gt in test_data])
    # print(len(test_data))
    # import pdb; pdb.set_trace()
    
    prompt_templates = {
        '': None
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
        best_prompt, best_res, best_gt = estimator.load_prompt_results(best_prompt_name)
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






