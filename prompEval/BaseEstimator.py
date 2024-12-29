

from typing import Dict, Callable
from collections import defaultdict
from tqdm import tqdm
import numpy as np



class BaseEstimator:
    def __init__(self, test_dataset: Dict[object, str], estimate_func: Callable[[str, str], float], gpt_func: Callable[[str], str]):
        self.test_dataset = test_dataset
        self.estimate_func = estimate_func
        self.gpt_func = gpt_func

        self.prompt_valuation = defaultdict(list)

    def gpt_experiment(self, prompt: str, gt: str):
        res = self.gpt_func(prompt)
        evaluate_res = self.estimate_func(res, gt)
        return res, evaluate_res
    
    def evaluate_prompts(self, prompt_templates: Dict[str, Callable[[object], str]]):
        '''
        prompt_templates: a dict of prompt templates, the key is the prompt template name, the value is a function that takes a obnject (X) and returns a prompt template
        '''
        step = 0
        for X, y in tqdm(self.test_dataset):
            for prompt_name, prompt_template in prompt_templates.items():        
                prompt = prompt_template(X)
                res, evaluate_res = self.gpt_experiment(prompt, y)
                self.prompt_valuation[prompt_name].append(evaluate_res)
                step += 1
                if step % 100 == 0:
                    print(f"Step: {step}")
                    self.read_out()




    def read_out(self):
        for prompt_name, prompt_valuation in self.prompt_valuation.items():
            print(f"Prompt: {prompt_name}")
            print(f"Mean: {np.mean(prompt_valuation)}")
            print(f"Std: {np.std(prompt_valuation)}")
            print(f"Median: {np.median(prompt_valuation)}")
            print(f"Min: {np.min(prompt_valuation)}")
            print(f"Max: {np.max(prompt_valuation)}")
            print("\n")
