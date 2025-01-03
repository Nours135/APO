from typing import Dict, Callable, Generator, Tuple, List
from collections import defaultdict, deque
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import random
from copy import deepcopy
import json


class BaseEstimator:
    def __init__(self, test_dataset: Tuple[object, str], estimate_func: Callable[[str, str], float], gpt_func: Callable[[str], str]):
        self.test_dataset = test_dataset
        self.estimate_func = estimate_func
        self.gpt_func = gpt_func

        self.prompt_valuation = defaultdict(list)
        self.prompt_name_2_detail_res = dict()

    def gpt_experiment(self, prompt: str, gt: str, prompt_name: str):
        res = self.gpt_func(prompt)
        evaluate_res = self.estimate_func(res, gt)
        if prompt_name not in self.prompt_name_2_detail_res:
            self.prompt_name_2_detail_res[prompt_name] = {
                'prompt': [], 'res': [], 'gt': [], 'evaluate_res': []
            }
        self.prompt_name_2_detail_res[prompt_name]['prompt'].append(prompt)
        self.prompt_name_2_detail_res[prompt_name]['res'].append(res)
        self.prompt_name_2_detail_res[prompt_name]['gt'].append(gt)
        self.prompt_name_2_detail_res[prompt_name]['evaluate_res'].append(evaluate_res)
        return res, evaluate_res

    def load_prompt_results(self, prompt_name: str) -> Tuple[List[str], List[str], List[str], List[float]]:
        return self.prompt_name_2_detail_res[prompt_name]['prompt'], self.prompt_name_2_detail_res[prompt_name]['res'], self.prompt_name_2_detail_res[prompt_name]['gt'], self.prompt_name_2_detail_res[prompt_name]['evaluate_res']
    
    def _evaluate_single_task(self, task):
        X, y, prompt_name, prompt_template = task
        prompt = prompt_template(X)
        res, evaluate_res = self.gpt_experiment(prompt, y, prompt_name)
        # if evaluate_res == 0:
        #     # pass
        #     # import pdb; pdb.set_trace()
        #     print(y)
        #     print(res)
        #     # print(prompt)
            # import pdb; pdb.set_trace()
        return prompt_name, evaluate_res

    def evaluate_prompts(self, prompt_templates: Dict[str, Callable[[object], str]], max_workers: int = 4):
        '''
        prompt_templates: a dict of prompt templates, the key is the prompt template name, the value is a function that takes a obnject (X) and returns a prompt template
        '''
        self.prompt_name_2_detail_res = dict()

        tasks = []
        for X, y in self.test_dataset:
            for prompt_name, prompt_template in prompt_templates.items():
                tasks.append((X, y, prompt_name, prompt_template))
        
        import random
        random.shuffle(tasks)
        if max_workers == 1:
            for i, task in enumerate(tqdm(tasks, desc="Processing")):
                prompt_name, evaluate_res = self._evaluate_single_task(task)
                self.prompt_valuation[prompt_name].append(evaluate_res)
                
                if (i + 1) % 100 == 0:
                    print(f"Step: {i + 1}")
                    self.read_out()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._evaluate_single_task, task) for task in tasks]
                
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing")):
                    prompt_name, evaluate_res = future.result()
                    self.prompt_valuation[prompt_name].append(evaluate_res)
                    
                    if (i + 1) % 100 == 0:
                        print(f"Step: {i + 1}")
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






class UCBEstimator(BaseEstimator):
    def __init__(self, test_dataset: Tuple[object, str], estimate_func: Callable[[str, str], float], gpt_func: Callable[[str], str], c: float = 1.0, tolerance: float = 0.001):
        '''
        test_dataset: 测试数据集
        estimate_func: 评估gpt_res and gt 的函数
        gpt_func: gpt 调用函数
        c: 探索参数，用于 UCB
        tolerance: 收敛条件，当最近 100 次 update 的平均 diff 小于 tolerance 时，认为收敛
        '''
        super().__init__(test_dataset, estimate_func, gpt_func)
        self.c = c  # Exploration parameter for UCB
        self.prompt_stats = defaultdict(lambda: {"mean": 0, "count": 0})
        self.total_count = 0
        self.prompt_templates = dict()

        # 收敛条件
        # self.previous_means = {name: 0 for name in self.prompt_stats.keys()}
        self.tolerance = tolerance  #
        self.update_diff = deque(maxlen=100)  # Only store last maxlen updates

        

    def _test_prompt_generator(self) -> Generator[str, None, None]:
        while True:
            prompt_name_2_upper_bound = {}
            for prompt_name, stats in self.prompt_stats.items():
                prompt_name_2_upper_bound[prompt_name] = stats['mean'] + self.c * np.sqrt(np.log(self.total_count) / stats['count'])
            yield max(prompt_name_2_upper_bound, key=prompt_name_2_upper_bound.get)


    def _update(self, prompt_name, reward):
        self.prompt_stats[prompt_name]['count'] += 1
        old_mean = self.prompt_stats[prompt_name]['mean']
        self.prompt_stats[prompt_name]['mean'] = ((self.prompt_stats[prompt_name]['count'] - 1) / self.prompt_stats[prompt_name]['count']) * self.prompt_stats[prompt_name]['mean'] + (1 / self.prompt_stats[prompt_name]['count']) * reward
        self.total_count += 1
        self.update_diff.append(abs(old_mean - self.prompt_stats[prompt_name]['mean']))


    def _check_convergence(self):
        '''检查最近 100 次 update 的平均 diff'''
        mean_diff = np.mean(self.update_diff)
        return mean_diff < self.tolerance
    
    def _evaluate_single_task(self, task):
        X, y, prompt_name, prompt_template = task
        prompt = prompt_template(X)
        res, evaluate_res = self.gpt_experiment(prompt, y, prompt_name)
        if evaluate_res == 0:
            pass
            # import pdb; pdb.set_trace()
            # print(y)
            # print(res)
            # print(prompt)
            # import pdb; pdb.set_trace()
        return prompt_name, evaluate_res
    
    def _task_generator(self, max_steps: int = 5) -> Generator[Tuple[object, str, str, Callable[[object], str]], None, None]:
        test_data_set = deepcopy(self.test_dataset)
        steps = 0
        while True:
            random.shuffle(test_data_set)
            max_i = len(test_data_set)
            for i, prompt_name in enumerate(self._test_prompt_generator()):
                if i >= max_i:
                    continue
                steps += 1
                if steps >= max_steps:
                    return
                X = test_data_set[i][0]
                y = test_data_set[i][1]
                yield X, y, prompt_name, self.prompt_templates[prompt_name]

    def add_prompt(self, prompt_name: str, one_peompt_template: Callable[[object], str]):
        self.prompt_templates[prompt_name] = one_peompt_template
        self.prompt_stats[prompt_name] = {"mean": 0, "count": 0}
    
    def add_prompts(self, prompts: Dict[str, Callable[[object], str]]):
        for prompt_name, one_peompt_template in prompts.items():
            self.add_prompt(prompt_name, one_peompt_template)

    def evaluate_prompts(self, max_workers: int = 4, max_steps: int = 500):
        self.prompt_name_2_detail_res = dict()  # clear out the old results

        if max_workers == 1:
            for i, task in enumerate(tqdm(self._task_generator(max_steps), desc="Processing", total=max_steps)):
                prompt_name, evaluate_res = self._evaluate_single_task(task)
                # import pdb; pdb.set_trace()
                self._update(prompt_name, evaluate_res)
                
                # if (i + 1) % 100 == 0:
                #     print(f"Step: {i + 1}")
                #     self.read_out()
        else:
            batch_size = 40
            task_generator = self._task_generator(max_steps)
            tasks = []
            batch_count = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pybar = tqdm(total=max_steps, desc="Processing")
                while True:
                    # 收集一批任务
                    for _ in range(batch_size):
                        try:
                            task = next(task_generator)
                            tasks.append(task)
                        except StopIteration:
                            break
                    
                    if not tasks:  # 如果没有新任务了，退出循环
                        break
                        
                    # 提交这一批任务
                    futures = [executor.submit(self._evaluate_single_task, task) for task in tasks]
                    
                    # 处理结果
                    for future in concurrent.futures.as_completed(futures):
                        pybar.update(1)
                        prompt_name, evaluate_res = future.result()
                        self._update(prompt_name, evaluate_res)
                    
                    batch_count += 1
                    # print(f"Completed batch {batch_count}")
        
                    # 在这里添加你的终止条件判断
                    # 例如：判断均值是否收敛
                    if self._check_convergence():  # 你需要实现这个方法
                        print("Convergence reached, stopping...")
                        break
                    
                    tasks = []  # 清空任务列表，准备下一批



    def read_out(self, logger):
        '''打印count，mean，upper bound'''
        for prompt_name, stats in self.prompt_stats.items():
            logger.info(f"Prompt: {prompt_name}, prompt_template: {str(self.prompt_templates[prompt_name])}\n"
                       f"explore count: {stats['count']}\n"
                       f"mean: {stats['mean']}\n" 
                       f"upper bound: {stats['mean'] + self.c * np.sqrt(np.log(self.total_count) / stats['count'])}\n\n")
        # log best
        best_prompt_name = self.get_best_prompt()
        logger.info(f"Best prompt: {best_prompt_name}, prompt_template: {str(self.prompt_templates[best_prompt_name])}\n"
                       f"explore count: {self.prompt_stats[best_prompt_name]['count']}\n"
                       f"mean: {self.prompt_stats[best_prompt_name]['mean']}\n" 
                       f"upper bound: {self.prompt_stats[best_prompt_name]['mean'] + self.c * np.sqrt(np.log(self.total_count) / self.prompt_stats[best_prompt_name]['count'])}\n\n")
        logger.info(json.dumps(self.prompt_stats, indent=4))

    def get_best_prompt(self) -> str:
        return max(self.prompt_stats, key=lambda x: self.prompt_stats[x]['mean'])