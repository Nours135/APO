
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import sys
import os
from utils.chatgpt_package import chatgpt_api
import string

def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result


class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, logger, max_threads=1, bf_eval=None):
        '''
        args should contains:
            n_gradients: int, 
            errors_per_gradient: int, 
            gradients_per_error: int, 
            steps_per_gradient: int, 
            mc_samples_per_step: int, 
            max_expansion_factor: int, 
            reject_on_errors: bool, 
            minibatch_size: int, 
            eval_rounds: int, 
            eval_prompts_per_round: int, 
            samples_per_eval: int, 
            max_threads: int
        '''
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.logger = logger
        # self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs):
        pass

class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """
    def _sample_error_str(self, texts, labels, preds, n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            # 在我的框架内，有些评估函数的返回值是 0 - 1 的浮点数，0表示极端的错误，在很多时候是因为数据缺失，导致没有价值！ 
            if self.evaluator_fn(l, p) < 0.5:  
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]
        error_string = ''
        num_errors = 0
        error_idx = 0
        for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f'## Example {error_idx+1}\n'
            error_string += f'Text: \'\'\'{t.strip()}\'\'\'\nLabel: \'\'\'{l}\'\'\'\nPrediction: \'\'\'{p}\'\'\'\n\n'
            error_idx += 1
        # import pdb; pdb.set_trace()
        self.logger.info(f"error_string: {error_string}")
        return error_string.strip()

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
        """ Get "gradients" for a prompt based on the error string."""
        gradient_prompt = f"""
        I'm trying to write a zero-shot classifier prompt.
    
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.
        Wrap each reason with <START> and <END>
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        res = [chatgpt_api(gradient_prompt, model_assign='gpt-4') for i in range(n)]   # , reload=True 这个开发的有点问题，然后需要 reload 的情况很少
        feedbacks = []
        new_prompts = []
        for r in res:    
            feedbacks += self.parse_tagged_text(r, "<START>", "<END>")
            self.logger.info(f"feedbacks: {feedbacks}")
        return feedbacks

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = f"""
        I'm trying to write a zero-shot classifier.
        
        My current prompt is:
        "{prompt}"

        But it gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.
        Make sure to keep any content wrapped by '__' as they are placeholders for instance information.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        res = [chatgpt_api(transformation_prompt, model_assign='gpt-4') for i in range(n)]  # , reload=True
        new_prompts = []
        for r in res:   
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
            self.logger.info(f"one new_prompt: {new_prompts}")
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = chatgpt_api(rewriter_prompt, model_assign='gpt-4')
        new_instructions = [x for x in new_instructions if x]
        return new_instructions

    def get_gradients(self, prompt, texts, labels, preds):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(
                texts, labels, preds, n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(
                prompt, error_string, self.opt['gradients_per_error'], n=1)
            prompt_feedbacks += [(t, error_string) for t in gradients]
        # import pdb; pdb.set_trace()
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, gpt4, train_exs):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=self.opt['minibatch_size'])

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()

            # evaluate prompt on minibatch
            _, texts, labels, preds = task.evaluate(gpt4, prompt, minibatch)

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                gradients = self.get_gradients(prompt, task_section, task, gpt4, texts, labels, preds)
                new_task_sections = []
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(
                        task_section, error_string, feedback, self.opt['steps_per_gradient'])
                    new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(
                        sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections)) # dedup
            tmp_new_prompts = [
                prompt.replace(task_section, tmp) 
                for tmp in new_sections
            ]
            
            # filter a little
            # if len(new_sections) > self.opt['max_expansion_factor']:
                # if self.opt['reject_on_errors']:
                #     error_exs = []
                #     for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                #         if l != p:
                #             error_exs.append({'text': t, 'label': l})
                #     error_exs = random.sample(error_exs, min(len(error_exs), 16))

                #     # speed up a little
                #     tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                #     error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads)
                #     tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                # else:
                    # tmp_new_prompts = random.sample(tmp_new_prompts, 
                    #     k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        new_prompts = list(set(new_prompts)) # dedup

        return new_prompts

    # def score_candidates(self, prompts, task, gpt4, train_exs):
    #     """ Score a list of prompts."""
    #     if len(prompts) == 1:
    #         return [1.0]

    #     evals = self.evaluator_fn(
    #         prompts, train_exs, task, gpt4,
    #         scorer=self.scorer,
    #         rounds=self.opt['eval_rounds'],
    #         num_prompts_per_round=self.opt['eval_prompts_per_round'],
    #         samples_per_eval=self.opt['samples_per_eval'],
    #         max_threads=self.max_threads
    #     )
    #     return evals

