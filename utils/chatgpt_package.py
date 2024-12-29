import pandas as pd
import openai
import multiprocessing
import time
import json
import math
import os
import logging
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, as_completed
from collections import defaultdict
import traceback
import sys
from billiard import Pool, Process
import requests

# get root path
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


from utils.cacher import CacheDecorator
cache_path = os.path.join(root_path, 'utils', 'cached')
cacheDecorator = CacheDecorator(cache_path)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

#################  llama3.1-70b vllm settings   #################
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://region-42.seetacloud.com:20634/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def openai_style_api(content, model_name, max_new_tokens=2048):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": content}
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
        stream=False
    )
    return completion.choices[0].message.content

####################################################################


model_date = '20231213'
root_path = '/root/autodl-fs/b2c_server/online/'

def read_file(file, sheet=None):  # 读取文件
    form = file.split(".")[-1]
    if form == "csv":
        data = pd.read_csv(file).fillna("")
    elif form == "xlsx":
        if sheet:
            data = pd.read_excel(file, sheet_name=sheet,  engine='openpyxl').fillna("")
        else:
            data = pd.read_excel(file,  engine='openpyxl').fillna("")
    elif form == "json":
        with open(file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    elif form == "txt":
        with open(file, "r", encoding="utf-8") as fp:
            data = fp.readlines()
            data = [a.strip() for a in data]
    return data

def gpt_status_create():
    """此函数用于gpt调用失败后,判断是否创建gpt_status/if_change文件"""
    if os.path.exists(f'gpt_status') == False:
        os.system(f'mkdir gpt_status')
    if os.path.exists(f'gpt_status/if_change') == False:
        os.system(f'touch gpt_status/if_change')

def check_gpt_status():
    """判断gpt_status/if_change文件是否存在,并根据不同情况判断使用第三方账号还是公司账号
        True表示if_change文件存在，第三方账号调不通,此时使用公司账号;
	False表示if_change文件不存在,此时使用第三方账号"""
    if os.path.exists(f'gpt_status/if_change') == True:
        try:
            file_creat_time = os.path.getctime(f'gpt_status/if_change')
            now_time = time.time()
            # if now_time - file_creat_time > 1800:
            if now_time - file_creat_time > 120: # 每隔20分钟尝试重新调用第三方账号
                os.system(f'rm -rf gpt_status/if_change')
                return False
            else:
                return True
        except:
            return False
    else:
        return False

def change_base(base):
    # 此处切换账号
    """账号切换逻辑"""
    if base != 'normal':
        base = base
    else: # 当base是默认时切换的逻辑
        if check_gpt_status():
            base = 'self'
        else:
            base = 'abc'
            # base = 'api'
    return base

class GPTRES_NONSENCE(Exception):
    def __init__(self, message="GPT response is nonsensical"):
        self.message = message
        super().__init__(self.message)


class GPTApiManager():
    """ 提供多进程调用chatgpt的接口
    未写完
    """    
    def __init__(self, model_assign: str, base: str = "normal", num_process: int = 5):
        self.model_assign = model_assign
        self.base = base
        self.pool = multiprocessing.Pool(num_process)
        
        self.task_list = []
    
    def chatgpt_api(self, question: str):
        '''append task to task_list'''
        return self.task_list.append(question)
    
    def __call__(self):
        pass
        

@cacheDecorator
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(GPTRES_NONSENCE))
def chatgpt_api(question: str, model_assign: str, base="self"):
    """ model_assign:   gpt-4   调用model为gpt-4-1106-preview
                        gpt-3.5-turbo   调用model为gpt-3.5-turbo
                        gpt-3.5-turbo-16k   调用model为gpt-3.5-turbo-16k
        base:           默认值为normal, 此情况优先第三方，调用失败使用公司账号；如果指定base，则会一直使用指定的账号    
                        self:公司自己账号
                        abc:三方abc的账号
                        api:三方api的账号
    """

    ##### vllm llama3.1-70b 逻辑
    try:
        gpt_res = None
        if model_assign == 'llama3.1-70b':
            gpt_res = openai_style_api(question, model_assign)
        if gpt_res is not None:
            return gpt_res
    except:
        print("call llama3.1-70b error")
    ############################## 

    
    
    base_assign = change_base(base)
    gpt_res = call_chatgpt(question, model_assign, base_assign)
    if gpt_res == '':
        raise GPTRES_NONSENCE()
        # gpt_status_create()   ### what is this for?
        # base_assign = change_base(base)
        # gpt_res = call_chatgpt(question, model_assign, base_assign)

    return gpt_res



def call_chatgpt(question, model_assign, base):
    url = "http://8.222.236.140/call_chatgpt"
    
    data = {"question":question, "base":base, "model_assign":model_assign}
    data = json.dumps(data)
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        ret = requests.post(url, headers=headers, data=data)
    except:
        return ''
    # ret = requests.post(url, data=data)
    if ret.status_code == 200:
        try:
            res = ret.text
            # print(f'ret.status_code == 200; llm result: {res}')
        except:
            res = ""
            error_type, error_value, error_trace = sys.exc_info()
            print(error_type)
            print(error_value)
            for info in traceback.extract_tb(error_trace):
                print(info)
    else:
        res = ""
        print("status_code != 200")
    return res


def strip_label(label):
    if label.startswith('*'):
        return label.replace('*', '').strip()
    if label.startswith('-'):
        return label.replace('-', '').strip()
    if label.find('.') != -1:
        return label.split('.')[1].strip()
    return label


def extract_table_data(gpt_res, col_cnt, filter_txt):
    res = []
    for line in gpt_res.split('\n'):
        line = line.replace('"', '')
        if line.find('|') == -1:
            continue
        fields = line.split('|')
        fields = [a.strip() for a in fields]
        if len(fields) == col_cnt+2:
            fields = fields[1:col_cnt+1]
        if len(fields) != col_cnt:
            continue
        if fields[0].find('--') != -1 or fields[0].find(filter_txt) != -1:
            continue
        if fields[2].find('-') != -1:
            continue
        if len(fields[0]) == 0:
            continue
        fields[0] = strip_label(fields[0])
        fields[2] = fields[2].replace('mentioned', '').replace('次', '')
        if not fields[2].isdigit():
            continue
        print(fields)
        res.append(fields)
    return res

def multi_process(input_df, sub_process, divide=5):
    df_num = len(input_df)
    each_epoch_num = math.ceil(df_num / divide)
    with ThreadPoolExecutor(max_workers=divide) as t:
        all_tasks = []
        for i in range(divide):
            if i < divide-1:
                df_tem = input_df[each_epoch_num * i: each_epoch_num * (i + 1)]
            else:
                df_tem = input_df[each_epoch_num * i:]
            all_tasks.append(t.submit(sub_process, *[df_tem]))
        res_list = []
        for future in as_completed(all_tasks):
            res = future.result()
            res_list.append(res)
        return res_list
    return []


if __name__ == '__main__':
    question = 'Who are you?'
    res = chatgpt_api(question, model_assign = 'gpt-4', base="self")
    
    print(res)
    import pdb; pdb.set_trace()