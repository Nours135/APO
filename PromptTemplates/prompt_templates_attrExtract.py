from typing import Dict, Tuple
import json
import pandas as pd

def load_property_def(property_def_file_path: str) -> Dict[str, Tuple[str, str]]:
    '''
    读取属性定义文件，返回一个字典，key 是属性名，value 是属性类别，属性的描述和属性值的枚举
    '''
    df = pd.read_excel(property_def_file_path)
    property_def = {}
    for _, row in df.iterrows():
        if row['参数名称'] == 'Brand':
            continue
        property_def[row['参数名称']] = (row['参数类型'], row['参数描述'], row['参数枚举值-枚举'])
    return property_def


class PromptTemplate_attrExtract:
    def __init__(self, template_str: str):
        self.template_str = template_str
        self.property_def = {
            'hair extensions': load_property_def('./data/jiafa/接发属性.xlsx'),
            'wigs': load_property_def('./data/jiafa/头套属性.xlsx')
        }
        if not '__item_class__' in self.template_str:
            print('Warning: __item_class__ not in template_str')
            self.template_str = 'Here is the product information about __item_class__.\n' + self.template_str
        if not '__property_def__' in self.template_str:
            print('Warning: __property_def__ not in template_str')
            self.template_str = self.template_str + '\nHere is the property definition:\n __property_def__'
        if not '__info_str__' in self.template_str:
            print('Warning: __info_str__ not in template_str')
            self.template_str = self.template_str + '\nHere is the product information:\n __info_str__'

    def _format_property_def(self, property_def: Dict[str, Tuple[str, str, str]]) -> str:
        '''
        就算是 single property， 也调用这个，不过 dict 中只有一个 key
        '''
        res = ''
        for key, value in property_def.items():
            if value[0] == '枚举':
                res += f'Property: {key}, Type: enum, Description: {value[1]}, example: {value[2]}\n'
            elif value[0] == '数值':
                res += f'Property: {key}, Type: number, Description: {value[1]}\n'
        return res

    def __call__(self, information: object):
        '''information 是 test dataset 中的第一个 X，是一个字典，包括三个字段， sku_id, item_class, info_str'''
        pass
    
    def __str__(self):
        return self.template_str


all_in_one_prompt_template_cot = '''You are an expert in information extraction. Your task is to extract the required attribute values from the given product information.

Here is the product information about __item_class__.

---
### Property Definition:
__property_def__

### Product Information string:
__info_str__
---

**Instructions:**
1. Carefully read the provided product information.
2. Identify the attribute values based on the given property definitions.
3. Think step by step to ensure accuracy.
4. Output only a valid JSON object containing the extracted attribute values.
5. Unit should be included in the value.

**Example JSON output format:**
```json
{
    "attribute_1": "value_1",
    "attribute_2": "value_2",
    "attribute_3": "value_3"
}
'''


all_in_one_prompt_template = '''You are an expert in information extraction. Your task is to extract the required attribute values from the given product information.

Here is the product information about __item_class__.

---
### Property Definition:
__property_def__

### Product Information string:
__info_str__
---

**Instructions:**
1. Carefully read the provided product information.
2. Identify the attribute values based on the given property definitions.
3. Output only a valid JSON object containing the extracted attribute values.
4. Unit should be included in the value.

**Example JSON output format:**
```json
{
    "attribute_1": "value_1",
    "attribute_2": "value_2",
    "attribute_3": "value_3"
}
'''



class PromptTemplate_attrExtract_allinone(PromptTemplate_attrExtract):
    # def __init__(self, template_str: str, property_def_file_path: str):
    #     super().__init__(template_str, property_def_file_path)
    def __call__(self, information: object):
        prompt = self.template_str.replace('__property_def__', self._format_property_def(self.property_def[information['item_class']]))
        prompt = prompt.replace('__info_str__', information['infomation'])
        prompt = prompt.replace('__item_class__', information['item_class'])
        return prompt



class PromptTemplate_attrExtract_single(PromptTemplate_attrExtract):
    def __call__(self, information: object, property_name: str):
        item_property_def = self.property_def[information['item_class']]
        prompt = self.template_str.replace('__property_def__', self._format_property_def({property_name: item_property_def[property_name]}))
        prompt = prompt.replace('__info_str__', information['infomation'])
        prompt = prompt.replace('__item_class__', information['item_class'])
        return prompt

