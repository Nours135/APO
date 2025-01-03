
class PromptTemplate_pcs:
    def __init__(self, template_str: str):
        self.template_str = template_str

        if not '__item_class__' in self.template_str:
            print('Warning: __item_class__ not in template_str')
            self.template_str = 'Here is the product information about __item_class__.\n' + self.template_str
        if not '__info_str__' in self.template_str:
            print('Warning: __info_str__ not in template_str')
            self.template_str = self.template_str + '\nHere is the product information:\n __info_str__'

    def __call__(self, information: object):
        item_class = information['item_class']
        info_str = information['infomation']
        prompt = self.template_str.replace('__item_class__', item_class)
        prompt = prompt.replace('__info_str__', info_str)
        return prompt
    
    def __str__(self):
        return self.template_str
    

template_cot = (
            "The following is product information about __item_class__. Your task is to extract the total quantity "
            "of the items mentioned in the information.\n\n"
            "Guidelines:\n"
            "Quantity Determination:\n"
            "If the product information explicitly specifies the total number of the items, use that value directly.\n"
            "If the total quantity is not explicitly stated, infer it based on the provided details.\n"
            "If no item count is provided and it is more likely that the total number is 1, mark the field as 1. "
            "Especially for products that are usually sold individually.\n"
            "If no item count is provided and it is more likely that the total number is multiple, mark the field as 2.\n"
            "If it is impossible to determine the total number, mark the field as 'unclear'. "
            "Here is the product information and title: __info_str__.\n\n"
            "Take a deep breath and analyze this problem step-by-step. "
            "Please explain every step of your reasoning process, focusing only on the most critical decision points. "
            "Finally, output the result in the format 'Number of Products|xxx,' including only the number. "
            "For example: 'Number of Products|2'."
        )


template_noncot = (
            "The following is product information about __item_class__. Your task is to extract the total quantity "
            "of the items mentioned in the information.\n\n"
            "Guidelines:\n"
            "Quantity Determination:\n"
            "If the product information explicitly specifies the total number of the items, use that value directly.\n"
            "If the total quantity is not explicitly stated, infer it based on the provided details.\n"
            "If no item count is provided and it is more likely that the total number is 1, mark the field as 1. "
            "Especially for products that are usually sold individually.\n"
            "If no item count is provided and it is more likely that the total number is multiple, mark the field as 2.\n"
            "If it is impossible to determine the total number, mark the field as 'unclear'. "
            "Here is the product information and title: __info_str__.\n\n"
            "Directly output the result in the format 'Number of Products|xxx,' including only the number. "
            "For example: 'Number of Products|2'."
        )
prompt_template_cot = PromptTemplate_pcs(template_cot)
prompt_template_noncot = PromptTemplate_pcs(template_noncot)
