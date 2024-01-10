import os
import re
import time
import torch
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TASK2DESC = {
    "high_school_physics": "高中物理",
    "fire_engineer": "注册消防工程师",
    "computer_network": "计算机网络",
    "advanced_mathematics": "高等数学",
    "logic": "逻辑学",
    "middle_school_physics": "初中物理",
    "clinical_medicine": "临床医学",
    "probability_and_statistics": "概率统计",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "operating_system": "操作系统",
    "middle_school_mathematics": "初中数学",
    "chinese_language_and_literature": "中国语言文学",
    "electrical_engineer": "注册电气工程师",
    "business_administration": "工商管理",
    "high_school_geography": "高中地理",
    "modern_chinese_history": "近代史纲要",
    "legal_professional": "法律职业资格",
    "middle_school_geography": "初中地理",
    "middle_school_chemistry": "初中化学",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "physician": "医师资格",
    "high_school_chinese": "高中语文",
    "tax_accountant": "税务师",
    "high_school_history": "高中历史",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
    "high_school_mathematics": "高中数学",
    "professional_tour_guide": "导游资格",
    "veterinary_medicine": "兽医学",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "basic_medicine": "基础医学",
    "education_science": "教育学",
    "urban_and_rural_planner": "注册城乡规划师",
    "middle_school_biology": "初中生物",
    "plant_protection": "植物保护",
    "middle_school_history": "初中历史",
    "high_school_politics": "高中政治",
    "metrology_engineer": "注册计量师",
    "art_studies": "艺术学",
    "college_economics": "大学经济学",
    "college_chemistry": "大学化学",
    "law": "法学",
    "sports_science": "体育学",
    "civil_servant": "公务员",
    "college_programming": "大学编程",
    "middle_school_politics": "初中政治",
    "teacher_qualification": "教师资格",
    "computer_architecture": "计算机组成",
    "college_physics": "大学物理",
    "discrete_mathematics": "离散数学",
    "marxism": "马克思主义基本原理",
    "accountant": "注册会计师",
}


def build_example(question, A, B, C, D, with_answer: bool = True):
    choice = "\n".join(
        [
            "A. " + A,
            "B. " + B,
            "C. " + C,
            "D. " + D,
        ]
    )
    answer = data["answer"].strip().upper() if with_answer else ""
    return f"{question}\n{choice}\n答案：{answer}"

def extract_answer_option(text):
    patterns = [
            "答案是?\s?([ABCD])",
            "答案是?\s?：([ABCD])",
            "答案是?\s?:([ABCD])",
            "答案应该?是\s?([ABCD])",
            "答案应该?选\s?([ABCD])",
            "答案为\s?([ABCD])",
            "选择\s?([ABCD])",
            "只有选?项?\s?([ABCD])\s?是?对",
            "只有选?项?\s?([ABCD])\s?是?错",
            "只有选?项?\s?([ABCD])\s?不?正确",
            "只有选?项?\s?([ABCD])\s?错误",
            "说法不?对选?项?的?是\s?([ABCD])",
            "说法不?正确选?项?的?是\s?([ABCD])",
            "说法错误选?项?的?是\s?([ABCD])",
            "([ABCD])\s?是正确的",
            "([ABCD])\s?是正确答案",
            "选项\s?([ABCD])\s?正确",
            "所以答\s?([ABCD])",
            "1.\s?([ABCD])[.。$]?$",
            "所以\s?([ABCD][.。$]?$)",
            "所有\s?([ABCD][.。$]?$)",
            "[\s，：:,]([ABCD])[。，,\.]?$",
            "[\s，,：:][故即]([ABCD])[。\.]?$",
            "[\s，,：:]因此([ABCD])[。\.]?$",
            "[是为。]\s?([ABCD])[。\.]?$",
            "因此\s?([ABCD])[。\.]?$",
            "显然\s?([ABCD])[。\.]?$",
            "1.\s?(.*?)$",
            "答案是\s?(\S+)(?:。|$)",
            "答案应该是\s?(\S+)(?:。|$)",
            "答案为\s?(\S+)(?:。|$)",
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.search(text)
        if match:
            return match.group(1)
    for i in text:
        if i in "ABCD": return i
    return "C"

def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('EVAL Task Parameters')
    group.add_argument(
        '--five_shot', action="store_true")
    group.add_argument(
        '--path', type=str, default=None)

    args = parser.parse_args()
    return args

args = get_args()
PATH = args.path
tokenizer = AutoTokenizer.from_pretrained(PATH,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",torch_dtype=torch.float16)
model.eval()


submit_dict = {}
filenames = os.listdir("test")
subject_list = [test_file.replace("_test.csv","") for test_file in filenames]


for index,subject_name in enumerate(subject_list):
    submit_dict[subject_name] = {}
    test_file_path=os.path.join('test',f'{subject_name}_test.csv')
    test_df=pd.read_csv(test_file_path)
    for idx in tqdm(range(len(test_df))):
        id = test_df["id"][idx]
        question = test_df["question"][idx]
        choice_A = test_df["A"][idx]
        choice_B = test_df["B"][idx]
        choice_C = test_df["C"][idx]
        choice_D = test_df["D"][idx]
        prompt = f"以下是中国关于{TASK2DESC[subject_name]}考试的单项选择题，请选出其中的正确答案。\n\n"
        prompt += build_example(question,choice_A,choice_B,choice_C,choice_D,with_answer=False)
        five_shot_prompt = '''<_user>以下是中国关于大学经济学考试的单项选择题，请从A、B、C、D四个选项中选出其中的正确答案。\n问题：考虑以下小型开放经济的数据：Y=1000，C=700，G=150，I=250-1000r*。如果世界利率为5%，那么小型开放经济的净出口为____。\nA. 50\nB. -50\nC. 150\nD. -150<_bot>答案: B<_end><_user>问题：已知一垄断企业成本函数为TC=5Q2+20Q+10，产品的需求函数为Q=140-P。利润最大化的产量为____。]\nA. 10\nB. 5\nC. 3\nD. 15<_bot>答案: A<_end><_user>问题：如果消费与实际利率负相关，则______。\nA. IS曲线更平坦\nB. IS曲线更陡峭\nC. LM曲线更平坦\nD. LM曲线更陡峭<_bot>答案: A<_end>问题：如果所有产品的生产都增加了10%，且所有价格都下降了10%，会发生以下哪一种情况？____\nA. 真实GDP增加10%，名义GDP减少10%\nB. 真实GDP增加10%，名义GDP不变\nC. 真实GDP不变，名义GDP增加10%\nD. 真实GDP不变，名义GDP减少10%<_bot>答案: B<_end><_user>问题：如果边际技术替代率MRTSLK小于劳动与资本的价格之比，为使成本最小，该厂商应该____。\nA. 同时增加资本和劳动\nB. 同时减少资本和劳动\nC. 减少资本，增加劳动\nD. 增加资本，减少劳动<_bot>答案: D<_end>'''
        if args.five_shot:
            prompt = five_shot_prompt + "<_user>" + prompt + "<_bot>"
        else:
            prompt = "<_user>" + prompt + "<_bot>"
        context_ids = tokenizer(prompt, return_tensors="pt")
        output = model.generate(context_ids["input_ids"].to(0), max_new_tokens=2000, temperature=0.2, top_p=0.95,
                                repetition_penalty=1.0, do_sample=False, eos_token_id=[160133, 160130])
        output_str = tokenizer.decode(output[0].tolist()).split("<_bot>")[-1]
        answer_extracted = extract_answer_option(output_str)
        submit_dict[subject_name][str(id)] = answer_extracted


with open("submission.json", 'w', encoding='utf-8') as f:
    json.dump(submit_dict, f, ensure_ascii=False, indent=4)



