import pandas as pd
import os,torch,re,jsonlines
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = "../models/7B" #checkpoint path
mmlu_dataset = "../mmlu/" #dataset path

def get_few_shot_prompt(filename):
    filepath = os.path.join(mmlu_dataset,"dev",filename.replace("test","dev"))
    df = pd.read_csv(filepath,header=None,names=["input","A","B","C","D","answer"])
    prompts = []
    _hint = f'There is a single choice question about {filename.replace("_test.csv", " ")}. Answer the question by replying A, B, C or D.'
    for i in range(len(df)):
        line = df.iloc[i]
        user_content = f"{_hint}\nQuestion: {line['input']}\nA. {line['A']}\nB. {line['B']}\nC. {line['C']}\nD. {line['D']}\nAnswer: "
        bot_content = f"{line[line['answer']]}\n"
        prompts.extend([{"role": "user", "content": user_content},{"role": "bot", "content": bot_content}])
    return prompts[:10]

# get result
def get_input_data(test_file_path,filename):
    _hint = f'There is a single choice question about {filename.replace("_test.csv", " ")}. Answer the question by replying A, B, C or D.'
    df = pd.read_csv(os.path.join(test_file_path,filename),header=None,names=["input","A","B","C","D","answer"])
    data = []
    for i in range(len(df)):
        line = df.iloc[i]
        input = f"{_hint}\nQuestion: {line['input']}\nA. {line['A']}\nB. {line['B']}\nC. {line['C']}\nD. {line['D']}\nAnswer: "
        target = line["answer"]
        data.append({"input":input,"target":target})
    return data
# post process
def get_capital_answer(text):
    patterns = [
        "the answer is ([A-E])",
        "the answer is([A-E])",
        "Answer: ([A-E])",
        "Answer: \(([A-E])\)",
        "Option \(([A-E])\)",
        "Answer:([A-E])",
        "Option ([A-E])",
        "Opt ([A-E])"
    ]
    for pattern in patterns:
        match = re.search(pattern,text,re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.findall("[A-D]", text)
    if match:
        return match[0]
    return ""
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",
                                                torch_dtype=torch.float16)
    generate_config = GenerationConfig.from_pretrained(PATH)
    generate_config.temperature = 0.1
    generate_config.top_k = 50
    generate_config.top_p = 0.95
    print(generate_config)
    model.eval()
    test_file_path = mmlu_dataset + "test"
    filenames = os.listdir(test_file_path)
    score_list = []
    for filename in filenames:
        score, total = 0, 0
        few_shot_prompt = get_few_shot_prompt(filename)
        input_list = get_input_data(test_file_path,filename)
        for line in tqdm(input_list):
            answer, history = model.chat(tokenizer = tokenizer, question=line["input"], history=few_shot_prompt, generation_config = generate_config,stream=False)
            answer = get_capital_answer(answer)
            if answer == line["target"]:
                score +=1
            total += 1
        score_list.append(score/total)
    final_score = sum(score_list)/len(score_list)
    print(final_score)