import os
from transformers import AutoTokenizer, GenerationConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
PATH = '../models/7B_8bit'


def main():
    # 加载模型相关
    tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
    model = TelechatGPTQForCausalLM.from_quantized(PATH, device="cuda:0", inject_fused_mlp=False,
                                                   inject_fused_attention=False, trust_remote_code=True)

    generate_config = GenerationConfig.from_pretrained(PATH)
    model.eval()

    #  chat(bot)模型多轮演示
    print("*" * 10 + "多轮输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    answer, history = model.chat(tokenizer=tokenizer, question=question, history=[], generation_config=generate_config,
                                 stream=False)
    print("回答:", answer)
    print("截至目前的聊天记录是:", history)

    question = "你是谁训练的"
    print("提问:", question)
    # 将history传入
    answer, history = model.chat(tokenizer=tokenizer, question=question, history=history,
                                 generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)

    # 也可以这么调用传入history
    history = [
        {"role": "user", "content": "你是谁"},
        {"role": "bot", "content": "我是telechat"},
    ]

    question = "你是谁训练的"
    print("提问:", question)
    answer, history = model.chat(tokenizer=tokenizer, question=question, history=history,
                                 generation_config=generate_config,
                                 stream=False)
    print("回答是:", answer)
    print("截至目前的聊天记录是:", history)

    # chat(bot)模型 流式返回演示
    print("*" * 10 + "流式输入演示" + "*" * 10)
    question = "你是谁？"
    print("提问:", question)
    gen = model.chat(tokenizer=tokenizer, question=question, history=[], generation_config=generate_config,
                     stream=True)
    for answer, history in gen:
        print("回答是:", answer)
        print("截至目前的聊天记录是:", history)

    # base模型 直接续写演示

    inputs = "hello"
    print("输入:", inputs)
    output = model.generate(**tokenizer(inputs, return_tensors="pt").to(model.device),
                            generation_config=generate_config)
    output = tokenizer.decode(output[0])
    print("续写结果:", output)


if __name__ == '__main__':
    main()
