import uvicorn
import os

from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PATH = '../models/7B'
tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",
                                             torch_dtype=torch.float16)
generate_config = GenerationConfig.from_pretrained(PATH)
model.eval()
print("=============AIGC服务启动==========")


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _gc()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
    flag = True
    try:
        if do_sample != None and (do_sample not in [True, False] or not isinstance(do_sample, bool)):
            flag = False
        if max_length != None and (not 0 < max_length <= 4096 or not isinstance(max_length, int)):
            flag = False
        if top_k != None and (not 0 < top_k < 100 or not isinstance(top_k, int)):
            flag = False
        if top_p != None and (not 0.0 < top_p < 1.0 or not isinstance(top_p, float)):
            flag = False
        if temperature != None and (not 0.0 < temperature < 1.0 or not isinstance(temperature, float)):
            flag = False
        if repetition_penalty != None and (
                not 1.0 < repetition_penalty < 100.0 or not isinstance(repetition_penalty, float)):
            flag = False
        return flag
    except Exception:
        flag = False
        return flag


def streamresponse_v2(tokenizer, query, history, do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
    result_generator = model.chat(tokenizer, query, history=history, generation_config=generate_config, stream=True,
                                  do_sample=do_sample, max_length=max_length, top_k=top_k, temperature=temperature,
                                  repetition_penalty=repetition_penalty, top_p=top_p)
    t_resp = ''
    while 1:
        try:
            char, _ = next(result_generator)
            if char is None:
                break
            else:
                t_resp += char

                yield char
        except StopIteration:
            break


def response_data(seqid, code, message, flag, data):
    res_dict = {
        "seqid": seqid,
        "code": code,
        "message": message,
        "flag": flag,
        "data": data
    }
    res = jsonable_encoder(res_dict)
    print("### 整个接口的返回结果: ", res)
    return res


def parse_data(dialog):
    history = dialog[:-1]
    query = dialog[-1].get("content")
    return history, query


@app.post('/telechat/gptDialog/v2')
async def doc_gptDialog_v2(item: dict):
    session_res = []
    # 参数校验
    try:
        dialog = item["dialog"]
    except:
        result_info = response_data("", "10301", "服务必填参数缺失", "0", "执行失败")
        return result_info
    # 暴露参数读取
    do_sample = item.get("do_sample", True)
    max_length = item.get("max_length", 4096)
    top_k = item.get("top_k", 20)
    top_p = item.get("top_p", 0.2)
    temperature = item.get("temperature", 0.1)
    repetition_penalty = item.get("repetition_penalty", 1.03)
    odd = [i for i in range(len(dialog)) if i % 2 == 0]
    even = [x for x in range(len(dialog)) if x % 2 == 1]
    for index in odd:
        if dialog[index].get("role", "") != "user":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    for index in even:
        if dialog[index].get("role", "") != "bot":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    if not check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
        result_info = response_data("", "10305", "请求参数范围错误", "0", "执行失败")
        return result_info
    try:
        history, query = parse_data(dialog)
        return StreamingResponse(
            streamresponse_v2(tokenizer, query, history, do_sample, max_length, top_k, top_p, temperature,
                              repetition_penalty),
            media_type="text/html")
    except Exception:
        import traceback
        traceback.print_exc()
        result_info = response_data('', "10903", "服务执行失败", "0", "执行失败")
        return result_info


@app.post('/telechat/gptDialog/v4')
async def doc_gptDialog_v3(item: dict, ):
    session_res = []
    try:
        dialog = item["dialog"]
    except:
        result_info = response_data("", "10301", "服务必填参数缺失", "0", "执行失败")
        return result_info
    odd = [i for i in range(len(dialog)) if i % 2 == 0]
    even = [x for x in range(len(dialog)) if x % 2 == 1]
    for index in odd:
        if dialog[index].get("role", "") != "user":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    for index in even:
        if dialog[index].get("role", "") != "bot":
            result_info = response_data("", "10904", "服务请求参数dialog错误", "0", "执行失败")
            return result_info
    do_sample = item.get("do_sample", True)
    max_length = item.get("max_length", 4096)
    top_k = item.get("top_k", 20)
    top_p = item.get("top_p", 0.2)
    temperature = item.get("temperature", 0.1)
    repetition_penalty = item.get("repetition_penalty", 1.03)
    if not check_ex(do_sample, max_length, top_k, top_p, temperature, repetition_penalty):
        result_info = response_data("", "10305", "请求参数范围错误", "0", "执行失败")
        return result_info
    try:
        history, query = parse_data(dialog)
        t_resp = model.chat(tokenizer, query, history=history, generation_config=generate_config, stream=False,
                            do_sample=do_sample, max_length=max_length, top_k=top_k, temperature=temperature,
                            repetition_penalty=repetition_penalty, top_p=top_p)

        res_data = {
            'role': "bot",
            'content': t_resp
        }
        result_info = res_data
    except Exception:
        import traceback
        traceback.print_exc()
        result_info = response_data('', "10903", "服务执行失败", "0", "执行失败")
    return result_info


if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 8070
    uvicorn.run(app, host=ip, port=port, reload=False)
