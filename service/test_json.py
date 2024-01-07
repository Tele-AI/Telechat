import json
import requests

headers = {"Content-Type": "application/json"}
url = 'http://0.0.0.0:8070/telechat/gptDialog/v2'
data = {
    "dialog": [{'role': 'user', 'content': "你好,请介绍一下自己。"}, {'role': 'bot', 'content': "我是telechat。"},
               {'role': 'user', 'content': "你是谁训练的"}],
    "do_sample": True, "top_p": 0.1, "temperature": 0.1, "repetition_penalty": 1.03
}

res = requests.post(url=url, data=json.dumps(data), headers=headers)
print(res.text)
