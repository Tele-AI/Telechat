# coding: utf-8
import json
import requests
import streamlit as st


st.set_page_config(page_title="Telechat")
st.title("Welcome Telechat")


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("bot", avatar="assistant"):
        st.markdown("您好，我是Telechat，很高兴为您服务")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = 'user' if message["role"] == "user" else "assistant"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    url = "http://0.0.0.0:8070/telechat/gptDialog/v2"
    messages = init_chat_history()
    prompt = st.chat_input("Shift + Enter 换行, Enter 发送")
    if prompt:
        with st.chat_message("user", avatar='user'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("bot", avatar="assistant"):
            placeholder = st.empty()
            data = {"dialog": messages}
            headers = {"Content-Type": "application/json"}
            res = requests.post(url=url, data=json.dumps(data), headers=headers, stream=True)
            response = ""
            for chunk in res.iter_content(chunk_size=1024):
                response += chunk.decode(encoding="utf-8")
                placeholder.markdown(response)

        messages.append({"role": "bot", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
