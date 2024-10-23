import re
import requests

request_url = 'https://llmaiadmin-test.classba.cn/api/chat/call'
authorization_key = "447c81f7f1893f3ff7bf1457b77f9363"


def chat(name: str, msg, params: dict = None, stream: bool = False, debug: bool = False):
    url = request_url
    headers = {
        'Content-Type': 'application/json',
        'authorization': authorization_key
    }
    data = {
        "name": name,
        "inputs": {
            "stream": stream,
            "msg": repr(msg),
        }
    }
    if params:
        data["inputs"].update(params)
    if stream:
        pattern = r'data: {"chunk": "(.*?)"}'
        with requests.post(url, headers=headers, json=data) as response:
            result = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(decoded_line)
                    if decoded_line.startswith("data: {\"chunk\":"):
                        result += re.findall(pattern, decoded_line)[0]
            return result
    else:
        if debug:
            print("------------------")
            print(f"DEBUG Message: {msg}")
            print("------------------")
        response = requests.post(url, headers=headers, json=data).json()
        if debug:
            print("------------------")
            print(f"DEBUG Response: {response}")
            print("------------------")
        return str(response["data"]), response


if __name__ == "__main__":
    # 创建包含系统消息和历史会话的消息列表
    messages = [
        {"role": "user", "content": "Question:A train travels at a constant speed of 60 kilometers per hour. If the distance between two stations is 180 kilometers, how long will it take for the train to travel between the two stations? A. 2 hour B. 2.5 hour C. 3 hours D. 3.5 hours What is the correct answer?"},
    ]

    response, _ = chat("Test_LLM", messages, debug=True)
    print(response)

    # 将助手的回复添加到消息列表中，继续维护历史会话
    messages.append({"role": "assistant", "content": response})
    print(messages)
