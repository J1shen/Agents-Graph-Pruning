import requests

class OpenAI:
    def __init__(self, api_key, base_url="https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def chat_completion(self, model, messages):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": model,
            "messages": messages
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


class Agent:
    def __init__(self, name):
        self.name = name
        self.api_key = "sk-NQUdJ4FQHJptUq3FIUkd82tmdWZ6jlQjV7eUoJvZFDbuuQ6r"
        self.client = OpenAI(api_key=self.api_key, base_url="https://api2.aigcbest.top/v1")
        self.system_message = {
            "role": "system",
            "content": ("You are currently an AI large language model, and we are conducting a test on your language "
                        "and mathematical abilities. If the questions we ask are multiple-choice, please try to answer "
                        "with the corresponding option. For example, if option A: 3 is the correct one, you should "
                        "simply answer with 'A'. If there are no options, ensure that your answers are logically "
                        "rigorous and respond in a structured manner. Always choose one of the given options.")
        }

    def send_to_api(self, msg: str):
        messages = [self.system_message, {"role": "user", "content": msg}]
        response = self.client.chat_completion(
            model="gpt-4o",
            messages=messages
        )
        return response['choices'][0]['message']['content']

agent = Agent("MyAgent")
response = agent.send_to_api("In the following equation, which option represents the correct solution?\n\n2x + 5 = 15\n\nA: x = 4\nB: x = 5\nC: x = 10\nD: x = 3")
print(response)


