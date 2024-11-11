from typing import List, Union, Optional
import requests
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry
import logging

logger = logging.getLogger(__name__)

@LLMRegistry.register('qian')
class QianLLM(LLM):
    def __init__(self, base_url: str = "https://api2.aigcbest.top/v1", max_tokens: int = 300) -> None:
        self.api_key = "sk-NQUdJ4FQHJptUq3FIUkd82tmdWZ6jlQjV7eUoJvZFDbuuQ6r"
        self.base_url = base_url
        self.max_tokens = max_tokens

    def chat_completion(self, model: str, messages: Union[str, List[dict]], stream: bool = False) -> Union[List[str], str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during request: {e}")
            raise

    async def agen(self, *args, **kwargs) -> Union[List[str], str]:
        try:
            if len(args) > 0 and isinstance(args[0], list):
                message_list = args[0]
                for message_obj in message_list:
                    if message_obj.role == 'user':
                        message = message_obj.content
                        break
            else:
                message = kwargs.get("message", "")
            system_message = {
                "role": "system",
                "content": ("You are currently an AI large language model, and we are conducting a test on your language "
                            "and mathematical abilities. For each multiple-choice question, please analyze the question "
                            "thoroughly before arriving at an answer.please provide your answer. Then, provide your answer "
                            "in the format: 'The correct answer is A/B/C/D.' For example, if the answer is option A, "
                            "respond with 'The correct answer is A.' You can only choose one option. "
                            "Answer all questions in this format only.")
            }
            user_message = {"role": "user", "content": message}
            messages = [system_message, user_message]

            response = self.chat_completion(model=kwargs.get("model", "gpt-4o"), messages=messages)
            return response
            
        except Exception as error:
            logger.error(f"Error in agen: {error}")
            raise

    def gen(self, *args, **kwargs) -> Union[List[str], str]:
        try:      
            if len(args) > 0 and isinstance(args[0], list):
                message_list = args[0]
                for message_obj in message_list:
                    if message_obj.role == 'user':
                        message = message_obj.content
                        break
            else:
                message = kwargs.get("message", "")
            system_message = {
                "role": "system",
                "content": ("You are currently an AI large language model, and we are conducting a test on your language "
                            "and mathematical abilities. If the questions we ask are multiple-choice, please try to answer "
                            "with the corresponding option. For example, if option A: 3 is the correct one, you should "
                            "simply answer with 'A'. If there are no options, ensure that your answers are logically "
                            "rigorous and respond in a structured manner. Always choose one of the given options.")
            }
            user_message = {"role": "user", "content": message}
            messages = [system_message, user_message]

            response = self.chat_completion(model=kwargs.get("model", "gpt-4o"), messages=messages)
            return response
        except Exception as error:
            logger.error(f"Error in gen: {error}")
            raise
