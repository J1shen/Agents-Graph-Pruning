from typing import List, Union, Optional
import re
import requests
import logging

from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry


request_url = 'https://llmaiadmin-test.classba.cn/api/chat/call'
authorization_key = "447c81f7f1893f3ff7bf1457b77f9363"

logger = logging.getLogger(__name__)

def chat(name: str, msg: Union[str, List[dict]], params: Optional[dict] = None, max_tokens: int = 300, stream: bool = False, debug: bool = False) -> str:
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
            "max_tokens": max_tokens,
        }
    }
    
    if params:
        data["inputs"].update(params)
    
    try:
        if stream:
            pattern = r'data: {"chunk": "(.*?)"}'
            with requests.post(url, headers=headers, json=data) as response:
                result = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        logger.debug(decoded_line)
                        if decoded_line.startswith("data: {\"chunk\":"):
                            result += re.findall(pattern, decoded_line)[0]
                return result
        else:
            if debug:
                logger.debug(f"DEBUG Message: {msg}")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            out = response.json()
            if debug:
                logger.debug(f"DEBUG Response: {out}")
            return str(out["data"])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during the request: {e}")
        raise

@LLMRegistry.register('squirrel')
class SquirrelLLM(LLM):
    def __init__(self, max_tokens: int = 300) -> None:
        self.max_tokens = max_tokens

    async def agen(self, *args, **kwargs) -> Union[List[str], str]:
        try:
            kwargs.pop("max_tokens", None)
            response = chat("GPT4o-xia", *args, **kwargs, max_tokens=self.max_tokens)
            return response
        except Exception as error:
            logger.error(f"Error in agen: {error}")
            raise

    def gen(self, *args, **kwargs) -> Union[List[str], str]:
        try:
            kwargs.pop("max_tokens", None)
            response = chat("GPT4o-xia", *args, **kwargs, max_tokens=self.max_tokens)
            return response
        except Exception as error:
            logger.error(f"Error in gen: {error}")
            raise
