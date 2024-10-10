# Editor: Junyi Shen

import asyncio
import copy
from dataclasses import asdict
from typing import Any, List, Union, Optional
from dotenv import load_dotenv
import async_timeout
from transformers import AutoModelForCausalLM, AutoTokenizer
from tenacity import retry, wait_random_exponential, stop_after_attempt
import torch

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry
load_dotenv()

# Global variables to store the model and tokenizer
# Note: Using this as class members will not work.
_tokenizer: Optional[Any] = None
_model: Optional[Any] = None

@LLMRegistry.register('LocalLLM')
class LocalLLM(LLM):
    def __init__(self, model_name: str):
        global _model, _tokenizer

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
        if _model is None:
            _model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = _tokenizer
        self.model = _model
        logger.info(f"Local LLM {model_name} loaded on {self.device}")
    
    def __deepcopy__(self, memo) -> "LocalLLM":
        # Overwrite deepcopy to avoid copying the model and tokenizer
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'model' or k == 'tokenizer':
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,        
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return llm_chat(
            self.model,
            self.tokenizer,
            messages,
            max_tokens,
            temperature,
            num_comps,
            device=self.device)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,        
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await llm_achat(
            self.model,
            self.tokenizer,
            messages,
            max_tokens,
            temperature,
            num_comps,
            device=self.device)

def llm_chat(
    model,
    tokenizer,
    messages: List[Message],
    max_tokens: int = 300,
    temperature: float = 0.7,
    num_comps=1,
    return_cost=False,
    device='cpu',
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    formated_messages = [message.content for message in messages]
    combined_text = "\n".join([f"{message['role']}: {message['content']}" for message in formated_messages])
    
    inputs = tokenizer(combined_text, return_tensors="pt").to(device)

    generation_params = {
        # "max_length": max_tokens,
        "max_new_tokens": max_tokens,
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature,
        "num_return_sequences": num_comps,
        "pad_token_id": tokenizer.eos_token_id,
    }

    try:
        outputs = model.generate(**inputs, **generation_params)
        print(outputs)
    except Exception as e:
        print(f'Error during generation: {e}')
        raise TimeoutError("LLM Timeout")
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    if num_comps == 1:
        return responses[0]

    return responses

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def llm_achat(
    model,
    tokenizer,
    messages: List[Message],
    max_tokens: int = 300,
    temperature: float = 0.7,
    num_comps=1,
    return_cost=False,
    device='cpu',
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    formated_messages = [asdict(message) for message in messages]
    combined_text = "\n".join([f"{message['role']}: {message['content']}" for message in formated_messages])

    inputs = tokenizer(combined_text, return_tensors="pt").to(device)
    
    generation_params = {
        # "max_length": max_tokens,
        "max_new_tokens": max_tokens,
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature,
        "num_return_sequences": num_comps,
        "pad_token_id": tokenizer.eos_token_id,
    }

    try:
        async with async_timeout.timeout(1000):
            outputs = await generate_response_async(model, inputs, generation_params)
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("LLM Timeout")
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    if num_comps == 1:
        return responses[0]
    
    return responses

async def generate_response_async(model, inputs, generation_params):
    outputs = await asyncio.to_thread(model.generate, **inputs, **generation_params)
    return outputs
