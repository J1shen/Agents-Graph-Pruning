# Creator: Junyi Shen

import asyncio
import copy
from dataclasses import asdict
from typing import Any, Dict, List, Union, Optional
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
loaded_tokenizers: Dict[str, Any] = {}
loaded_models: Dict[str, Any] = {}

@LLMRegistry.register('LocalLLM')
class LocalLLM(LLM):
    global loaded_tokenizers
    global loaded_models
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = loaded_tokenizers.get(model_name)
        if tokenizer is None:
            loaded_models[model_name] = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model = loaded_models[model_name]
            loaded_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = loaded_tokenizers[model_name]
        else:
            self.tokenizer = tokenizer
            self.model = loaded_models[model_name]
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


def _preprocess_messages(
    tokenizer,
    messages: List[Message],
    max_tokens: int,
    temperature: float,
    num_comps: int,
    device: str,
) -> Dict[str, Any]:
    chat = [asdict(message) for message in messages]
    formatted_chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_chat, return_tensors="pt").to(device)

    generation_params = {
        "max_new_tokens": max_tokens,
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature,
        "num_return_sequences": num_comps,
        "pad_token_id": tokenizer.eos_token_id,
    }
    return inputs | generation_params


def llm_chat(
    model,
    tokenizer,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
    device='cpu',
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    inputs = _preprocess_messages(
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        num_comps=num_comps,
        device=device,
    )

    try:
        outputs = model.generate(**inputs)
        print(outputs)
    except Exception as e:
        logger.error(f'Error during generation: {e}')
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
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
    device='cpu',
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    inputs = _preprocess_messages(
        tokenizer=tokenizer,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        num_comps=num_comps,
        device=device,
    )

    try:
        async with async_timeout.timeout(1000):
            outputs = await generate_response_async(model, **inputs)
    except asyncio.TimeoutError:
        logger.error('Timeout')
        raise TimeoutError("LLM Timeout")
    
    responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    if num_comps == 1:
        return responses[0]
    
    return responses

async def generate_response_async(model, **kwargs):
    outputs = await asyncio.to_thread(model.generate, **kwargs)
    return outputs
