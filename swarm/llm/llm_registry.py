from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM
from swarm.llm import BENCHMARK_MODEL_PREFIX, OPENAI_MODEL_PREFIX


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)

    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        if model_name == 'mock':
            model = cls.registry.get(model_name)

        elif model_name == 'squirrel':
            model = cls.registry.get(model_name)

        elif model_name == 'qian':
            model = cls.registry.get(model_name)

        elif model_name.startswith('gpt-'): 
            # any version of GPTChat like "gpt-4-1106-preview"
            model = cls.registry.get('GPTChat', model_name)
        elif model_name.startswith(OPENAI_MODEL_PREFIX):
            model = cls.registry.get('OpenAIChat', model_name[len(OPENAI_MODEL_PREFIX):])
        elif model_name.startswith(BENCHMARK_MODEL_PREFIX):
            model = cls.registry.get(
                "BenchmarkLLM", model_name[len(BENCHMARK_MODEL_PREFIX) :]
            )
        else:
            model = cls.registry.get('LocalLLM', model_name)
        return model
