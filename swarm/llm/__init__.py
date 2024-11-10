from swarm.llm.format import Message, Status

OPENAI_MODEL_PREFIX = "[openai]"  # defined here to prevent circular import
BENCHMARK_MODEL_PREFIX = "[benchmark]"  # defined here to prevent circular import

from swarm.llm.llm import LLM
from swarm.llm.mock_llm import MockLLM # must be imported before LLMRegistry
from swarm.llm.gpt_chat import GPTChat # must be imported before LLMRegistry
from swarm.llm.local_llm import LocalLLM  # must be imported before LLMRegistry
from swarm.llm.openai_chat import OpenAIChat  # must be imported before LLMRegistry
from swarm.llm.llm_registry import LLMRegistry
from swarm.llm.squirrel_llm import SquirrelLLM
from swarm.llm.qian_llm import QianLLM
from swarm.llm.visual_llm import VisualLLM
from swarm.llm.mock_visual_llm import MockVisualLLM # must be imported before VisualLLMRegistry
from swarm.llm.gpt4v_chat import GPT4VChat # must be imported before VisualLLMRegistry
from swarm.llm.visual_llm_registry import VisualLLMRegistry

__all__ = [
    "Message",
    "Status",

    "LLM",
    "LLMRegistry",

    "VisualLLM",
    "VisualLLMRegistry"
]
