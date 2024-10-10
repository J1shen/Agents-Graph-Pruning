import os
import asyncio
import argparse

para = argparse.ArgumentParser()
para.add_argument("--task", type=str, default="What is the capital of Jordan?")
para.add_argument("--run_mode", type=int, default=0)
para.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
para.add_argument("--host", type=str, default="127.0.0.1")
para.add_argument("--port", type=str, default="6376")

args = para.parse_args()

os.environ["DEFAULT_HOST"] = args.host
os.environ["DEFAULT_PORT"] = args.port

# Import after setting up the environment variables
from swarm.graph.swarm import Swarm
from swarm.llm import OPENAI_MODEL_PREFIX


async def arun(model_name):
    swarm = Swarm(["IO"], "gaia", model_name=model_name)
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = await swarm.arun(inputs)
    print(answer)


def run(model_name):
    swarm = Swarm(["IO"], "gaia", model_name=model_name)
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = swarm.run(inputs)
    print(answer)


model_name = OPENAI_MODEL_PREFIX + args.model_name

if args.run_mode:
    asyncio.run(arun(model_name))
else:
    run(model_name)
