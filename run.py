import asyncio
from swarm.graph.swarm import Swarm
import argparse
para = argparse.ArgumentParser()
para.add_argument("--task", type=str, default="What is the capital of Jordan?")
para.add_argument('--run_mode', type=int, default=0)
para.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
args = para.parse_args()

async def arun(model_name):
    swarm = Swarm(
        ["IO"], 
        "gaia",
        model_name=model_name
    )
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = await swarm.arun(inputs)

def run(model_name):
    swarm = Swarm(
        ["IO"], 
        "gaia",
        model_name=[model_name]*2
    )
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = swarm.run(inputs)
    
if args.run_mode:
    asyncio.run(arun(args.model_name))
else:
    run(args.model_name)