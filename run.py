import asyncio
from swarm.graph.swarm import Swarm
import argparse
para = argparse.ArgumentParser()
para.add_argument("--task", type=str, default="What is the capital of Jordan?")
para.add_argument('--run_mode', type=int, default=0)
args = para.parse_args()

async def arun():
    swarm = Swarm(
        ["IO"], 
        "gaia",
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    )
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = await swarm.arun(inputs)
    print(answer)

def run():
    swarm = Swarm(
        ["IO"], 
        "gaia",
        model_name="meta-llama/Llama-3.2-1B-Instruct"
    )
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = swarm.run(inputs)
    print(answer)
    
if args.run_mode:
    asyncio.run(arun())
else:
    run()