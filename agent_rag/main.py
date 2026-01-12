#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:19:11 2025

@author: oleg
"""
from config import ModelConfig, AgentConfig
from llm import LocalLLM
from agent import Agent

MODEL_PATH = "/media/oleg/D(ubuntu)/do_not_touch/rag/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"


def main():
    llm_cfg = ModelConfig(
        model_path=MODEL_PATH,
        temperature=0.5,
        repeat_penalty=1.3,
        top_p=0.9,
        n_ctx=4096,
        n_gpu_layers=0,
        max_tokens=512
    )

    agent_cfg = AgentConfig()

    llm = LocalLLM(llm_cfg)
    agent = Agent(llm, agent_cfg)

    print("Local AI Agent REPL")
    print("Type a goal. 'exit' to quit.\n")

    while True:
        try:
            goal = input(">> ").strip()
            if not goal:
                continue
            if goal.lower() in {"exit", "quit"}:
                print("Bye.")
                break

            agent.run(goal, max_steps=10)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()


# Let's create a python code for multyplication table
# show all files in current folder
# se cicles and print nice ultiplication table
# What do you know about Business Solutions? Answer in no more than two sentences.
# how many files does in current folder?
# check first page of Business Solutions company in Slovenia
# What do you know about Business Solutions company in Slovenia? use python to pull data from their web site
