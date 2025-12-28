#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:18:55 2025

@author: oleg
"""
from llm import LocalLLM
from memory import VectorMemory
from tools import Tools
from config import AgentConfig


class Agent:
    def __init__(self, llm: LocalLLM, cfg: AgentConfig):
        self.llm = llm
        self.memory = VectorMemory()
        self.cfg = cfg
        self.memories = []
        self.memories_short = []

    def step(self, goal: str, n_step):
        self.memories.append([self.memory.search(goal)])
        print('memories', self.memories)

        prompt = f"""
{self.cfg.system_prompt}

Goal:
{goal}

Relevant memory:
{self.memories}

Мemory of previous steps
{self.memories_short}

step number = {n_step}


general instruction of processing. use step number to know what is your current step:
    1 step - define tool or give final answer if you have enough information
    2 step - if you have enough information summarize it and give answer

Rules:
- You may think internally (THOUGHT).
- If you need an external action, respond with exactly ONE tool call.
- If you have enough information, respond with FINAL.
- You always should give simple answer

Output format (choose exactly one):
- TOOL:python <code>
- TOOL:shell <command>
- TOOL:read <path>
- TOOL:search <goal>
- FINAL: <answer>

Do NOT repeat the same tool call.
Do NOT loop.
Maximum steps are limited.

You may include THOUGHT before TOOL or FINAL.

"""
        print('prompt', prompt)
        response = self.llm.generate(prompt)
        print('response=====================')
        print(response)
        print('end response=================')
        self.memories_short.append([f'step number = {n_step}' + response])

        print('=========================self.memories_short')
        print(self.memories_short)
        print('end self.memories_short=====================')

        if response.startswith("TOOL:"):
            return self.handle_tool(response, goal)

        else:
            self.memory.add(response, True)
            return response

    def handle_tool(self, response: str, goal: str):
        try:
            if 'FINAL:' in response:
                answer = 'FINAL: ' + response.split('FINAL:')[1]
                answer = answer.replace("```", "")
                return answer
            tool, payload = response.split(" ", 1)
            tool = tool.replace("TOOL:", "")
            print('tool', tool)
            payload = payload.replace("```", "")
            if tool == "python":
                result = Tools.run_python(payload)
            elif tool == "shell":
                result = Tools.run_shell(payload)
            elif tool == "read":
                result = Tools.read_file(payload)
            elif tool == "search":
                result = self.memory.search(goal)
                return result
            else:
                result = "Unknown tool"

            self.memory.add(f"Tool result: {result}", True)
            return result

        except Exception as e:
            return f"Tool parsing error: {e}"

    def run(self, goal: str, max_steps: int = 10):
        print(f"\n[GOAL] {goal}\n")

        for step in range(max_steps):
            print(f"[STEP {step+1}]")
            out = self.step(goal, step+1)
            print('==============')
            print(out)
            print('end out=======')

            if "DONE" in out or "COMPLETED" in out or "FINAL:" in out:
                break
