#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:18:55 2025

@author: oleg
"""
from llm import LocalLLM
from memory import VectorMemory
from agent_mcp.mcp_client import MCPClient
import asyncio
from config import AgentConfig

import logging
import sys
# Выводим логи в stderr, чтобы не мешать каналу связи stdio
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
logger = logging.getLogger("mcp")
logger.setLevel(logging.DEBUG)


# TODO падает внутри ответа на этапе использования тулов с ошибкой:
# MCP tool error: stdio_client() missing 1 required positional argument: 'server'
# вопрос был hi. what is your MCP possibilities


#  этой строке
# self.memory.add(f"Tool result ({tool}): {result}", True)
# нужно сохранять не в общую память, а во временную


class Agent:
    def __init__(self, llm: LocalLLM, cfg: AgentConfig):
        self.llm = llm
        self.memory = VectorMemory()
        self.mcp = MCPClient()

        self.cfg = cfg
        self.memories = []
        self.memories_short = []

    async def step(self, goal: str, n_step):
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

        if response.startswith("FINAL:"):
            self.memory.add(response, True)
            return response

        if response.startswith("TOOL:"):
            return await self.handle_tool(response, goal, n_step)

        # всё остальное — это THOUGHT
        self.memory.add(response, True)
        return response

    async def handle_tool(self, response: str, goal: str, n_step):
        try:
            tool, payload = response.split(" ", 1)
            tool = tool.replace("TOOL:", "").strip()
            payload = payload.replace("```", "").strip()

            print(f"[MCP] calling tool={tool}, payload={payload}")

            if tool == "shell":
                result = await self.mcp.call(
                    "shell", {"command": payload}
                )

            elif tool == "read":
                result = await self.mcp.call(
                    "read", {"path": payload}
                )

            elif tool == "search":
                result = await self.mcp.call(
                    "search", {"query": payload}
                )

            else:
                return f"Unknown tool: {tool}"

            self.memories_short.append(
                [f'step number = {n_step}' + f"Tool result ({tool}): {result}"])
            return result

        except Exception as e:
            return f"MCP tool error: {e}"

    async def run(self, goal: str, max_steps: int = 10):
        print(f"\n[GOAL] {goal}\n")

        for step in range(max_steps):
            print(f"[STEP {step+1}]")
            out = await self.step(goal, step+1)
            print('==============')
            print(out)
            print('end out=======')

            if "DONE" in out or "COMPLETED" in out or "FINAL:" in out:
                break
