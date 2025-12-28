#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:29:45 2025

@author: oleg
"""
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_path: str
    n_ctx: int = 8192
    n_gpu_layers: int = 0     # >0 если GPU
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512


class AgentConfig(BaseModel):
    system_prompt: str = (
        "You are a local autonomous AI agent. "
        "You think step by step, plan actions, "
        "use tools when needed, and store useful memory."
    )
