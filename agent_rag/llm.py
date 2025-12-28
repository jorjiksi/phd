#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:17:49 2025

@author: oleg
"""
from llama_cpp import Llama
from config import ModelConfig


class LocalLLM:
    def __init__(self, cfg: ModelConfig):
        self.llm = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            verbose=False
        )
        self.max_tokens = cfg.max_tokens

    def generate(self, prompt: str) -> str:
        out = self.llm(
            prompt,
            max_tokens=self.max_tokens,
            stop=[
                "\nTOOL:",   # ← остановка перед повтором
                "\nTHOUGHT:",
                "\nDONE:"
            ]
        )
        return out["choices"][0]["text"].strip()
