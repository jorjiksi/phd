#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:26:46 2025

@author: oleg
"""
import os
from memory import VectorMemory


def ingest_directory(path: str, memory: VectorMemory):
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith((".txt", ".md", ".py")):
                full = os.path.join(root, f)
                with open(full, "r", errors="ignore") as fh:
                    text = fh.read()
                memory.add_text(text, source=full)


memory = VectorMemory()
ingest_directory("/home/oleg/agent_rag_bs/docs/", memory)
