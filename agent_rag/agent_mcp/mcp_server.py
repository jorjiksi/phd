#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 22:13:16 2026

@author: oleg
"""
from mcp.server.fastmcp import FastMCP
import subprocess
import os
import sys

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # КРИТИЧЕСКИ ВАЖНО
)

mcp = FastMCP("local-agent")


@mcp.tool()
def shell(command: str) -> str:
    try:
        return subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT
        ).decode()
    except Exception as e:
        return str(e)


@mcp.tool()
def read(path: str) -> str:
    if not os.path.exists(path):
        return "File not found"
    return open(path, "r", errors="ignore").read()


@mcp.tool()
def search(query: str) -> str:
    return f"Search requested for: {query}"


if __name__ == "__main__":
    mcp.run()
