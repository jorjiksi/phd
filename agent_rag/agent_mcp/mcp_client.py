#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 22:14:28 2026

@author: oleg
"""
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


import logging

# Настраиваем вывод в консоль
logging.basicConfig(level=logging.DEBUG)
# Фокусируемся на обмене данными MCP
logging.getLogger("mcp").setLevel(logging.DEBUG)


class MCPClient:
    def __init__(self, server_cmd=None):
        self.server_cmd = server_cmd or ["python", "agent_mcp/mcp_server.py"]
        self.session = None
        self._cm = None  # context manager

    async def _connect(self):
        if self.session is not None:
            return  # 🔒 КРИТИЧНО
        print(
            f"[DEBUG] Пытаюсь запустить сервер: {self.server_cmd}", flush=True)
        try:
            self._cm = stdio_client(
                server=StdioServerParameters(
                    command=self.server_cmd[0],
                    args=self.server_cmd[1:]
                )
            )
            entered = await self._cm.__aenter__()

            # entered = (read_stream, write_stream)
            read_stream, write_stream = entered

            # 🔥 ВОТ ЭТО ГЛАВНОЕ
            self.session = ClientSession(read_stream, write_stream)
            # 🔥 ВОТ ЭТОГО НЕ ХВАТАЛО
            await self.session.initialize()
            tools = await self.session.list_tools()
            print("[DEBUG] MCP tools:", [t.name for t in tools.tools])

            print("[DEBUG] MCP initialized")
            print("[DEBUG] MCP handshake completed")

            print("[DEBUG] Подключение установлено успешно", flush=True)
        except Exception as e:
            print(f"[DEBUG] Ошибка при подключении: {e}", flush=True)
            raise

    async def close(self):
        if self._cm:
            await self._cm.__aexit__(None, None, None)
            self._cm = None
            self.session = None

    async def call(self, tool_name: str, args: dict):
        if self.session is None:
            raise RuntimeError("MCP not connected")

        await self._connect()
        print('args==========================================')
        print(args)
        print(type(args))
        print('end args ==========================================')
        result = await self.session.call_tool(tool_name, args)

        return result.content
