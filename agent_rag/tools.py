#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:18:37 2025

@author: oleg
"""
import subprocess
import os
import io
import contextlib


class Tools:

    @staticmethod
    def run_python(code: str) -> str:
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, {})
            output = stdout.getvalue()
            return output if output else "(no output)"
        except Exception as e:
            return f"Python error: {e}"

    @staticmethod
    def run_shell(cmd: str) -> str:
        try:
            out = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.STDOUT
            )
            print('out ', out)
            print('out.decode() ', out.decode())
            return out.decode()
        except Exception as e:
            return f"Shell error: {e}"

    @staticmethod
    def read_file(path: str) -> str:
        if not os.path.exists(path):
            return "File not found"
        return open(path).read()
