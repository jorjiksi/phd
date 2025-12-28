#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 14:18:19 2025

@author: oleg
"""
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, size=300, overlap=50):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


class VectorMemory:
    def __init__(self, dim=384, path="memory"):
        self.path = path
        os.makedirs(path, exist_ok=True)

        self.index_path = os.path.join(path, "index.faiss")
        self.texts_path = os.path.join(path, "texts.pkl")

        self.index_path_short = os.path.join(path, "index_short.faiss")
        self.texts_path_short = os.path.join(path, "texts_short.pkl")

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def add(self, text: str, short: bool = False):
        vec = self.embedder.encode([text]).astype("float32")
        self.index.add(vec)
        self.texts.append(text)
        if short:
            self._save_short()
        else:
            self._save()

    def add_text(self, text: str, source: str = ""):
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            payload = f"[SOURCE: {source} | CHUNK {i}]\n{chunk}"
            vec = self.embedder.encode([payload]).astype("float32")
            self.index.add(vec)
            self.texts.append(payload)

        self._save()

    def search(self, query: str, k=2):
        if not self.texts:
            return []
        q = self.embedder.encode([query]).astype("float32")
        _, idx = self.index.search(q, k)
        return [self.texts[i] for i in idx[0] if i < len(self.texts)]

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def _save_short(self):
        faiss.write_index(self.index, self.index_path_short)
        with open(self.texts_path_short, "wb") as f:
            pickle.dump(self.texts, f)
