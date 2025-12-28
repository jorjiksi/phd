#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 09:48:57 2025

@author: oleg
"""
from SyntheticElliottGenerator import SyntheticElliottGenerator
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class FractalWaveVisualizer:
    """
    Визуализатор фрактальной декомпозиции волн.
    Работает как дерево: уровень 0 = весь ряд,
    уровень 1 = сегменты между экстремумами,
    уровень 2 = под-сегменты внутри сегментов и т.д.
    """

    def __init__(self, prominence=0.05, distance=5):
        """
        prominence — чувствительность к экстремумам
        distance — минимальный шаг между экстремумами
        """
        self.prominence = prominence
        self.distance = distance

    # -------------------------------------------------------
    # Поиск максимальных и минимальных точек
    # -------------------------------------------------------
    def _find_extrema(self, data):
        peaks, _ = find_peaks(
            data, prominence=self.prominence, distance=self.distance)
        troughs, _ = find_peaks(-data, prominence=self.prominence,
                                distance=self.distance)
        extrema = np.sort(np.concatenate([peaks, troughs]))
        return extrema

    # -------------------------------------------------------
    # Рекурсивная фрактальная декомпозиция
    # -------------------------------------------------------
    def _decompose(self, data, depth, max_depth):
        if depth >= max_depth:
            return []

        extrema = self._find_extrema(data)

        if len(extrema) < 2:
            return []

        segments = []
        for i in range(len(extrema) - 1):
            s = extrema[i]
            e = extrema[i + 1]
            seg = data[s:e+1]
            segments.append((depth, s, e, seg))

        subtree = []
        for (d, s, e, seg) in segments:
            # рекурсивный спуск
            sub = self._decompose(seg, depth + 1, max_depth)
            subtree.append({
                "level": d,
                "start": s,
                "end": e,
                "data": seg,
                "children": sub
            })

        return subtree

    # -------------------------------------------------------
    # Публичный метод: построить дерево фракталов
    # -------------------------------------------------------
    def build_tree(self, data, max_depth=4):
        tree = {
            "level": 0,
            "start": 0,
            "end": len(data) - 1,
            "data": data,
            "children": self._decompose(data, 1, max_depth)
        }
        return tree

    # -------------------------------------------------------
    # Визуализация дерева в виде графика
    # -------------------------------------------------------
    def plot(self, tree):
        cmap = plt.cm.get_cmap("tab10")

        def draw_node(node, offset=0):
            level = node["level"]
            data = node["data"]
            color = cmap(level % 10)

            # рисуем текущий уровень
            plt.plot(np.arange(offset, offset + len(data)),
                     data, color=color, linewidth=2)

            # рисуем детей
            for child in node["children"]:
                # offset смещается относительно локального отрезка
                draw_node(child, offset + child["start"])

        plt.figure(figsize=(14, 5))
        draw_node(tree)
        plt.grid(True, alpha=0.3)
        plt.title("Fractal Wave Decomposition")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    # Текстовое отображение дерева
    # -------------------------------------------------------
    def print_tree(self, tree, indent=0):
        prefix = "  " * indent
        print(
            f"{prefix}Level {tree['level']} | range [{tree['start']}:{tree['end']}]")

        for child in tree["children"]:
            self.print_tree(child, indent + 1)


# создаём синтетику
gen = SyntheticElliottGenerator(window=400, max_depth=3)
series = gen.generate()

# создаём визуализатор
viz = FractalWaveVisualizer(prominence=0.05, distance=5)

# строим фрактальное дерево
tree = viz.build_tree(series, max_depth=4)

# визуализируем
viz.plot(tree)

# печатаем иерархию
viz.print_tree(tree)
