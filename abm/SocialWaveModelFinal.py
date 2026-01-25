#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:34:48 2026

@author: oleg
"""
import os
import pickle
import gzip
from datetime import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SocialWaveModelFinal:
    def __init__(self, n_agents=200, influence=0.2, damping=0.05, noise=0.01,
                 connections=2, steps=500, save_interval=None, save_dir_base="swm_",
                 new_model=True):
        """
        Финальный класс модели SocialWave.

        Параметры
        ----------
        n_agents : int
            Количество агентов
        influence : float
            Сила влияния соседей
        damping : float
            Само-демпфирование агента
        noise : float
            Стохастический шум
        connections : int
            Количество связей на шаг
        steps : int
            Количество шагов симуляции
        save_interval : int, optional
            Интервал сохранения (по умолчанию считается автоматически)
        save_dir_base : str
            Базовое имя папки для сохранений
        new_model : bool
            True, если создаём новую модель
        """
        self.n = n_agents
        self.steps = steps
        self.influence = influence
        self.damping = damping
        self.noise = noise
        self.connections = connections
        self.new_model = new_model

        # индивидуальные параметры
        self.sensitivity = np.random.lognormal(mean=0, sigma=0.3, size=self.n)
        self.agent_damping = np.random.uniform(
            0.5*self.damping, 1.5*self.damping, self.n)

        # дата/время создания модели
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # уникальная папка для сохранения
        self.save_dir = os.path.join(save_dir_base,
                                     f"N{self.n}_steps{self.steps}_inf{self.influence}_damp{self.damping}_{self.timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)

        # save_interval: автоматически для больших моделей
        if save_interval is None:
            # примерно 100–500 файлов на модель, адаптируем под размер
            approx_saves = 500
            self.save_interval = max(1, steps // approx_saves)
        else:
            self.save_interval = save_interval

        # граф сети
        self.G = nx.watts_strogatz_graph(self.n, min(6, self.n-1), 0.1)
        # фиксированное расположение для анимации
        self.pos = nx.spring_layout(self.G, seed=42)

        # инициализация состояний
        np.random.seed(42)
        self.moods = [np.random.uniform(-0.2, 0.2, self.n)]
        self.edge_lists = []
        self.history = []

        # если новая модель — сохраняем параметры
        self.swm_params = {
            'n_agents': self.n,
            'steps': self.steps,
            'influence': self.influence,
            'damping': self.damping,
            'noise': self.noise,
            'connections': self.connections,
            'timestamp': self.timestamp,
            'save_interval': self.save_interval
        }
        if new_model:
            self._save_params()

    # ---------------------- Step simulation ----------------------
    def step(self, step_number):
        if step_number >= len(self.moods):
            raise ValueError("Step number exceeds recorded moods length!")

        delayed_moods = np.array(self.moods[step_number])
        new_moods = np.copy(delayed_moods)
        edge_list_t = []

        for i in range(self.n):
            possible = np.delete(np.arange(self.n), i)
            neighbors = np.random.choice(
                possible, self.connections, replace=False)
            for nb in neighbors:
                edge_list_t.append((i, nb))

            neighbor_avg = delayed_moods[neighbors].mean()
            delta = self.influence * \
                np.tanh(self.sensitivity[i] *
                        (neighbor_avg - delayed_moods[i]))
            new_moods[i] += delta - self.agent_damping[i] * \
                delayed_moods[i] + np.random.normal(0, self.noise)

        self.moods.append(new_moods)
        self.history.append(np.mean(new_moods))
        self.edge_lists.append(edge_list_t)

        # инкрементальное сохранение
        if ((step_number + 1) % self.save_interval == 0) or (step_number + 1 == self.steps):
            self._save_step(step_number+1)
            self.edge_lists = []

    # ---------------------- Run simulation ----------------------
    def run(self, start_step=0):
        """
        Запуск симуляции с любого шага (для продолжения после остановки)
        """
        for step_number in range(start_step, self.steps):
            self.step(step_number)
            print(f"Step {step_number+1}/{self.steps} completed.")

        print("Simulation finished.")

    # ---------------------- Save model and data ----------------------
    def _save_params(self):
        fname = os.path.join(self.save_dir, "swm_params.pkl")
        with open(fname, "wb") as f:
            pickle.dump(self.swm_params, f)

    def _save_step(self, step_number):
        # сохраняем edge lists
        fname_edges = os.path.join(
            self.save_dir, f"edges_step{step_number}.pkl.gz")
        with gzip.open(fname_edges, "wb") as f:
            pickle.dump(self.edge_lists, f)

        # moods
        fname_moods = os.path.join(self.save_dir, "moods.npy")
        np.save(fname_moods, self.moods)

        # история
        fname_history = os.path.join(self.save_dir, "history.npy")
        np.save(fname_history, self.history)

        print(f"Saved step {step_number}")

    # ---------------------- Load model ----------------------
    @staticmethod
    def load_from_disk(save_dir):
        """
        Загружает модель из указанной папки и позволяет продолжить симуляцию
        """
        # параметры модели
        with open(os.path.join(save_dir, "swm_params.pkl"), "rb") as f:
            params = pickle.load(f)

        model = SocialWaveModelFinal(
            n_agents=params['n_agents'],
            influence=params['influence'],
            damping=params['damping'],
            noise=params['noise'],
            connections=params['connections'],
            steps=params['steps'],
            save_interval=params['save_interval'],
            new_model=False
        )

        # состояние
        moods = np.load(os.path.join(save_dir, "moods.npy"), allow_pickle=True)
        history = np.load(os.path.join(save_dir, "history.npy"))
        model.moods = moods.tolist()
        model.history = history.tolist()

        # граф
        model.G = nx.watts_strogatz_graph(model.n, min(6, model.n-1), 0.1)
        model.pos = nx.spring_layout(model.G, seed=42)

        return model

    # ---------------------- Get graph ----------------------
    def get_graph(self, step_number):
        fname_edges = os.path.join(
            self.save_dir, f"edges_step{step_number}.pkl.gz")
        if os.path.exists(fname_edges):
            with gzip.open(fname_edges, "rb") as f:
                edge_list_t = pickle.load(f)
        else:
            edge_list_t = self.edge_lists

        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(edge_list_t)
        return G

    # ---------------------- Plot time series ----------------------
    def plot_time_series(self, window=21, save_chart=False, filename="mood_chart"):
        import pandas as pd
        df = pd.Series(self.history)
        smooth = df.rolling(window=window, center=True).mean()
        plt.figure(figsize=(12, 6))
        plt.plot(self.history, alpha=0.4, label="Raw mood")
        plt.plot(smooth, color='red', linewidth=2, label="Smoothed trend")
        plt.title(f"Evolution of collective mood ({self.n} agents)")
        plt.xlabel("Time step")
        plt.ylabel("Average mood")
        plt.legend()
        plt.grid(True)
        if save_chart:
            plt.savefig(os.path.join(self.save_dir,
                        f"{filename}.png"), dpi=600)
        plt.show()

    # ---------------------- Static network ----------------------
    def plot_network_static(self, step=-1):
        moods = self.moods[step]
        plt.figure(figsize=(8, 8))
        nodes = nx.draw_networkx_nodes(
            self.G, self.pos, node_color=moods, cmap="RdBu", node_size=80)
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.3)
        plt.colorbar(nodes, label="Agent mood")
        plt.title(f"Social network snapshot (step {step})")
        plt.axis("off")
        plt.show()

    # ---------------------- Dynamic network + time series ----------------------
    def plot_network_dynamic(self, steps=None, interval=200, repeat=True):
        if steps is None:
            steps = len(self.moods)
        steps = min(steps, len(self.moods))

        fig, (ax_net, ax_ts) = plt.subplots(1, 2, figsize=(
            14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
        ax_net.axis("off")
        ax_net.set_title("Social network dynamics")

        nodes = nx.draw_networkx_nodes(
            self.G, self.pos, node_color=self.moods[0], cmap="RdBu", node_size=80, ax=ax_net)
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.3, ax=ax_net)
        cbar = plt.colorbar(nodes, ax=ax_net, fraction=0.046)
        cbar.set_label("Agent mood")

        ax_ts.set_title("Average collective mood")
        ax_ts.set_xlabel("Time step")
        ax_ts.set_ylabel("Mean mood")
        ax_ts.set_xlim(0, steps)
        ymin = min(self.history[:steps])
        ymax = max(self.history[:steps])
        margin = 0.1 * (ymax - ymin + 1e-9)
        ax_ts.set_ylim(ymin-margin, ymax+margin)
        ts_line, = ax_ts.plot([], [], color="black", linewidth=2)

        def update(frame):
            nodes.set_array(self.moods[frame])
            xdata = np.arange(frame+1)
            ydata = self.history[:frame+1]
            ts_line.set_data(xdata, ydata)
            ax_net.set_title(f"Network (step {frame})")
            return nodes, ts_line

        anim = FuncAnimation(fig, update, frames=steps,
                             interval=interval, blit=False, repeat=repeat)
        plt.tight_layout()
        plt.show()
        return anim


model = SocialWaveModelFinal()
model.run()
model.plot_network_dynamic(steps=300, interval=150)
