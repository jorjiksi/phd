#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 08:48:40 2026

@author: oleg
"""
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community

from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation


class SocialWaveSmallWorld:
    def __init__(
        self,
        n_agents=100,
        k=6,
        p=0.1,
        influence=0.4,
        damping=0.02,
        noise=0.03,
        seed=42,
        save_interval=100
    ):
        np.random.seed(seed)

        self.n = n_agents
        self.influence = influence
        self.damping = damping
        self.noise = noise
        self.save_interval = save_interval
        self.delay = 0

        # --- small-world topology ---
        self.G = nx.watts_strogatz_graph(n_agents, k, p)

        # fixed layout (ВАЖНО для динамики)
        self.pos = nx.spring_layout(self.G, seed=seed)

        # initial agent moods
        self.mood_history = [np.random.uniform(-0.2, 0.2, n_agents)]

        # history
        self.history = []
        self.network_metrics = []

    def step(self, step_number):
        if step_number >= self.delay:
            delayed_moods = np.array(
                self.mood_history[step_number - self.delay])
        else:
            delayed_moods = np.array(self.mood_history[step_number])
        new_moods = np.copy(delayed_moods)

        for i in range(self.n):
            neighbors = list(self.G.neighbors(i))
            if not neighbors:
                continue

            neighbor_avg = np.mean(new_moods[neighbors])

            delta = self.influence * \
                (neighbor_avg - delayed_moods[i])
            noise = np.random.normal(0, self.noise)

            new_moods[i] += delta - self.damping * \
                delayed_moods[i] + noise

        self.history.append(np.mean(new_moods))
        self.mood_history.append(new_moods)

        self._collect_network_metrics()

    def run(self, steps=500):
        self.steps = steps
        for step_number in range(steps):
            start_step = perf_counter()
            self.step(step_number)
            if ((step_number + 1) % self.save_interval == 0
                    or (step_number + 1 == self.steps)):
                end_step = perf_counter()
                print(
                    f"Step {step_number + 1} completed. took: {round(end_step-start_step, 2)} sec")

        return np.array(self.history)

    def get_states(self):
        """
        Возвращает состояния агентов в виде np.array
        """
        return np.array(self.moods)

    def _collect_network_metrics(self):
        metrics = {
            "clustering": nx.average_clustering(self.G),
            "path_length": nx.average_shortest_path_length(self.G),
            "degree_var": np.var([d for _, d in self.G.degree()])
        }
        self.network_metrics.append(metrics)

    def plot_network_dynamic(self, steps=200, interval=200, auto_stop=True):

        steps = min(steps, len(self.mood_history))

        fig = plt.figure(figsize=(15, 6))

        ax_net = fig.add_axes([0.05, 0.15, 0.45, 0.75])
        ax_ts = fig.add_axes([0.55, 0.15, 0.40, 0.75])

        # ---------------- Network ----------------
        ax_net.axis("off")
        nodes = nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color=self.mood_history[0],
            cmap="RdBu",
            node_size=80,
            ax=ax_net
        )
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.3, ax=ax_net)

        cbar = fig.colorbar(nodes, ax=ax_net, fraction=0.046)
        cbar.set_label("Agent mood")

        # ---------------- Time series ----------------
        ax_ts.set_title("Average collective mood")
        ax_ts.set_xlabel("Time step")
        ax_ts.set_ylabel("Mean mood")
        ax_ts.set_xlim(0, steps)

        ymin = min(self.history[:steps])
        ymax = max(self.history[:steps])
        pad = 0.1 * (ymax - ymin + 1e-6)
        ax_ts.set_ylim(ymin - pad, ymax + pad)

        ts_line, = ax_ts.plot([], [], color="black", linewidth=2)

        # ---------------- State ----------------
        self._paused = True
        self._auto_stopped = False
        self._last_trend = None

        def update(frame):
            if self._paused:
                return nodes, ts_line

            nodes.set_array(self.mood_history[frame])

            xdata = np.arange(frame + 1)
            ydata = self.history[:frame + 1]
            ts_line.set_data(xdata, ydata)

            if frame >= 2 and auto_stop and not self._auto_stopped:
                d1 = self.history[frame - 1] - self.history[frame - 2]
                d2 = self.history[frame] - self.history[frame - 1]

                t1 = np.sign(d1)
                t2 = np.sign(d2)

                if self._last_trend is None:
                    self._last_trend = t1

                if t2 != self._last_trend and t2 != 0:
                    self._paused = True
                    self._auto_stopped = True
                    anim.event_source.stop()
                    ax_ts.axvline(frame, color="red",
                                  linestyle="--", alpha=0.6)
                    ax_net.set_title(f"Auto-stop at step {frame}")
                    return nodes, ts_line

                self._last_trend = t2

            ax_net.set_title(f"Step {frame}")
            return nodes, ts_line

        anim = FuncAnimation(
            fig,
            update,
            frames=steps,
            interval=interval,
            repeat=False,
            blit=False
        )

        # ❗️ОЧЕНЬ ВАЖНО
        anim.event_source.stop()

        # ---------------- Buttons ----------------
        ax_play = fig.add_axes([0.40, 0.02, 0.1, 0.06])
        ax_pause = fig.add_axes([0.52, 0.02, 0.1, 0.06])

        btn_play = Button(ax_play, "Play")
        btn_pause = Button(ax_pause, "Pause")

        def play(event):
            self._paused = False
            self._auto_stopped = False
            anim.event_source.start()

        def pause(event):
            self._paused = True
            anim.event_source.stop()

        btn_play.on_clicked(play)
        btn_pause.on_clicked(pause)

        plt.show(block=True)
        return anim


class SocialWaveVisualizer:

    def __init__(self, model, cluster_size=15):
        """
        model — экземпляр SocialWaveSmallWorld
        cluster_size — сколько узлов брать в локальный кластер
        """
        self.model = model
        self.G = model.G
        self.cluster_size = cluster_size

        self.global_mean = []
        self.variance = []
        self.cluster_means = {}

        self._init_clusters()

    def _init_clusters(self):
        """Find clusters byt Louvain method"""
        np.random.seed(42)

        self.cluster_nodes = community.louvain_communities(self.model.G)

        for i in range(len(self.cluster_nodes)):
            self.cluster_means[i] = []

    def collect_step(self, step_number):
        """Собираем наблюдаемые на текущем шаге"""
        states = self.model.mood_history[step_number]
        self.global_mean.append(states.mean())
        self.variance.append(states.var())

        for i, cluster in enumerate(self.cluster_nodes):
            self.cluster_means[i].append(states[list(cluster)].mean())

    def plot_results(self, history=None):

        t = range(len(self.global_mean))

        fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

        # --- Global mean ---
        axes[0].plot(t, self.global_mean, lw=2)
        axes[0].set_title("Global mean (волны исчезают)")
        axes[0].set_ylabel("⟨emotion⟩")

        # --- Cluster means ---
        for i, series in self.cluster_means.items():
            axes[i+1].plot(t, series, alpha=0.8)
            if i == 0:
                break

        axes[1].set_title("Local cluster means (волны живут)")
        axes[1].set_ylabel("⟨emotion⟩_cluster")

        # --- Variance ---
        axes[2].plot(t, self.variance, color="black", lw=2)
        axes[2].set_title("Variance (скрытая коллективная динамика)")
        axes[2].set_ylabel("Var(emotion)")
        axes[2].set_xlabel("time step")

        plt.tight_layout()
        plt.show()

    def plot_time_series(self,  df=None,  window=21, time_frame='',  save_chart=False, filename="mood_chart"):
        """
        Save simulation data for a given step interval to disk.

        The method serializes and compresses:
        - the accumulated edge lists for the interval,
        - the full mood history up to the current step.

        This design minimizes RAM usage during long simulations by
        incrementally persisting intermediate results.

        Parameters
        ----------
        step_number : int
            Simulation step index corresponding to the saved data.
        edge_list_t : list of list of tuple
            Edge lists accumulated since the last save operation.
        """
        # smooth = np.convolve(self.history, np.ones(window)/window, mode='same')

        if time_frame == '':
            df = pd.Series(self.history).copy(deep=True)
        smooth = pd.Series(df).rolling(
            window=window, center=True).mean()
        plt.figure(figsize=(12, 6))
        if time_frame == '':
            plt.plot(self.history, alpha=0.4, label="Raw mood")
        plt.plot(smooth, color='red', linewidth=2, label="Smoothed trend")
        plt.title(
            f"Evolution of collective mood {time_frame} for {self.model.n} agents")
        plt.xlabel("Time step")
        plt.ylabel("Average mood")
        plt.legend()
        plt.grid(True)
        if save_chart:
            filename_jpg = filename + time_frame + '.jpg'
            path_jpg = f'{self.save_dir}/{filename_jpg}'
            plt.savefig(path_jpg, format="jpg", dpi=600, bbox_inches="tight")

            # high quality
            filename_png = filename + time_frame + '.png'
            path_png = f'{self.save_dir}/{filename_png}'
            plt.savefig(path_png, dpi=600, bbox_inches="tight")

            # vecrtor quality
            filename_svg = filename + time_frame + '.svg'
            path_svg = f'{self.save_dir}/{filename_svg}'
            plt.savefig(path_svg, format="svg", bbox_inches="tight")
        plt.show()


sizes = 100  # [10, 100, 1000]
results = {}

model = SocialWaveSmallWorld(
    n_agents=sizes,
    k=min(5, sizes-1),
    p=0.1,
    influence=0.45,
    damping=0.02,
    noise=0.02
)

ts = model.run(steps=500)

results[sizes] = {
    "time_series": ts,
    "network_metrics": model.network_metrics
}

# model.plot_network_static(step=150)
# model.plot_network_dynamic(steps=200, interval=150, auto_stop=False)

viz = SocialWaveVisualizer(model)

for step in range(model.steps):
    viz.collect_step(step)

viz.plot_results()

# for i in range(len(viz.cluster_means)):
#     viz.plot_time_series(viz.cluster_means[i], window=7,
#                          time_frame='yes')
