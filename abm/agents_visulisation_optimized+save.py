#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:42:36 2025

@author: oleg
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocialWaveModel with disk-saving and time series tracking.
Optimized for large simulations without excessive RAM usage.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized SocialWaveModel
- Stores networks and moods on disk incrementally to save RAM
- Maintains time series
- Supports 3D dynamic visualization
"""


import glob
import os
import pickle
import gzip
from time import perf_counter
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
pio.renderers.default = "browser"


class SocialWaveModel:
    def __init__(self, n_agents=200, influence=0.2, damping=0.05, noise=0.01,
                 connections=2, save_interval=100, save_dir="swm_", new_model=True):
        """
               n_agents: Number of agents
               influence: Strength of influence of neighbors
               damping: Strength of self-damping
               noise: Random noise
               connections: Number of connections per agent
               save_interval: how often to save edge lists and moods to disk
               save_dir: directory for storing data
               """
        self.n = n_agents
        self.delay = 1
        self.influence = influence
        self.damping = damping
        self.noise = noise
        self.connections = connections
        self.save_dir = save_dir
        self.new_model = new_model

        # individual sensitivity to the environment
        self.sensitivity = np.random.lognormal(
            mean=0.0, sigma=0.3, size=self.n)

        # individual damping
        self.agent_damping = np.random.uniform(
            0.5 * self.damping,
            1.5 * self.damping,
            self.n
        )

        self.swm_params = {'n_agents': self.n,
                           'influence': self.influence,
                           'damping': self.damping,
                           'noise': self.noise,
                           'connections': self.connections,
                           'creat_date': datetime.now()}

        self.save_interval = save_interval
        if new_model:
            self.save_dir = save_dir + f'{self.swm_params}'
            os.makedirs(self.save_dir, exist_ok=True)

            with open(f"{self.save_dir}/{self.save_dir}_swm_params.txt", "w", encoding="utf-8") as f:
                for key, value in self.swm_params.items():
                    f.write(f"{key}: {value}\n")

        # initial moods
        np.random.seed(42)
        self.moods = [np.random.uniform(-0.2, 0.2, n_agents)]
        # self.moods = [np.random.normal(-1, 1, n_agents)] - for cases with a predetermined trend
        # list of edge lists per step (saved on disk)
        self.edge_lists = []
        # full history of average moods
        self.history = []

        # ---------------------- Step simulation ----------------------

    def step(self, step_number):
        """
               Perform a single simulation step.

               At each step, every agent randomly selects a fixed number of neighbors
               and updates its mood based on:
               - the average mood of selected neighbors,
               - a self-damping term,
               - additive Gaussian noise.

               The method also:
               - records the updated moods,
               - stores the generated edge list for the current step,
               - updates the global mood history,
               - periodically saves data to disk based on `save_interval`.

               Parameters
               ----------
               step_number : int
                   Index of the current simulation step.
               """
        # delayed moods (если истории мало — берём текущее)
        if step_number >= self.delay:
            delayed_moods = np.array(self.moods[step_number - self.delay])
        else:
            delayed_moods = np.array(self.moods[step_number])
        new_moods = np.copy(delayed_moods)
        edge_list_t = []

        for i in range(self.n):
            # choose random neighbors
            possible = np.delete(np.arange(self.n), i)
            neighbors = np.random.choice(
                possible, self.connections, replace=False)

            # add undirected edges
            for nb in neighbors:
                edge_list_t.append((i, nb))

            # update mood
            neighbor_avg = delayed_moods[neighbors].mean()
            # delta = self.influence * \
            #     (neighbor_avg - self.moods[step_number][i])
            delta = self.influence * np.tanh(
                self.sensitivity[i] *
                (neighbor_avg - delayed_moods[i])
            )
            # new_moods[i] += delta - self.damping * \
            #     self.moods[step_number][i] + \
            #     np.random.normal(0, self.noise)
            new_moods[i] += (delta
                             - self.agent_damping[i] *
                             delayed_moods[i]
                             + np.random.normal(0, self.noise)
                             )
        self.moods.append(new_moods)
        self.history.append(np.mean(new_moods))
        self.edge_lists.append(edge_list_t)

        # save periodically
        if (((step_number + 1) % self.save_interval == 0)
                or (step_number + 1 == self.steps)):

            self._save_step(step_number + 1, self.edge_lists)

            # keep in-memory edge list for last interval
            self.edge_lists = []

        # ---------------------- Run simulation ----------------------

    def run(self, steps=500):
        """
           Run the full simulation for a given number of steps.

           This method iteratively calls `step()` to evolve agent moods and network
           structure over time. Execution time statistics are recorded and stored
           in the model metadata. Model parameters are saved to disk if the model
           was initialized as a new instance.

           Parameters
           ----------
           steps : int, optional
               Number of simulation steps to execute (default is 500).

           Returns
           -------
           numpy.ndarray
               Array containing the time series of average agent moods.
           """
        self.steps = steps
        print('start proccessing')
        start = perf_counter()
        for step_number in range(steps):
            start_step = perf_counter()
            self.step(step_number)
            if ((step_number + 1) % self.save_interval == 0
                    or (step_number + 1 == self.steps)):
                end_step = perf_counter()
        print(
            f"Step {step_number + 1} completed. took: {round(end_step-start_step, 2)} sec")

        end = perf_counter()
        self.full_train = f'{round(end-start, 2)} sec'
        self.swm_params['full_train'] = self.full_train

        if self.new_model:
            with open(f"{self.save_dir}/{self.save_dir}_swm_params.txt", "w", encoding="utf-8") as f:
                for key, value in self.swm_params.items():
                    f.write(f"{key}: {value}\n")

        print(f'Script done. took: {end-start} sec')

        return np.array(self.history)

        # ---------------------- Save step ----------------------

    def _save_step(self, step_number, edge_list_t):
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
        # save edge_list
        fname_edges = os.path.join(
            self.save_dir, f"edges_step{step_number}.pkl.gz")
        with gzip.open(fname_edges, "wb") as f:
            pickle.dump(edge_list_t, f, protocol=5)

        # save moods incrementally
        fname_moods = os.path.join(self.save_dir, "moods.npy")

        np.save(fname_moods, self.moods)

        print(f"Saved step {step_number}")

        # ---------------------- Get graph ----------------------

    def get_graph(self, step_number):
        """
               Return a NetworkX Graph object for given step.
               Loads edge list from disk if saved, otherwise uses last in-memory.
               """
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
    def plot_time_series(self, df=None,  window=21, time_frame='',  save_chart=False, filename="mood_chart"):
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
        smooth = df.rolling(
            window=window, center=True).mean()
        plt.figure(figsize=(12, 6))
        if time_frame == '':
            plt.plot(self.history, alpha=0.4, label="Raw mood")
        plt.plot(smooth, color='red', linewidth=2, label="Smoothed trend")
        plt.title(
            f"Evolution of collective mood {time_frame} for {self.n} agents")
        plt.xlabel("Time step")
        plt.ylabel("Average mood")
        plt.legend()
        plt.grid(True)
        if save_chart:
            filename_jpg = filename + time_frame + '.jpg'
            path_jpg = f'{self.save_dir}/{filename_jpg}'
            plt.savefig(path_jpg, format="jpg",
                        dpi=600, bbox_inches="tight")

            # high quality
            filename_png = filename + time_frame + '.png'
            path_png = f'{self.save_dir}/{filename_png}'
            plt.savefig(path_png, dpi=600, bbox_inches="tight")

            # vecrtor quality
            filename_svg = filename + time_frame + '.svg'
            path_svg = f'{self.save_dir}/{filename_svg}'
            plt.savefig(path_svg, format="svg", bbox_inches="tight")
        plt.show()

    def time_framed_series(self):
        """
               Generate and plot aggregated time series at different temporal scales.

               The method computes:
               - weekly averages (7-step aggregation),
               - monthly averages (30-step aggregation),

               and visualizes them using `plot_time_series()` with appropriate
               smoothing windows. Resulting plots are automatically saved to disk.

               This is intended for long simulations where coarse-grained trends
               are more informative than step-level fluctuations.
               """
        time_series = pd.Series(self.history)
        weekly_time_series = time_series.groupby(
            time_series.index // 7).mean()
        monthly_time_series = time_series.groupby(
            time_series.index // 30).mean()
        self.plot_time_series(weekly_time_series, window=52,
                              time_frame='weekly', save_chart=True)
        self.plot_time_series(monthly_time_series, window=12,
                              time_frame='monthly', save_chart=True)

    # ---------------------- 3D dynamic network ----------------------

    def create_3d_dynamic_network(self, step_numbers=None):
        """
               Create an interactive 3D animated visualization of the evolving network.

               The visualization uses Plotly to display agents as nodes positioned
               via a 3D spring layout, with edges representing social connections.
               Node colors encode agent moods at each time step.

               Edge lists and moods are loaded from disk to support large simulations
               without excessive memory usage.

               Parameters
               ----------
               step_numbers : list of int or None, optional
                   Specific step indices to visualize. If None, all saved steps
                   are included in the animation.
               """
        # load edge lists
        edge_files = sorted(glob.glob(os.path.join(
            self.save_dir, "edges_step*.pkl.gz")))
        moods_on_disk = np.load(os.path.join(self.save_dir, "moods.npy"))

        if step_numbers is None:
            step_numbers = list(range(len(edge_files)))

        # build graph once for node positions
        G0 = self.get_graph(step_numbers[0]*self.save_interval)
        pos = nx.spring_layout(G0, dim=3, seed=42)
        x = np.array([pos[i][0] for i in range(self.n)])
        y = np.array([pos[i][1] for i in range(self.n)])
        z = np.array([pos[i][2] for i in range(self.n)])

        # initial frame
        G0_edges = list(G0.edges())
        edge_x, edge_y, edge_z = [], [], []
        for e in G0_edges:
            edge_x += [x[e[0]], x[e[1]], None]
            edge_y += [y[e[0]], y[e[1]], None]
            edge_z += [z[e[0]], z[e[1]], None]

        moods0 = moods_on_disk[0]
        norm0 = (moods0 - moods0.min()) / (np.ptp(moods0)+1e-9)

        fig = go.Figure(
            data=[
                go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                             mode="lines",
                             line=dict(width=2, color="lightgray"),
                             hoverinfo="none"),
                go.Scatter3d(x=x, y=y, z=z,
                             mode="markers",
                             marker=dict(size=6, color=norm0,
                                         colorscale="RdBu", cmin=0, cmax=1,
                                         opacity=0.9),
                             hoverinfo="text",
                             text=[f"Node {i}<br>Mood: {m:.3f}" for i, m in enumerate(moods0)])
            ]
        )

        # frames
        frames = []
        for idx, fname in enumerate(edge_files):
            step_edges = None
            with gzip.open(fname, "rb") as f:
                step_edges = pickle.load(f)[0]

            edge_x, edge_y, edge_z = [], [], []
            for e in step_edges:
                edge_x += [x[e[0]], x[e[1]], None]
                edge_y += [y[e[0]], y[e[1]], None]
                edge_z += [z[e[0]], z[e[1]], None]

            moods_t = moods_on_disk[idx]
            norm_t = (moods_t - moods_t.min()) / (np.ptp(moods_t)+1e-9)

            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                                 mode="lines", line=dict(width=2, color="lightgray")),
                    go.Scatter3d(x=x, y=y, z=z,
                                 mode="markers",
                                 marker=dict(size=6, color=norm_t,
                                             colorscale="RdBu", cmin=0, cmax=1,
                                             opacity=0.9),
                                 hoverinfo="text",
                                 text=[f"Node {i}<br>Mood: {m:.3f}" for i, m in enumerate(moods_t)])
                ],
                name=f"frame{idx}"
            ))

        fig.frames = frames
        fig.update_layout(
            title="Dynamic 3D Social Network Visualization",
            scene=dict(xaxis=dict(visible=False),
                       yaxis=dict(visible=False),
                       zaxis=dict(visible=False)),
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play", method="animate",
                                            args=[None, {"frame": {"duration": 600}, "fromcurrent": True}]),
                                       dict(label="Pause", method="animate",
                                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])],
                              showactive=True)]
        )
        fig.show()

    # ---------------------- Load full or partial saved data ----------------------

    @staticmethod
    def load_from_disk(save_dir, model_params):
        """
               Load a previously saved SocialWaveModel from disk.

               The method reconstructs the model state by loading:
               - agent mood histories,
               - references to stored edge lists.

               Network structures are loaded lazily on demand via `get_graph()`.

               Parameters
               ----------
               save_dir : str
                   Base directory where simulation data is stored.
               model_params : dict
                   Dictionary of model parameters used to identify the saved run.

               Returns
               -------
               SocialWaveModel
                   Reconstructed model instance with history and disk-backed data access.
               """
        full_path = os.getcwd()
        save_dir_path = full_path + f'/{save_dir}' + f'{model_params}'
        moods_on_disk = np.load(os.path.join(save_dir_path, "moods.npy"))
        edge_files = sorted([f for f in os.listdir(save_dir_path) if f.startswith(
            "edges_step") and f.endswith(".pkl.gz")])
        n_agents = moods_on_disk.shape[1]

        model = SocialWaveModel(
            n_agents=n_agents, save_dir=save_dir_path, new_model=False)
        model.moods = [moods_on_disk[i]
                       for i in range(moods_on_disk.shape[0])]
        model.history = [m.mean() for m in model.moods]
        # edge_lists will be loaded step by step through get_graph()
        return model


model = SocialWaveModel(n_agents=10, influence=0.6,
                        damping=0.02, noise=0.02, connections=2, save_interval=1000)

# run for 10800 step - if 1 step is 1 real day, then 10800 it is around 30 years
# 30 year will give option analyse almost with all possible tools fot time series
history = model.run(steps=10800)


# # to load model and/or visualize uncomment rows bellow
# # saved model parameters
# swm_params = {'n_agents': 10,
#               'influence': 0.25,
#               'damping': 0.005,
#               'noise': 0.2,
#               'connections': 2,
#               'creat_date': datetime(2025, 12, 27, 21, 44, 5, 899429)}
#               'influence': 0.45,
#               'damping': 0.02,
#               'noise': 0.02,
#               'connections': 5,
#               'creat_date': datetime(2026, 1, 12, 23, 49, 54, 861448)}

# swm_params = {'n_agents': 100,
#               'influence': 0.45,
#               'damping': 0.02,
#               'noise': 0.02,
#               'connections': 5,
#               'creat_date': datetime(2026, 1, 12, 23, 49, 37, 385874)}


# swm_params = {'n_agents': 1000,
#               'influence': 0.45,
#               'damping': 0.02,
#               'noise': 0.02,
#               'connections': 5,
#               'creat_date': datetime(2026, 1, 12, 22, 55, 17, 857993)}

# swm_params = {'n_agents': 5000,
# swm_params = {'n_agents': 10000,
#               'influence': 0.45,
#               'damping': 0.02,
#               'noise': 0.02,
#               'connections': 5,
#               'creat_date': datetime(2026, 1, 12, 0, 47, 15, 185192)}
#               'creat_date': datetime(2026, 1, 12, 23, 46, 20, 197166)}

#  load data
# model = model.load_from_disk('swm_', swm_params)

# # Plotting a time series
model.plot_time_series(window=7, save_chart=True)
# model.time_framed_series()

# # Creating 3D animation (using only the latest saved data from RAM)
model.create_3d_dynamic_network()
