from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
import pickle
import gzip
from datetime import datetime
import numpy as np
import networkx as nx


class SocialWaveSmallWorld:
    """
    Agent-based model of collective emotional dynamics
    on a Watts–Strogatz small-world network
    with full checkpointing and recovery support.
    """

    def __init__(
        self,
        n_agents=1000,
        k=4,
        p_rewire=0.1,
        influence=0.45,
        damping=0.02,
        noise=0.02,
        steps=1000,
        seed=None,
        base_dir="runs",
        resume_from=None,
    ):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.n_agents = int(n_agents)
        self.k = int(k)
        self.p_rewire = float(p_rewire)
        self.influence = float(influence)
        self.damping = float(damping)
        self.noise = float(noise)
        self.steps = int(steps)

        # Time bookkeeping
        self.current_step = 0

        # === Directory & run ID ===
        if resume_from is None:
            self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_id = (
                f"N{self.n_agents}_k{self.k}_p{self.p_rewire}_"
                f"inf{self.influence}_damp{self.damping}_"
                f"noise{self.noise}_{self.start_time}"
            )
            self.run_dir = os.path.join(base_dir, self.run_id)
            os.makedirs(self.run_dir, exist_ok=True)

            self._init_model()
            self._init_storage()

        else:
            self.run_dir = resume_from
            self._load_checkpoint()

        # === Adaptive save interval ===
        self.save_every = self._compute_save_interval()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_model(self):
        self.graph = nx.watts_strogatz_graph(
            n=self.n_agents,
            k=self.k,
            p=self.p_rewire,
            seed=self.seed,
        )

        # Agent emotional states
        self.emotions = np.random.uniform(-0.2, 0.2, self.n_agents)

    def _init_storage(self):
        self.mean_emotion_ts = []

        self.params = {
            "n_agents": self.n_agents,
            "k": self.k,
            "p_rewire": self.p_rewire,
            "influence": self.influence,
            "damping": self.damping,
            "noise": self.noise,
            "steps": self.steps,
            "seed": self.seed,
            "start_time": self.start_time,
        }

        self._save_params()

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(self):
        new_emotions = self.emotions.copy()

        for i in range(self.n_agents):
            neighbors = list(self.graph.neighbors(i))
            if neighbors:
                neighbor_avg = np.mean(self.emotions[neighbors])
                delta = self.influence * (neighbor_avg - self.emotions[i])
                noise = np.random.normal(0, self.noise)

                new_emotions[i] += delta - self.damping * \
                    new_emotions[i] + noise

        self.emotions = new_emotions
        self.current_step += 1

        self.mean_emotion_ts.append(float(np.mean(self.emotions)))

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run(self):
        try:
            while self.current_step < self.steps:
                self.step()

                if self.current_step % self.save_every == 0:
                    self._save_checkpoint()

        except Exception as e:
            print("⚠️ Model interrupted, saving checkpoint...")
            self._save_checkpoint()
            raise e

        self._save_checkpoint(final=True)

    # ------------------------------------------------------------------
    # Saving logic
    # ------------------------------------------------------------------

    def _compute_save_interval(self):
        # Heuristic for large-scale runs
        target_checkpoints = min(1000, max(10, int(1e8 / self.n_agents)))
        return max(1, self.steps // target_checkpoints)

    def _save_params(self):
        path = os.path.join(self.run_dir, "params.pkl")
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    def _save_checkpoint(self, final=False):
        state = {
            "current_step": self.current_step,
            "emotions": self.emotions,
            "mean_emotion_ts": self.mean_emotion_ts,
        }

        name = (
            "checkpoint_final.pkl.gz"
            if final
            else f"checkpoint_step_{self.current_step}.pkl.gz"
        )

        path = os.path.join(self.run_dir, name)

        with gzip.open(path, "wb") as f:
            pickle.dump(state, f)

    # ------------------------------------------------------------------
    # Loading logic
    # ------------------------------------------------------------------

    def _load_checkpoint(self):
        # Load params
        with open(os.path.join(self.run_dir, "params.pkl"), "rb") as f:
            self.params = pickle.load(f)

        for k, v in self.params.items():
            setattr(self, k, v)

        # Find latest checkpoint
        checkpoints = [
            f for f in os.listdir(self.run_dir)
            if f.startswith("checkpoint_step")
        ]

        if not checkpoints:
            raise RuntimeError("No checkpoints found to resume from.")

        latest = max(
            checkpoints,
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        with gzip.open(os.path.join(self.run_dir, latest), "rb") as f:
            state = pickle.load(f)

        self.current_step = state["current_step"]
        self.emotions = state["emotions"]
        self.mean_emotion_ts = state["mean_emotion_ts"]

        # Rebuild network
        self.graph = nx.watts_strogatz_graph(
            n=self.n_agents,
            k=self.k,
            p=self.p_rewire,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_saved_run(cls, run_dir):
        return cls(resume_from=run_dir)


class SocialWaveVisualizer:
    """
    Visualization layer for SocialWaveSmallWorld.
    Handles static network plots, dynamic animations,
    and time series visualization with controls.
    """

    def __init__(
        self,
        model,
        max_nodes_draw=1000,
        interval=50
    ):
        self.model = model
        self.interval = interval
        self.max_nodes_draw = max_nodes_draw

        self.anim = None
        self.running = True

        # --- node sampling for large systems ---
        if model.n_agents > max_nodes_draw:
            self.draw_nodes = np.random.choice(
                model.n_agents,
                max_nodes_draw,
                replace=False
            )
        else:
            self.draw_nodes = np.arange(model.n_agents)

        self.subgraph = model.graph.subgraph(self.draw_nodes)
        self.pos = nx.spring_layout(self.subgraph, seed=42)

    # ==========================================================
    # Static network plot
    # ==========================================================

    def plot_network_static(self):
        emotions = self.model.emotions[self.draw_nodes]

        plt.figure(figsize=(8, 8))
        nx.draw(
            self.subgraph,
            self.pos,
            node_color=emotions,
            cmap="coolwarm",
            node_size=30,
            edge_color="gray",
            alpha=0.8
        )
        plt.title("Static snapshot of emotional state")
        plt.colorbar(
            plt.cm.ScalarMappable(cmap="coolwarm"),
            label="Emotion"
        )
        plt.show()

    # ==========================================================
    # Dynamic animation
    # ==========================================================

    def plot_network_dynamic(self, steps=None):
        if steps is None:
            steps = self.model.steps - self.model.current_step

        self.fig, (self.ax_net, self.ax_ts) = plt.subplots(
            1, 2, figsize=(14, 6)
        )

        # --- network plot ---
        self.nodes = nx.draw_networkx_nodes(
            self.subgraph,
            self.pos,
            node_color=self.model.emotions[self.draw_nodes],
            cmap="coolwarm",
            node_size=30,
            ax=self.ax_net
        )
        nx.draw_networkx_edges(
            self.subgraph,
            self.pos,
            ax=self.ax_net,
            alpha=0.3
        )

        self.ax_net.set_title("Network dynamics")
        self.ax_net.axis("off")

        # --- time series plot ---
        self.ts_line, = self.ax_ts.plot(
            self.model.mean_emotion_ts,
            lw=2
        )
        self.ax_ts.set_title("Mean emotion over time")
        self.ax_ts.set_xlabel("Step")
        self.ax_ts.set_ylabel("Mean emotion")

        self._add_buttons()

        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=steps,
            interval=self.interval,
            repeat=False
        )

        plt.show()

    # ==========================================================
    # Update function
    # ==========================================================

    def _update(self, frame):
        if not self.running:
            return

        self.model.step()

        emotions = self.model.emotions[self.draw_nodes]
        self.nodes.set_array(emotions)

        self.ts_line.set_data(
            np.arange(len(self.model.mean_emotion_ts)),
            self.model.mean_emotion_ts
        )
        self.ax_ts.relim()
        self.ax_ts.autoscale_view()

        return self.nodes, self.ts_line

    # ==========================================================
    # Buttons
    # ==========================================================

    def _add_buttons(self):
        ax_play = plt.axes([0.15, 0.02, 0.1, 0.05])
        ax_pause = plt.axes([0.27, 0.02, 0.1, 0.05])
        ax_reset = plt.axes([0.39, 0.02, 0.1, 0.05])

        self.btn_play = Button(ax_play, "Play")
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_reset = Button(ax_reset, "Reset")

        self.btn_play.on_clicked(self._play)
        self.btn_pause.on_clicked(self._pause)
        self.btn_reset.on_clicked(self._reset)

    def _play(self, event):
        self.running = True

    def _pause(self, event):
        self.running = False

    def _reset(self, event):
        self.running = False

        # reset model state
        self.model.current_step = 0
        self.model.mean_emotion_ts = []
        self.model.emotions = np.random.uniform(-0.2, 0.2, self.model.n_agents
                                                )

        self.nodes.set_array(
            self.model.emotions[self.draw_nodes]
        )
        self.ts_line.set_data([], [])

        self.ax_ts.relim()
        self.ax_ts.autoscale_view()

        self.fig.canvas.draw_idle()


# new run
model = SocialWaveSmallWorld(
    n_agents=10,
    steps=500,
    p_rewire=0.1,
    seed=42
)
model.run()

viz = SocialWaveVisualizer(model)
viz.plot_network_dynamic()


# # continue after break
# model = SocialWaveSmallWorld.from_saved_run(
#     "runs/N100000_k4_p0.05_inf0.2_damp0.05_noise0.01_2026-01-27_10-42-11"
# )
# model.run()
