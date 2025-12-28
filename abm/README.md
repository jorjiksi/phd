# SocialWaveModel  
Agent-Based Simulation of Emergent Collective Dynamics

## Abstract
This repository contains an agent-based computational model designed to study
the emergence of collective wave-like dynamics under local stochastic interactions.
The model focuses on how simple interaction rules, combined with noise and damping,
can generate persistent macroscopic patterns over time.

The implementation is optimized for long-horizon simulations through incremental
disk storage, enabling experiments with thousands of agents and tens of thousands
of time steps.

---

## Research Motivation
Understanding how macroscopic collective dynamics emerge from microscopic
interactions is a central question in complex systems, computational social science,
and econophysics.

This model is intended as:
- a **conceptual research tool**
- an **experimental testbed** for emergent wave phenomena
- a bridge between **agent-based modeling** and **time-series analysis**

---

## Model Summary
- Agents are represented as nodes in a dynamically evolving network
- Each agent holds a continuous internal state (“mood”)
- At every time step:
  - agents interact with a small random neighborhood
  - local averaging competes with self-damping
  - stochastic perturbations introduce variability

The collective mean state is tracked and analyzed as a macroscopic observable.

---

## Implementation Highlights
- Disk-based storage of network snapshots and agent states
- Rolling-window smoothing and time aggregation (weekly / monthly)
- High-resolution static plots (PNG, SVG)
- Interactive 3D dynamic network visualization (Plotly)

---

## File Structure
```text
agents_visulisation_optimized+save.py   # core simulation + analysis
moods.npy                               # saved agent states
edges_step<N>.pkl.gz                   # compressed network snapshots


Requirements

Python ≥ 3.9

numpy

pandas

networkx

matplotlib

plotly

Intended Use

This code is designed for:

exploratory research

methodological experiments

illustrative simulations for academic publications

It is not intended as a calibrated predictive model.

Author

Oleg Maiorov
PhD Research — Complex Systems / Agent-Based Modeling
