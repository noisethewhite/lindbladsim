Quantum-Resonant Netting (QRN)
Numerical simulations for Figures 2–4

Author: Oleg Dolgikh
Date: December 2025

This repository contains a self-contained Python script used to generate
Figures 2–4 in the accompanying manuscript on Quantum-Resonant Netting (QRN).

────────────────────────────────────────────────────────────
Purpose
────────────────────────────────────────────────────────────
The code provides a numerical illustration of environment-assisted quantum
transport (ENAQT) in a minimal open quantum system defined on a graph.
Its purpose is qualitative demonstration, not parameter fitting or
biophysical realism.

Specifically, the simulations illustrate:
• interference trapping at low dephasing
• overdamped (Zeno-like) behavior at high dephasing
• an intermediate-noise optimum (ENAQT regime)

────────────────────────────────────────────────────────────
Model summary
────────────────────────────────────────────────────────────
• Hypotheses are represented as nodes of a simple graph (1D chain, N = 10)
• Dynamics follow a Lindblad master equation
• Hamiltonian:
    H = −γ L + V
  where L is the graph Laplacian and V is a diagonal potential
• Dephasing and irreversible readout are modeled by Lindblad operators
• Performance metric:
    P_success(T) — cumulative probability of capture into a sink node

────────────────────────────────────────────────────────────
Figures
────────────────────────────────────────────────────────────
Figure 2:
    Time evolution of P_success(t) for:
    - classical random walk
    - purely coherent dynamics (κ = 0)
    - QRN netting regime (κ > 0)

Figure 3:
    Time evolution of the expected potential ⟨V⟩(t),
    used as a proxy for variational free energy.

Figure 4:
    Efficiency landscape P_success(T) over (γ, κ),
    demonstrating an ENAQT-like optimum at intermediate dephasing.

────────────────────────────────────────────────────────────
Parameters and reproducibility
────────────────────────────────────────────────────────────
All parameters are explicitly fixed in the code and match those reported
in the manuscript. The script is deterministic up to a fixed random seed.

The intent is exact reproducibility of the published figures rather than
interactive parameter exploration.

────────────────────────────────────────────────────────────
Requirements
────────────────────────────────────────────────────────────
Python 3.9+
numpy
scipy
matplotlib

────────────────────────────────────────────────────────────
Disclaimer
────────────────────────────────────────────────────────────
This is a toy model intended for conceptual illustration.
No claim is made regarding direct neurobiological implementation.

────────────────────────────────────────────────────────────
License
────────────────────────────────────────────────────────────
MIT License (or specify otherwise)