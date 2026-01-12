# PDE Framework — 1D Reaction–Diffusion Solver

A lightweight 1D reaction–diffusion framework for simulating PDE systems of the form:

\[
\frac{\partial u_i}{\partial t} = D_i \frac{\partial^2 u_i}{\partial x^2} + f_i(\mathbf{u})
\]

Designed for quick experiments with one- and two-species (and generally multi-species) reaction–diffusion models on a 1D spatial domain with simple boundary conditions.

---

## Features

- ✅ 1D reaction–diffusion PDEs
- ✅ User-defined reaction terms (plug in your own kinetics)
- ✅ Species-specific diffusion coefficients
- ✅ Basic boundary conditions (e.g. `zero-flux`)
- ✅ Time integration with user-chosen `dt`, spatial discretisation via `dx`
- ✅ Records and returns time series for each species
- ✅ Save simulation output to `.npz`

---

## Installation / Setup

Make sure your project structure allows importing the module:

```python
from pde_framework import ReactionDiffusion1D
