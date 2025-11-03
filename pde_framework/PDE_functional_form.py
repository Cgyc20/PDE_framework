import numpy as np
import matplotlib.pyplot as plt

class ReactionDiffusion1D:
    def __init__(self, n=100, dt=0.1, dx=1.0):
        self.n = n
        self.dt = dt
        self.dx = dx
        self.u = None
        self.v = None
        self.model_type = None
        self.reaction_func = None
        self.diffusion_coefficients = None

    def input_system(self, reaction_func, diffusion_coefficients, u0, v0=None):
        self.reaction_func = reaction_func
        self.diffusion_coefficients = diffusion_coefficients

        # Assign initial conditions
        self.u = np.array(u0)
        if v0 is not None:
            self.v = np.array(v0)
            self.model_type = "two species model"
        else:
            self.model_type = "one species model"

        print(f"Detected: {self.model_type}")

# Example usage:

def du_dt(u):
    return u - u**3  # simple one-species reaction

def dv_dt(u, v):
    return u - v, 0.1*(u - 0.5*v)  # two-species reaction

# One-species
u0 = np.random.rand(100)
model1 = ReactionDiffusion1D()
model1.input_system(du_dt, diffusion_coefficients=[0.1], u0=u0)

# Two-species
u0 = np.random.rand(100)
v0 = np.random.rand(100)
model2 = ReactionDiffusion1D()
model2.input_system(dv_dt, diffusion_coefficients=[0.1, 0.05], u0=u0, v0=v0)
