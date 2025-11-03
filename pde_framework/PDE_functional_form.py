import numpy as np
import inspect
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
        # Check number of arguments
        num_args = len(inspect.signature(reaction_func).parameters)
        if num_args > 2:
            raise ValueError("Reaction function can only have 1 or 2 input arguments!")

        
        self.reaction_func = reaction_func
        self.diffusion_coefficients = diffusion_coefficients

        # Assign initial conditions
        self.u = np.array(u0)
        if v0 is not None:
            if num_args != 2:
                raise ValueError("Reaction function must have 2 input arguments for two-species model!")
            self.v = np.array(v0)
            self.model_type = "two species model"
            # Test that reaction function returns exactly 2 outputs
            test_out = reaction_func(self.u, self.v)
            if not (isinstance(test_out, tuple) and len(test_out) == 2):
                raise ValueError("Two-species function must return exactly 2 outputs!")
        else:

            if num_args != 1:
                raise ValueError("Reaction function must have 1 input argument for one-species model!")
            self.model_type = "one species model"
            # Test that reaction function returns a single output
            test_out = reaction_func(self.u)
            if isinstance(test_out, tuple):
                raise ValueError("One-species function must return a single output!")

        print(f"Detected: {self.model_type}")

def main():


    # Example usage
    def reaction_one_species(u):
        return u * (1 - u)

    def reaction_two_species(u, v):
        du = u * (1 - u) - v * u
        dv = v * (u - 0.5)
        return du, dv

    n = 100
    dt = 0.01
    dx = 1.0

    # One-species model
    model1 = ReactionDiffusion1D(n=n, dt=dt, dx=dx)
    u0 = np.random.rand(n)
    model1.input_system(reaction_one_species, diffusion_coefficients=[0.1], u0=u0)

    # Two-species model
    model2 = ReactionDiffusion1D(n=n, dt=dt, dx=dx)
    u0 = np.random.rand(n)
    v0 = np.random.rand(n)
    model2.input_system(reaction_two_species, diffusion_coefficients=[0.1, 0.05], u0=u0, v0=v0)