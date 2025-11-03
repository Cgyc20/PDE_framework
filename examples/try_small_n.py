from pde_framework import ReactionDiffusion1D
import numpy as np


def main():


    # Example usage
    def reaction_one_species(u):
        return u * (1 - u)

    def reaction_two_species(u, v):
        du = u * (1 - u) - v * u
        dv = v * (u - 0.5)
        return du, dv

    L = 1.0
    dt = 0.01
    dx = 0.2

    n = int(L // dx)
    # One-species model
    model1 = ReactionDiffusion1D(L=1.0, dt=dt, dx=dx)
    u0 = np.random.rand(n)
    model1.input_system(reaction_one_species, diffusion_coefficients=[0.1], u0=u0)
    print(model1.n)
    model1.print_system()
    A = model1._build_finite_difference_matrix(boundary_type='zero-flux')
    print(A)
    # Two-species model
    model2 = ReactionDiffusion1D(L=1.0, dt=dt, dx=dx)
    u0 = np.random.rand(n)
    v0 = np.random.rand(n)
    model2.input_system(reaction_two_species, diffusion_coefficients=[0.1, 0.05], u0=u0, v0=v0)
    model2.print_system()


if __name__ == "__main__":
    main()