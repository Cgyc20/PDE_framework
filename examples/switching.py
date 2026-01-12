from pde_framework import ReactionDiffusion1D
import numpy as np


def main():


    # Example usage
    r_1 = 0.2
    r_2 = 0.2

    def reaction_two_species(u, v):
        du = r_2*v-r_1*u
        dv = r_1*u-r_2*v
        return du, dv

    L = 1.0
    dt = 0.0001
    dx = 0.005

    n = int(L // dx)
    # One-species model

    # Two-species model
    model2 = ReactionDiffusion1D(L=1.0, dt=dt, dx=dx, boundary_type='zero-flux')
    u0 = np.zeros(n)
    u0[:n//4+1] = 1.0
    v0 = np.zeros(n)
    v0[3*n//4:] = 1.0

    model2.input_system(reaction_two_species, diffusion_coefficients=[0.01, 0.01], u0=u0, v0=v0)
    model2.print_system()
    u_record, v_record = model2.run_simulation(total_time=10.0)

    model2.save_data(u_record, v_record, filename='two_species_simulation.npz')

    
if __name__ == "__main__":
    main()