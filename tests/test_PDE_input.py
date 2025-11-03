from pde_framework import ReactionDiffusion1D
import numpy as np



def test_incorrect_outputs_one_species():
    rd = ReactionDiffusion1D()

    def reaction_func(u):
        return u, u  # Incorrect: returns two outputs instead of one

    u0 = [0.1] * rd.n

    try:
        rd.input_system(reaction_func, diffusion_coefficients=[0.1], u0=u0)
    except ValueError as e:
        assert str(e) == "One-species function must return a single output!"
    else:
        assert False, "ValueError not raised for incorrect outputs in one-species model"


def test_incorrect_outputs_two_species():
    rd = ReactionDiffusion1D()

    def reaction_func(u, v):
        return u  # Incorrect: returns one output instead of two

    u0 = [0.1] * rd.n
    v0 = [0.2] * rd.n

    try:
        rd.input_system(reaction_func, diffusion_coefficients=[0.1, 0.05], u0=u0, v0=v0)
    except ValueError as e:
        assert str(e) == "Two-species function must return exactly 2 outputs!"
    else:
        assert False, "ValueError not raised for incorrect outputs in two-species model"

