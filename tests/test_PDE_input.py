from pde_framework import ReactionDiffusion1D
import numpy as np


def test_incorrect_outputs_one_species():
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

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
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

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


def test_correct_one_species():
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

    def reaction_func(u):
        return u * (1 - u)

    u0 = [0.1] * rd.n

    try:
        rd.input_system(reaction_func, diffusion_coefficients=[0.1], u0=u0)
    except Exception:
        assert False, "Correct one-species function raised an exception!"


def test_correct_two_species():
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

    def reaction_func(u, v):
        du = u * (1 - u) - u * v
        dv = v * (u - 0.5)
        return du, dv

    u0 = [0.1] * rd.n
    v0 = [0.2] * rd.n

    try:
        rd.input_system(reaction_func, diffusion_coefficients=[0.1, 0.05], u0=u0, v0=v0)
    except Exception:
        assert False, "Correct two-species function raised an exception!"


def test_missing_diffusion_coefficients():
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

    def reaction_func(u):
        return u * (1 - u)

    u0 = [0.1] * rd.n

    try:
        rd.input_system(reaction_func, diffusion_coefficients=[], u0=u0)
    except ValueError as e:
        assert "diffusion_coefficients" in str(e)
    else:
        assert False, "Number of diffusion coefficients must match number of species in reaction function."


def test_mismatched_initial_conditions():
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

    def reaction_func(u):
        return u * (1 - u)

    u0 = [0.1] * (rd.n - 1)  # Wrong length

    try:
        rd.input_system(reaction_func, diffusion_coefficients=[0.1], u0=u0)
    except ValueError as e:
        assert "length" in str(e).lower()
    else:
        assert False, "ValueError not raised for mismatched u0 length"


def test_print_pde_one_species(capsys):
    rd = ReactionDiffusion1D(L=100.0, dt=0.1, dx=1.0, boundary_type='zero-flux')

    def reaction_func(u):
        return u * (1 - u)

    u0 = [0.1] * rd.n
    rd.input_system(reaction_func, diffusion_coefficients=[0.1], u0=u0)
    rd.print_system()

    captured = capsys.readouterr()
    assert "∂u/∂t = 0.1 ∂²u/∂x² + u * (1 - u)" in captured.out


def test_invalid_boundary_type():
    try:
        rd = ReactionDiffusion1D(L=10.0, dt=0.1, dx=1.0, boundary_type="invalid")
    except ValueError as e:
        assert str(e) == "Boundary type must be either 'zero-flux' or 'periodic'."
    else:
        assert False, "ValueError not raised for invalid boundary type"

def test_diffusion_coeff_type():
    rd = ReactionDiffusion1D(L=10.0, dt=0.1, dx=1.0, boundary_type="zero-flux")
    
    def reaction(u):
        return u
    
    u0 = [0.1] * rd.n
    try:
        rd.input_system(reaction, diffusion_coefficients=["0.1"], u0=u0)
    except AssertionError as e:
        assert str(e) == "Diffusion coefficients must be floats."
    else:
        assert False, "AssertionError not raised for non-float diffusion coefficient"


def test_periodic_finite_difference_matrix():
    L, dx = 5.0, 1.0
    rd = ReactionDiffusion1D(L=L, dt=0.1, dx=dx, boundary_type='periodic')
    expected_matrix = np.array([
        [-2, 1, 0, 0, 1],
        [1, -2, 1, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 1, -2, 1],
        [1, 0, 0, 1, -2]
    ])
    assert np.all(rd.finite_difference_operator == expected_matrix), "Periodic FD matrix is incorrect"

def test_zero_flux_finite_difference_matrix():
    L, dx = 5.0, 1.0
    rd = ReactionDiffusion1D(L=L, dt=0.1, dx=dx, boundary_type='zero-flux')
    expected_matrix = np.array([
        [-1, 1, 0, 0, 0],
        [1, -2, 1, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 1, -2, 1],
        [0, 0, 0, 1, -1]
    ])
    assert np.all(rd.finite_difference_operator == expected_matrix), "Zero-flux FD matrix is incorrect"


def test_scaled_periodic_matrix():
    L, dx = 5.0, 1.0
    D = 0.2
    rd = ReactionDiffusion1D(L=L, dt=0.1, dx=dx, boundary_type='periodic')

    # Scale matrix manually
    expected_matrix = np.array([
        [-2, 1, 0, 0, 1],
        [1, -2, 1, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 1, -2, 1],
        [1, 0, 0, 1, -2]
    ]) * (D / dx**2)

    # Set a simple reaction to initialise
    rd.input_system(lambda u: -u, diffusion_coefficients=[D], u0=np.ones(rd.n))

    # Check that the diffusion matrix matches the scaled expected
    assert np.allclose(rd.finite_difference_matrix_u, expected_matrix), \
        "Scaled periodic FD matrix is incorrect"


def test_scaled_zero_flux_matrix():
    L, dx = 5.0, 1.0
    D = 0.2
    rd = ReactionDiffusion1D(L=L, dt=0.1, dx=dx, boundary_type='zero-flux')

    expected_matrix = np.array([
        [-1, 1, 0, 0, 0],
        [1, -2, 1, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 1, -2, 1],
        [0, 0, 0, 1, -1]
    ]) * (D / dx**2)

    rd.input_system(lambda u: -u, diffusion_coefficients=[D], u0=np.ones(rd.n))

    assert np.allclose(rd.finite_difference_matrix_u, expected_matrix), \
        "Scaled zero-flux FD matrix is incorrect"

