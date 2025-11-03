import numpy as np
import inspect
import matplotlib.pyplot as plt

class ReactionDiffusion1D:
    def __init__(self, L : float, dt: float, dx: float):
       
        assert isinstance(L, float), "Domain length must be a float."
        assert isinstance(dt, float), "Time step dt must be a float."
        assert isinstance(dx, float), "Spatial step dx must be a float."
        
        self.L = L
        self.n = int(self.L // dx)
        self.dt = dt
        self.dx = dx
        self.u = None
        self.v = None
        self.model_type = None
        self.reaction_func = None
        self.diffusion_coefficients = None

        assert self.n > 2, "Number of spatial points n must be greater than 2."
    def input_system(self, reaction_func: callable, diffusion_coefficients: list, u0: np.ndarray, v0: np.ndarray =None):
        # Check number of arguments
        num_args = len(inspect.signature(reaction_func).parameters)

        assert isinstance(diffusion_coefficients,list), "Diffusion coefficients must be provided as a list."
        
        if len(diffusion_coefficients) != num_args:
             raise ValueError("diffusion_coefficients must match the number of species in reaction function.")


        for diff_values in diffusion_coefficients:
            assert isinstance(diff_values,float), "Diffusion coefficients must be floats."


        if num_args > 2:
            raise ValueError("Reaction function can only have 1 or 2 input arguments!")

        
        self.reaction_func = reaction_func
        self.diffusion_coefficients = diffusion_coefficients

        # Assign initial conditions
        # Assign initial conditions
        self.u = np.array(u0)

        # Check u0 length
        if len(self.u) != self.n:
            raise ValueError(f"Length of u0 ({len(self.u)}) must match n ({self.n})")

        if v0 is not None:
            if num_args != 2:
                raise ValueError("Reaction function must have 2 input arguments for two-species model!")
            self.v = np.array(v0)

            if len(self.v) != len(self.u):
                raise ValueError(
                    f"The two species must be defined over the same grid size. "
                    f"The first species has length {len(self.u)} but the second species has length {len(self.v)}."
                )
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


    def print_system(self):
        """
        Pretty-print the reaction-diffusion PDE(s) with explicit reaction terms.
        """
        if self.model_type is None:
            print("System not defined yet!")
            return

        try:
            source = inspect.getsource(self.reaction_func).splitlines()
            source = [line.strip() for line in source if line.strip() and not line.strip().startswith("#")]

            if self.model_type == "one species model":
                # find return statement
                return_line = next(line for line in source if line.startswith("return"))
                return_expr = return_line[len("return "):]
                D = self.diffusion_coefficients[0]
                print(f"∂u/∂t = {D} ∂²u/∂x² + {return_expr}")

            else:
                # two-species model: extract du and dv assignments
                du_line = next(line for line in source if line.startswith("du"))
                dv_line = next(line for line in source if line.startswith("dv"))
                f_expr = du_line.split("=", 1)[1].strip()
                g_expr = dv_line.split("=", 1)[1].strip()
                D_u, D_v = self.diffusion_coefficients
                print(f"∂u/∂t = {D_u} ∂²u/∂x² + {f_expr}")
                print(f"∂v/∂t = {D_v} ∂²v/∂x² + {g_expr}")

        except Exception as e:
            print("Could not extract reaction expressions:", e)
            print("Fallback: just show function name(s)")
            if self.model_type == "one species model":
                print(f"∂u/∂t = {self.diffusion_coefficients[0]} ∂²u/∂x² + {self.reaction_func.__name__}(u)")
            else:
                D_u, D_v = self.diffusion_coefficients
                print(f"∂u/∂t = {D_u} ∂²u/∂x² + f(u,v)")
                print(f"∂v/∂t = {D_v} ∂²v/∂x² + g(u,v)")


    def _build_finite_difference_matrix(self, boundary_type: str):
        """
        Builds the finite difference matrix depending on which boundary type we want.
        """

        if boundary_type == 'periodic':
            # Periodic boundary conditions
            A = np.zeros((self.n, self.n))
            for i in range(self.n):
                A[i, i] = -2
                A[i, (i - 1) % self.n] = 1
                A[i, (i + 1) % self.n] = 1
            return A
        
        elif boundary_type == 'zero-flux':
            # Zero-flux (Neumann) boundary conditions
            A = np.zeros((self.n, self.n))
            for i in range(self.n):
                if i == 0:
                    A[i, i] = -1
                    A[i, i + 1] = 1
                elif i == self.n - 1:
                    A[i, i] = -1
                    A[i, i - 1] = 1
                else:
                    A[i, i] = -2
                    A[i, i - 1] = 1
                    A[i, i + 1] = 1
            return A
        

    