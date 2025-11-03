import numpy as np
import inspect
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class ReactionDiffusion1D:
    def __init__(self, L : float, dt: float, dx: float, boundary_type: str):
        
        """
        Initializes a 1D reaction-diffusion PDE model. Up to two species supported.

        Parameters:
        -----------
        L : float
            Length of the 1D spatial domain.
        dt: float
            Timestep of the numerical scheme.
        dx: float
            Spatial step size.
        boundary_type: str
            Type of boundary condition ('zero-flux' or 'periodic').
        """
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



        assert isinstance(boundary_type, str), "Boundary type must be a string."

        if boundary_type != 'zero-flux' and boundary_type != 'periodic':
            raise ValueError("Boundary type must be either 'zero-flux' or 'periodic'.")
        else:
            self.boundary_type = boundary_type

        assert self.n > 2, "Number of spatial points n must be greater than 2."

        self.finite_difference_operator = self._build_finite_difference_matrix(boundary_type=self.boundary_type)


    def input_system(self, reaction_func: callable, diffusion_coefficients: list, u0: np.ndarray, v0: np.ndarray =None):
        """
        Inputs the reaction function, diffusion coefficients, and initial conditions for the PDE system.

        Parameters:
        -----------
        reaction_func : callable
            Function defining the reaction terms. Should accept 1 or 2 arguments (for one or two species).
        diffusion_coefficients : list
            List of diffusion coefficients for each species.
        u0 : np.ndarray
            Initial condition for species u.
        v0 : np.ndarray, optional (If there exists the species V)
        """
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
            
            #The finite difference matrices
            self.finite_difference_matrix_u = self.finite_difference_operator*(self.diffusion_coefficients[0]/self.dx**2)
            self.finite_difference_matrix_v = self.finite_difference_operator*(self.diffusion_coefficients[1]/self.dx**2)

            
            
        else:

            if num_args != 1:
                raise ValueError("Reaction function must have 1 input argument for one-species model!")
            self.model_type = "one species model"
            # Test that reaction function returns a single output
            test_out = reaction_func(self.u)
            if isinstance(test_out, tuple):
                raise ValueError("One-species function must return a single output!")

            self.finite_difference_matrix_u = self.finite_difference_operator*(self.diffusion_coefficients[0]/self.dx**2)

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
        
    def RK4(self, u, v = None):
        """
        Performs a single RK4 timestep for the one-species model.

        Parameters:
        -----------
        u : np.ndarray
            Current state of species u.
        v : np.ndarray, optional (If there exists the species V)
        """

        if v is None:
            k1 = self.dt * (self.finite_difference_matrix_u @ u + self.reaction_func(u))
            k2 = self.dt * (self.finite_difference_matrix_u @ (u + 0.5 * k1) + self.reaction_func(u + 0.5 * k1))
            k3 = self.dt * (self.finite_difference_matrix_u @ (u + 0.5 * k2) + self.reaction_func(u + 0.5 * k2))
            k4 = self.dt * (self.finite_difference_matrix_u @ (u + k3) + self.reaction_func(u + k3))

            u_next = u + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            return u_next
        else: #Else we are basically going to run the two spcies simulation
            k1_u = self.dt * (self.finite_difference_matrix_u @ u + self.reaction_func(u, v)[0])
            k1_v = self.dt * (self.finite_difference_matrix_v @ v + self.reaction_func(u, v)[1])

            k2_u = self.dt * (self.finite_difference_matrix_u @ (u + 0.5 * k1_u) + self.reaction_func(u + 0.5 * k1_u, v + 0.5 * k1_v)[0])
            k2_v = self.dt * (self.finite_difference_matrix_v @ (v + 0.5 * k1_v) + self.reaction_func(u + 0.5 * k1_u, v + 0.5 * k1_v)[1])

            k3_u = self.dt * (self.finite_difference_matrix_u @ (u + 0.5 * k2_u) + self.reaction_func(u + 0.5 * k2_u, v + 0.5 * k2_v)[0])
            k3_v = self.dt * (self.finite_difference_matrix_v @ (v + 0.5 * k2_v) + self.reaction_func(u + 0.5 * k2_u, v + 0.5 * k2_v)[1])

            k4_u = self.dt * (self.finite_difference_matrix_u @ (u + k3_u) + self.reaction_func(u + k3_u, v + k3_v)[0])
            k4_v = self.dt * (self.finite_difference_matrix_v @ (v + k3_v) + self.reaction_func(u + k3_u, v + k3_v)[1])

            u_next = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6
            v_next = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

            return u_next, v_next
    

    def _one_species_simulation(self, total_time: float):
        """
        Runs the simulation for the single species model.

        Parameters:
        -----------
        total_time : float
            Total time to simulate.
        """

        self.timevector = np.arange(0, total_time, self.dt)

        u_record = np.zeros((len(self.timevector), self.n))
        u_record[0, :] = self.u.copy()
        u_current = self.u.copy()
        for t_idx in range(1, len(self.timevector)):
            u_next = self.RK4(u_current)
            u_record[t_idx, :] = u_next
            u_current = u_next.copy()

        return u_record
    
    def _two_species_simulation(self, total_time: float):
        """
        Runs the simulation for the two species model.

        Parameters:
        -----------
        total_time : float
            Total time to simulate.
        """

        self.timevector = np.arange(0, total_time, self.dt)

        u_record = np.zeros((len(self.timevector), self.n))
        v_record = np.zeros((len(self.timevector), self.n))
        u_record[0, :] = self.u.copy()
        v_record[0, :] = self.v.copy()
        u_current = self.u.copy()
        v_current = self.v.copy()
        for t_idx in range(1, len(self.timevector)):
            u_next, v_next = self.RK4(u_current, v_current)
            u_record[t_idx, :] = u_next
            v_record[t_idx, :] = v_next
            u_current = u_next.copy()
            v_current = v_next.copy()

        return u_record, v_record
    

    def run_simulation(self, total_time: float):
        """
        Runs the simulation for the defined PDE system.

        Parameters:
        -----------
        total_time : float
            Total time to simulate.
        """

        if self.model_type == "one species model":
            return self._one_species_simulation(total_time)
        else:
            return self._two_species_simulation(total_time)

        
    def save_data(self, u_record: np.ndarray, v_record: np.ndarray = None, filename: str ="simulation_data.npz"):
        """
        Saves the simulation data to a .npz file.

        Parameters:
        -----------
        u_record : np.ndarray
            Recorded data for species u.
        v_record : np.ndarray, optional (If there exists the species V)
            Recorded data for species v.
        filename : str
            Name of the output file.
        """
        # Build save dictionary with simulation data and metadata
        save_kwargs = {}
        save_kwargs["u_record"] = u_record
        if v_record is not None:
            save_kwargs["v_record"] = v_record

        # Add numerical and domain metadata (only include values that exist)
        save_kwargs["L"] = self.L
        save_kwargs["n"] = self.n
        save_kwargs["dx"] = self.dx
        save_kwargs["dt"] = self.dt
        # boundary_type always set in __init__
        save_kwargs["boundary_type"] = self.boundary_type
        if self.diffusion_coefficients is not None:
            save_kwargs["diffusion_coefficients"] = np.array(self.diffusion_coefficients)
        if self.model_type is not None:
            save_kwargs["model_type"] = self.model_type

        # timevector may be set after running a simulation
        if hasattr(self, "timevector") and self.timevector is not None:
            save_kwargs["timevector"] = self.timevector

        # Save initial conditions (the stored u/v are treated as initial states)
        if self.u is not None:
            save_kwargs["u0"] = self.u
        if self.v is not None:
            save_kwargs["v0"] = self.v

        # Save everything in a compressed .npz archive
        np.savez_compressed(filename, **save_kwargs)

   