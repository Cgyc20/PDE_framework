import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys

# ============================================================================
# ANALYTICAL SOLUTION
# ============================================================================

# Parameters
L = 1.0
D = 0.01
alpha = 0.2
beta = 0.2
u0 = 1.0
v0 = 1.0
N_terms = 200
Nx = 200
T_total = 2.0
Nt = 100

# Spatial and time arrays
x = np.linspace(0, L, Nx)
t_values = np.linspace(0, T_total, Nt)

def compute_analytical_solution(t):
    """Compute analytical solution using Fourier series"""
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    
    # Handle n=0 mode separately (constant term)
    U_00 = (1/L) * u0 * (L/4)  # ∫u(x,0)dx from 0 to L/4
    V_00 = (1/L) * v0 * (L/4)  # ∫v(x,0)dx from 3L/4 to L
    
    # For n=0, the eigenvalues are:
    lambda1_0 = 0  # -D*0^2
    lambda2_0 = -alpha - beta  # -D*0^2 - alpha - beta
    
    # Constants for n=0
    c1_0 = (beta * (V_00 + U_00)) / (alpha + beta)
    c2_0 = (-beta * V_00 + alpha * U_00) / (alpha + beta)
    
    # Time evolution for n=0
    U_0t = c1_0 * np.exp(lambda1_0 * t) + c2_0 * np.exp(lambda2_0 * t)
    V_0t = (alpha/beta) * c1_0 * np.exp(lambda1_0 * t) - c2_0 * np.exp(lambda2_0 * t)
    
    # Add n=0 term (constant)
    u += U_0t
    v += V_0t
    
    # Handle n >= 1 terms
    for n in range(1, N_terms + 1):
        k_n = n * np.pi / L
        
        # Initial Fourier coefficients (with proper normalization)
        U_n0 = (2/L) * u0 * (L/(n*np.pi)) * np.sin(n * np.pi / 4)
        V_n0 = (2/L) * v0 * (L/(n*np.pi)) * (np.sin(n * np.pi) - np.sin(3 * n * np.pi / 4))
        
        # Eigenvalues
        lambda1 = -D * k_n**2
        lambda2 = -D * k_n**2 - alpha - beta
        
        # Constants
        c1 = (beta * (V_n0 + U_n0)) / (alpha + beta)
        c2 = (-beta * V_n0 + alpha * U_n0) / (alpha + beta)
        
        # Time evolution
        U_nt = c1 * np.exp(lambda1 * t) + c2 * np.exp(lambda2 * t)
        V_nt = (alpha/beta) * c1 * np.exp(lambda1 * t) - c2 * np.exp(lambda2 * t)
        
        u += U_nt * np.cos(k_n * x)
        v += V_nt * np.cos(k_n * x)
    
    return u, v

# ============================================================================
# PDE DATA LOADING
# ============================================================================

def load_data(filename):
    """Load u and optionally v and timevector/dt from a .npz file."""
    data = np.load(filename, allow_pickle=True)
    keys = set(data.files)
    
    u = None
    v = None
    for k in ['u_record', 'u_rec', 'u']:
        if k in keys:
            u = data[k]
            break
    for k in ['v_record', 'v_rec', 'v']:
        if k in keys:
            v = data[k]
            break

    if u is None:
        raise KeyError(f"No recognizable u array found in {filename}. Available keys: {data.files}")

    # Try to find a timevector or dt
    timevector = None
    if 'timevector' in keys:
        timevector = np.asarray(data['timevector'])
    elif 'time' in keys:
        timevector = np.asarray(data['time'])
    elif 't' in keys:
        timevector = np.asarray(data['t'])
    elif 'dt' in keys:
        try:
            dt = float(data['dt'].tolist())
            T = int(u.shape[0])
            timevector = np.arange(0, T * dt, dt)
        except Exception:
            timevector = None

    return u, v, timevector

# ============================================================================
# COMBINED ANIMATION
# ============================================================================

def create_combined_animation(pde_filename=None):
    """Create animation showing both analytical and PDE solutions"""
    
    # Load PDE data if provided
    pde_u = None
    pde_v = None
    pde_timevector = None
    pde_x = None
    
    if pde_filename and Path(pde_filename).exists():
        pde_u, pde_v, pde_timevector = load_data(pde_filename)
        
        # Subsample PDE data to match animation speed
        skip = max(1, len(pde_u) // Nt)
        pde_u = pde_u[::skip]
        if pde_v is not None:
            pde_v = pde_v[::skip]
        if pde_timevector is not None:
            pde_timevector = pde_timevector[::skip]
        
        # Create spatial grid for PDE data
        pde_N = pde_u.shape[1]
        pde_x = np.linspace(0, L, pde_N)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Reaction-Diffusion System: Analytical vs PDE Solution', fontsize=14)
    
    # Initial analytical solution
    u_analytical, v_analytical = compute_analytical_solution(0)
    
    # Plot analytical solution
    line_u_analytical, = ax1.plot(x, u_analytical, 'b-', linewidth=2, label='u (Analytical)')
    line_v_analytical, = ax1.plot(x, v_analytical, 'r-', linewidth=2, label='v (Analytical)')
    
    # Plot PDE solution if available
    line_u_pde = None
    line_v_pde = None
    if pde_u is not None:
        line_u_pde, = ax1.plot(pde_x, pde_u[0], 'b--', linewidth=1.5, alpha=0.7, label='u (PDE)')
        if pde_v is not None:
            line_v_pde, = ax1.plot(pde_x, pde_v[0], 'r--', linewidth=1.5, alpha=0.7, label='v (PDE)')
    
    # Setup analytical plot
    ax1.set_ylabel('Concentration')
    ax1.set_ylim(-0.1, 1.2)
    ax1.set_xlim(0, L)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axvline(x=0.25, color='blue', linestyle=':', alpha=0.5)
    ax1.axvline(x=0.75, color='red', linestyle=':', alpha=0.5)
    
    # Plot difference (PDE - Analytical) if available
    if pde_u is not None:
        diff_u_initial = pde_u[0] - np.interp(pde_x, x, u_analytical)
        line_diff_u, = ax2.plot(pde_x, diff_u_initial, 'b-', linewidth=1.5, label='u difference')
        
        if pde_v is not None:
            diff_v_initial = pde_v[0] - np.interp(pde_x, x, v_analytical)
            line_diff_v, = ax2.plot(pde_x, diff_v_initial, 'r-', linewidth=1.5, label='v difference')
    
    # Setup difference plot
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Difference (PDE - Analytical)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Time text
    time_text = ax1.text(0.02, 0.98, f't = {0:.3f}', transform=ax1.transAxes,
                        ha='left', va='top', fontsize=12,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    def animate(frame):
        # Update analytical solution
        t = t_values[frame]
        u_analytical, v_analytical = compute_analytical_solution(t)
        
        line_u_analytical.set_ydata(u_analytical)
        line_v_analytical.set_ydata(v_analytical)
        
        # Update PDE solution if available
        if pde_u is not None and frame < len(pde_u):
            line_u_pde.set_ydata(pde_u[frame])
            if pde_v is not None:
                line_v_pde.set_ydata(pde_v[frame])
            
            # Update differences
            diff_u = pde_u[frame] - np.interp(pde_x, x, u_analytical)
            line_diff_u.set_ydata(diff_u)
            if pde_v is not None:
                diff_v = pde_v[frame] - np.interp(pde_x, x, v_analytical)
                line_diff_v.set_ydata(diff_v)
            
            # Update y-limits for difference plot
            all_diffs = diff_u
            if pde_v is not None:
                all_diffs = np.concatenate([all_diffs, diff_v])
            ax2.set_ylim(all_diffs.min() - 0.1, all_diffs.max() + 0.1)
        
        time_text.set_text(f't = {t:.3f}')
        
        # Return all line objects to update
        lines = [line_u_analytical, line_v_analytical, time_text]
        if pde_u is not None:
            lines.extend([line_u_pde, line_diff_u])
            if pde_v is not None:
                lines.extend([line_v_pde, line_diff_v])
        
        return lines
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    # Mass conservation check for analytical solution
    print("Analytical solution mass conservation check:")
    for t in [0, 0.5, 1.0, 2.0]:
        u, v = compute_analytical_solution(t)
        total_mass = np.trapz(u + v, x)
        print(f"t = {t:.1f}: Total mass = {total_mass:.6f}")
    
    return ani

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Use command line argument or default filename
    pde_filename = sys.argv[1] if len(sys.argv) > 1 else 'two_species_simulation.npz'
    
    if not Path(pde_filename).exists():
        print(f"PDE data file not found: {pde_filename}")
        print("Creating animation with analytical solution only...")
        pde_filename = None
    
    create_combined_animation(pde_filename)