import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 1.0
D = 0.01
alpha = 1.0
beta = 1.0
u0 = 1.0
v0 = 1.0
N_terms = 200
Nx = 200
T_total = 2.0
Nt = 100

# Spatial and time arrays
x = np.linspace(0, L, Nx)
t_values = np.linspace(0, T_total, Nt)

# Compute analytical solution at time t
def compute_solution(t):
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    
    # Handle n=0 mode separately (constant term)
    # For n=0, k_n = 0
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

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Initial plot
u_init, v_init = compute_solution(0)
line_u, = ax.plot(x, u_init, 'b-', linewidth=2, label='u(x,t)')
line_v, = ax.plot(x, v_init, 'r-', linewidth=2, label='v(x,t)')

# Setup plot
ax.set_xlabel('Position x')
ax.set_ylabel('Concentration')
ax.set_ylim(-0.1, 1.2)
ax.set_xlim(0, L)
ax.grid(True, alpha=0.3)
ax.legend()

# Add vertical lines to show initial regions
ax.axvline(x=0.25, color='blue', linestyle='--', alpha=0.5)
ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.5)

# Animation function
def animate(frame):
    t = t_values[frame]
    u, v = compute_solution(t)
    
    line_u.set_ydata(u)
    line_v.set_ydata(v)
    
    ax.set_title(f'Reaction-Diffusion System: t = {t:.2f}')
    
    return line_u, line_v

# Create animation
anim = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)

plt.tight_layout()
plt.show()

# Print mass conservation check
print("Mass conservation check:")
for t in [0, 0.5, 1.0, 2.0]:
    u, v = compute_solution(t)
    total_mass = np.trapz(u + v, x)
    print(f"t = {t:.1f}: Total mass = {total_mass:.6f}")