import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from pathlib import Path


def load_data(filename):
    """Load u and optionally v and timevector/dt from a .npz file.

    Recognises common keys: 'u_record', 'u', 'u_rec' for u and similarly for v.
    Also looks for 'timevector' or 't' or a scalar 'dt' to build a time vector.
    Returns (u, v, timevector or None).
    """
    data = np.load(filename, allow_pickle=True)
    keys = set(data.files)
    # candidate names
    u_keys = ['u_record', 'u_rec', 'u']
    v_keys = ['v_record', 'v_rec', 'v']

    u = None
    v = None
    for k in u_keys:
        if k in keys:
            u = data[k]
            break
    for k in v_keys:
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


def animate_1D(u, v=None, interval=50, x=None, timevector=None, show=True):
    """Animate 1D time-series arrays.

    Parameters:
    - u: array-like, shape (T, N)
    - v: optional array-like, shape (T, N)
    - interval: ms between frames
    - x: spatial coordinates (length N). If None, uses 0..N-1
    - show: whether to call plt.show()
    """
    u = np.asarray(u)
    if u.ndim != 2:
        raise ValueError("u must be a 2D array with shape (time, space)")

    T, N = u.shape
    if x is None:
        x = np.arange(N)
    else:
        x = np.asarray(x)
        if x.shape[0] != N:
            raise ValueError("x must have length equal to the spatial dimension of u")

    fig, ax = plt.subplots()
    line_u, = ax.plot(x, u[0], label='u')
    lines = [line_u]

    if v is not None:
        v = np.asarray(v)
        if v.shape != u.shape:
            raise ValueError("v must have the same shape as u")
        line_v, = ax.plot(x, v[0], label='v')
        lines.append(line_v)

    # determine y-limits
    ymin = u.min() if v is None else min(u.min(), v.min())
    ymax = u.max() if v is None else max(u.max(), v.max())
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5

    ax.set_xlim(x.min(), x.max())
    # small padding
    pad_low = ymin - 0.05 * (ymax - ymin if ymax != ymin else 1.0)
    pad_high = ymax + 0.05 * (ymax - ymin if ymax != ymin else 1.0)
    ax.set_ylim(pad_low, pad_high)
    ax.set_xlabel('Spatial index')
    ax.set_ylabel('Concentration')
    ax.legend()

    # time counter text in top-right corner (axes fraction coordinates)
    if timevector is None:
        timevector = np.arange(0, T)
    # If timevector length mismatches frames, fallback to frame indices
    if len(timevector) != T:
        timevector = np.arange(0, T)
    time_text = ax.text(0.98, 0.98, f't = {timevector[0]:.3f}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

    def update(frame):
        line_u.set_ydata(u[frame])
        if v is not None:
            line_v.set_ydata(v[frame])
        # update time counter
        time_text.set_text(f't = {timevector[frame]:.3f}')
        return tuple(lines) + (time_text,)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=interval, blit=True)

    if show:
        plt.show()

    return ani


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else 'two_species_simulation.npz'
    if not Path(filename).exists():
        print(f"File not found: {filename}", file=sys.stderr)
        sys.exit(2)

    u, v, timevector = load_data(filename)

    # subsample frames
    skip = 40
    u = u[::skip]
    if v is not None:
        v = v[::skip]
    if timevector is not None:
        timevector = timevector[::skip]

    animate_1D(u, v, timevector=timevector, interval=30)


    
