import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

hbar = 1.0
L = 1.0
N = 2**10
m = 1.0
x = np.linspace(0, L, N, endpoint=False)
dx = L / N

V_0 = 0.0
V = V_0 * np.sin(2 * np.pi * x / L)      
#V = V_0 * (x - 0.5)**2               

g = 0.0
initial_state = "gaussian"

def periodic_delta(x, x0, L): 
    return ((x - x0 + L/2) % L) - L/2

def psi0_soliton(x, L, a=2.0, x0=0.25*L, v=10.0):
    dxp = periodic_delta(x, x0, L)
    psi = a / np.cosh(a * dxp) * np.exp(1j * v * dxp) 
    norm = np.linalg.norm(psi) #np.sqrt(np.sum(np.abs(psi)**2) * (L/len(x)))
    return psi / norm
 
if initial_state == "soliton":
    psi0 = psi0_soliton(x, L, a=2.0, x0=0.5, v=0.0) 
elif initial_state == "flat":
    length = len(x)
    psi = np.ones(length)
    psi0 = psi / np.linalg.norm(psi) #np.sqrt(np.sum(np.abs(psi)**2) * (L/len(x)))
elif initial_state == "gaussian":
    center = 0.5
    sigma = 0.1
    psi0 = np.exp(-((x - center) ** 2) / (2 * sigma**2)).astype(np.complex128)
    psi0 /= np.linalg.norm(psi0) #np.sqrt(np.sum(np.abs(psi0)**2) * dx)
elif initial_state == "dirac": 
    center = int(np.floor(len(x) / 2) + 1)    
    psi0 = np.zeros_like(x)
    psi0[center] = 1.0 
 
k_wave = np.fft.fftfreq(N, d=dx)  #2.0 * np.pi 
k_wave = np.fft.fftshift(k_wave)
print(k_wave)

EVOLVE_DT = 0.01
kinetic_phase_dt = np.exp(-1j * (hbar * (k_wave**2) / (2.0 * m)) * EVOLVE_DT)

def evolve_step(psi, dt=EVOLVE_DT, g=g, enable_potential=True):
    if enable_potential:
        interaction_potential = V + g * np.abs(psi)**2
        psi = np.exp(-1j * interaction_potential * dt / 2.0) * psi

    psi_k = np.fft.fft(psi)
    if dt == EVOLVE_DT:
        psi_k *= kinetic_phase_dt
    else:
        phase = np.exp(-1j * (hbar * (k_wave**2) / (2.0 * m)) * dt)
        psi_k *= phase
    psi = np.fft.ifft(psi_k)

    if enable_potential:
        interaction_potential = V + g * np.abs(psi)**2
        psi = np.exp(-1j * interaction_potential * dt / 2.0) * psi
 
    #norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    norm = np.linalg.norm(psi)
    if norm != 0:
        psi /= norm
    return psi

def save_snapshots(density_rows, time_marks, save_prefix="splitting_classical"):
    """Save snapshots at specific time points: t=0, 0.2, 0.4, 0.6, 0.8, 1.0"""
    # Define specific time points
    target_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, target_time in enumerate(target_times):
        # Find the closest time index
        time_idx = np.argmin(np.abs(np.array(time_marks) - target_time))
        actual_time = time_marks[time_idx]
        
        if time_idx < len(density_rows):
            axes[i].plot(x, density_rows[time_idx], 'b-', linewidth=2)
            axes[i].set_title(f't = {actual_time:.3f}')
            axes[i].set_xlabel('Position x')
            axes[i].set_ylabel('$|\\psi(x)|^2$')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, np.max(density_rows) * 1.1)
    
    plt.suptitle(f'Time Evolution Snapshots (Classical Splitting Method)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_snapshots.pdf', dpi=600, bbox_inches='tight')
    plt.show()

T_MAX = 20.0
NUM_FRAMES = 1000
total_steps = int(np.round(T_MAX / EVOLVE_DT))
NUM_FRAMES = min(NUM_FRAMES, max(1, total_steps))
STEPS_PER_FRAME = max(1, total_steps // NUM_FRAMES)
REMAINDER_STEPS = total_steps - STEPS_PER_FRAME * (NUM_FRAMES - 1)
if REMAINDER_STEPS < 1:
    REMAINDER_STEPS = STEPS_PER_FRAME

psi_anim = psi0.copy()
current_time = 0.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
 
line, = ax1.plot(x, np.abs(psi_anim)**2, lw=2, zorder=3)
ax1.set_xlim(0, L)
ax1.set_xlabel("x")
ax1.set_ylabel(r"$|\psi(x,t)|^2$")
ax1.set_ylim(0, 1.2 * np.max(np.abs(psi0)**2))
title1 = ax1.set_title(f"t = {current_time:.3f} | V0={V_0}, g={g}")

ax1V = ax1.twinx()
ax1V.set_ylabel("V(x)")

Vmin, Vmax = float(np.min(V)), float(np.max(V))
Vpad = 0.05 * (Vmax - Vmin if Vmax > Vmin else (abs(Vmax) + 1.0))
ax1V.set_ylim(Vmin - Vpad, Vmax + Vpad)
 
V_line, = ax1V.plot(x, V, lw=1.0, alpha=0.7, zorder=1)
ax1V.fill_between(x, V, Vmin - Vpad, alpha=0.15, zorder=0)
 
ax1V.patch.set_alpha(0.0)
 
vmax0 = float(np.max(np.abs(psi0)**2)) * 1.2
im = ax2.imshow((np.abs(psi_anim)**2).reshape(N, 1),
                aspect='auto', origin='lower',
                extent=[0, EVOLVE_DT, 0, L],
                vmin=0.0, vmax=vmax0)
ax2.set_xlabel('Time')
ax2.set_ylabel('Position x')
ax2.set_title(r'Time Evolution of $|\psi(x,t)|^2$')
plt.colorbar(im, ax=ax2, label=r'$|\psi(x,t)|^2$')

density_rows = [np.abs(psi_anim)**2]
time_marks = [current_time]

def init():
    line.set_data(x, np.abs(psi_anim)**2)
    return line,

def update(frame_idx):
    global psi_anim, current_time, density_rows, time_marks

    steps_this_frame = STEPS_PER_FRAME if frame_idx < NUM_FRAMES - 1 else REMAINDER_STEPS
    for _ in range(steps_this_frame):
        psi_anim = evolve_step(psi_anim, dt=EVOLVE_DT)
        current_time += EVOLVE_DT

    prob_density = np.abs(psi_anim)**2
    density_rows.append(prob_density)
    time_marks.append(current_time)
 
    line.set_ydata(prob_density)
 
    ymax = max(ax1.get_ylim()[1], float(1.2 * np.max(prob_density)))
    ax1.set_ylim(0, ymax)
    title1.set_text(f"t = {current_time:.3f} | V0={V_0}, g={g}")
 
    density_grid = np.vstack(density_rows)   
    im.set_data(density_grid.T)            
    im.set_extent([0, current_time, 0, L])

    return line,

anim = FuncAnimation(fig, update, frames=NUM_FRAMES, init_func=init,
                     interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()

# Save snapshots after animation
save_snapshots(density_rows, time_marks, "splitting_classical")
 
#anim.save('splitting_classical_evolution.gif', writer='pillow', fps=10)
