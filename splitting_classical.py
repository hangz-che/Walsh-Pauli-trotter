import numpy as np
import matplotlib.pyplot as plt
 
hbar = 1.0
L = 1.0          
N = 2**10
m = 1 
x = np.linspace(0, L, N, endpoint=False)
dx = L / N
  
V_0 = 0.0
V = V_0 * (x - 0.5)**2  
 
g = 0.0

center = 0.5
sigma = 0.1
psi0 = np.exp(-((x - center) ** 2) / (2 * sigma**2)) 
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)
  
k = np.fft.fftfreq(N, d=dx) #* 2 * np.pi
   
def evolve_with_potential(psi_initial, t, hbar=1.0, m=1.0, dt=0.005, g=1.0):
     
    psi = psi_initial.copy()
     
    n_steps = int(t / dt)
    
    for _ in range(n_steps):
        # e^(-i(V+I)t/2)  
        interaction_potential = V + g * np.abs(psi)**2
        psi = np.exp(-1j * interaction_potential * dt / 2) * psi
        
        # e^(-iKt)  
        psi_k = np.fft.fft(psi)
        phase = np.exp(-1j * (hbar * k**2 / (2 * m)) * dt)
        psi_k = psi_k * phase
        psi = np.fft.ifft(psi_k)
        
        # e^(-i(V+I)t/2)  
        interaction_potential = V + g * np.abs(psi)**2
        psi = np.exp(-1j * interaction_potential * dt / 2) * psi

    return psi
  
times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
 
for i, t in enumerate(times):
    psi_t = evolve_with_potential(psi0, t, hbar=hbar, m=m, g=g)
    prob_density = np.abs(psi_t)**2 
    print(f"Sum over discrete points at time {t} = {np.sum(prob_density) * dx:.6f}")
    axes[i//3, i%3].plot(x, prob_density)
    axes[i//3, i%3].set_title(f"t = {t:.1f}")
    axes[i//3, i%3].set_xlabel("x")
    axes[i//3, i%3].set_ylabel(r"$|\psi(x,t)|^2$")

fig.suptitle(f"Gaussian Evolution with Potential V = {V_0} (x-1/2)² + Interaction {g}|\psi|² (g = {g})", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(f'gaussian_evolution_with_potential_{V_0}_interaction_g{g}.png', dpi=300, bbox_inches='tight')
plt.show()
