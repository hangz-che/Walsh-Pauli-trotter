import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate  
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector  
from walsh_laplacian_pauli import decompose_walsh_laplacian, decompose_walsh_laplacian_base
from walsh_potential_pauli import build_walsh_pauli_potential
 

def periodic_delta(x, x0, L): 
    return ((x - x0 + L/2) % L) - L/2

def psi0_soliton(x, L, a=2.0, x0=0.5, v=0.0):
    dxp = periodic_delta(x, x0, L)
    psi = a / np.cosh(a * dxp) * np.exp(1j * v * dxp) 
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (L/len(x)))
    return psi / norm

def create_initial_wavefunction(N, initial_state="soliton"):
    size = 2**N
    L = 1.0
    x_values = np.linspace(0, L, size, endpoint=False)
    dx = L / N
 
    if initial_state == "gaussian": 
        center = 0.5  
        sigma = 0.1  
        psi = np.exp(-((x_values - center)**2) / (2 * sigma**2))
        psi = psi / np.linalg.norm(psi)  
        #psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

    elif initial_state == "dirac": 
        center = int(np.floor(len(x_values) / 2) + 1)   
        psi = np.zeros_like(x_values)
        psi[center] = 1.0
        psi = psi / np.linalg.norm(psi)  

    elif initial_state == "flat":
        length = len(x_values)
        psi = np.ones(length)
        psi = psi / np.linalg.norm(psi)  

    elif initial_state == "soliton": 
        psi = psi0_soliton(x_values, L, a=2.0, x0=0.5, v=0.0)    
    else:
        raise ValueError(f"Unknown initial state: {initial_state}")
 
    psi_flipped = psi[::-1]
    symmetry_error = np.max(np.abs(psi - psi_flipped))
    print(f"Initial state symmetry error: {symmetry_error:.2e}")

    return psi

 

def create_quantum_amplitude_encoding(N, psi_classical):
     
    size = 2**N
    if len(psi_classical) != size:
        raise ValueError(f"Wavefunction length {len(psi_classical)} must equal 2^{N} = {size}")
     
    return Statevector(psi_classical)

def create_quantum_initial_state(N, initial_state="gaussian"):
     
    psi_classical = create_initial_wavefunction(N, initial_state)
     
    quantum_statevector = create_quantum_amplitude_encoding(N, psi_classical)
    
    return psi_classical, quantum_statevector

def create_kinetic_wp_circuit(N, dt, walsh_factor=None, power_factor=None):
     
    qc = QuantumCircuit(N)
     
    if walsh_factor is None:
        walsh_factor = (np.pi**2) / 6
    if power_factor is None:
        power_factor = 4**N
     
    kinetic_decomp = decompose_walsh_laplacian_base(N, walsh_factor, power_factor)
     
    for pauli_str, coeff in kinetic_decomp.items():
        # exp(-i * coeff * Z_i Z_j * dt)
        phase = coeff * dt  

        z_qubits = [i for i, p in enumerate(pauli_str) if p == 'Z']   
 
        if len(z_qubits) == 2:
            # ZZ interaction: exp(-i θ Z_i Z_j) = CNOT_ij RZ(2θ) CNOT_ij
            qc.cx(z_qubits[0], z_qubits[1])
            qc.rz(2 * phase, z_qubits[1])
            qc.cx(z_qubits[0], z_qubits[1])
    
    return qc

def create_potential_wp_circuit(N, dt, V0=1.0, potential_factor=None):
    
    qc = QuantumCircuit(N)
     
    if potential_factor is None:
        potential_factor = V0 / 12
 
    for l1 in range(1, N+1):
        for l2 in range(l1+1, N+1):
            coeff = 3 * potential_factor * (2**(-(l1 + l2 - 1)))
            phase = coeff * dt 
            # exp(-i θ Z_i Z_j) = CNOT_ij RZ(2θ) CNOT_ij
            qubit1 = l1 - 1  
            qubit2 = l2 - 1
            
            qc.cx(qubit1, qubit2)
            qc.rz(2 * phase, qubit2)
            qc.cx(qubit1, qubit2)
    
    return qc

def create_trotter_step_circuit(N, dt, V0=1.0, walsh_factor=None, power_factor=None, potential_factor=None):
 
    qc = QuantumCircuit(N)
    
    # Second-order Trotter: exp(-i V dt/2) exp(-i K dt) exp(-i V dt/2)
    
    # First half of potential evolution: exp(-i V_WP dt/2)
    if V0 != 0:
        potential_circuit_half = create_potential_wp_circuit(N, dt/2, V0, potential_factor)
        qc.compose(potential_circuit_half, inplace=True)
    
    # Kinetic evolution in momentum space: exp(-i K_WP dt)
    qft_gate = QFTGate(N) 
    qc.append(qft_gate, range(N))
    
    #qc.x(0)  
    
    kinetic_circuit = create_kinetic_wp_circuit(N, dt, walsh_factor, power_factor)
    qc.compose(kinetic_circuit, inplace=True)
     
    #qc.x(0)  

    qft_gate = QFTGate(N) 
    qc.append(qft_gate, range(N))
  
    # Second half of potential evolution: exp(-i V_WP dt/2)
    if V0 != 0:
        potential_circuit_half2 = create_potential_wp_circuit(N, dt/2, V0, potential_factor)
        qc.compose(potential_circuit_half2, inplace=True)
    
    return qc

def simulate_quantum_time_evolution(N, T_final=1.0, n_steps=50, V0=1.0, initial_state="gaussian", walsh_factor=None, power_factor=None, potential_factor=None, use_quantum_encoding=True):
   
    dt = T_final / n_steps
    times = np.linspace(0, T_final, n_steps + 1)
     
    if use_quantum_encoding:
        # Use quantum amplitude encoding
        psi_initial, initial_quantum_statevector = create_quantum_initial_state(N, initial_state)
        print(f"Using quantum amplitude encoding for initial state")
    else: 
        psi_initial = create_initial_wavefunction(N, initial_state)
        initial_quantum_statevector = None
        print(f"Using classical initial state preparation")
     
    simulator = AerSimulator(method='statevector')
     
    size = 2**N
    psi_evolution = np.zeros((n_steps + 1, size), dtype=complex)
    probability_evolution = np.zeros((n_steps + 1, size))
     
    psi_evolution[0] = psi_initial
    probability_evolution[0] = np.abs(psi_initial)**2
    
    print(f"Walsh-Pauli: N={N}, steps={n_steps}")
      
    trotter_circuit = create_trotter_step_circuit(N, dt, V0, walsh_factor, power_factor, potential_factor)
     
    if use_quantum_encoding:
        # Use quantum amplitude encoded initial state
        current_statevector = initial_quantum_statevector
    else: 
        current_statevector = Statevector((psi_initial))
      
    for i in range(n_steps): 
        current_statevector = current_statevector.evolve(trotter_circuit).copy()
         
        psi_current = current_statevector.data
        psi_evolution[i + 1] = psi_current
        probability_evolution[i + 1] = np.abs(psi_current)**2
         
        norm = np.linalg.norm(psi_current)
        prob_sum = np.sum(probability_evolution[i + 1])
         
        if (i + 1) % 100 == 0 or i < 5:
            print(f"Step {i+1}: norm={norm:.6f}, prob_sum={prob_sum:.6f}, max_prob={np.max(probability_evolution[i + 1]):.6f}")
        
        if abs(norm - 1.0) > 1e-10:
            print(f"Warning: norm deviation at step {i+1}: {norm:.10f}")
        
        if abs(prob_sum - 1.0) > 1e-10:
            print(f"Warning: probability sum deviation at step {i+1}: {prob_sum:.10f}")
    
    return times, psi_evolution, probability_evolution



def analyze_trotter_circuit(N, dt, V0=1.0, walsh_factor=None, power_factor=None, potential_factor=None):
       
    qft_circuit = QuantumCircuit(N)

    kinetic_circuit = create_kinetic_wp_circuit(N, dt, walsh_factor, power_factor)
    potential_circuit = create_potential_wp_circuit(N, dt, V0, potential_factor)
    trotter_circuit = create_trotter_step_circuit(N, dt, V0, walsh_factor, power_factor, potential_factor)
    
    print(f"QFT circuit:")
    print(f"  Depth: {qft_circuit.depth()}")
    print(f"  Gate count: {len(qft_circuit.data)}")
    
    print(f"\nKinetic evolution circuit exp(-i K_WP dt):")
    print(f"  Depth: {kinetic_circuit.depth()}")
    print(f"  Gate count: {len(kinetic_circuit.data)}")
    
    print(f"\nPotential evolution circuit exp(-i V_WP dt):")
    print(f"  Depth: {potential_circuit.depth()}")
    print(f"  Gate count: {len(potential_circuit.data)}")
    
    print(f"\nComplete Trotter step circuit:")
    print(f"  Depth: {trotter_circuit.depth()}")
    print(f"  Gate count: {len(trotter_circuit.data)}")
     
    kinetic_decomp = decompose_walsh_laplacian(N)
    print(f"\nWalsh-Pauli decomposition:")
    print(f"  Kinetic terms: {len(kinetic_decomp)}")
   
    potential_terms = N * (N - 1) // 2 + 1   
    print(f"  Potential terms: {potential_terms}")
    
    return trotter_circuit



def visualize_time_evolution(times, probability_evolution, N, save_prefix="walsh_evolution"):
    
    size = 2**N
    x_values = np.linspace(0, 1, size, endpoint=False)
     
    plt.rcParams['mathtext.default'] = 'regular'
     
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
     
    im1 = ax1.imshow(probability_evolution.T, aspect='auto', origin='lower', 
                     extent=[0, times[-1], 0, 1], cmap='viridis')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position x')
    ax1.set_title(f'Time Evolution of $|\\psi(x,t)|^2$ (N={N} qubits)')
    plt.colorbar(im1, ax=ax1, label='$|\\psi(x,t)|^2$')
     
    time_indices = np.linspace(0, len(times)-1, 6, dtype=int)
    time_labels = [f't={times[idx]:.3f}' for idx in time_indices]
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    for idx, (time_idx, label, color) in enumerate(zip(time_indices, time_labels, colors)):
        ax2.plot(x_values, probability_evolution[time_idx], 
                label=label, color=color, linewidth=2)
    
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('$|\\psi(x)|^2$')
    ax2.set_title('Probability Density at Different Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Set the same y-axis range as the second image
    ax2.set_ylim(0, np.max(probability_evolution) * 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_N{N}.pdf', dpi=600, bbox_inches='tight')
    plt.show()
     
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    snapshot_indices = time_indices
    print(snapshot_indices)
    
    for i, time_idx in enumerate(snapshot_indices):
        axes[i].plot(x_values, probability_evolution[time_idx], 'b-', linewidth=2)
        axes[i].set_title(f't = {times[time_idx]:.3f}')
        axes[i].set_xlabel('Position x')
        axes[i].set_ylabel('$|\\psi(x)|^2$')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, np.max(probability_evolution) * 1.1)
    
    plt.suptitle(f'Time Evolution Snapshots (N={N} qubits)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_snapshots_N{N}.pdf', dpi=600, bbox_inches='tight')
    plt.show()
 

def show_simulation_circuit(N, dt, V0=0):
    walsh_factor = (np.pi**2) / 6
    power_factor = 4**N
    potential_factor = V0 / 12
     
    qft_circuit = QuantumCircuit(N)
     
    kinetic_circuit = create_kinetic_wp_circuit(N, dt, walsh_factor, power_factor)
    potential_circuit = create_potential_wp_circuit(N, dt, V0, potential_factor)
    trotter_circuit = create_trotter_step_circuit(N, dt, V0, walsh_factor, power_factor, potential_factor)
     
    fig = trotter_circuit.draw(output='mpl', fold=0)
    plt.title(f'Walsh-Pauli Trotter Circuit (N={N})')
    plt.savefig(f'trotter_circuit_N{N}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
           
    return trotter_circuit
 
if __name__ == "__main__":

    N = 10   
    T_final = 1 
    dt = 0.01 
    n_steps = int(T_final / dt)
    V0 = 0   
    initial_state = "gaussian"   
     
    walsh_factor = (np.pi**2) / 6  
    power_factor = 4**N         
    potential_factor = V0 / 12    
      
 
    simulation_circuit = show_simulation_circuit(N, dt, V0)
      
 
    times_wp, psi_wp, prob_wp = simulate_quantum_time_evolution(
            N=N, T_final=T_final, n_steps=n_steps, V0=V0, initial_state=initial_state,
            walsh_factor=walsh_factor, power_factor=power_factor, potential_factor=potential_factor,
            use_quantum_encoding=True  
        )
     
     
    visualize_time_evolution(times_wp, prob_wp, N, "walsh_pauli_simulation")
         