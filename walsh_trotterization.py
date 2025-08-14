import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
 
from walsh_laplacian_pauli import decompose_walsh_laplacian, decompose_walsh_laplacian_base
from walsh_potential_pauli import build_walsh_pauli_potential


def create_initial_wavefunction(N, initial_state="gaussian"):
    
    size = 2**N
    
    if initial_state == "gaussian": 
        x_values = np.linspace(0, 1, size, endpoint=False)
        center = 0.25
        sigma = 0.15
        psi = np.exp(-((x_values - center)**2) / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown initial state: {initial_state}")
    
    psi = psi / np.linalg.norm(psi)
    return psi.astype(complex)

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
            
            if abs(phase) < 1e-12:   
                continue
            
            # exp(-i θ Z_i Z_j) = CNOT_ij RZ(2θ) CNOT_ij
            qubit1 = l1 - 1  
            qubit2 = l2 - 1
            
            qc.cx(qubit1, qubit2)
            qc.rz(2 * phase, qubit2)
            qc.cx(qubit1, qubit2)
    
    return qc

def create_trotter_step_circuit(N, dt, V0=1.0, walsh_factor=None, power_factor=None, potential_factor=None):
    """
    Create a second-order Trotter step circuit: exp(-i V dt/2) exp(-i K dt) exp(-i V dt/2)
    
    The circuit structure is:
    1. exp(-i V_WP dt/2) - First half of potential evolution
    2. QFT - Transform to momentum space
    3. exp(-i K_WP dt) - Kinetic evolution in momentum space  
    4. QFT† - Transform back to position space
    5. exp(-i V_WP dt/2) - Second half of potential evolution
    """
     
    qc = QuantumCircuit(N)
    
    # Second-order Trotter: exp(-i V dt/2) exp(-i K dt) exp(-i V dt/2)
    
    # First half of potential evolution: exp(-i V_WP dt/2)
    potential_circuit_half = create_potential_wp_circuit(N, dt/2, V0, potential_factor)
    qc.compose(potential_circuit_half, inplace=True)
    
    # Kinetic evolution in momentum space: exp(-i K_WP dt)
    qft_gate = QFTGate(N)
    qc.append(qft_gate, range(N))
    
    qc.x(0)
    
    kinetic_circuit = create_kinetic_wp_circuit(N, dt, walsh_factor, power_factor)
    qc.compose(kinetic_circuit, inplace=True)
     
    qc.x(0)
    
    # QFT†
    qft_dagger = QFTGate(N).inverse()
    qc.append(qft_dagger, range(N))
    
    # Second half of potential evolution: exp(-i V_WP dt/2)
    potential_circuit_half2 = create_potential_wp_circuit(N, dt/2, V0, potential_factor)
    qc.compose(potential_circuit_half2, inplace=True)
    
    return qc

def simulate_quantum_time_evolution(N, T_final=1.0, n_steps=50, V0=1.0, initial_state="gaussian", walsh_factor=None, power_factor=None, potential_factor=None):
   
    dt = T_final / n_steps
    times = np.linspace(0, T_final, n_steps + 1)
     
    psi_initial = create_initial_wavefunction(N, initial_state)
     
    simulator = AerSimulator(method='statevector')
     
    size = 2**N
    psi_evolution = np.zeros((n_steps + 1, size), dtype=complex)
    probability_evolution = np.zeros((n_steps + 1, size))
     
    psi_evolution[0] = psi_initial
    probability_evolution[0] = np.abs(psi_initial)**2
    
    print(f"Walsh-Pauli: N={N}, steps={n_steps}")
      
    trotter_circuit = create_trotter_step_circuit(N, dt, V0, walsh_factor, power_factor, potential_factor)
     
    current_statevector = Statevector((psi_initial))
      
    for i in range(n_steps): 
        current_statevector = current_statevector.evolve(trotter_circuit).copy()
         
        psi_current = current_statevector.data
        psi_evolution[i + 1] = psi_current
        probability_evolution[i + 1] = np.abs(psi_current)**2
         
        norm = np.linalg.norm(psi_current)
        if abs(norm - 1.0) > 1e-10:
            print(f"Warning: norm deviation at step {i+1}: {norm:.10f}")
    
    return times, psi_evolution, probability_evolution



def analyze_trotter_circuit(N, dt, V0=1.0, walsh_factor=None, power_factor=None, potential_factor=None):
       
    qft_circuit = QuantumCircuit(N)
    qft_circuit.append(QFTGate(N), range(N))
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
     
    time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
    time_labels = ['t=0.0', f't={times[len(times)//4]:.2f}', f't={times[len(times)//2]:.2f}', 
                   f't={times[3*len(times)//4]:.2f}', f't={times[-1]:.2f}']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for idx, (time_idx, label, color) in enumerate(zip(time_indices, time_labels, colors)):
        ax2.plot(x_values, probability_evolution[time_idx], 
                label=label, color=color, linewidth=2)
    
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('$|\\psi(x)|^2$')
    ax2.set_title('Probability Density at Different Times')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_N{N}.pdf', dpi=600, bbox_inches='tight')
    plt.show()
     
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    snapshot_indices = np.linspace(0, len(times)-1, 6, dtype=int)
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
    qft_circuit.append(QFTGate(N), range(N))
    
    kinetic_circuit = create_kinetic_wp_circuit(N, dt, walsh_factor, power_factor)
    potential_circuit = create_potential_wp_circuit(N, dt, V0, potential_factor)
    trotter_circuit = create_trotter_step_circuit(N, dt, V0, walsh_factor, power_factor, potential_factor)
    
 
    fig = trotter_circuit.draw(output='mpl', fold=-1)
    plt.title(f'Walsh-Pauli Trotter Circuit (N={N})')
    plt.savefig(f'trotter_circuit_N{N}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
           
    return trotter_circuit

if __name__ == "__main__":

    N = 8  
    T_final = 1 
    n_steps = 100
    V0 = 10   
    initial_state = "gaussian"   
     
    walsh_factor = (np.pi**2) / 6  
    power_factor = 4**N         
    potential_factor = V0 / 12    
    
    dt = T_final / n_steps   
      
    simulation_circuit = show_simulation_circuit(N, dt, V0)
      
    times_wp, psi_wp, prob_wp = simulate_quantum_time_evolution(
            N=N, T_final=T_final, n_steps=n_steps, V0=V0, initial_state=initial_state,
            walsh_factor=walsh_factor, power_factor=power_factor, potential_factor=potential_factor
        )
    
     
    visualize_time_evolution(times_wp, prob_wp, N, "walsh_pauli_simulation")
         