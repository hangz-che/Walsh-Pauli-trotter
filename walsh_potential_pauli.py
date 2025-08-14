import numpy as np
import matplotlib.pyplot as plt 

def build_classical_diagonal_potential(N, V0=1.0):
   
    size = 2**N
    V_classical = np.zeros((size, size), dtype=float)
    
    for j in range(size):
        x = j / size  # x_j ∈ [0, 1)
        potential_value = V0 * (x - 0.5)**2
        V_classical[j, j] = potential_value
    
    return V_classical

def build_walsh_pauli_potential(N, V0=1.0):
   
    factor = V0 / 12
    size = 2**N

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    V_walsh = np.zeros((size, size), dtype=complex)
 
    identity_term = factor
    for i in range(size):
        V_walsh[i, i] += identity_term
     
    for l1 in range(1, N+1):
        for l2 in range(l1+1, N+1):
            coeff = 3 * factor * (2**(-(l1 + l2 - 1)))
             
            term_matrix = np.array([[1]], dtype=complex)
            for pos in range(1, N+1):
                if pos == l1 or pos == l2:
                    term_matrix = np.kron(term_matrix, Z)
                else:
                    term_matrix = np.kron(term_matrix, I)
            
            V_walsh += coeff * term_matrix
    
    return V_walsh

def compute_operator_norm(matrix): 
    return np.linalg.norm(matrix, ord=2)

def compute_frobenius_norm(matrix): 
    return np.linalg.norm(matrix, ord='fro')

def compare_potential_matrices(N_max=6, V0=1.0): 
    N_values = list(range(2, N_max + 1))
    classical_norms_frobenius = []
    walsh_norms_frobenius = []
    difference_norms_spectral = []
    difference_norms_frobenius = []
    relative_errors = []
    
    print(f"Comparing potential matrix Frobenius norms for different N (V0={V0}):")
    print("=" * 100)
    print(f"{'N':<3} {'||V_diag^N||_F':<15} {'||V_WP^N||_F':<15} {'||V1-V2||_F':<15} {'Relative Error':<15}")
    print("-" * 100)
    
    for N in N_values: 
        V_classical = build_classical_diagonal_potential(N, V0)
        V_walsh = build_walsh_pauli_potential(N, V0) 
         
        norm_classical_frobenius = compute_frobenius_norm(V_classical)
        norm_walsh_frobenius = compute_frobenius_norm(V_walsh)
        
        difference_matrix = V_classical - V_walsh
        difference_norm_spectral = compute_operator_norm(difference_matrix)  # Spectral norm
        difference_norm_frobenius = compute_frobenius_norm(difference_matrix)  # Frobenius norm
        
        relative_error = difference_norm_frobenius / norm_classical_frobenius if norm_classical_frobenius != 0 else float('inf')
        
        classical_norms_frobenius.append(norm_classical_frobenius)
        walsh_norms_frobenius.append(norm_walsh_frobenius)
        difference_norms_spectral.append(difference_norm_spectral)
        difference_norms_frobenius.append(difference_norm_frobenius)
        relative_errors.append(relative_error)
        
        print(f"{N:<3} {norm_classical_frobenius:<15.6f} {norm_walsh_frobenius:<15.6f} {difference_norm_frobenius:<15.6f} {relative_error:<15.6f}")
    
    return N_values, classical_norms_frobenius, walsh_norms_frobenius, difference_norms_spectral, difference_norms_frobenius, relative_errors

def visualize_potential_convergence(N_values, classical_norms, walsh_norms, difference_norms_frobenius, relative_errors, V0=1.0):
    
    plt.rcParams['mathtext.default'] = 'regular'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
     
    ax1.semilogy(N_values, classical_norms, 'bo-', label=r'$\|V_{\mathrm{diag}}^N\|_F$', linewidth=2, markersize=8)
    ax1.semilogy(N_values, walsh_norms, 'ro-', label=r'$\|V_{\mathrm{WP}}^N\|_F$', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of qubits')
    ax1.set_ylabel('Frobenius norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
     
    ax2.semilogy(N_values, relative_errors, 'go-', linewidth=2, markersize=8, label='Relative error')
    
    # Add 1/2^N decay baseline
    baseline_decay = [1.0 / (2**N) for N in N_values]
    ax2.semilogy(N_values, baseline_decay, 'k--', linewidth=2, alpha=0.7, label=r'$1/2^N$ baseline')
    
    ax2.set_xlabel('Number of qubits')
    ax2.set_ylabel(r'$\frac{\|V_{\mathrm{diag}}^N - V_{\mathrm{WP}}^N\|_F}{\|V_{\mathrm{diag}}^N\|_F}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'potential_convergence_V0_{V0}.pdf', dpi=600, bbox_inches='tight')
    plt.show()
 

if __name__ == "__main__": 
    V0 = 1.0 
    N_values, classical_norms_frobenius, walsh_norms_frobenius, difference_norms_spectral, difference_norms_frobenius, relative_errors = compare_potential_matrices(N_max=8, V0=V0)
   
    visualize_potential_convergence(N_values, classical_norms_frobenius, walsh_norms_frobenius, difference_norms_frobenius, relative_errors, V0)
 
 