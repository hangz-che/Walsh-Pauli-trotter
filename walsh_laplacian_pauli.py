import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def build_classical_diagonal_matrix(N):
    
    size = 2**N
    K_classical = np.zeros((size, size), dtype=float)
    
    for j in range(size):
        zeta = j / size  # ζ ∈ Ω_N
        eigenval = 2 * (np.pi**2) * (4**N) * (zeta - 0.5)**2
        K_classical[j, j] = eigenval
    
    return K_classical

def decompose_walsh_laplacian_base(N, walsh_factor=None, power_factor=None):
    
    if walsh_factor is None:
        walsh_factor = (np.pi**2) / 6
    if power_factor is None:
        power_factor = 4**N
        
    decomp = {}
    
    # walsh_factor * power_factor * I^⊗N
    decomp['I' * N] = walsh_factor * power_factor
    
    # 3 * walsh_factor * Σ Z_ℓ₁ Z_ℓ₂ 
    for l1 in range(1, N+1):
        for l2 in range(l1+1, N+1):
            pauli_str = ['I'] * N
            pauli_str[l1-1] = 'Z'
            pauli_str[l2-1] = 'Z'
            pauli_str = ''.join(pauli_str)
            coeff = 3 * walsh_factor * (2**(2*N - (l1 + l2 - 1)))
            decomp[pauli_str] = coeff
    
    return decomp

def decompose_walsh_laplacian(N): 
    return decompose_walsh_laplacian_base(N)

def build_walsh_pauli_matrix(N):
    """ 
    K^N_diag,WP = (π²/6)(4^N I^⊗N + 3 Σ_{1≤ℓ₁<ℓ₂≤N} 2^{2N-(ℓ₁+ℓ₂-1)} Z_ℓ₁ Z_ℓ₂)
    """
    factor = (np.pi**2) / 6
    size = 2**N

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    K_walsh = np.zeros((size, size), dtype=complex)

    identity_term = factor * (4**N)
    for i in range(size):
        K_walsh[i, i] += identity_term
    
    for l1 in range(1, N+1):
        for l2 in range(l1+1, N+1):
            coeff = 3 * factor * (2**(2*N - (l1 + l2 - 1)))
            
            term_matrix = np.array([[1]], dtype=complex)
            for pos in range(1, N+1):
                if pos == l1 or pos == l2:
                    term_matrix = np.kron(term_matrix, Z)
                else:
                    term_matrix = np.kron(term_matrix, I)
            
            K_walsh += coeff * term_matrix
    
    return K_walsh

def compute_operator_norm(matrix):
    return np.linalg.norm(matrix, ord=2)

def compute_frobenius_norm(matrix):
    return np.linalg.norm(matrix, ord='fro')

def compare_matrices(N_max=6): 
    N_values = list(range(2, N_max + 1))
    classical_norms_frobenius = []
    walsh_norms_frobenius = []
    difference_norms_spectral = []
    difference_norms_frobenius = []
    relative_errors = []
    
    print("Comparing Frobenius norms for different N:")
    print("=" * 100)
    print(f"{'N':<3} {'||K_diag^N||_F':<15} {'||K_diag,WP^N||_F':<15} {'||K1-K2||_F':<15} {'Relative Error':<15}")
    print("-" * 100)
    
    for N in N_values: 
        K_classical = build_classical_diagonal_matrix(N)
        K_walsh = build_walsh_pauli_matrix(N) 
         
        norm_classical_frobenius = compute_frobenius_norm(K_classical)
        norm_walsh_frobenius = compute_frobenius_norm(K_walsh)
        
   
        difference_matrix = K_classical - K_walsh
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

def visualize_convergence(N_values, classical_norms, walsh_norms, difference_norms_frobenius, relative_errors):
    
    plt.rcParams['mathtext.default'] = 'regular'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogy(N_values, classical_norms, 'bo-', label=r'$\|K_{\mathrm{diag}}^N\|_F$', linewidth=2, markersize=8)
    ax1.semilogy(N_values, walsh_norms, 'ro-', label=r'$\|K_{\mathrm{diag,WP}}^N\|_F$', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of qubits')
    ax1.set_ylabel('Frobenius norm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(N_values, relative_errors, 'go-', linewidth=2, markersize=8, label='Relative error')
    
    # Add 1/2^N decay baseline
    baseline_decay = [1 / (2**N) for N in N_values]
    ax2.semilogy(N_values, baseline_decay, 'k--', linewidth=2, alpha=0.7, label=r'$1/2^N$ baseline')
    
    ax2.set_xlabel('Number of qubits')
    ax2.set_ylabel(r'$\frac{\|K_{\mathrm{diag}}^N - K_{\mathrm{diag,WP}}^N\|_F}{\|K_{\mathrm{diag}}^N\|_F}$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laplacian_convergence.pdf', dpi=600, bbox_inches='tight')
    plt.show()
 
if __name__ == "__main__": 
    N_values, classical_norms_frobenius, walsh_norms_frobenius, difference_norms_spectral, difference_norms_frobenius, relative_errors = compare_matrices(N_max=8)
     
    visualize_convergence(N_values, classical_norms_frobenius, walsh_norms_frobenius, difference_norms_frobenius, relative_errors)
 