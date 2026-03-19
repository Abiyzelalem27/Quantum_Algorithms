

import numpy as np
from scipy.sparse import kron, identity, csc_matrix
from qutip import qeye, sigmax, tensor, sigmay, sigmaz
from qutip import *

def H0_transverse(N):
    """Transverse field Hamiltonian H0 = -sum σx_i"""
    sx_list = []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
    H0 = -sum(sx_list)
    return H0 

def H1_ising(N):
    """Ising Hamiltonian H1 = -sum σz_i σz_{i+1}"""
    sz_list = []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))
    H1 = -sum(sz_list[i] * sz_list[i+1] for i in range(N-1))
    return H1 


def initial_state_for_ADST(N): 
    psi_single = np.array([1, 1], dtype=complex)/np.sqrt(2)
    psi0 = psi_single
    for _ in range(N-1):
        psi0 = np.kron(psi0, psi_single)
    return psi0

def build_operators(N):
    """Construct σx and σz tensor operators for N qubits."""
    sx_list = []
    sz_list = []
    for i in range(N):
        op_list = [qeye(2)] * N
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list = [qeye(2)] * N
        op_list[i] = sigmaz() 
        sz_list.append(tensor(op_list)) 
    return sx_list, sz_list 

def build_hamiltonians(N):
    """Build H0 (transverse field) and H1 (Ising) Hamiltonians."""
    sx_list, sz_list = build_operators(N)
    H0 = sum(-sx for sx in sx_list)
    H1 = sum(-sz_list[i]*sz_list[i+1] for i in range(N-1))
    return H0, H1

def instantaneous_spectrum(H0, H1, times, t_final):
    """Compute eigenvalues and eigenstates of H(t) = (1-λ) H0 + λ H1."""
    eigenvals = []
    eigenstates = []
    for t in times:
        lam = t / t_final
        Ht = (1-lam)*H0 + lam*H1
        evals, evecs = Ht.eigenstates()
        eigenvals.append(evals)
        eigenstates.append(evecs)
    return np.array(eigenvals), eigenstates


def lam(t, args):
    """Time-dependent interpolation parameter λ(t)."""
    return t / args['t_final']

def simulate_dynamics(H0, H1, times, t_final):
    """Time evolution under H(t) using mesolve."""
    psi0 = H0.groundstate()[1]
    H = [H0, [H1, lam]]
    args = {'t_final': t_final}
    result = mesolve(H, psi0, times, c_ops=[], e_ops=[], args=args) 
    return result.states

def linear_schedule(t, T):
    """Linear ramp: λ(t) = t/T, evolves uniformly from H0 → H1."""
    return t / T

def power_law_schedule(t, T, p=2):
    """Power-law ramp: λ(t) = (t/T)^p, slows start/end depending on p."""
    return (t / T)**p

def optimized_step_schedule(t, T, min_gap_t):
    """S-curve ramp: slows near min energy gap at t=min_gap_t using tanh."""
    normalized_t = t / T
    return 0.5 * (1 + np.tanh(5 * (normalized_t - (min_gap_t / T))))

def run_simulation(schedule_func, schedule_name): 
    H_td = [H0, [H1, lambda t, args: schedule_func(t, args['T'])]] if schedule_name != "Optimized" else \
           [H0, [H1, lambda t, args: schedule_func(t, args['T'], min_gap_t)]]
    
    mesolve(H_td, psi0, times, c_ops=[], e_ops=[], args={'T': T})  # evolution
    lam_vals = np.array([schedule_func(t, T) if schedule_name != "Optimized" else schedule_func(t, T, min_gap_t) for t in times])
    plt.plot(times, lam_vals, label=schedule_name)
    return lam_vals

