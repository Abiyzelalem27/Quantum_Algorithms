


import qutip as qt 
import numpy as np

def H_single_qubit(j, a, b, Omega, phi=0.0, Delta=0.0, dim=2):
    op_ab = qt.basis(dim, a) * qt.basis(dim, b).dag()
    op_ba = op_ab.dag()
    op_bb = qt.basis(dim, b) * qt.basis(dim, b).dag()
    H_local = (Omega/2)*(np.exp(1j*phi)*op_ab + np.exp(-1j*phi)*op_ba) - Delta*op_bb
    if j == 1:
        return qt.tensor(H_local, qt.qeye(dim))
    elif j == 2:
        return qt.tensor(qt.qeye(dim), H_local)
    else:
        raise ValueError("Qubit index must be 1 or 2")


def H_two_qubit(j, k, a, b, c, d, V, dim=2):
    ket_ab = qt.tensor(qt.basis(dim, a), qt.basis(dim, b))
    ket_cd = qt.tensor(qt.basis(dim, c), qt.basis(dim, d))
    H = (V/2)*(ket_ab*ket_cd.dag() + ket_cd*ket_ab.dag())
    return H

def time_evolution_operator(H, tau):
    """
    Compute the time evolution operator U = exp(-i H tau).
    Used to generate pulse unitaries from Hamiltonians.
    """
    return (-1j * H * tau).expm()


def U1_pulse(Omega):
    """
    First π-pulse on the control atom driving |1> ↔ |r>.
    Implements excitation of the control qubit to the Rydberg state.
    """
    dim = 3
    op_r1 = qt.basis(dim,2) * qt.basis(dim,1).dag()
    op_1r = op_r1.dag()
    Hc = (Omega/2) * (op_r1 + op_1r)
    Hc_full = qt.tensor(Hc, qt.qeye(dim))
    tau1 = np.pi / Omega
    return time_evolution_operator(Hc_full, tau1)


def U2_pulse(Omega, V):
    """
    Second pulse acting on the target atom including blockade interaction.
    The |rr> state acquires a large energy shift V enforcing Rydberg blockade.
    """
    dim = 3
    op_r1 = qt.basis(dim,2) * qt.basis(dim,1).dag()
    op_1r = op_r1.dag()
    Ht = (Omega/2) * (op_r1 + op_1r)
    Ht_full = qt.tensor(qt.qeye(dim), Ht)
    r = qt.basis(dim,2)
    rr = qt.tensor(r,r)
    H_block = V * rr * rr.dag()
    tau2 = 2*np.pi / Omega
    H_total = Ht_full + H_block
    return time_evolution_operator(H_total, tau2)


def U3_pulse(Omega):
    """
    Third pulse identical to the first π-pulse on the control atom.
    De-excites the control qubit from the Rydberg state.
    """
    return U1_pulse(Omega)


def rydberg_CZ_gate(Omega, V):
    """
    Construct the full Rydberg blockade CZ gate.
    Gate sequence: U = U3 * U2 * U1.
    """
    U1 = U1_pulse(Omega)
    U2 = U2_pulse(Omega, V)
    U3 = U3_pulse(Omega)
    return U3 * U2 * U1

s
def computational_subspace_gate(U):
    """
    Extract the 4x4 gate matrix in the computational basis |00>, |01>, |10>, |11>.
    Works for two 3-level atoms.
    """
    dim = 3
    zero = qt.basis(dim,0)
    one  = qt.basis(dim,1)
    basis = [
        qt.tensor(zero, zero),
        qt.tensor(zero, one),
        qt.tensor(one, zero),
        qt.tensor(one, one)
    ] 
    # Use matrix_element instead of .tr()
    U_comp = qt.Qobj([[ U.matrix_element(b1, b2) for b2 in basis ] for b1 in basis])
    return U_comp


# Define Z gate on a qubit
def Z_gate(qubit, dim=3):
    """
    Single-qubit Z gate acting on qubit 0 or 1 in two-atom space.
    """
    Z = qt.Qobj([[1,0,0],[0,-1,0],[0,0,1]])  # acts on |0>,|1>,|r> leaves |r> unchanged
    if qubit==0:
        return qt.tensor(Z, qt.qeye(dim))
    else:
        return qt.tensor(qt.qeye(dim), Z)

def process_fidelity_manual(U, U_ideal):
    """
    Compute normalized process fidelity between two unitary gates.
    Works without qutip-qip.
    """
    d = U.shape[0]
    # trace of U† * U_ideal, squared and normalized
    return np.abs((U.dag() * U_ideal).tr())**2 / d**2

U_ideal = qt.Qobj([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,-1] 
])
     



