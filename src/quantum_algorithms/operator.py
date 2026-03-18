

import numpy as np
from collections import Counter
import itertools 
from scipy import sparse 
import scipy
import matplotlib.pyplot as plt 
import math
import random 
I = np.array([[1, 0],
              [0, 1]], dtype=complex)
I8 = np.eye(8, dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
              [1j,  0]], dtype=complex)
Z = np.array([[1,  0],
              [0, -1]], dtype=complex)
H = 1 / np.sqrt(2) * np.array([[1,  1],
                               [1, -1]], dtype=complex)
P0 = np.array([[1, 0],
               [0, 0]], dtype=complex)
P1 = np.array([[0, 0],
               [0, 1]], dtype=complex)

# 3-qubit identity
I8 = np.kron(np.kron(I,I),I)

# Bit-flip operators
X1 = np.kron(np.kron(X,I),I)   
X2 = np.kron(np.kron(I,X),I)   
X3 = np.kron(np.kron(I,I),X)   

# Phase-flip operators
Z1 = np.kron(np.kron(Z,I),I)   
Z2 = np.kron(np.kron(I,Z),I)   
Z3 = np.kron(np.kron(I,I),Z)   

def projectors(dim):
    """
    Generate computational basis projectors {|i><i|} with the given dimension.
    """
    projectors = []
    for i in range(dim):
        ket = np.zeros(dim, dtype=complex)
        ket[i] = 1
        P = np.outer(ket, ket)
        projectors.append(P)
    return projectors
    
def rotation_gate(theta, n):
    """
    This function implements a unitary rotation of a single qubit
    by an angle `theta` around an axis `n` on the Bloch sphere.

    The rotation generator is constructed as N = n · σ,
    where σ = (X, Y, Z) are the Pauli matrices.

    Parameters
    theta : Rotation angle
    n : Rotation axis
    """
    nx, ny, nz = n
    N = nx * X + ny * Y + nz * Z
    R = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * N
    return R
    
def U_N_qubits(ops):
    """
    Constructs an N-qubit operator using tensor products.

    Parameters
    ops : single-qubit operators.
    """
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U


def U_one_gate(V, i, N):
    """
    Applies a single-qubit gate to qubit i
    in an N-qubit system.

    Parameters
    V : Single-qubit gate.
    i : Target qubit index.
    N : Total number of qubits.
    """
    ops = [I] * N
    ops[i] = V
    return U_N_qubits(ops)


def U_two_gates(V, W, i, j, N):
    """
    Applies two single-qubit gates to an N-qubit system.
    If i != j:
        applies V on qubit i and W on qubit j.
    If i == j:
        applies the composed gate V @ W on qubit i,
        preserving operator ordering.
    """
    ops = [I] * N
    if i == j:
        ops[i] = V @ W
    else:
        ops[i] = V
        ops[j] = W
    return U_N_qubits(ops)

def rho(states, probabilities):
    """
    Constructs a density matrix from pure states.
    Parameters
    states  and probabilities 
    """
    return sum(p * np.outer(psi, psi.conj())
               for psi, p in zip(states, probabilities))

def evolve(state, U):
    """
    Evolves a quantum state using a unitary operator.
    Parameters
    state : State vector or density matrix.
    U :  Unitary operator.
    """
    if state.ndim == 1:
        # Pure state evolution
        return U @ state
    elif state.ndim == 2:
        # Density matrix evolution
        return U @ state @ U.conj().T
    else:
        raise ValueError("State must be a vector or a density matrix")


def controlled_gate(U, control, target, N):
    """
    Controlled-U gate on an N-qubit register.
    Implements the projector decomposition:
        C_U = P0(control) ⊗ I  +  P1(control) ⊗ U(target)
    """
    if control == target:
        raise ValueError("Control and target must be different")
    # Operator acting on the subspace where the control qubit is |0⟩
    P0_ops = [
        P0 if i == control else I
        for i in range(N)
    ]

    # Operator acting on the subspace where control qubit is |1⟩
    P1_ops = [
        P1 if i == control else U if i == target else I
        for i in range(N)
    ]

    return U_N_qubits(P0_ops) + U_N_qubits(P1_ops)

def normalize_state(psi):
    """
    Normalize a pure state vector |psi>.
    """
    norm = np.linalg.norm(psi)
    if np.isclose(norm, 0):
        raise ValueError("State vector has zero norm.")
    return psi/norm 

def born_rule_probs(rho, projectors):
    """
    Compute measurement outcome probabilities using the Born rule:
    p(i)= Tr(Pi * rho) for each projector
    """
    probs = np.array([np.real(np.trace(Pi @ rho)) for Pi in projectors])
    probs = np.clip(probs, 0, 1)
    probs = probs / np.sum(probs)
    return probs


def sample_from_probs(probs):
    """
    Return a sampled index based on probs.
    """
    return np.random.choice(len(probs), p=probs)

def measure_pure_state(psi, projectors):
    """
    Measure pure state |psi> using projectors.
    Returns:
        outcome (int)
        psi_post (np.ndarray)
        probs (np.ndarray)
    """
    psi = normalize_state(psi)
    probs = born_probs_pure(psi, projectors)
    outcome = sample_from_probs(probs)
    Pk = projectors[outcome]
    psi_post_unnormalized = Pk @ psi
    norm_post = np.linalg.norm(psi_post_unnormalized)
    if np.isclose(norm_post, 0):
        raise ValueError("Outcome probability ~0 (numerical issue).")
    psi_post = psi_post_unnormalized / norm_post
    
    return outcome, psi_post, probs
    

def doMeasurement(state, projectors):
    """
     Perform a projective measurement on a quantum state (pure or density matrix)
    using a list of projectors
    Returns:
        outcome :Index of the measured outcome
        post_state : Collapsed state after measurement (normalized)
        probs : Probability 
    """
    # Determine if pure state or density matrix
    pure = state.ndim == 1

    if pure:
        # Normalize pure state
        state = state / np.linalg.norm(state)
        # Probabilities via Born rule
        probs = np.array([np.real(np.vdot(state, P @ state)) for P in projectors])
    else:
        # Density matrix case
        probs = np.array([np.real(np.trace(P @ state)) for P in projectors])

    # Normalize probabilities (safety)
    probs = np.clip(probs, 0, 1)
    probs /= probs.sum()
    # Sample outcome
    outcome = np.random.choice(len(projectors), p=probs)
    Pk = projectors[outcome]

    # Collapse state
    if pure:
        post_unnorm = Pk @ state
        norm_post = np.linalg.norm(post_unnorm)
        if np.isclose(norm_post, 0):
            raise ValueError("Outcome probability ~0 (numerical issue).")
        post_state = post_unnorm / norm_post
    else:
        post_state = Pk @ state @ Pk
        denom = np.trace(post_state)
        if np.isclose(denom, 0):
            raise ValueError("Outcome probability ~0 (numerical issue).")
        post_state = post_state / denom
    return outcome, post_state, probs

def measurement_density_matrix(rho, projectors):
    """
    Perform measurement using the given projectors
    """
    probs = born_rule_probs(rho, projectors)
    outcome = sample_from_probs(probs)
    Pk = projectors[outcome]
    numerator = Pk @ rho @ Pk
    denom = np.trace(numerator)
    if np.isclose(denom, 0):
        raise ValueError("Outcome probability ~0 (numerical issue).")
    rho_post = numerator / denom
    return outcome, rho_post, probs

def initial_state(n):
    """
    Prepared the initial state
    """
    total = n + 1
    state = np.zeros(2**total, dtype=complex)
    state[1] = 1.0  # basis index where ancilla=1 and inputs all zero
    return state

def apply_hadamards(state, total_qubits):
    """
    Apply Hadamards gate on the prepared initial state to create a superposition.
    """
    H_full = U_N_qubits([H] * total_qubits)
    return H_full.dot(state)

def sample_probs(probs, shots, rng=None):
    """
    Sample measurement outcomes based on a given probability distribution.
    
    It draws a specified number of random samples ("shots") according to the
    probability distribution `probs`,

    """
    if rng is None:
        rng = np.random.default_rng()
    outcomes = rng.choice(len(probs), size=shots, p=probs)
    return Counter(outcomes)
    
def oracle_function(f, n):
    """
    Build a function that applies the oracle operator U_f to the statevector of n+1 qubits.
    The oracle implements the transformation:
        U_f |x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
    Parameters
        f : Boolean function f(x) -> {0, 1} for x in [0, 2^n)
        n : Number of input qubits
    """

    def apply_Uf(state):
        new = np.copy(state)
        for x in range(2**n):
            fx = f(x)
            idx0 = (x << 1) | 0
            idx1 = (x << 1) | 1
            if fx == 1:
                # swap amplitudes between ancilla 0 and 1 for this x
                new[idx0], new[idx1] = state[idx1], state[idx0]
        return new   
    return apply_Uf  

def f_constant_0(x):
    return 0 

def f_constant_1(x):
    return 1

def f_balanced_parity(x):
    return x % 2  # 0 for even, 1 for odd 

def measure_probs_first_n(state, n):
    """Compute prob distribution over first n qubits (sum over ancilla)."""
    probs = np.zeros(2**n)
    for x in range(2**n):
        # Apply bitwise operations to find the correct index for each state
        idx0 = (x << 1) | 0  # ancilla = 0
        idx1 = (x << 1) | 1  # ancilla = 1
        probs[x] = np.abs(state[idx0])**2 + np.abs(state[idx1])**2
    return probs 

def sample_measurements_input(state, n, shots, rng=None):
    """
    Measurement outcomes from the full-register distribution given by state,
    then aggregate counts over the input register (i.e., ignore ancilla).
    """
    if rng is None:
        rng = np.random.default_rng()
    probs_full = np.abs(state)**2
    probs_full = probs_full / probs_full.sum()
    samples = rng.choice(len(probs_full), size=shots, p=probs_full)
    input_samples = samples >> 1   # removes ancilla qubit (shift right)
    return Counter(input_samples) 

def bloch_vector(rho):
    """
    Compute the Bloch vector (rX, rY, rZ) for a single-qubit density matrix rho.
    r_J = Tr(rho * J), J = X, Y, Z
    """
    rX = np.real(np.trace(rho @ X))
    rY = np.real(np.trace(rho @ Y))
    rZ = np.real(np.trace(rho @ Z))
    return np.array([rX, rY, rZ])


def bloch_visualization(channel_kraus_ops, n_samples=1000, seed=None):
    """
    Visualize the effect of a single-qubit quantum channel on the Bloch sphere.

    Parameters
    channel_kraus_ops : Kraus operators for quantum channel.
    n_samples : Number of random pure states to sample (default 1000).
    seed :  Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    bloch_vectors_out = np.zeros((n_samples, 3))

    for i in range(n_samples):
        psi = random_pure_state(rng)
        rho = dm(psi)
        rho_after = apply_channel(rho, channel_kraus_ops)
        bloch_vectors_out[i, :] = bloch_vector(rho_after)

def apply_kraus(rho, kraus_ops):
    """
    Apply a quantum channel to a density matrix using Kraus operators.
    """
    rho_out = np.zeros_like(rho, dtype=complex)
    for E in kraus_ops:
        rho_out += E @ rho @ E.conj().T  # E rho E†
    
    return rho_out 

def rotation_channel(p, R):
    """
    Random unitary single-qubit channel using rotation R.
    Returns list of Kraus operators [M0, M1].
    """
    M0 = np.sqrt(1-p) * I
    M1 = np.sqrt(p) * R
    return [M0, M1]

def apply_channel(rho, kraus_ops):
    """
    Applies a quantum channel to a single-qubit density matrix.
    """
    rho_out = np.zeros_like(rho, dtype=complex)
    for M in kraus_ops:
        rho_out += M @ rho @ M.conj().T
    return rho_out 
    
def bit_flip_kraus(p):
    """
    Bit flip channel (X noise).

    Kraus operators:
        E0 = sqrt(1-p) I
        E1 = sqrt(p)   X
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * X
    return [E0, E1]


def phase_flip_kraus(p):
    """
    Phase flip channel (Z noise).

    Kraus operators:
        E0 = sqrt(1-p) I
        E1 = sqrt(p)   Z
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z
    return [E0, E1]

def amplitude_damping_kraus(gamma):
    """
    Amplitude damping channel.

    Physical meaning:
        |1> -> |0> with probability gamma

    Kraus operators:
        E0 = [[1, 0],
              [0, sqrt(1-gamma)]]

        E1 = [[0, sqrt(gamma)],
              [0, 0]]
    """
    E0 = np.array([[1, 0],
                   [0, np.sqrt(1 - gamma)]], dtype=complex)

    E1 = np.array([[0, np.sqrt(gamma)],
                   [0, 0]], dtype=complex)

    return [E0, E1]


def phase_damping_kraus(lmbda):
    """
    Phase damping channel (dephasing).

    Physical meaning:
        Off-diagonal terms decay but populations stay unchanged.
    """
    E0 = np.array([[1, 0],
                   [0, np.sqrt(1 - lmbda)]], dtype=complex)

    E1 = np.array([[0, 0],
                   [0, np.sqrt(lmbda)]], dtype=complex)

    return [E0, E1]


def depolarizing_kraus(p):
    """
    Depolarizing channel.

    Kraus operators:
        E0 = sqrt(1 - 3p/4) I
        E1 = sqrt(p/4) X
        E2 = sqrt(p/4) Y
        E3 = sqrt(p/4) Z
    """
    E0 = np.sqrt(1 - 3*p/4) * I
    E1 = np.sqrt(p/4) * X
    E2 = np.sqrt(p/4) * Y
    E3 = np.sqrt(p/4) * Z
    return [E0, E1, E2, E3]

def ket0():
    return np.array([1, 0], dtype=complex)

def ket1():
    return np.array([0, 1], dtype=complex)

def ket_plus():
    return (ket0() + ket1()) / np.sqrt(2)

def ket_minus():
    return (ket0() - ket1()) / np.sqrt(2)

def dm(psi):
    """
    Construct a density matrix from a pure state |psi⟩.

    ρ = |psi⟩⟨psi|
    """
    psi = psi/np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def random_pure_state(rng=None):
    """
    Generate a random single-qubit pure state |psi⟩.
    Useful for Monte-Carlo simulations.
    """
    rng = np.random.default_rng() if rng is None else rng
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v = v / np.linalg.norm(v)
    return v

def pauli_kraus_channel(pX, pY, pZ):
    """
    General Pauli channel:

    E(rho) = pI*rho + pX XrhoX + pY YrhoY + pZ ZrhoZ

    where:
        pI = 1 - (pX+pY+pZ)
    """
    pI = 1 - (pX + pY + pZ)
    if pI < 0:
        raise ValueError("Probabilities must satisfy pX+pY+pZ <= 1")
    E0 = np.sqrt(pI) * I
    E1 = np.sqrt(pX) * X
    E2 = np.sqrt(pY) * Y
    E3 = np.sqrt(pZ) * Z

    return [E0, E1, E2, E3]

def E1_rho(psi, p):
    """
    Q3: One-qubit bit flip channel:
        E1(rho) = (1-p)rho + pXrhoX
    """
    rho = dm(psi)
    return (1 - p) * rho + p * (X @ rho @ X)

def deutsch_jozsa(n, f):
    """
    Deutsch–Jozsa Algorithm(DJA) the Boolean function 
        is constant or balanced using a single oracle query. 
    Parameters:
        n : Number of input qubits
        f : Oracle function
    """
    total_qubits = n + 1
    state = initial_state(n)
    state = apply_hadamards(state, total_qubits)
    U = oracle_function(f, n)
    state = U(state)
    H_first_n = U_N_qubits([H] * n + [I])
    state = H_first_n @ state
    return state  

def deutsch_jozsa_error1(n, f, theta, target_qubit, axis):
    """
    DJA with a single-qubit rotation error applied before the first Hadamard gates.
    """
    total_qubits = n + 1
    state = initial_state(n)
    R = rotation_gate(theta, axis)
    state = U_one_gate(R, target_qubit, total_qubits) @ state
    state = apply_hadamards(state, total_qubits)
    U = oracle_function(f, n)
    state = U(state)
    H_first_n = U_N_qubits([H]*n + [I])
    state = H_first_n @ state

    return state

def deutsch_jozsa_error2(n, f, theta, target_qubit, axis):
    """
    DJA with a single-qubit rotation error applied after the first Hadamard gates.
    """
    total_qubits = n + 1
    state = initial_state(n)
    state = apply_hadamards(state, total_qubits)
    R = rotation_gate(theta, axis)
    state = U_one_gate(R, target_qubit, total_qubits) @ state
    U = oracle_function(f, n)
    state = U(state)
    H_first_n = U_N_qubits([H]*n + [I])
    state = H_first_n @ state
    
    return state

def deutsch_jozsa_error3(n, f, theta, target_qubit, axis):
    """
    DJA with a single-qubit error applied after the oracle U_f.
    """
    total_qubits = n + 1
    state = initial_state(n)
    state = apply_hadamards(state, total_qubits)
    U = oracle_function(f, n)
    state = U(state)
    R = rotation_gate(theta, axis)
    state = U_one_gate(R, target_qubit, total_qubits) @ state
    H_first_n = U_N_qubits([H]*n + [I])
    state = H_first_n @ state

    return state

def deutsch_jozsa_error4(n, f, theta, target_qubit, axis):
    """
    Deutsch–Jozsa algorithm with a single-qubit rotation error
    applied after the final Hadamard gates.
    """
    total_qubits = n + 1
    state = initial_state(n)
    state = apply_hadamards(state, total_qubits)
    U = oracle_function(f, n)
    state = U(state)
    H_first_n = U_N_qubits([H]*n + [I])
    state = H_first_n @ state
    R = rotation_gate(theta, axis)
    state = U_one_gate(R, target_qubit, total_qubits) @ state

    return state

def encode_3_qubit_bit_flip_code(psi):
    """
    Encode a single qubit state into the 3-qubit bit-flip code.
    """
    psi = np.kron(psi, np.kron(ket0(), ket0()))
    CNOT12 = controlled_gate(X, 0, 1, 3)
    CNOT13 = controlled_gate(X, 0, 2, 3)
    psi = CNOT13 @ CNOT12 @ psi
    return psi 

def syndrome_measurement(psi):
    """Measure parity checks Z1Z2 and Z2Z3"""
    Z1Z2 = np.kron(Z, Z)
    Z1Z2 = np.kron(Z1Z2, I)  # qubits 1 and 2
    Z2Z3 = np.kron(I, np.kron(Z, Z))  # qubits 2 and 3
    s1 = np.vdot(psi, Z1Z2 @ psi).real
    s2 = np.vdot(psi, Z2Z3 @ psi).real
    # Convert to +1/-1
    s1 = 1 if s1 > 0 else -1
    s2 = 1 if s2 > 0 else -1
    return (s1, s2)
    
def correct_phase_flip(psi):
    """
    Correct a single phase-flip using the 3-qubit phase-flip code.
    """
    s1,s2 = syndrome_measurement(psi)
    # Map syndromes to Z corrections
    if (s1,s2) == (1,1):
        return psi          
    elif (s1,s2) == (-1,1):
        return Z1 @ psi      
    elif (s1,s2) == (-1,-1):
        return Z2 @ psi      
    elif (s1,s2) == (1,-1):
        return Z3 @ psi      
    else:
        raise ValueError("Invalid syndrome")

def correct_bit_flip(psi):
    """
    Correct a single bit-flip using the syndrome.
    """
    s1, s2 = syndrome_measurement(psi)
    if (s1, s2) == (1, 1):
        return psi
    elif (s1, s2) == (-1, 1):
        return X1@ psi
    elif (s1, s2) == (-1, -1):
        return X2@ psi
    elif (s1, s2) == (1, -1):
        return X3@ psi
    else:
        raise ValueError("Invalid syndrome") 

def bit_flip_channel_3qubits(psi, p):
    """
    Apply the bit-flip channel.
    """
    # Single-qubit Kraus operators
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * X  
    # Generate all 8 three-qubit Kraus operator
    kraus_ops = []
    for k0 in [E0, E1]:
        for k1 in [E0, E1]:
            for k2 in [E0, E1]:
                kraus_ops.append(np.kron(k0, np.kron(k1, k2)))        
    rho_out = np.zeros_like(psi)
    for K in kraus_ops:
        rho_out += K @ psi
    return rho_out 
    
def bit_flip_kraus_nqubits(p, n):
    """
    Kraus operators for a bit-flip channel applied
    to each qubit in an n-qubit system.
    Parameters
    p : probability 
    n:  Number of qubits
    """
    single_qubit_ops = [np.sqrt(1 - p) * I, np.sqrt(p) * X]
    kraus_ops = [np.array([[1]], dtype=complex)] 
    for _ in range(n):
        new_ops = []
        for K in kraus_ops:
            for E in single_qubit_ops:
                new_ops.append(np.kron(K, E))
        kraus_ops = new_ops
    return kraus_ops
    

# Sparse helper functions
def buildSparseGateSingle(n, i, gate):
    return sparse.kron(sparse.kron(sparse.identity(2**i), gate), sparse.identity(2**(n-i-1)))

def buildSparseCNOT(n, ic, it):
    return buildSparseGateSingle(n, ic, P0) + buildSparseGateSingle(n, ic, P1) @ buildSparseGateSingle(n, it, X)

def dm_sparse(psi):
    """Density matrix from state vector (sparse)"""
    return psi @ psi.getH()

def ket0_sparse(n=1):
    """n-qubit |0>"""
    return sparse.csr_matrix(np.array([[1], [0]], dtype=complex)) if n==1 else sparse.kron(ket0_sparse(), sparse.identity(2**(n-1), dtype=complex))

# Multi-qubit bit-flip Kraus (sparse)
def bit_flip_kraus_nqubits_sparse(p, n):
    single_ops = [np.sqrt(1-p) * I, np.sqrt(p) * X]
    # generate all combinations
    kraus_ops = []
    for combo in product(single_ops, repeat=n):
        K = combo[0]
        for E in combo[1:]:
            K = sparse.kron(K, E)
        kraus_ops.append(K)
    return kraus_ops

def buildSparseCNOT(n, ic, it):
    P0ic = buildSparseGateSingle(n, ic, P0)
    P1ic = buildSparseGateSingle(n, ic, P1)
    Xit  = buildSparseGateSingle(n, it, X)
    return P0ic + P1ic @ Xit


# helper function for initializing all qubits in state zero
def initRegisterPsi(n):
    return basisvec(n,0)

def initRegisterRho(n):
    ini = basisvec(n,0)
    return np.outer(ini.conj(),ini)

def apply_kraus_sparse(rho, kraus_ops):
    rho_out = sparse.csr_matrix(rho.shape, dtype=complex)
    for K in kraus_ops:
        rho_out += K @ rho @ K.getH()
    return rho_out 

def depolarizing_kraus_nqubits(p, n):
    """
    Depolarizing channel for n qubits.

    E(ρ) = (1-p)ρ + p * I/d

    Kraus operators are tensor products of Pauli matrices.
    Total number = 4^n
    """
    paulis = [I, X, Y, Z]
    pauli_strings = list(itertools.product(paulis, repeat=n))
    d = 2**n
    kraus_ops = []

    for P_string in pauli_strings:
        P = P_string[0]
        for op in P_string[1:]:
            P = np.kron(P, op)
        if np.array_equal(P, np.eye(d)):
            coeff = np.sqrt(1 - (4**n - 1)*p/(4**n))
        else:
            coeff = np.sqrt(p/(4**n))
        kraus_ops.append(coeff * P)

    return kraus_ops
    

def single_qubit_channel_n_register(kraus_single, n, target):
    """
    Lift single-qubit Kraus operators to act on qubit 'target' in an n-qubit register.

    Parameters:
        kraus_single : list of 2x2 Kraus operators for the single qubit
        n: total number of qubits in the register
        target: index of qubit to apply the channel (0-based)

    Returns:
        list of 2^n x 2^n Kraus operators acting on the full register
    """
    kraus_n = []

    for K in kraus_single:
        full_op = np.array([[1]], dtype=complex)  
        for qubit in range(n):
            op = K if qubit == target else I
            full_op = np.kron(full_op, op)
        kraus_n.append(full_op)
    return kraus_n 

def recovery_bit_flip(rho, syndrome):
    """
    Apply the recovery operation for the 3-qubit bit-flip code.
    
    Parameters:
      rho : 8x8 density matrix
      syndrome : int (0=no error, 1=qubit1, 2=qubit2, 3=qubit3)
    Returns:
    - corrected density matrix
    """
    recovery_ops = [I8, X1, X2, X3]
    M = recovery_ops[syndrome]
    return M @ rho @ M.conj().T

def recovery_phase_flip(rho, syndrome):
    """
    Apply the recovery operation for the 3-qubit phase-flip code.
    Parameters:
     rho: 8x8 density matrix
     syndrome: int (0=no error, 1=qubit1, 2=qubit2, 3=qubit3)
    Returns:
    - corrected_state: 8x8 density matrix after applying Z correction
    """
    recovery_ops = [I8, Z1, Z2, Z3]
    M = recovery_ops[syndrome]
    return M @ rho @ M.conj().T 

def encode_3_qubit_phase_flip_code(psi):
    """
    Encode a 1-qubit state |psi> = [alpha, beta] 
    into the 3-qubit phase-flip code: 
        |0_L> = |+++>, |1_L> = |--->
    """
    alpha, beta = psi
    # Logical 3-qubit states using the ket_plus / ket_minus functions
    zero_L = np.kron(ket_plus(), np.kron(ket_plus(), ket_plus()))
    one_L  = np.kron(ket_minus(), np.kron(ket_minus(), ket_minus()))
    # Encoded state
    encoded = alpha * zero_L + beta * one_L
    return encoded

def syndrome_measurement_bit_flip(psi):
    """Measure parity checks Z1Z2 and Z2Z3"""
    Z1Z2 = np.kron(np.kron(Z,Z), I)
    Z2Z3 = np.kron(np.kron(I,Z), Z)
    s1 = 1 if np.vdot(psi, Z1Z2 @ psi).real > 0 else -1
    s2 = 1 if np.vdot(psi, Z2Z3 @ psi).real > 0 else -1
    return (s1, s2)

def syndrome_measurement_phase_flip(psi):
    """
    Measure X parity checks for phase-flip code: X1X2, X2X3
    """
    X1X2 = np.kron(np.kron(X,X), I)
    X2X3 = np.kron(np.kron(I,X), X)
    s1 = 1 if np.vdot(psi, X1X2 @ psi).real > 0 else -1
    s2 = 1 if np.vdot(psi, X2X3 @ psi).real > 0 else -1
    return (s1, s2)  



