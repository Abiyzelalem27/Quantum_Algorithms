
import numpy as np
from collections import Counter
I = np.array([[1, 0],
              [0, 1]], dtype=complex)
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
    states : list of numpy.ndarray
        State vectors.
    probabilities : list of float
        Classical probabilities.
    """
    return sum(p * np.outer(psi, psi.conj())
               for psi, p in zip(states, probabilities))

def evolve(state, U):
    """
    Evolves a quantum state using a unitary operator.

    Parameters
    state : numpy.ndarray
        State vector or density matrix.
    U : numpy.ndarray
        Unitary operator.
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

    # Operator acting on the subspace where control qubit is |0⟩
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

    import numpy as np


def normalize_state(psi):
    """Normalize a pure state vector |psi>."""
    norm = np.linalg.norm(psi)
    if np.isclose(norm, 0):
        raise ValueError("State vector has zero norm.")
    return psi / norm

def born_rule_probs(rho, projectors):
    """
    Compute measurement outcome probabilities using the Born rule:
    p_i = Tr(P_i * rho) for each projector P_i.

    Returns a normalized probability vector.
    """
    probs = np.array([np.real(np.trace(Pi @ rho)) for Pi in projectors])

    # safety: numeric cleanup + normalization
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

def measurement_density_matrix(rho, projectors):
    """
    Perform measurement using GIVEN projectors.

    Args:
        rho (np.ndarray): density matrix
        projectors (list[np.ndarray]): measurement projectors P_i

    Returns:
        outcome (int)
        rho_post (np.ndarray)
        probs (np.ndarray)
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
    """prepared the intitial state"""
    total = n + 1
    state = np.zeros(2**total, dtype=complex)
    state[1] = 1.0  # basis index where ancilla=1 and inputs all zero
    return state

def apply_hadamards(state, total_qubits):
    """apply hadamards on the prepared initial state to create a superposition"""
    H_full = U_N_qubits([H] * total_qubits)
    return H_full.dot(state)

def sample_probs(probs, shots, rng=None):
    """Sample measurement outcomes based on a given probability distribution.
    
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

def apply_kraus(rho, kraus_ops):
    """
    Apply a quantum noise channel using Kraus operators

    E(rho) = Σ_k (Ek rho Ek†)
    """
    out = np.zeros_like(rho, dtype=complex)

    for E in kraus_ops:
        out += E @ rho @ E.conj().T

    return out

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
    """
    Return the computational basis state |0⟩.

    |0⟩ = [1, 0]^T
    """
    return np.array([1, 0], dtype=complex)


def ket1():
    """
    Return the computational basis state |1⟩.

    |1⟩ = [0, 1]^T
    """
    return np.array([0, 1], dtype=complex)


def dm(psi):
    """
    Construct a density matrix from a pure state |psi⟩.

    ρ = |psi⟩⟨psi|
    """
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

def apply_bitflips(state, p):
    noisy = state.copy()
    for i in range(3):  # 3-qubit repetition
        if np.random.rand() < p:
            noisy = apply_X(noisy, i)  # your X-on-qubit-i function
    return noisy

def deutsch_jozsa(n, f):
    """
    Deutsch–Jozsa Algorithm (DJA).if the Boolean function f(x) : {0,1}^n -> {0,1}
        is constant or balanced using a single oracle query.
    Steps:
        1. Prepare |0...0>|1>
        2. Apply Hadamard to all qubits
        3. Apply oracle U_f
        4. Apply Hadamard to the first n qubits
        5. Measure first n qubits
    Parameters:
        n : int
            Number of input qubits
        f : function
            Oracle function f(x) -> 0 or 1
    Returns:
        state : np.ndarray
            Final state vector after algorithm  
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
    Deutsch–Jozsa algorithm with a single-qubit rotation error
    applied **before the first Hadamard gates**.

    Parameters:
        theta : float
            Rotation angle in radians
        target_qubit : int
            Qubit to apply the rotation
        axis : tuple
            Rotation axis vector (nx, ny, nz)
    Returns:
        state : np.ndarray
            Final state vector after algorithm  
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
    Deutsch–Jozsa algorithm with a single-qubit rotation error
    applied after the first Hadamard gates.
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
    Deutsch–Jozsa algorithm with a single-qubit rotation error
    applied after the oracle U_f.
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