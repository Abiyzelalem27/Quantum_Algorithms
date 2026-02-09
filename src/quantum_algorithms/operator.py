


import numpy as np
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
S = np.array([[1, 0],
              [0, 1j]], dtype=complex)
T = np.array([[1, 0],
              [0, np.e**(1j * np.pi / 4)]], dtype=complex)


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
    
# Control: first qubit
# Target: second qubit
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


def U_N_qubits(ops):
    """
    Constructs an N-qubit operator using tensor products.

    Parameters
    ops : single-qubit operators.

    Returns
        N-qubit operator.
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


# DENSITY MATRIX REPRESENTATION

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


# QUANTUM STATE EVOLUTION

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
        raise ValueError("Control and target must be different.")

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


def deutsch_jozsa(n, oracle_type):
    """
    n : number of input qubits
    oracle_type : 'constant' or 'balanced'
    """
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))

    if oracle_type == "constant":
        pass

    elif oracle_type == "balanced":
        for i in range(n):
            qc.cx(i, n)
    else:
        raise ValueError("oracle_type must be 'constant' or 'balanced'")

    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def run_dj(n, oracle_type, shots):
    simulator = AerSimulator()
    qc = deutsch_jozsa(n, oracle_type)

    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=shots).result()

    return result.get_counts() 
   s 
