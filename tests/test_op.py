


import numpy as np 
import pytest
from numpy.testing import assert_allclose

from quantum_algorithms import (
    I, X, Y, Z, H, S, T, 
    P0, P1, CNOT,
    projectors,
    U_N_qubits, U_one_gate, U_two_gates,
    controlled_gate,
    born_rule_probs,
    sample_from_probs,
)



def test_projector_operators():
    assert np.allclose(P0 @ P0, P0)
    assert np.allclose(P1 @ P1, P1)

    assert np.allclose(P0.conj().T, P0)
    assert np.allclose(P1.conj().T, P1)


def test_I_gate():
    """check that I is unitary"""
    assert np.allclose(I.conj().T @ I, np.eye(2))


def test_X_gate():
    """check that X is unitary and self-inverse"""
    assert np.allclose(X.conj().T @ X, I)
    assert np.allclose(X @ X, I)


def test_Y_gate():
    """check that Y is unitary and self-inverse"""
    assert np.allclose(Y.conj().T @ Y, I)
    assert np.allclose(Y @ Y, I)


def test_Z_gate():
    """check that Z is unitary and self-inverse"""
    assert np.allclose(Z.conj().T @ Z, I)
    assert np.allclose(Z @ Z, I)


def test_S_gate():
    """check that S is unitary and squares to Z"""
    assert np.allclose(S.conj().T @ S, I)
    assert np.allclose(S @ S, Z)


def test_T_gate():
    """check that T is unitary and generates S and Z by powers"""
    assert np.allclose(T.conj().T @ T, I)
    assert np.allclose(T @ T, S)
    assert np.allclose(T @ T @ T @ T, Z)


def test_H_gate():
    assert np.allclose(H.conj().T @ H, I)
    assert np.allclose(H @ H, I)


# ==========================================================
# IMPROVED CONTROLLED-GATE TESTS (WITH LOOPS + ERROR CHECK)
# ==========================================================

def test_controlled_gate_raises_error_when_control_equals_target():
    """
    Explicitly check that controlled_gate raises ValueError
    when control == target, for multiple system sizes.
    """
    for N in [2, 2, 4]:
        for i in range(N):
            with pytest.raises(ValueError, match="Control and target must be different"):
                controlled_gate(X, i, i, N)


def test_controlled_gate_matches_projector_definition_all_indices():
    """
    Verify controlled_gate(U, control, target, N) matches the projector form:

        C_U = P0(control) ⊗ I  +  P1(control) ⊗ U(target)

    for all valid (control, target) pairs in N qubits.
    """
    for N in [2, 3, 4]:
        for control in range(N):
            for target in range(N):

                if control == target:
                    continue

                C_U = controlled_gate(X, control, target, N)

                # Expected operator (projector decomposition)
                P0_ops = [
                    P0 if i == control else I
                    for i in range(N)
                ]

                P1_ops = [
                    P1 if i == control else X if i == target else I
                    for i in range(N)
                ]

                expected = U_N_qubits(P0_ops) + U_N_qubits(P1_ops)

                assert np.allclose(C_U, expected)


def test_controlled_gate_unitary_all_indices():
    """
    Controlled version of a unitary gate must also be unitary.
    """
    for N in [2, 3, 4]:
        dim = 2**N
        Id = np.eye(dim)

        for control in range(N):
            for target in range(N):

                if control == target:
                    continue

                C_U = controlled_gate(X, control, target, N)

                assert np.allclose(C_U.conj().T @ C_U, Id)
                assert np.allclose(C_U @ C_U.conj().T, Id)


def test_controlled_gate_01_X():
    """verify CNOT matrix for control qubit 0 and target qubit 1 (N=2)"""
    N = 2
    control = 0
    target = 1

    C_X0_1 = controlled_gate(X, control, target, N)

    CNOT0_1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

    assert np.allclose(C_X0_1, CNOT0_1)


def test_controlled_gate_10_X():
    """verify CNOT matrix for control qubit 1 and target qubit 0 (N=2)"""
    N = 2
    control = 1
    target = 0

    C_X1_0 = controlled_gate(X, control, target, N)

    CNOT1_0 = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])

    assert np.allclose(C_X1_0, CNOT1_0)


def test_controlled_gate_01_Z():
    """verify CZ matrix for control qubit 0 and target qubit 1 (N=2)"""
    N = 2
    control = 0
    target = 1

    C_Z = controlled_gate(Z, control, target, N)

    CZ0_1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])

    assert np.allclose(C_Z, CZ0_1)


def test_controlled_gate_psi():
    """controlled-Z applies phase only when control qubit is 1"""
    N = 2
    control = 0
    target = 1

    C_Z = controlled_gate(Z, control, target, N)

    psi = np.zeros(4)
    psi[3] = 1  # |11⟩

    psi_out = C_Z @ psi

    expected = np.zeros(4)
    expected[3] = -1  # -|11⟩

    assert np.allclose(psi_out, expected)


def test_controlled_gate_02_X():
    """Controlled-X gate on a 3-qubit register with control qubit 0 and target qubit 2.

    C_X = (P0 ⊗ I ⊗ I) + (P1 ⊗ I ⊗ X)
    """
    N = 3
    control = 0
    target = 2

    C_X = controlled_gate(X, control, target, N)

    C_X0_2 = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])

    assert np.allclose(C_X, C_X0_2)


def test_controlled_gate_02_psi():
    """target gate not applied when control qubit is 0"""
    N = 3
    control = 0
    target = 2

    C_X = controlled_gate(X, control, target, N)

    psi = np.zeros(8)
    psi[1] = 1  # |001⟩

    psi_out = C_X @ psi

    assert np.allclose(psi_out, psi)


def test_projectors_hermitian_and_idempotent():
    """
    Each projector must be Hermitian and idempotent.
    """
    dim = 2  # arbitrary dimension
    P = projectors(dim)

    for Pi in P:
        assert np.allclose(Pi.conj().T, Pi)   # Hermiticity
        assert np.allclose(Pi @ Pi, Pi)       # Idempotence


def test_projectors_orthogonal():
    """
    Different projectors must be orthogonal.
    """
    dim = 4
    P = projectors(dim)

    for i in range(dim):
        for j in range(dim):
            if i != j:
                assert np.allclose(P[i] @ P[j], 0)


def test_projectors_identity():
    """
    Sum of projectors equals the identity.
    """
    dim = 6
    P = projectors(dim)
    Id = np.eye(dim)

    assert np.allclose(sum(P), Id)


def test_U_two_gates():
    """
    Verify correct embedding of two single-qubit gates
    on an N-qubit system, for both i != j and i == j.
    """
    N = 3

    # Case 1: i != j
    i, j = 0, 2
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H, i, N) @ U_one_gate(X, j, N)
    assert np.allclose(U_two, expected)

    # Case 2: i == j
    i = j = 1
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H @ X, i, N)
    assert np.allclose(U_two, expected)

    # Case 3: reversed indices
    i, j = 2, 0
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H, i, N) @ U_one_gate(X, j, N)
    assert np.allclose(U_two, expected)


def test_U_two_gates_different_qubits():
    N = 3

    i, j = 0, 2
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H, i, N) @ U_one_gate(X, j, N)

    assert np.allclose(U_two, expected)


def test_U_two_gates_same_qubit():
    N = 3

    i = j = 1
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H @ X, i, N)

    assert np.allclose(U_two, expected)


def test_U_two_gates_reversed_indices():
    N = 3

    i, j = 2, 0
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H, i, N) @ U_one_gate(X, j, N)

    assert np.allclose(U_two, expected) 


def test_born_rule_probs_properties():
    rho = np.array([[0.7, 0.0],
                    [0.0, 0.3]], dtype=complex)

    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)

    P1 = np.array([[0, 0],
                   [0, 1]], dtype=complex)

    projectors = [P0, P1]

    probs = born_rule_probs(rho, projectors)

    # Property 1: probabilities are real
    assert np.allclose(np.imag(probs), 0.0)

    # Property 2: probabilities are non-negative
    assert np.all(probs >= 0)

    # Property 3: probabilities sum to 1
    assert np.isclose(np.sum(probs), 1.0)


def test_born_rule_probs_diagonal_case():
    rho = np.array([[0.7, 0.0],
                    [0.0, 0.3]], dtype=complex)

    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)

    P1 = np.array([[0, 0],
                   [0, 1]], dtype=complex)

    probs = born_rule_probs(rho, [P0, P1])

    # For diagonal rho: probabilities match diagonal entries
    assert_allclose(probs, np.array([0.7, 0.3]))


def test_controlled_gate_matches_projector_definition_all_indices():
    """
    Verify controlled_gate(U, control, target, N) matches:
        C(U) = P0(control) ⊗ I + P1(control) ⊗ U(target)
    for all valid (control != target).
    """
    for N in [2, 2, 4]:
        for control in range(N):
            for target in range(N):

                if control == target:
                    continue  # invalid, tested separately

                C_U = controlled_gate(X, control, target, N)

                # Expected operator (projector decomposition)
                P0_ops = [P0 if i == control else I for i in range(N)]
                P1_ops = [P1 if i == control else (X if i == target else I) for i in range(N)]

                expected = U_N_qubits(P0_ops) + U_N_qubits(P1_ops)

                assert np.allclose(C_U, expected)



