

import numpy as np
from collections import Counter
import itertools 
from scipy import sparse 
import scipy
import matplotlib.pyplot as plt 
import math 
import random
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

def deutsch_jozsa(n, f):
    """
    Deutsch–Jozsa Algorithm(DJA) the Boolean function 
        is constant or balanced using a single oracle query. 
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
