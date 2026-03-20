
import numpy as np 
import pytest
from numpy.testing import assert_allclose

from quantum_algorithms import ( 
    rotation_gate, 
    oracle_function,
    deutsch_jozsa,
    f_constant_0,
    f_constant_1,
    f_balanced_parity,
    measure_probs_first_n,
    sample_measurements_input,
    X, Y, Z, I,
)

def test_rotation_gate():
    """Test rotation_gate produces correct rotations around X, Y, Z
    
     i*Ri (up to global phase)
    """

    theta = np.pi # 180-degree
    #X-axis 
    Rx = rotation_gate(theta, (1, 0, 0))
    expected_Rx = -1j * X
    assert_allclose(Rx, expected_Rx)

    #Y-axis
    Ry = rotation_gate(theta, (0, 1, 0))
    expected_Ry = -1j * Y #i*Y
    assert_allclose(Ry, expected_Ry)

    # Z-axis 
    Rz = rotation_gate(theta, (0, 0, 1))
    expected_Rz = -1j * Z #i*z 
    assert_allclose(Rz, expected_Rz)

def test_constant_functions_n():
    """Test constant functions 0 and 1 for multiple qubit sizes, probs[0]=1"""
    for n in [2, 3, 4]:
        # Constant zero function
        f0 = lambda x: 0
        state0 = deutsch_jozsa(n, f0)
        probs0 = measure_probs_first_n(state0, n)
        assert np.isclose(probs0[0], 1.0, atol=1e-12)

        # Constant one function
        f1 = lambda x: 1
        state1 = deutsch_jozsa(n, f1)
        probs1 = measure_probs_first_n(state1, n)
        assert np.isclose(probs1[0], 1.0, atol=1e-12) 

def test_balanced_function():
    """Test balanced function, probs[0]=0"""
    for n in [2, 3, 4]:
        f_balanced = lambda x: x % 2
        state = deutsch_jozsa(n, f_balanced)
        probs = measure_probs_first_n(state, n)
        # probs=o 
        assert np.isclose(probs[0], 0.0, atol=1e-12)
        
def test_measure_probs_basis_state():
    n = 3
    state = np.zeros(2**n)
    state[0] = 1  # |000>
    probs = measure_probs_first_n(state, n)
    assert np.isclose(probs[0], 1.0)
    assert np.allclose(probs[1:], 0.0)
    
def test_measure_probs_uniform_superposition():
    n = 2
    state = np.ones(2**n) / np.sqrt(2**n)
    probs = measure_probs_first_n(state, n)
    expected = np.ones(2**n) / (2**n)
    assert np.allclose(probs, expected)
    
def test_measure_probs_first_qubit_only():
    # state = (|00> + |10>) / sqrt(2)
    state = np.zeros(4)
    state[0] = 1/np.sqrt(2)
    state[2] = 1/np.sqrt(2)
    probs = measure_probs_first_n(state, 1)
    # first qubit always 0
    assert np.allclose(probs, [1.0, 0.0])
    
def test_measure_probs_entangled():
    state = np.zeros(4)
    state[0] = 1/np.sqrt(2)
    state[3] = 1/np.sqrt(2)
    probs = measure_probs_first_n(state, 1)
    assert np.allclose(probs, [0.5, 0.5])

def test_measure_probs_properties():
    n = 3
    state = np.random.rand(2**n) + 1j*np.random.rand(2**n)
    state = state / np.linalg.norm(state) 
    probs = measure_probs_first_n(state, n)
    # real
    assert np.allclose(np.imag(probs), 0)
    # non-negative
    assert np.all(probs >= 0)
    # sum to 1
    assert np.isclose(np.sum(probs), 1.0) 
