

# Quantum Algorithms

This repository contains implementations and simulations of several **fundamental quantum algorithms** and **quantum system simulations** using Python and scientific quantum computing libraries. It explores how quantum mechanical principles such as **superposition**, **interference**, and **entanglement** enable computational advantages over classical algorithms, allowing certain problems to be solved more efficiently.

# Implemented Quantum Algorithms

## 1. Deutsch Algorithm
The **Deutsch Algorithm** determines whether a function `f(x)` is **constant** or **balanced** using a single quantum query to an oracle. It represents one of the earliest demonstrations of **quantum computational advantage**.

## 2. Deutsch–Jozsa Algorithm
The **Deutsch–Jozsa Algorithm** generalizes the Deutsch algorithm to functions with **n-bit inputs**. It determines whether a function is **constant or balanced** using **one quantum evaluation**, while a classical algorithm may require exponentially many queries.

## 3. Grover Search Algorithm
**Grover's Algorithm** searches for a marked element in an **unsorted database** and provides a **quadratic speedup** compared to classical search algorithms.

Classical complexity:
```
O(N)
```

Quantum complexity:
```
O(√N)
```

## 4. Quantum Fourier Transform (QFT)
The **Quantum Fourier Transform (QFT)** is the quantum analogue of the classical discrete Fourier transform. It is a key component used in many quantum algorithms.

## 5. Quantum Phase Estimation (QPE)
The **Quantum Phase Estimation algorithm** determines the **phase (eigenvalue)** associated with an eigenvector of a unitary operator. It is a central subroutine used in many advanced quantum algorithms.

## 6. Shor's Factoring Algorithm
**Shor’s Algorithm** efficiently factors large integers using quantum computation. It combines **Quantum Phase Estimation** and the **Quantum Fourier Transform** to achieve an **exponential speedup** over classical factoring algorithms.

---

# Quantum Simulations

In addition to algorithm implementations, this repository includes simulations of quantum systems and time-dependent quantum dynamics.

## Adiabatic State Preparation
This notebook demonstrates **adiabatic quantum evolution**, where a quantum system evolves slowly from an initial Hamiltonian \(H_0\) to a final Hamiltonian \(H_1\). If the evolution is slow enough, the system remains in the ground state, allowing preparation of complex quantum states.

Key topics:
- Time-dependent Hamiltonians
- Instantaneous eigenstates
- Adiabatic theorem
- Ground state preparation

## Noisy Quantum Simulations
Simulation of **noise effects in quantum systems** to study how decoherence and imperfections influence quantum algorithms.

## Rydberg Atom Simulations
Numerical simulations of **Rydberg atom systems**, which are an important physical platform for quantum computing and quantum simulation.

---

# Repository Structure

```
Quantum_Algorithms
│
├── src/
│   └── quantum_algorithms/
│
├── notebooks/
│   ├── deutsch_jozsa.ipynb
│   ├── DJA_with_error.ipynb
│   ├── shor_QFT_PE.ipynb
│   ├── Adiabatic_state_preparation.ipynb
│   ├── Rydberg_atoms.ipynb
│   └── Simulating_noisy.ipynb
│
├── tests/
├── README.md
├── CHANGELOG.md
└── pyproject.toml
```

---

# Requirements

Main dependencies:

- Python
- NumPy
- Matplotlib
- QuTiP
- SciPy

Install dependencies with:

```bash
pip install numpy matplotlib qutip scipy
```

---

# References

- Nielsen & Chuang — *Quantum Computation and Quantum Information*
- QuTiP Documentation — https://qutip.org/docs/

---

# Author

**Abiy Zelalem Tegegne** 

GitHub:  
https://github.com/Abiyzelalem27