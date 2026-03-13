

import numpy as np
from collections import Counter
import itertools 
from scipy import sparse 
import scipy
import matplotlib.pyplot as plt 
import math 


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

def lcm(a, b):
    """
    Compute the least common multiple (LCM) of two integers a and b.
    Uses the formula: lcm(a,b) = |a*b| / gcd(a,b).
    """
    return abs(a*b) // math.gcd(a, b) 

def encrypt_number(m, public_key): 
    """
    Encrypt a message number m using the RSA public key.
    Parameters:
    m : int
        The message represented as an integer smaller than N.
    public_key : tuple (N, e)
        The RSA public key where:
        N = modulus (p*q)
        e = public exponent.
    Returns:
    int
        The ciphertext c = m^e mod N.
    """
    N, e = public_key
    return pow(m, e, N)


def decrypt_number(c, private_key):
    """
    Decrypt a ciphertext number c using the RSA private key.
    Parameters:
    c : int
        The encrypted message (ciphertext).
    private_key : tuple (N, d)
        The RSA private key where:
        N = modulus (p*q)
        d = private exponent.
    Returns:
    int
        The original message m = c^d mod N.
    """
    N, d = private_key
    return pow(c, d, N) 

def generate_RSA_keys(p, q, e, use_carmichael=False):
    """
    Generate RSA public and private keys from two prime numbers p and q
    and a public exponent e.
    Parameters:
    p, q : int
        Prime numbers used to construct the RSA modulus N = p*q.
    e : int
        Public exponent. It must satisfy gcd(e, φ(N)) = 1.
    use_carmichael : bool, optional
        If True, use Carmichael's function λ(N) = lcm(p-1, q-1)
        instead of Euler's totient φ(N) = (p-1)(q-1).

    Returns
    public_key : tuple (N, e)
        The RSA public key used for encryption. 
    private_key : tuple (N, d)
        The RSA private key used for decryption, where
        d is the modular inverse of e modulo φ(N) or λ(N).
    Notes
    The private exponent d satisfies:
        d * e ≡ 1 (mod φ(N))   or   d * e ≡ 1 (mod λ(N)).
    """ 
    N = p * q
    if use_carmichael:
        phi = lcm(p-1, q-1)   # Carmichael's function λ(N)
    else:
        phi = (p-1)*(q-1)     # Euler's totient φ(N)
    if math.gcd(e, phi) != 1:
        raise ValueError("e must be coprime with φ(N)")
    d = modinv(e, phi)
    public_key = (N, e)
    private_key = (N, d)
    return public_key, private_key  

def extended_gcd(a, b):
    """
    Compute the extended Euclidean algorithm.
    Returns a tuple (g, x, y) such that g = gcd(a, b) and g = a*x + b*y.
    
    Parameters:
        a (int): First integer.
        b (int): Second integer.
    Returns:
        Tuple[int, int, int]: (g, x, y) where g = gcd(a, b) and x, y satisfy g = a*x + b*y.
    """
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y
        

def modinv(a, m):
    """
    Return the modular inverse of a modulo m.
    Finds x such that (a * x) % m == 1.
    
    Parameters:
        a (int): The number to invert.
        m (int): The modulus.
    Returns:
        int: Modular inverse of a modulo m.
    Raises:
        ValueError: If no modular inverse exists.
    """
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse exists")
    return x % m

def text_to_numbers(text):
    """
    Convert a text string (A-Z) to a list of numbers (A=0, ..., Z=25).
    Parameters:
        text (str): Uppercase or lowercase text.
    Returns:
        List[int]: List of numbers representing each letter.
    """
    return [ord(c) - ord('A') for c in text.upper()]

def numbers_to_text(numbers):
    """
    Convert a list of numbers (0-25) back to uppercase text (A-Z).
    Parameters:
        numbers (List[int]): List of numbers 0-25.
    Returns:
        str: Uppercase text string.
    """
    return ''.join(chr(n + ord('A')) for n in numbers)

def encrypt_text(text, public_key):
    """
    Encrypt a text string using an RSA public key.
    Parameters:
        text (str): Plaintext string to encrypt.
        public_key (Tuple[int, int]): RSA public key (e, n).
    Returns:
        List[int]: List of encrypted integers.
    """
    nums = text_to_numbers(text)
    return [encrypt_number(n, public_key) for n in nums]

def decrypt_text(cipher, private_key): 
    """
    Decrypt a list of integers using an RSA private key.
    Parameters:
        cipher_list (List[int]): List of encrypted integers.
        private_key (Tuple[int, int]): RSA private key (d, n).
    Returns:
        str: Decrypted text string.
    """
    decrypted_nums = [decrypt_number(c, private_key) for c in cipher]
    return numbers_to_text(decrypted_nums)

def order_mod(x, N):
    """
    Return the order r of x mod N.
    The order r is the smallest positive integer such that
    x^r ≡ 1 (mod N). The function determines r by brute force
    by repeatedly multiplying x modulo N until 1 is obtained.
    """
    r = 1
    value = x % N
    
    while value != 1:
        value = (value * x) % N
        r += 1
        
    return r


def shor_success(x, N):
    """
    Check whether Shor's classical post-processing step
    succeeds for a given x.

    The algorithm succeeds if:
    - x is coprime with N
    - the order r of x mod N is even
    - x^(r/2) is not congruent to -1 mod N

    If these conditions hold, gcd(x^(r/2) ± 1, N) yields
    a non-trivial factor of N.
    """
    
    if math.gcd(x, N) != 1:
        return True
    
    r = order_mod(x, N)
    
    if r % 2 != 0:
        return False
    
    xr2 = pow(x, r//2, N)
    
    if xr2 == N - 1:
        return False
    
    f1 = math.gcd(xr2 - 1, N)
    f2 = math.gcd(xr2 + 1, N)
    
    return (1 < f1 < N) or (1 < f2 < N)

def test_success_rate(N):
    """
    Test for how many numbers x (with 1 < x < N and gcd(x,N)=1)
    Shor's classical conditions succeed.

    Returns
    -------
    success : int
        Number of successful x
    total : int
        Total number of x coprime with N
    rate : float
        Success probability 
    """
    total = 0
    success = 0

    for x in range(2, N):
        if math.gcd(x, N) == 1:
            total += 1
            if shor_success(x, N):
                success += 1

    return success, total, success / total


def find_factor(x, N):
    """
    Attempt to find a non-trivial factor of N using
    Shor's classical post-processing step for a given x.

    Returns a factor of N if successful, otherwise None.
    """
    
    # Step 1: Check gcd
    if math.gcd(x, N) != 1:
        return math.gcd(x, N)
    
    # Step 2: Find order r of x modulo N
    r = order_mod(x, N)
    
    if r % 2 != 0:  # r must be even
        return None
    
    xr2 = pow(x, r // 2, N)
    
    if xr2 == N - 1:  # trivial case
        return None
    
    # Step 3: Compute potential factors
    f1 = math.gcd(xr2 - 1, N)
    f2 = math.gcd(xr2 + 1, N)
    
    if 1 < f1 < N:
        return f1
    if 1 < f2 < N:
        return f2
    
    return None


def iqft(state, n, n1):
    """
    Apply the inverse Quantum Fourier Transform (iQFT) to the first n1 qubits
    of an n-qubit quantum register.

    The function sequentially applies the gates of the iQFT circuit directly
    to the state vector instead of constructing the full unitary matrix.
    This avoids building a dense 2^n x 2^n operator and keeps the simulation
    memory efficient.

    The circuit consists of:
        1. Controlled phase rotations R_k^{-1}
        2. Hadamard gates
        3. Swap gates to reverse qubit order

    Only the first n1 qubits are transformed, while the remaining qubits
    remain unchanged (identity operation).

    Parameters
    ----------
    state : numpy.ndarray
        State vector of the n-qubit register in the computational basis.
        The dimension must be 2^n.
    n : int
        Total number of qubits in the register.
    n1 : int
        Number of qubits on which the iQFT should act (starting from qubit 0).

    Returns
    -------
    numpy.ndarray
        The transformed state vector after applying the iQFT.
    """

    psi = state

    # Main iQFT circuit
    for j in range(n1):

        # Apply controlled phase rotations
        for k in range(j):
            CR = buildSparseCRk(n, j, k, j-k+1, inverse=True) 
            psi = CR @ psi 

        # Apply Hadamard on qubit j
        op = 1
        for q in range(n):
            if q == j:
                op = np.kron(op, H)
            else:
                op = np.kron(op, I)

        psi = op @ psi

    # Final swaps (reverse qubit order)
    for i in range(n1 // 2):
        SW = swap_gate(n, i, n1 - i - 1)
        psi = SW @ psi

    return psi

def buildSparseGateSingle(n, i, gate):
    """
    Embed a single-qubit gate into an n-qubit register using sparse matrices.
    """
    sgate = sparse.csr_matrix(gate)
    left = sparse.identity(2**i, format="csr")
    right = sparse.identity(2**(n-i-1), format="csr")
    return sparse.kron(sparse.kron(left, sgate), right)


def buildSparseCNOT(n, ic, it):
    """
    Sparse n-qubit CNOT gate with control qubit ic and target qubit it.
    """
    P0ic = buildSparseGateSingle(n, ic, P0)
    P1ic = buildSparseGateSingle(n, ic, P1)
    Xit  = buildSparseGateSingle(n, it, X)
    return P0ic + P1ic @ Xit

def buildSparseCRk(n, ic, it, k, inverse=False):
    """
    Sparse controlled-Rk gate on n qubits.

    Parameters
    ----------
    n : int - total number of qubits
    ic : int - control qubit index
    it : int - target qubit index
    k : int - Rk parameter
    inverse : bool - if True, use Rk^\dagger
    """
    phase = np.exp(2j * np.pi / 2**k)
    if inverse:
        phase = np.conj(phase)
    R = np.array([[1,0],[0,phase]])

    P0ic = buildSparseGateSingle(n, ic, P0)
    P1ic = buildSparseGateSingle(n, ic, P1)
    Rt = buildSparseGateSingle(n, it, R)

    return P0ic + P1ic @ Rt


def swap_gate(n, q1, q2):
    """
    Sparse SWAP gate between qubits q1 and q2 in an n-qubit register.
    """
    dim = 2**n
    U = sparse.lil_matrix((dim, dim), dtype=complex)
    for i in range(dim):
        bits = list(format(i, f'0{n}b'))
        bits[q1], bits[q2] = bits[q2], bits[q1]
        j = int("".join(bits), 2)
        U[j,i] = 1
    return U.tocsr()


def Rk(k, inverse=False):
    """
    Single-qubit Rk gate:
        Rk = [[1, 0],
              [0, exp(2πi / 2^k)]]
    """
    phase = np.exp(2j * np.pi / 2**k)
    if inverse:
        phase = np.conj(phase)
    return np.array([[1,0],[0,phase]]) 


def qubits_for_number(N):
    """Return the minimum number of qubits needed to encode integers 0..N-1"""
    return int(np.ceil(np.log2(N)))

def build_x_tothe_z(t, x, N):
    """
    Construct the modular exponentiation gate U_x as a sparse matrix.

    Parameters
    ----------
    t : int
        Number of qubits in the first register.
    x : int
        Base of the exponentiation.
    N : int
        Modulus.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix representing U_x.
    """
    L = qubits_for_number(N)
    dim1 = 2**t          # first register dimension
    dim2 = 2**L          # second register dimension
    dim = dim1 * dim2    # total Hilbert space dimension

    row = np.arange(dim)  # row indices
    col = np.arange(dim)  # column indices (initialized to identity)

    # Loop over all states of the first register
    for j in range(dim1):
        # Loop over all states of the second register
        for y in range(dim2):
            if y < N:
                # Map |y> -> |x^j * y mod N> in the block
                col[j*dim2 + y] = j*dim2 + (y * pow(x, j, N)) % N

    # Build sparse matrix with 1s at positions (row, col)
    return sparse.csr_matrix((np.ones(dim), (row, col)))


def order_finding_state(t, x, N):
    """
    Simulates the quantum order finding algorithm (without measurement).

    Parameters
    ----------
    t : int
        Number of qubits in the first register (phase estimation register)
    x : int
        Base whose order modulo N should be found
    N : int
        Modulus

    Returns
    -------
    psi : numpy array
        Final state vector of the quantum circuit
    """

    L = int(np.ceil(np.log2(N)))

    dim1 = 2**t
    dim2 = 2**L
    dim = dim1 * dim2

    psi = np.zeros((dim1, dim2), dtype=complex)

    # initial state |0>|1>
    psi[0,1] = 1

    # Hadamard on first register -> uniform superposition
    psi = np.tile(psi[0], (dim1,1)) / np.sqrt(dim1)

    # modular exponentiation
    for a in range(dim1):
        y = pow(x, a, N)
        psi[a,:] = 0
        psi[a,y] = 1/np.sqrt(dim1)

    # apply inverse QFT
    psi = iqft(psi, t, L)
 
    return psi.reshape(dim)



