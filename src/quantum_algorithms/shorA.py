

import numpy as np
from collections import Counter
import itertools 
from scipy import sparse 
import scipy
import matplotlib.pyplot as plt 
import math 
from fractions import Fraction 

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
    #Check gcd
    if math.gcd(x, N) != 1:
        return math.gcd(x, N)
    #Find order r of x modulo N
    r = order_mod(x, N)
    if r % 2 != 0:  # r must be even
        return None
    xr2 = pow(x, r // 2, N)
    if xr2 == N - 1:  # trivial case
        return None
    #Compute potential factors
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
    """
    L = qubits_for_number(N)
    dim1=2**t # first register dimension
    dim2=2**L # second register dimension
    dim=dim1*dim2 # total Hilbert space dimension
    row=np.arange(dim) # indexes all rows
    col=np.arange(dim) # will contain the position (col) of the non-zero entry for each row
    for j in range(dim1): # loop over states of the first register
        for y in range(dim2): # loop over states of the second register
            if y < N: # if y>=N, we want the identity, so we leave the column index unchanged (it was initialized to be equal to the row index)
                col[j*dim2+y]=j*dim2 + np.mod(y*pow(x,j,N),N)
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

    
def cont_frac(phi, max_denom):
    """
    Compute the continued fraction representation of a number phi.

    The function returns a list of integers representing the continued fraction
    expansion of phi, truncated so that the resulting rational approximation
    has a denominator at most max_denom.
    Args:
        phi (float): The number to approximate as a continued fraction.
        max_denom (int): The maximum allowed denominator for the rational approximation.

    Returns:
        list[int]: The continued fraction coefficients of phi.
    """
    frac = []

    # integer part
    a = phi // 1
    r = phi - a
    frac.append(int(a))

    while eval_contfrac_rational(frac)[1] <= max_denom and r > 1 / max_denom:
        a = (1 / r) // 1
        r = (1 / r) - a
        frac.append(int(a))

    if r < 1 / max_denom:
        return frac
    else:
        return frac[:-1] 

def eval_contfrac(frac):
    """
    Evaluate a continued fraction as a floating-point number.
    Args:
        frac (list[int]): A list of integers representing a continued fraction.
    Returns:
        float: The decimal value of the continued fraction.
    """
    n = len(frac)
    a = 0
    for i in range(n - 1, 0, -1):
        a = 1 / (frac[i] + a)
    a = frac[0] + a
    return a


def eval_contfrac_rational(frac):
    """
    Evaluate a continued fraction as an exact rational number.
    Args:
        frac (list[int]): A list of integers representing a continued fraction.
    Returns:
        list[int]: A two-element list [numerator, denominator] representing
                   the exact rational value of the continued fraction.
    """
    n = len(frac)
    if n == 1:
        return [frac[0], 1]
    numer = 1
    denom = frac[n - 1]
    for i in range(n - 2, 0, -1):
        denom_new = denom * frac[i] + numer
        numer = denom
        denom = denom_new
    numer = frac[0] * denom + numer
    return [numer, denom]

def qft(vector):
    """
    Discrete Fourier Transform using NumPy (quantum-style).
    
    """
    N = len(vector)
    omega = np.exp(2j * np.pi / N)
    result = np.zeros(N, dtype=complex)
    for k in range(N):
        result[k] = sum(vector[n] * omega**(k*n) for n in range(N))
    return result / np.sqrt(N)

def simulates_shor_algorithm(N):
    """simulates_Shor_algorithm classically for small integer N."""
    if N % 2 == 0:
        return [2, N//2]

    for attempt in range(5):  # try a few random numbers
        a = random.randint(2, N-1)
        if gcd(a, N) != 1:
            return [gcd(a, N), N//gcd(a, N)]

        # Find period r of f(x) = a^x mod N
        r = 1
        while pow(a, r, N) != 1:
            r += 1

        if r % 2 != 0:
            continue

        factor1 = gcd(pow(a, r//2) - 1, N)
        factor2 = gcd(pow(a, r//2) + 1, N)

        if factor1 not in [1, N] or factor2 not in [1, N]:
            return [factor1, factor2]
    return [N]  # failure 


def is_coprime(a, N):
    """
    Returns True if a and N are coprime (gcd(a, N) == 1)
    """
    return math.gcd(a, N) == 1


def continued_fraction_expansion(num, den, max_den=100):
    """
    Approximates num/den as a fraction with denominator <= max_den
    This is used to guess the order r from the measurement result
    """
    frac = Fraction(num, den).limit_denominator(max_den)
    return frac.denominator


# Simulate quantum order-finding
def orderFindingSim(N, x, n_first_register):
    """
    Returns:
        pvec: probability vector for measurement outcomes
        r_true: actual order of x mod N
    """
    # Compute the true order r of x mod N
    r_true = 1
    while pow(x, r_true, N) != 1:
        r_true += 1
    
    # Build probability vector: peaks at multiples of 2^n / r
    M = 2**n_first_register
    pvec = np.zeros(M)
    for k in range(M):
        # Each peak corresponds to multiples of M / r_true
        for j in range(r_true):
            peak = int(j * M / r_true)
            if peak < M:
                pvec[peak] = 1
    pvec = pvec / pvec.sum()  # normalize
    return pvec, r_true

def factor(N):
    """
    Classical simulation of Shor's factoring algorithm
    1. Pick random x coprime with N
    2. Use simulated quantum order-finding to get r
    3. Compute potential factors using gcd(x^(r/2) ± 1, N)
    """
    n_first_register = 2 * N.bit_length()  #register size
    while True:
        x = np.random.randint(2, N)
        if not is_coprime(x, N):
            return gcd(x, N)  # found a factor trivially
        pvec, true_r = orderFindingSim(N, x, n_first_register) #Returns probability vector pvec
        m = doMeasurement(pvec) 
        r = continued_fraction_expansion(m, 2**n_first_register, max_den=N)
        trials = 0
        while trials < 5:
            m2 = doMeasurement(pvec)
            r2 = continued_fraction_expansion(m2, 2**n_first_register, max_den=N)
            r = np.lcm(r, r2)
            trials += 1
        if r % 2 == 0 and pow(x, r//2, N) != N-1:
            f1 = gcd(pow(x, r//2) - 1, N)
            f2 = gcd(pow(x, r//2) + 1, N)
            if f1 not in [1, N]:
                return f1
            if f2 not in [1, N]:
                return f2
