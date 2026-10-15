from qulacs import QuantumCircuit
import numpy as np
import random
from grover import Grover

def hadamard(nqubits: int, depth: int = 11) -> QuantumCircuit:
    circuit = QuantumCircuit(nqubits)
    for i in range(nqubits):
        for d in range(depth):
            circuit.add_H_gate(i)
    return circuit

def seq_hadamard(nqubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(nqubits)
    for i in reversed(range(nqubits-5, nqubits)):
        circuit.add_H_gate(i)
    for i in range(0, nqubits-4):
        circuit.add_H_gate(i)
    circuit.add_H_gate(nqubits-1)
    circuit.add_H_gate(0)
    circuit.add_H_gate(nqubits-2)
    circuit.add_H_gate(1)
    return circuit

def GHZ(nqubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(nqubits)
    circuit.add_H_gate(0)
    for i in range(0, nqubits-1):
        circuit.add_CNOT_gate(i, i+1)
    return circuit

def local_GHZ(nqubits: int, nlocalqubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(nqubits)
    circuit.add_H_gate(0)
    for i in range(0, nlocalqubits-1):
        circuit.add_CNOT_gate(i, i+1)
    return circuit

def QFT(nqubits: int) -> QuantumCircuit:
    from qulacs.gate import CNOT, H, RZ

    def cu1(qc, l, a, b):
        qc.add_gate(RZ(a, l/2))
        qc.add_gate(CNOT(a, b))
        qc.add_gate(RZ(b, -l/2))
        qc.add_gate(CNOT(a, b))
        qc.add_gate(RZ(b, l/2))

    circuit = QuantumCircuit(nqubits)
    circuit.add_X_gate(0)
    circuit.add_X_gate(2)
    for j in range(nqubits):
        for k in range(j):
            cu1(circuit, np.pi/float(2**(j-k)), j, k)
        circuit.add_gate(H(j))

    return circuit

def ising(nqubits: int, M: int = 1) -> QuantumCircuit:
    Bx = 2.0
    total_T = 15.0
    random.seed(555)

    Bz = [random.uniform(-2.0, 2.0) for i in range(0, nqubits)]
    J = [random.uniform(-2.0, 2.0) for i in range(0, nqubits)]

    def CZ(qc, q1, q2, phi):
        qc.add_RZ_gate(q2, 0.5*phi)
        qc.add_CNOT_gate(q1, q2)
        qc.add_RZ_gate(q2, -0.5*phi)
        qc.add_CNOT_gate(q1, q2)

    def ZcrossZ(qc, q1, q2, phi):
        qc.add_RZ_gate(q1, phi)
        qc.add_RZ_gate(q2, -phi)
        CZ(qc, q1, q2, -2.0*phi)

    def initialize(qc):
        for i in range(0, nqubits):
            qc.add_H_gate(i)

    def red_hamiltonian(qc, m):
        for i in range(0, nqubits-1, 2):
            phi = J[i] * (2.0 * m-1) / M
            ZcrossZ(qc, i, i+1, phi)

    def black_hamiltonian(qc, m):
        for i in range(1, nqubits-1, 2):
            phi = J[i] * (2.0*m - 1) / M
            ZcrossZ(qc, i, i+1, phi)

    def Bz_hamiltonian(qc, m):
        for i in range(0, nqubits):
            theta1 = (1.0 - (2.0*m - 1.0) / M) * -2.0 * Bx * total_T / M
            theta2 = (1.0 - (2.0*m - 1.0) / M) * -2.0 * Bz[i] * total_T / M
            qc.add_H_gate(i)
            qc.add_RZ_gate(i, theta1)
            qc.add_H_gate(i)
            qc.add_RZ_gate(i, theta2)

    circuit = QuantumCircuit(nqubits)
    initialize(circuit)
    for i in range(1, M+1):
        red_hamiltonian(circuit, i)
        black_hamiltonian(circuit, i)
        Bz_hamiltonian(circuit, i)
    return circuit

def HEA(nqubits: int, depth: int = 1) -> QuantumCircuit:
    def add_layer(circuit: QuantumCircuit):
        for i in range(0, nqubits):
            circuit.add_RY_gate(i, random.uniform(-np.pi, np.pi))
            circuit.add_RZ_gate(i, random.uniform(-np.pi, np.pi))
        for i in range(0, nqubits-1):
            circuit.add_CNOT_gate(i, i+1)

    random.seed(111)
    circuit = QuantumCircuit(nqubits)
    for _ in range(depth):
        add_layer(circuit)
    for i in range(0, nqubits):
        circuit.add_RY_gate(i, random.uniform(-np.pi, np.pi))
        circuit.add_RZ_gate(i, random.uniform(-np.pi, np.pi))
    return circuit

def HEA_within_block(nqubits: int, block_size: int, depth: int = 1) -> QuantumCircuit:
    def add_layer(circuit: QuantumCircuit):
        for i in range(0, block_size):
            circuit.add_RY_gate(i, random.uniform(-np.pi, np.pi))
            circuit.add_RZ_gate(i, random.uniform(-np.pi, np.pi))
        for i in range(0, block_size-1):
            circuit.add_CNOT_gate(i, i+1)

    random.seed(111)
    circuit = QuantumCircuit(nqubits)
    for _ in range(depth):
        add_layer(circuit)
    for i in range(0, block_size):
        circuit.add_RY_gate(i, random.uniform(-np.pi, np.pi))
        circuit.add_RZ_gate(i, random.uniform(-np.pi, np.pi))
    return circuit

def CC(nqubits: int) -> QuantumCircuit:
    """
    Quantum Coin Problem circuit implementation in Qulacs.
    Translated from the Qiskit implementation in cc.py.
    
    Args:
        nqubits: Number of qubits (nCoins = nqubits - 1)
    
    Returns:
        QuantumCircuit: The quantum coin circuit
    """
    nCoins = nqubits - 1
    circuit = QuantumCircuit(nqubits)
    
    random.seed(555)
    indexOfFalseCoin = random.randint(0, nCoins - 1)
            
    # Simplified version without measurements and conditional operations
    # This captures the core quantum operations but changes the algorithm semantics
    
    # Initial Hadamard gates on all coin qubits
    for i in range(nCoins):
        circuit.add_H_gate(i)
    
    # CNOT gates from all coin qubits to the ancilla qubit
    for i in range(nCoins):
        circuit.add_CNOT_gate(i, nCoins)
    
    # Apply operations that would happen in one branch of the original conditional logic
    # This is a deterministic approximation
    circuit.add_X_gate(nCoins)  # X gate on ancilla
    circuit.add_H_gate(nCoins)  # H gate on ancilla
    
    # CNOT from false coin to ancilla
    circuit.add_CNOT_gate(indexOfFalseCoin, nCoins)
    
    # H gates on all coin qubits
    for i in range(nCoins):
        circuit.add_H_gate(i)
    
    return circuit

def BV(nqubits: int) -> QuantumCircuit:
    """
    Bernstein-Vazirani quantum algorithm implementation in Qulacs.
    Translated from the Qiskit implementation in bv.py.
    
    Args:
        nqubits: Total number of qubits (n_qubits-1 for data + 1 ancilla)
    
    Returns:
        QuantumCircuit: The Bernstein-Vazirani circuit
    """

    def generate_hidden_string(n_qubits: int, prob: float = 0.5) -> str:
        """
        Generate a random binary string for Bernstein-Vazirani algorithm.
        
        Args:
            n_qubits: Number of qubits for the string
            prob: Probability of each bit being '1'
        
        Returns:
            str: Random binary string
        """
        answer = []
        for i in range(n_qubits):
            if random.random() <= prob:
                answer.append("1")
            else:
                answer.append("0")
        return "".join(answer)

    circuit = QuantumCircuit(nqubits)
    
    # Generate hidden string for n_qubits-1 data qubits
    random.seed(555)  # For reproducibility
    hidden_string = generate_hidden_string(nqubits - 1, 0.5)
    
    # Apply Hadamard gates to all data qubits (first n_qubits-1 qubits)
    for i in range(nqubits - 1):
        circuit.add_H_gate(i)
    
    # Prepare ancilla qubit (last qubit) in |-> state
    circuit.add_X_gate(nqubits - 1)  # |1⟩
    circuit.add_H_gate(nqubits - 1)  # |->
    
    # Apply oracle: CNOT gates based on hidden string
    # Reverse the string to match the original indexing
    hidden_string_reversed = hidden_string[::-1]
    for i in range(len(hidden_string_reversed)):
        if hidden_string_reversed[i] == "1":
            circuit.add_CNOT_gate(i, nqubits - 1)
    
    # Apply Hadamard gates to data qubits again
    for i in range(nqubits - 1):
        circuit.add_H_gate(i)
    
    return circuit

def QSVM(nqubits: int) -> QuantumCircuit:
    """
    Quantum Support Vector Machine circuit implementation in Qulacs.
    Translated from the Qiskit implementation in qsvm_raw.py.
    
    Args:
        nqubits: Number of qubits
    
    Returns:
        QuantumCircuit: The QSVM circuit
    """
    circuit = QuantumCircuit(nqubits)
    
    # Set random seed for reproducibility (matching the original)
    np.random.seed(42)  # Using a fixed seed for reproducible results
    
    def hadamard_circuit():
        """Apply Hadamard gates to all qubits"""
        for i in range(nqubits):
            circuit.add_H_gate(i)
    
    def phase_addition():
        """Apply phase addition operations"""
        # Phase gates on all qubits (P gate = RZ gate in this context)
        for i in range(nqubits):
            phase = np.random.rand() * np.pi
            circuit.add_RZ_gate(i, phase)
        
        # Forward pass: CNOT-RZ-CNOT on adjacent qubits
        for i in range(nqubits - 1):
            control_qubit = i
            target_qubit = i + 1
            circuit.add_CNOT_gate(control_qubit, target_qubit)
            rz_angle = np.random.rand() * np.pi
            circuit.add_RZ_gate(target_qubit, rz_angle)
            circuit.add_CNOT_gate(control_qubit, target_qubit)
        
        # Backward pass: CNOT-RZ-CNOT on adjacent qubits (reversed order)
        for i in range(nqubits - 1, 0, -1):
            control_qubit = i
            target_qubit = i - 1
            circuit.add_CNOT_gate(control_qubit, target_qubit)
            rz_angle = np.random.rand() * np.pi
            circuit.add_RZ_gate(target_qubit, rz_angle)
            circuit.add_CNOT_gate(control_qubit, target_qubit)
        
        # Final RZ gates on all qubits
        for i in range(nqubits):
            rz_angle = np.random.rand() * np.pi
            circuit.add_RZ_gate(i, rz_angle)
    
    # Build the circuit following the original structure
    hadamard_circuit()    # First Hadamard layer
    phase_addition()      # Phase addition operations
    hadamard_circuit()    # Second Hadamard layer
    
    return circuit

def QAOA(nqubits: int, depth: int = 1) -> QuantumCircuit:
    """
    Quantum Approximate Optimization Algorithm (QAOA) implementation in Qulacs for Maxcut.
    
    Args:
        nqubits: Number of qubits
        depth: Number of QAOA layers (p parameter)
    
    Returns:
        QuantumCircuit: The QAOA circuit
    """
    circuit = QuantumCircuit(nqubits)
    
    # Set random seed for reproducible parameters
    random.seed(777)
    np.random.seed(777)
    
    # Initialize state with Hadamard gates (equal superposition)
    for i in range(nqubits):
        circuit.add_H_gate(i)
    
    # QAOA layers
    for layer in range(depth):
        # Cost Hamiltonian (Problem-specific)
        # MaxCut: ZZ interactions on edges
        gamma = random.uniform(0, 2 * np.pi)
        for i in range(nqubits - 1):
            # ZZ interaction = CNOT-RZ-CNOT
            circuit.add_CNOT_gate(i, i + 1)
            circuit.add_RZ_gate(i + 1, gamma)
            circuit.add_CNOT_gate(i, i + 1)
        
        # Mixer Hamiltonian (X rotations)
        beta = random.uniform(0, np.pi)
        for i in range(nqubits):
            circuit.add_RX_gate(i, beta)
    
    return circuit

def Wstate(nqubits: int) -> QuantumCircuit:
    import math
    
    def F_gate(qc, i, j, n, k):
        theta = math.acos(math.sqrt(1./(n-k+1)))
        qc.add_RY_gate(j, -theta)
        # qc.add_CZ_gate(i, j)
        qc.add_H_gate(j)
        qc.add_CNOT_gate(i, j)
        qc.add_H_gate(j)
        qc.add_RY_gate(j, theta)

    circuit = QuantumCircuit(nqubits)
    n = nqubits

    circuit.add_X_gate(n-1)

    for i in range(0, n-1):
        F_gate(circuit, n-1-i, n-2-i, n, i+1)

    for i in range(0, n-1):
        circuit.add_CNOT_gate(n-2-i, n-1-i)
    
    return circuit

def get_circuit(name: str, nqubits: int, block_size: int|None = None,
                local_size: int|None = None) -> QuantumCircuit:
    if name == "hadamard":
        return hadamard(nqubits)
    elif name == "seq_hadamard":
        return seq_hadamard(nqubits)
    elif name == "ghz":
        return GHZ(nqubits)
    elif name == "local_ghz":
        if local_size is None:
            raise ValueError("local_size must be provided for local_ghz circuit")
        return local_GHZ(nqubits, local_size)
    elif name == "qft":
        return QFT(nqubits)
    elif name == "ising":
        return ising(nqubits)
    elif name == "hea":
        return HEA(nqubits, depth=1)
    elif name == "hea_block":
        return HEA_within_block(nqubits, block_size=block_size, depth=1)
    elif name == "grover":
        return Grover(nqubits, nsteps=1)
    elif name == "cc":
        return CC(nqubits)
    elif name == "bv":
        return BV(nqubits)
    elif name == "qsvm":
        return QSVM(nqubits)
    elif name == "qaoa":
        return QAOA(nqubits, depth=1)
    elif name == "wstate":
        return Wstate(nqubits)
    else:
        raise ValueError(f"Unknown circuit name: {name}")