#!/usr/bin/env python3
"""
Grover's Search Algorithm using Qulacs

Implements Grover's Search algorithm to find the square root of x
in the polynomial ring GF(2)[x] mod (x^n + x + 1).

This is a Python implementation using Qulacs that corresponds to
the C implementation in grover_search.c

Author: Generated for Qulacs comparison
Date: December 22, 2025
"""

import math
from qulacs import QuantumCircuit
# from qulacs.gate import T, Tdag #TOFFOLI

def my_toffoli(circuit, control1, control2, target):
    """
    Implement TOFFOLI gate using 1-qubit and 2-qubit gates
    Using RZ gates for better compatibility
    """
    import numpy as np
    
    circuit.add_H_gate(target)
    circuit.add_CNOT_gate(control2, target)
    circuit.add_Tdag_gate(target)
    circuit.add_CNOT_gate(control1, target)
    circuit.add_T_gate(target)
    circuit.add_CNOT_gate(control2, target)
    circuit.add_Tdag_gate(target)
    circuit.add_CNOT_gate(control1, target)
    circuit.add_T_gate(control2)
    circuit.add_T_gate(target)
    circuit.add_CNOT_gate(control1, control2)
    circuit.add_H_gate(target)
    circuit.add_T_gate(control1)
    circuit.add_Tdag_gate(control2)
    circuit.add_CNOT_gate(control1, control2)

def diffuse(circuit, n, q_indices, x_indices):
    """
    Diffusion operator - implementation of C diffuse() function
    
    Args:
        circuit: QuantumCircuit object
        q_indices: List of n qubit indices for the search register
        x_indices: List of n-1 auxiliary qubit indices (scratch register)
    """
    # Hadamard applied to q
    for j in range(n):
        circuit.add_H_gate(q_indices[j])
    
    # Want to phase flip on q = 00...0
    # So invert q to compute q[0] and q[1] and ... and q[n-1]
    for j in range(n):
        circuit.add_X_gate(q_indices[j])
    
    # Compute x[n-2] = q[0] and q[1] and ... and q[n-1]
    # Build up the AND of all q qubits using auxiliary qubits
    my_toffoli(circuit, q_indices[1], q_indices[0], x_indices[0])
    for j in range(1, n-1):
        my_toffoli(circuit, x_indices[j-1], q_indices[j+1], x_indices[j])
    
    # Phase flip conditioned on x[n-2]
    circuit.add_Z_gate(x_indices[n-2])  # Phase Flip==Z if q=00...0, i.e. x[n-2]==1
    
    # Undo the local registers
    for j in range(n-2, 0, -1):
        my_toffoli(circuit, x_indices[j-1], q_indices[j+1], x_indices[j])
    my_toffoli(circuit, q_indices[1], q_indices[0], x_indices[0])
    
    # Restore q
    for j in range(n):
        circuit.add_X_gate(q_indices[j])
    
    # Complete the diffusion
    for j in range(n):
        circuit.add_H_gate(q_indices[j])

def Sqr(circuit, n, a_indices, b_indices):
    """
    Squaring operation - implementation of C Sqr() function
    Computes b(x) = a(x) * a(x) mod (x^n + x + 1)
    
    Args:
        circuit: QuantumCircuit object
        a_indices: List of n qubit indices for input register a
        b_indices: List of n qubit indices for output register b
    """
    # First loop: for i from 0 to (n-1)//2
    for i in range((n-1)//2 + 1):
        k = 2 * i
        if k < n:
            circuit.add_CNOT_gate(a_indices[i], b_indices[k])
    
    # Second loop: for i from (n+1)//2 to n-1
    for i in range((n+1)//2, n):
        k = (2*i) - n
        if k < n:
            circuit.add_CNOT_gate(a_indices[i], b_indices[k])
        if k + 1 < n:
            circuit.add_CNOT_gate(a_indices[i], b_indices[k + 1])

def EQxMark(circuit, n, b_indices, t_indices, x_indices, tF):
    """
    Test if b(x) == x - implementation of C EQxMark() function
    Tests if the input polynomial b(x) equals x over GF(2)[x] mod (x^n + x + 1)
    
    Args:
        circuit: QuantumCircuit object
        b_indices: List of n qubit indices for register b
        t_indices: List with 1 qubit index for test result
        x_indices: List of n-1 auxiliary qubit indices (scratch register)
        tF: If tF!=0, set return result in qubit t[0], else apply Z
    """
    # Change b to reflect testing for the polynomial x
    for j in range(n):
        if j != 1:
            circuit.add_X_gate(b_indices[j])
    
    # Compute x[n-2] = b[0] and b[1] and ... and b[n-1]
    my_toffoli(circuit, b_indices[1], b_indices[0], x_indices[0])
    for j in range(1, n-1):
        my_toffoli(circuit, x_indices[j-1], b_indices[j+1], x_indices[j])
    
    # Either return result in t or Phase flip conditioned on x[n-2]
    if tF != 0:
        circuit.add_CNOT_gate(x_indices[n-2], t_indices[0])  # Returns result in t
    else:
        circuit.add_Z_gate(x_indices[n-2])  # Phase Flip==Z if b=01...0 == 'x', i.e. x[n-2]==1
    
    # Undo the local registers
    for j in range(n-2, 0, -1):
        my_toffoli(circuit, x_indices[j-1], b_indices[j+1], x_indices[j])
    my_toffoli(circuit, b_indices[1], b_indices[0], x_indices[0])
    
    # Restore b
    for j in range(n):
        if j != 1:
            circuit.add_X_gate(b_indices[j])

def Grover(nqubits: int, nsteps: int | None = 1) -> QuantumCircuit:
    """
    Create the complete Grover search quantum circuit - implementation of C main() function
    
    Returns:
        QuantumCircuit: Complete circuit for Grover's algorithm
        tuple: (a_indices, b_indices, t_indices) for measurement
    """
    # Calculate total number of qubits needed
    # a[n] + b[n] + t[1] + 2 * x[n-1] (two sets of auxiliary qubits for diffuse and EQxMark)
    assert (nqubits + 1) % 4 == 0, "nqubits + 1 must be divisible by 4"
    n = (nqubits + 1) // 4
    # n_qubits = n + n + 1 + 2*(n-1)
    circuit = QuantumCircuit(nqubits)
    
    # Define qubit indices - matching C variable layout
    a_indices = list(range(n))                               # a[0] to a[n-1]
    b_indices = list(range(n, 2*n))                         # b[0] to b[n-1]
    t_indices = [2*n]                                       # t[0]
    x_diffuse_indices = list(range(2*n+1, 2*n+1+n-1))      # x[0] to x[n-2] for diffuse
    x_eq_indices = list(range(2*n+1+n-1, 2*n+1+2*(n-1)))   # x[0] to x[n-2] for EQxMark
    
    # Grover parameters - matching C implementation
    N = int(2**n)
    if nsteps is None:
        nsteps = int(math.floor((math.pi/4) * math.sqrt(N)))
    
    print(f"Grover Search for sqrt(x) in GF(2)[x]/(x^{n}+x+1)")
    print(f"Problem size: n = {n}")
    print(f"Database size: N = 2^{n} = {N}")
    print(f"Number of iterations: {nsteps}")
    print(f"Total qubits: {nqubits}")
    
    # Initialize a[0..n-1] into uniform superposition
    for i in range(n):
        circuit.add_H_gate(a_indices[i])
    
    # Grover iteration: Mark then diffuse
    for istep in range(1, nsteps + 1):
        # Sqr(a, b) - Sets b(x) = a(x) * a(x)
        Sqr(circuit, n, a_indices, b_indices)
        
        # EQxMark(b, t, 0) - Tests if b(x) == x and if so Phase Flips
        EQxMark(circuit, n, b_indices, t_indices, x_eq_indices, 0)
        
        # Sqr(a, b) - Note: Sqr is its own inverse
        Sqr(circuit, n, a_indices, b_indices)
        
        # Diffuse
        diffuse(circuit, n, a_indices, x_diffuse_indices)
    
    # For the final measurement, compute causal state
    Sqr(circuit, n, a_indices, b_indices)
    EQxMark(circuit, n, b_indices, t_indices, x_eq_indices, 1)  # tF=1 implies test result returned in t
    
    return circuit

if __name__ == "__main__":
    """
    Run the Grover search algorithm and measure results - implementation of C main() function
    """
    print("Creating Grover search quantum circuit...")
    circuit = Grover(nqubits=31, nsteps=1)
    
    print(f"Circuit created: {circuit.get_gate_count()} gates, {circuit.get_qubit_count()} qubits")