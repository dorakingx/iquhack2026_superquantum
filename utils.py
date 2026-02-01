"""
Utility functions for quantum circuit compilation in the Superquantum challenge.

This module provides functions for:
1. Calculating operator norm distance between target unitaries and circuits
2. Counting T and T_dag gates in circuits
3. Exporting circuits to OpenQASM 2.0 format
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import svd


def operator_norm_distance(target_unitary, circuit, optimize_phase=True):
    """
    Calculate the operator norm distance between a target unitary and a quantum circuit.
    
    The operator norm distance is defined as:
        d(U, Ũ) = min_φ ||U - e^{iφ}Ũ||_op
    
    where ||·||_op is the operator norm (largest singular value).
    
    Args:
        target_unitary (np.ndarray): Target unitary matrix (2^n × 2^n complex matrix)
        circuit (QuantumCircuit): Qiskit quantum circuit to compare
        optimize_phase (bool): If True, optimize over global phase φ. Default True.
    
    Returns:
        float: The operator norm distance (non-negative)
    
    Raises:
        ValueError: If dimensions don't match or circuit is invalid
    """
    # Convert circuit to unitary matrix
    circuit_operator = Operator(circuit)
    circuit_unitary = circuit_operator.data
    
    # Validate dimensions
    if target_unitary.shape != circuit_unitary.shape:
        raise ValueError(
            f"Dimension mismatch: target_unitary has shape {target_unitary.shape}, "
            f"circuit_unitary has shape {circuit_unitary.shape}"
        )
    
    # Handle empty circuits
    if circuit.num_qubits == 0:
        return np.linalg.norm(target_unitary, ord=2)
    
    if optimize_phase:
        # Closed-form solution: optimal phase φ = arg(Tr(U†Ũ))
        # This minimizes ||U - e^{iφ}Ũ||_op
        trace_product = np.trace(np.conj(target_unitary).T @ circuit_unitary)
        
        # Handle case where trace is zero (orthogonal unitaries)
        if np.abs(trace_product) < 1e-15:
            # If trace is zero, any phase gives same result, use φ=0
            optimal_phase = 0.0
        else:
            optimal_phase = np.angle(trace_product)
        
        # Apply optimal phase
        phase_factor = np.exp(1j * optimal_phase)
        difference = target_unitary - phase_factor * circuit_unitary
    else:
        # No phase optimization, use φ=0
        difference = target_unitary - circuit_unitary
    
    # Compute operator norm (largest singular value)
    _, singular_values, _ = svd(difference)
    operator_norm = np.max(singular_values)
    
    return float(operator_norm)


def count_t_gates(circuit):
    """
    Count the total number of T and T_dag gates in a quantum circuit.
    
    This counts both 't' and 'tdg' gates, which is the T-count metric
    used in Clifford+T synthesis.
    
    Args:
        circuit (QuantumCircuit): Qiskit quantum circuit
    
    Returns:
        int: Total count of T and T_dag gates
    """
    gate_counts = circuit.count_ops()
    
    t_count = gate_counts.get('t', 0)
    tdg_count = gate_counts.get('tdg', 0)
    
    return int(t_count + tdg_count)


def export_to_openqasm(circuit, filename=None):
    """
    Export a quantum circuit to OpenQASM 2.0 format.
    
    Args:
        circuit (QuantumCircuit): Qiskit quantum circuit to export
        filename (str, optional): If provided, write OpenQASM to this file.
                                  If None, return the OpenQASM string.
    
    Returns:
        str or None: OpenQASM 2.0 string if filename is None,
                     None if written to file
    
    Raises:
        IOError: If file writing fails
    """
    qasm_string = circuit.qasm()
    
    if filename is not None:
        with open(filename, 'w') as f:
            f.write(qasm_string)
        return None
    else:
        return qasm_string
