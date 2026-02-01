"""
Main solver script for the Superquantum challenge.

This module implements a class-based synthesis pipeline for compiling
target unitaries into Clifford+T circuits using Qiskit.
"""

import numpy as np
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from utils import operator_norm_distance, count_t_gates, export_to_openqasm


class ChallengeSolver(ABC):
    """
    Base class for quantum circuit synthesis solvers.
    
    Provides common functionality for evaluating synthesized circuits,
    including T-count calculation and operator norm distance measurement.
    """
    
    def __init__(self, target_unitary, problem_name):
        """
        Initialize the solver with a target unitary and problem name.
        
        Args:
            target_unitary (np.ndarray): Target unitary matrix (2^n × 2^n)
            problem_name (str): Name/description of the problem
        """
        self.target_unitary = target_unitary
        self.problem_name = problem_name
        self.circuit = None
        
        # Validate unitary dimensions
        dim = target_unitary.shape[0]
        if dim != target_unitary.shape[1]:
            raise ValueError("Target unitary must be square")
        if not np.isclose(np.linalg.det(target_unitary), 1.0, atol=1e-6):
            # Check if unitary (up to global phase)
            pass  # Allow global phase differences
    
    def check_t_count(self):
        """
        Count T and T_dag gates in the synthesized circuit.
        
        Returns:
            int: Total T-count (T + T_dag gates)
        
        Raises:
            ValueError: If circuit has not been synthesized yet
        """
        if self.circuit is None:
            raise ValueError("Circuit has not been synthesized yet. Call synthesize() first.")
        return count_t_gates(self.circuit)
    
    def calculate_distance(self):
        """
        Calculate operator norm distance between target and synthesized circuit.
        
        Returns:
            float: Operator norm distance (with global phase optimization)
        
        Raises:
            ValueError: If circuit has not been synthesized yet
        """
        if self.circuit is None:
            raise ValueError("Circuit has not been synthesized yet. Call synthesize() first.")
        return operator_norm_distance(self.target_unitary, self.circuit, optimize_phase=True)
    
    def export_qasm(self, filename=None):
        """
        Export the synthesized circuit to OpenQASM 2.0 format.
        
        Args:
            filename (str, optional): If provided, write to file; otherwise return string
        
        Returns:
            str or None: OpenQASM string or None if written to file
        
        Raises:
            ValueError: If circuit has not been synthesized yet
        """
        if self.circuit is None:
            raise ValueError("Circuit has not been synthesized yet. Call synthesize() first.")
        return export_to_openqasm(self.circuit, filename)
    
    def evaluate_circuit(self):
        """
        Evaluate the synthesized circuit: calculate T-count and distance.
        
        Returns:
            dict: Dictionary with 't_count' and 'distance' keys
        """
        t_count = self.check_t_count()
        distance = self.calculate_distance()
        return {
            't_count': t_count,
            'distance': distance
        }
    
    @abstractmethod
    def synthesize(self):
        """
        Synthesize the target unitary into a Clifford+T circuit.
        
        This method must be implemented by subclasses.
        It should set self.circuit to a QuantumCircuit instance.
        """
        pass
    
    def solve(self):
        """
        Main workflow: synthesize the circuit and evaluate it.
        
        Returns:
            dict: Results dictionary with 't_count', 'distance', and 'circuit' keys
        """
        self.synthesize()
        results = self.evaluate_circuit()
        results['circuit'] = self.circuit
        return results


class UnitarySolver(ChallengeSolver):
    """
    Solver for direct unitary matrix synthesis.
    
    Used for Problems 1 and 2 where we have explicit unitary matrices.
    """
    
    def synthesize(self):
        """
        Synthesize a unitary matrix into a Clifford+T circuit.
        
        Process:
        1. Create QuantumCircuit with appropriate number of qubits
        2. Embed the unitary using circuit.unitary()
        3. Transpile to basis ['h', 't', 'tdg', 'cx']
        4. Store result in self.circuit
        """
        # Determine number of qubits from unitary dimension
        dim = self.target_unitary.shape[0]
        num_qubits = int(np.log2(dim))
        
        if 2**num_qubits != dim:
            raise ValueError(f"Unitary dimension {dim} is not a power of 2")
        
        # Create circuit and embed unitary
        circuit = QuantumCircuit(num_qubits)
        circuit.unitary(self.target_unitary, range(num_qubits), label='target')
        
        # Transpile to Clifford+T basis
        # Define target basis gates
        basis_gates = ['h', 't', 'tdg', 'cx']
        
        # Use transpile with basis gates and optimization
        transpiled_circuit = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=3
        )
        
        self.circuit = transpiled_circuit


class HamiltonianSolver(ChallengeSolver):
    """
    Solver for Hamiltonian exponential synthesis.
    
    Used for Problem 3+ where we need to synthesize exp(i*H) for some Hamiltonian H.
    This will use Trotterization or KAK decomposition approaches.
    """
    
    def synthesize(self):
        """
        Synthesize a Hamiltonian exponential into a Clifford+T circuit.
        
        This is a stub implementation for future Problem 3.
        Will implement Trotterization/KAK decomposition here.
        """
        raise NotImplementedError(
            "HamiltonianSolver.synthesize() not yet implemented. "
            "This will be used for Problem 3 (exp(i*pi/7 * Z⊗Z))"
        )


def create_controlled_y():
    """
    Create the Controlled-Y gate unitary matrix.
    
    Returns:
        np.ndarray: 4×4 complex matrix representing CY gate
    """
    CY = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype=complex)
    return CY


def create_controlled_ry(theta):
    """
    Create the Controlled-Ry gate unitary matrix.
    
    Args:
        theta (float): Rotation angle
    
    Returns:
        np.ndarray: 4×4 complex matrix representing CRy(θ) gate
    """
    cry_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta/2), 0, -np.sin(theta/2)],
        [0, 0, 1, 0],
        [0, np.sin(theta/2), 0, np.cos(theta/2)]
    ], dtype=complex)
    return cry_matrix


def main():
    """
    Main execution: solve Problems 1 and 2.
    """
    # Define problems: (problem_name, target_unitary, solver_class)
    problems = {
        "Problem 1: Controlled-Y": (
            create_controlled_y(),
            UnitarySolver
        ),
        "Problem 2: Controlled-Ry(π/7)": (
            create_controlled_ry(np.pi / 7),
            UnitarySolver
        )
    }
    
    print("=" * 60)
    print("Superquantum Challenge Solver")
    print("=" * 60)
    print()
    
    results_summary = []
    
    for problem_name, (target_unitary, solver_class) in problems.items():
        print(f"Solving {problem_name}...")
        print("-" * 60)
        
        try:
            solver = solver_class(target_unitary, problem_name)
            results = solver.solve()
            
            t_count = results['t_count']
            distance = results['distance']
            
            print(f"  T-count: {t_count}")
            print(f"  Operator Norm Distance: {distance:.6e}")
            print()
            
            results_summary.append({
                'problem': problem_name,
                't_count': t_count,
                'distance': distance
            })
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            print()
            results_summary.append({
                'problem': problem_name,
                'error': str(e)
            })
    
    # Print summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for result in results_summary:
        if 'error' in result:
            print(f"{result['problem']}: ERROR - {result['error']}")
        else:
            print(f"{result['problem']}: T-count={result['t_count']}, Distance={result['distance']:.6e}")
    print()


if __name__ == "__main__":
    main()
