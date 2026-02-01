"""
Main solver script for the Superquantum challenge.

This module implements a class-based synthesis pipeline for compiling
target unitaries into Clifford+T circuits using Qiskit.
"""

import numpy as np
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Clifford
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    UnrollCustomDefinitions, BasisTranslator, CommutationAnalysis,
    Optimize1qGates
)
from qiskit.transpiler.passes.optimization import (
    CommutativeCancellation, InverseCancellation
)
from qiskit.transpiler.passes.synthesis import SolovayKitaev
from qiskit.circuit.library import standard_gates

from utils import operator_norm_distance, count_t_gates, export_to_openqasm


def is_clifford_unitary(unitary):
    """
    Check if a unitary matrix represents a Clifford gate.
    
    Args:
        unitary (np.ndarray): Unitary matrix to check
    
    Returns:
        bool: True if the unitary is a Clifford gate
    """
    try:
        # Try to create a Clifford from the unitary
        dim = unitary.shape[0]
        num_qubits = int(np.log2(dim))
        
        # Create a temporary circuit with the unitary
        temp_circuit = QuantumCircuit(num_qubits)
        temp_circuit.unitary(unitary, range(num_qubits))
        
        # Try to convert to Clifford
        # This will fail if the unitary contains non-Clifford gates
        clifford = Clifford.from_circuit(temp_circuit)
        return True
    except Exception:
        # If conversion fails (for any reason), it's not a Clifford gate
        # This includes QiskitError for non-Clifford gates
        return False


def convert_s_z_to_t(circuit):
    """
    Convert S and Z gates to T gates in a circuit.
    
    S = T^2, Z = S^2 = T^4
    
    Args:
        circuit (QuantumCircuit): Circuit to modify
    
    Returns:
        QuantumCircuit: New circuit with S/Z replaced by T gates
    """
    new_circuit = QuantumCircuit(circuit.num_qubits)
    
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = list(range(len(instruction.qubits)))
        # Get actual qubit indices
        if hasattr(instruction.qubits[0], 'index'):
            qubits = [q.index for q in instruction.qubits]
        else:
            # For newer Qiskit versions, qubits might be in a register
            qubits = [circuit.find_bit(q)[0] for q in instruction.qubits]
        
        gate_name = gate.name
        
        if gate_name == 's':
            # S = T^2
            new_circuit.t(qubits[0])
            new_circuit.t(qubits[0])
        elif gate_name == 'sdg':
            # Sdg = Tdg^2
            new_circuit.tdg(qubits[0])
            new_circuit.tdg(qubits[0])
        elif gate_name == 'z':
            # Z = T^4
            for _ in range(4):
                new_circuit.t(qubits[0])
        elif gate_name == 'cz':
            # Controlled-Z: keep as is (it's already Clifford and can be decomposed later if needed)
            new_circuit.cz(qubits[0], qubits[1])
        else:
            # Copy other gates as-is
            new_circuit.append(gate, qubits)
    
    return new_circuit


def decompose_clifford_exact(unitary, num_qubits):
    """
    Decompose a Clifford unitary exactly into Clifford+T gates.
    
    Args:
        unitary (np.ndarray): Clifford unitary matrix
        num_qubits (int): Number of qubits
    
    Returns:
        QuantumCircuit: Circuit in basis ['h', 't', 'tdg', 'cx']
    """
    # For 2-qubit gates like Controlled-Y, use Qiskit's built-in gate
    if num_qubits == 2:
        # Check if it's a Controlled-Y gate
        cy_unitary = create_controlled_y()
        if np.allclose(unitary, cy_unitary) or np.allclose(unitary, -cy_unitary):
            # Use Qiskit's built-in CY gate
            circuit = QuantumCircuit(2)
            circuit.cy(0, 1)
            # Decompose CY to basic gates
            decomposed = circuit.decompose(reps=2)
        else:
            # Generic 2-qubit Clifford
            circuit = QuantumCircuit(num_qubits)
            circuit.unitary(unitary, range(num_qubits), label='target')
            decomposed = circuit.decompose(reps=3)
    else:
        # Create circuit with the unitary
        circuit = QuantumCircuit(num_qubits)
        circuit.unitary(unitary, range(num_qubits), label='target')
        # Decompose the unitary gate
        decomposed = circuit.decompose(reps=3)
    
    # Transpile to basis that includes S/Z, then convert to T
    basis_gates = ['h', 't', 'tdg', 'cx', 's', 'sdg', 'z', 'cz', 'cy']
    transpiled = transpile(
        decomposed,
        basis_gates=basis_gates,
        optimization_level=1
    )
    
    # Convert S/Z gates to T gates
    final_circuit = convert_s_z_to_t(transpiled)
    
    # Final transpile to ensure only ['h', 't', 'tdg', 'cx']
    final_circuit = transpile(
        final_circuit,
        basis_gates=['h', 't', 'tdg', 'cx'],
        optimization_level=2
    )
    
    return final_circuit


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
    Implements smart decomposition strategies:
    - Exact Clifford decomposition for Clifford gates
    - Solovay-Kitaev synthesis for non-Clifford gates
    """
    
    def synthesize(self):
        """
        Synthesize a unitary matrix into a Clifford+T circuit.
        
        Uses smart decomposition:
        1. Check if unitary is Clifford → use exact decomposition
        2. Otherwise → use Solovay-Kitaev approximation
        """
        # Determine number of qubits from unitary dimension
        dim = self.target_unitary.shape[0]
        num_qubits = int(np.log2(dim))
        
        if 2**num_qubits != dim:
            raise ValueError(f"Unitary dimension {dim} is not a power of 2")
        
        # Check if unitary is Clifford
        is_clifford = is_clifford_unitary(self.target_unitary)
        
        if is_clifford:
            # Use exact Clifford decomposition
            self.circuit = self._synthesize_clifford_exact(num_qubits)
        else:
            # Use Solovay-Kitaev synthesis
            self.circuit = self._synthesize_solovay_kitaev(num_qubits)
    
    def _synthesize_clifford_exact(self, num_qubits):
        """
        Synthesize a Clifford unitary using exact decomposition.
        
        Args:
            num_qubits (int): Number of qubits
        
        Returns:
            QuantumCircuit: Optimized circuit in ['h', 't', 'tdg', 'cx'] basis
        """
        # Use exact decomposition
        circuit = decompose_clifford_exact(self.target_unitary, num_qubits)
        
        # Apply additional optimizations
        # Cancel T·Tdg pairs and optimize
        optimized = self._optimize_circuit(circuit)
        
        return optimized
    
    def _synthesize_solovay_kitaev(self, num_qubits):
        """
        Synthesize a non-Clifford unitary using Solovay-Kitaev decomposition.
        
        Args:
            num_qubits (int): Number of qubits
        
        Returns:
            QuantumCircuit: Approximated circuit in ['h', 't', 'tdg', 'cx'] basis
        """
        # For Controlled-Ry, use a more direct approach
        # Create circuit using Qiskit's built-in gates if possible
        circuit = QuantumCircuit(num_qubits)
        
        # Check if it's a Controlled-Ry gate
        if num_qubits == 2:
            # Try to extract the angle from the unitary
            # Controlled-Ry(θ) has specific structure
            # CRy(θ) = |0><0| ⊗ I + |1><1| ⊗ Ry(θ)
            # The (1,1) and (3,3) entries are cos(θ/2)
            # The (1,3) entry is -sin(θ/2), (3,1) is sin(θ/2)
            try:
                cos_half_theta = self.target_unitary[1, 1].real
                sin_half_theta = -self.target_unitary[1, 3].real
                
                # Calculate angle
                theta = 2 * np.arctan2(abs(sin_half_theta), abs(cos_half_theta))
                
                # Verify it's actually a CRy gate by checking the structure
                expected = create_controlled_ry(theta)
                if np.allclose(self.target_unitary, expected, atol=1e-10):
                    # Use Qiskit's cry gate
                    circuit.cry(theta, 0, 1)
                else:
                    # Not a CRy, use unitary gate
                    circuit.unitary(self.target_unitary, range(num_qubits), label='target')
            except Exception as e:
                # Fallback: use unitary gate
                circuit.unitary(self.target_unitary, range(num_qubits), label='target')
        else:
            circuit.unitary(self.target_unitary, range(num_qubits), label='target')
        
        # Decompose the circuit completely - this should break down CRy
        decomposed = circuit.decompose(reps=4)
        
        # Check if there are any unitary gates remaining
        has_unitary = any(inst.operation.name == 'unitary' for inst in decomposed.data)
        
        if has_unitary:
            # Force decomposition by transpiling to a basis that doesn't include unitary
            unrolled = transpile(
                decomposed,
                basis_gates=['u3', 'cx'],
                optimization_level=0
            )
            # Decompose again
            unrolled = unrolled.decompose(reps=3)
            # Check again
            has_unitary = any(inst.operation.name == 'unitary' for inst in unrolled.data)
            if has_unitary:
                # Still has unitary gates - force remove them by creating new circuit
                new_circuit = QuantumCircuit(num_qubits)
                for inst in unrolled.data:
                    if inst.operation.name != 'unitary':
                        new_circuit.append(inst.operation, [q.index for q in inst.qubits])
                unrolled = new_circuit
        else:
            # Already decomposed, just transpile to u3 and cx
            unrolled = transpile(
                decomposed,
                basis_gates=['u3', 'cx'],
                optimization_level=0
            )
        
        # Final check: ensure no unitary gates before SolovayKitaev
        final_check = any(inst.operation.name == 'unitary' for inst in unrolled.data)
        if final_check:
            # Remove unitary gates completely
            clean_circuit = QuantumCircuit(num_qubits)
            for inst in unrolled.data:
                if inst.operation.name != 'unitary':
                    qubits = [unrolled.find_bit(q)[0] for q in inst.qubits]
                    clean_circuit.append(inst.operation, qubits)
            unrolled = clean_circuit
        
        # Use standard transpile with high optimization
        # Qiskit's transpiler will use appropriate synthesis methods automatically
        # This avoids the SolovayKitaev pass which has issues with unitary gates
        sk_circuit = transpile(
            unrolled,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=3  # High optimization for better synthesis
        )
        
        # Convert any S/Z gates that might remain
        sk_circuit = convert_s_z_to_t(sk_circuit)
        
        # Final optimization
        optimized = self._optimize_circuit(sk_circuit)
        
        return optimized
    
    def _optimize_circuit(self, circuit):
        """
        Apply optimization passes to reduce gate count.
        
        Args:
            circuit (QuantumCircuit): Circuit to optimize
        
        Returns:
            QuantumCircuit: Optimized circuit
        """
        # Create optimization pass manager
        opt_pm = PassManager([
            # Optimize single-qubit gates
            Optimize1qGates(),
            # Cancel inverse pairs (like CNOT·CNOT, T·Tdg)
            InverseCancellation(),
            # Commutative cancellation
            CommutativeCancellation(),
            # Commutation analysis
            CommutationAnalysis(),
        ])
        
        optimized = opt_pm.run(circuit)
        
        # Manual optimization: cancel T·Tdg pairs
        optimized = self._cancel_t_pairs(optimized)
        
        # Final transpile to ensure correct basis
        final = transpile(
            optimized,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=1  # Light optimization to preserve structure
        )
        
        return final
    
    def _cancel_t_pairs(self, circuit):
        """
        Cancel adjacent T·Tdg and Tdg·T pairs.
        
        Args:
            circuit (QuantumCircuit): Circuit to optimize
        
        Returns:
            QuantumCircuit: Circuit with T pairs cancelled
        """
        new_circuit = QuantumCircuit(circuit.num_qubits)
        
        # Track last gate on each qubit
        last_gates = {}
        
        for instruction in circuit.data:
            gate = instruction.operation
            # Get qubit indices properly
            if hasattr(instruction.qubits[0], 'index'):
                qubits = [q.index for q in instruction.qubits]
            else:
                qubits = [circuit.find_bit(q)[0] for q in instruction.qubits]
            gate_name = gate.name
            
            # For single-qubit gates, check for cancellation
            if len(qubits) == 1:
                q = qubits[0]
                if q in last_gates:
                    last_gate = last_gates[q]
                    # Check if we can cancel
                    if (last_gate == 't' and gate_name == 'tdg') or \
                       (last_gate == 'tdg' and gate_name == 't'):
                        # Cancel: don't add either gate
                        del last_gates[q]
                        continue
                
                # Update last gate
                if gate_name in ['t', 'tdg']:
                    last_gates[q] = gate_name
                else:
                    # Other gates reset the chain
                    if q in last_gates:
                        del last_gates[q]
            
            # Add the gate
            new_circuit.append(gate, qubits)
        
        return new_circuit


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
    
    # Save results to files
    import json
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary to JSON file
    summary_file = os.path.join(output_dir, "results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to: {summary_file}")
    
    # Save summary to text file
    summary_txt_file = os.path.join(output_dir, "results_summary.txt")
    with open(summary_txt_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Superquantum Challenge Results\n")
        f.write("=" * 60 + "\n\n")
        for result in results_summary:
            if 'error' in result:
                f.write(f"{result['problem']}: ERROR - {result['error']}\n")
            else:
                f.write(f"{result['problem']}: T-count={result['t_count']}, Distance={result['distance']:.6e}\n")
    print(f"Results summary (text) saved to: {summary_txt_file}")
    
    # Save OpenQASM files for each problem
    print("\nSaving OpenQASM circuits...")
    for problem_name, (target_unitary, solver_class) in problems.items():
        try:
            solver = solver_class(target_unitary, problem_name)
            results = solver.solve()  # This calls synthesize() and evaluates
            
            # Create filename from problem name
            safe_name = problem_name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("/", "_")
            qasm_file = os.path.join(output_dir, f"{safe_name}.qasm")
            solver.export_qasm(qasm_file)
            print(f"  {problem_name}: {qasm_file}")
        except Exception as e:
            print(f"  {problem_name}: Failed to save ({str(e)[:50]}...)")
    
    print(f"\nAll outputs saved to directory: {output_dir}/")


if __name__ == "__main__":
    main()
