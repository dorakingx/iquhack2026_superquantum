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

from utils import operator_norm_distance, count_t_gates, export_to_openqasm, map_rz_to_clifford_t


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
    
    def __init__(self, target_unitary, problem_name, recursion_degree=2):
        """
        Initialize UnitarySolver.
        
        Args:
            target_unitary (np.ndarray): Target unitary matrix
            problem_name (str): Name/description of the problem
            recursion_degree (int): Recursion degree for SolovayKitaev (default 2)
        """
        super().__init__(target_unitary, problem_name)
        self.recursion_degree = recursion_degree
    
    def synthesize(self):
        """
        Synthesize a unitary matrix into a Clifford+T circuit.
        
        Uses smart decomposition:
        1. Check if unitary matches QFT → use exact QFT synthesis
        2. Check if unitary is Clifford → use exact decomposition
        3. Otherwise → use Solovay-Kitaev approximation
        """
        # Determine number of qubits from unitary dimension
        dim = self.target_unitary.shape[0]
        num_qubits = int(np.log2(dim))
        
        if 2**num_qubits != dim:
            raise ValueError(f"Unitary dimension {dim} is not a power of 2")
        
        # Check if this is Problem 9 (Structured Unitary 2) - use QFT synthesis
        if "Structured Unitary 2" in self.problem_name or "Problem 9" in self.problem_name:
            # Use exact QFT synthesis
            self.circuit = self._synthesize_qft_exact(num_qubits)
            return
        
        # Check if unitary matches QFT (up to global phase) - fallback check
        from qiskit.circuit.library import QFT
        qft_circuit = QFT(num_qubits)
        qft_unitary = Operator(qft_circuit).data
        
        # Compare with target (accounting for global phase)
        # Check if U_target ≈ e^(iφ) * U_QFT
        # Use trace to find optimal phase
        trace_product = np.trace(np.conj(self.target_unitary).T @ qft_unitary)
        if np.abs(trace_product) > 1e-10:
            optimal_phase = np.angle(trace_product)
            qft_unitary_phased = np.exp(1j * optimal_phase) * qft_unitary
            if np.allclose(self.target_unitary, qft_unitary_phased, atol=1e-10):
                # Use exact QFT synthesis
                self.circuit = self._synthesize_qft_exact(num_qubits)
                return
        
        # Check if unitary is Clifford
        is_clifford = is_clifford_unitary(self.target_unitary)
        
        if is_clifford:
            # Use exact Clifford decomposition
            self.circuit = self._synthesize_clifford_exact(num_qubits)
        else:
            # Use Solovay-Kitaev synthesis
            self.circuit = self._synthesize_solovay_kitaev(num_qubits)
    
    def _synthesize_qft_exact(self, num_qubits):
        """
        Synthesize QFT using exact Qiskit QFT circuit.
        
        Args:
            num_qubits (int): Number of qubits
        
        Returns:
            QuantumCircuit: Optimized circuit in ['h', 't', 'tdg', 'cx'] basis
        """
        from qiskit.circuit.library import QFT
        
        # Create QFT circuit
        qft_circuit = QFT(num_qubits)
        
        # Decompose and transpile to target basis
        # QFT contains CP gates which transpile well to T gates
        decomposed = qft_circuit.decompose(reps=2)
        
        # Transpile to target basis
        transpiled = transpile(
            decomposed,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=2
        )
        
        # Convert S/Z gates to T gates
        circuit_with_t = convert_s_z_to_t(transpiled)
        
        # Apply optimization
        optimized = self._optimize_circuit(circuit_with_t)
        
        return optimized
    
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
        
        # Apply Solovay-Kitaev explicitly
        # Initialize SolovayKitaev with recursion_degree (default 2, can be increased for better approximation)
        skd = SolovayKitaev(recursion_degree=self.recursion_degree)
        
        # Create PassManager with SolovayKitaev
        # First ensure all gates are in ['u3', 'cx'] basis
        pm = PassManager([
            # Unroll any remaining custom definitions to u3 and cx
            UnrollCustomDefinitions(standard_gates, ['u3', 'cx']),
            # Explicitly apply SolovayKitaev to approximate u3 gates
            skd,
        ])
        
        # Run the pass manager to apply SolovayKitaev
        try:
            sk_circuit = pm.run(unrolled)
        except Exception as e:
            # Fallback: if SolovayKitaev fails, use standard transpile
            print(f"  Warning: SolovayKitaev pass failed ({str(e)[:50]}...), using standard transpile")
            sk_circuit = transpile(
                unrolled,
                basis_gates=['h', 't', 'tdg', 'cx'],
                optimization_level=2
            )
        
        # Translate to target basis ['h', 't', 'tdg', 'cx']
        sk_circuit = transpile(
            sk_circuit,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=1  # Light optimization to preserve SolovayKitaev structure
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
        
        # Convert S/Z gates back to T gates (Optimize1qGates might merge t,t -> s)
        optimized = convert_s_z_to_t(optimized)
        
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


class StatePrepSolver(ChallengeSolver):
    """
    Solver for state preparation problems.
    
    Used for Problem 7 where we need to design a unitary that maps |00⟩ to a target statevector.
    Uses Initialize to extract the unitary, then delegates to UnitarySolver logic.
    """
    
    def __init__(self, seed, problem_name):
        """
        Initialize StatePrepSolver with a seed for random statevector generation.
        
        Args:
            seed (int): Random seed for generating target statevector
            problem_name (str): Name/description of the problem
        """
        from qiskit.quantum_info import random_statevector
        
        # Generate target statevector
        self.target_statevector = random_statevector(4, seed=seed)
        
        # Construct unitary that maps |00⟩ to target statevector
        # We need a unitary U such that U|00⟩ = |ψ⟩
        # Use QR decomposition: start with |ψ⟩ as first column, complete basis
        
        psi = self.target_statevector.data
        
        # Create a matrix with |ψ⟩ as first column
        basis_matrix = np.zeros((4, 4), dtype=complex)
        basis_matrix[:, 0] = psi
        
        # Complete the basis with orthogonal vectors
        # Use Gram-Schmidt: start with standard basis vectors
        standard_basis = np.eye(4, dtype=complex)
        
        # Project out |ψ⟩ component from each standard basis vector
        for i in range(1, 4):
            v = standard_basis[:, i]
            # Project out components already in basis
            for j in range(i):
                v = v - np.vdot(basis_matrix[:, j], v) * basis_matrix[:, j]
            # Normalize
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                basis_matrix[:, i] = v / norm
            else:
                # If vector is too small, use next standard basis vector
                # and project out |ψ⟩
                v = standard_basis[:, (i + 1) % 4]
                v = v - np.vdot(psi, v) * psi
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    basis_matrix[:, i] = v / norm
                else:
                    # Fallback: use random vector
                    v = np.random.randn(4) + 1j * np.random.randn(4)
                    for j in range(i):
                        v = v - np.vdot(basis_matrix[:, j], v) * basis_matrix[:, j]
                    norm = np.linalg.norm(v)
                    if norm > 1e-10:
                        basis_matrix[:, i] = v / norm
        
        # Ensure orthonormality using QR decomposition
        Q, R = np.linalg.qr(basis_matrix)
        
        # Ensure first column matches |ψ⟩ (up to global phase)
        # Adjust phase
        phase_diff = np.angle(np.vdot(psi, Q[:, 0]))
        Q[:, 0] = Q[:, 0] * np.exp(-1j * phase_diff)
        
        # Verify first column matches
        if not np.allclose(Q[:, 0], psi, atol=1e-10):
            # If not matching, try adjusting the entire matrix
            # Find the column closest to |ψ⟩
            overlaps = [abs(np.vdot(psi, Q[:, i])) for i in range(4)]
            best_col = np.argmax(overlaps)
            if best_col != 0:
                # Swap columns
                Q[:, [0, best_col]] = Q[:, [best_col, 0]]
        
        target_unitary = Q
        
        # Initialize base class with extracted unitary
        super().__init__(target_unitary, problem_name)
        
        # Store seed for reference
        self.seed = seed
    
    def synthesize(self):
        """
        Synthesize the state preparation unitary into a Clifford+T circuit.
        
        Delegates to UnitarySolver synthesis logic.
        """
    def _synthesize_qft_exact(self, num_qubits):
        """
        Synthesize QFT using exact Qiskit QFT circuit.
        
        Args:
            num_qubits (int): Number of qubits
        
        Returns:
            QuantumCircuit: Optimized circuit in ['h', 't', 'tdg', 'cx'] basis
        """
        from qiskit.circuit.library import QFT
        
        # Create QFT circuit
        qft_circuit = QFT(num_qubits)
        
        # Decompose and transpile to target basis
        # QFT contains CP gates which transpile well to T gates
        decomposed = qft_circuit.decompose(reps=2)
        
        # Transpile to target basis
        transpiled = transpile(
            decomposed,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=2
        )
        
        # Convert S/Z gates to T gates
        circuit_with_t = convert_s_z_to_t(transpiled)
        
        # Apply optimization
        optimized = self._optimize_circuit(circuit_with_t)
        
        return optimized
    
    def _synthesize_clifford_exact(self, num_qubits):
        """Synthesize Clifford unitary using exact decomposition."""
        circuit = decompose_clifford_exact(self.target_unitary, num_qubits)
        optimized = self._optimize_circuit(circuit)
        return optimized
    
    def _synthesize_solovay_kitaev(self, num_qubits):
        """Synthesize non-Clifford unitary using Solovay-Kitaev decomposition."""
        # Create circuit with the unitary
        circuit = QuantumCircuit(num_qubits)
        circuit.unitary(self.target_unitary, range(num_qubits), label='target')
        
        # Decompose the circuit completely
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
                # Still has unitary gates - force remove them
                new_circuit = QuantumCircuit(num_qubits)
                for inst in unrolled.data:
                    if inst.operation.name != 'unitary':
                        qubits = [unrolled.find_bit(q)[0] for q in inst.qubits]
                        new_circuit.append(inst.operation, qubits)
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
        
        # Apply Solovay-Kitaev explicitly
        skd = SolovayKitaev(recursion_degree=2)
        
        # Create PassManager with SolovayKitaev
        pm = PassManager([
            UnrollCustomDefinitions(standard_gates, ['u3', 'cx']),
            skd,
        ])
        
        # Run the pass manager to apply SolovayKitaev
        try:
            sk_circuit = pm.run(unrolled)
        except Exception as e:
            # Fallback: if SolovayKitaev fails, use standard transpile
            print(f"  Warning: SolovayKitaev pass failed ({str(e)[:50]}...), using standard transpile")
            sk_circuit = transpile(
                unrolled,
                basis_gates=['h', 't', 'tdg', 'cx'],
                optimization_level=2
            )
        
        # Translate to target basis ['h', 't', 'tdg', 'cx']
        sk_circuit = transpile(
            sk_circuit,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=1
        )
        
        # Convert any S/Z gates that might remain
        sk_circuit = convert_s_z_to_t(sk_circuit)
        
        # Final optimization
        optimized = self._optimize_circuit(sk_circuit)
        
        return optimized
    
    def _optimize_circuit(self, circuit):
        """Apply optimization passes to reduce gate count."""
        opt_pm = PassManager([
            Optimize1qGates(),
            InverseCancellation(),
            CommutativeCancellation(),
            CommutationAnalysis(),
        ])
        
        optimized = opt_pm.run(circuit)
        
        # Manual optimization: cancel T·Tdg pairs
        optimized = self._cancel_t_pairs(optimized)
        
        # Final transpile to ensure correct basis
        final = transpile(
            optimized,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=1
        )
        
        return final
    
    def _cancel_t_pairs(self, circuit):
        """Cancel adjacent T·Tdg and Tdg·T pairs."""
        new_circuit = QuantumCircuit(circuit.num_qubits)
        last_gates = {}
        
        for instruction in circuit.data:
            gate = instruction.operation
            if hasattr(instruction.qubits[0], 'index'):
                qubits = [q.index for q in instruction.qubits]
            else:
                qubits = [circuit.find_bit(q)[0] for q in instruction.qubits]
            gate_name = gate.name
            
            if len(qubits) == 1:
                q = qubits[0]
                if q in last_gates:
                    last_gate = last_gates[q]
                    if (last_gate == 't' and gate_name == 'tdg') or \
                       (last_gate == 'tdg' and gate_name == 't'):
                        del last_gates[q]
                        continue
                
                if gate_name in ['t', 'tdg']:
                    last_gates[q] = gate_name
                else:
                    if q in last_gates:
                        del last_gates[q]
            
            new_circuit.append(gate, qubits)
        
        return new_circuit


class DiagonalSolver(ChallengeSolver):
    """
    Solver for diagonal unitary synthesis with exact phase-to-gate mapping.
    
    Used for Problem 11 where we have a diagonal unitary with phases that are
    multiples of π/4. This allows exact synthesis without approximation.
    """
    
    def __init__(self, phases, problem_name):
        """
        Initialize DiagonalSolver with phase array.
        
        Args:
            phases (list): List of 16 phase values (for 4 qubits) in order 0000 to 1111
            problem_name (str): Name/description of the problem
        """
        # Create diagonal unitary: diag([exp(i*phi) for phi in phases])
        diagonal_elements = [np.exp(1j * phi) for phi in phases]
        target_unitary = np.diag(diagonal_elements)
        
        # Initialize base class
        super().__init__(target_unitary, problem_name)
        
        # Store phases for reference
        self.phases = phases
    
    def synthesize(self):
        """
        Synthesize diagonal unitary using exact decomposition.
        
        Uses qiskit.circuit.library.DiagonalGate to decompose, then replaces
        each Rz gate with exact T/S/Z mapping.
        """
        from qiskit.circuit.library import DiagonalGate
        
        # Determine number of qubits from phases length
        num_qubits = int(np.log2(len(self.phases)))
        
        # Convert phases to diagonal elements (complex numbers with abs=1)
        diagonal_elements = [np.exp(1j * phi) for phi in self.phases]
        
        # Create diagonal circuit using DiagonalGate
        diagonal_circuit = QuantumCircuit(num_qubits)
        diagonal_circuit.append(DiagonalGate(diagonal_elements), range(num_qubits))
        
        # Decompose the diagonal gate
        # This will break it down into CNOTs and Rz gates
        decomposed = diagonal_circuit.decompose(reps=3)
        
        # Create new circuit and replace Rz gates with exact T/S/Z gates
        new_circuit = QuantumCircuit(num_qubits)
        
        for instruction in decomposed.data:
            gate = instruction.operation
            # Get qubit indices
            if hasattr(instruction.qubits[0], 'index'):
                qubits = [q.index for q in instruction.qubits]
            else:
                qubits = [decomposed.find_bit(q)[0] for q in instruction.qubits]
            
            gate_name = gate.name
            
            if gate_name == 'rz':
                # Extract angle from Rz gate
                angle = float(gate.params[0])
                
                # Map to exact T/S/Z gates
                gate_sequence = map_rz_to_clifford_t(angle)
                
                # Apply gates in sequence
                for g in gate_sequence:
                    if g == 't':
                        new_circuit.t(qubits[0])
                    elif g == 'tdg':
                        new_circuit.tdg(qubits[0])
                    elif g == 's':
                        new_circuit.s(qubits[0])
                    elif g == 'sdg':
                        new_circuit.sdg(qubits[0])
                    elif g == 'z':
                        new_circuit.z(qubits[0])
            else:
                # Copy other gates as-is (CNOTs, etc.)
                new_circuit.append(gate, qubits)
        
        # Convert S/Z gates to T gates (S = T^2, Z = T^4)
        circuit_with_t = convert_s_z_to_t(new_circuit)
        
        # Apply optimization
        optimized = self._optimize_circuit(circuit_with_t)
        
        self.circuit = optimized
    
    def _optimize_circuit(self, circuit):
        """Apply optimization passes to reduce gate count."""
        opt_pm = PassManager([
            Optimize1qGates(),
            InverseCancellation(),
            CommutativeCancellation(),
            CommutationAnalysis(),
        ])
        
        optimized = opt_pm.run(circuit)
        
        # Convert S/Z gates back to T gates (Optimize1qGates might merge t,t -> s)
        optimized = convert_s_z_to_t(optimized)
        
        # Manual optimization: cancel T·Tdg pairs
        optimized = self._cancel_t_pairs(optimized)
        
        # Final transpile to ensure correct basis
        final = transpile(
            optimized,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=1
        )
        
        return final
    
    def _cancel_t_pairs(self, circuit):
        """Cancel adjacent T·Tdg and Tdg·T pairs."""
        new_circuit = QuantumCircuit(circuit.num_qubits)
        last_gates = {}
        
        for instruction in circuit.data:
            gate = instruction.operation
            if hasattr(instruction.qubits[0], 'index'):
                qubits = [q.index for q in instruction.qubits]
            else:
                qubits = [circuit.find_bit(q)[0] for q in instruction.qubits]
            gate_name = gate.name
            
            if len(qubits) == 1:
                q = qubits[0]
                if q in last_gates:
                    last_gate = last_gates[q]
                    if (last_gate == 't' and gate_name == 'tdg') or \
                       (last_gate == 'tdg' and gate_name == 't'):
                        del last_gates[q]
                        continue
                
                if gate_name in ['t', 'tdg']:
                    last_gates[q] = gate_name
                else:
                    if q in last_gates:
                        del last_gates[q]
            
            new_circuit.append(gate, qubits)
        
        return new_circuit


class HamiltonianSolver(ChallengeSolver):
    """
    Solver for Hamiltonian exponential synthesis.
    
    Used for Problem 3+ where we need to synthesize exp(i*H) for some Hamiltonian H.
    Uses Pauli gadget decomposition and Trotterization.
    """
    
    def __init__(self, hamiltonian_config, problem_name):
        """
        Initialize Hamiltonian solver with configuration.
        
        Args:
            hamiltonian_config (dict): Configuration with keys:
                - 'pauli_terms': list of Pauli strings (e.g., ['ZZ'], ['XX', 'YY'])
                - 'angle': float (rotation angle)
                - 'trotter_steps': int (for non-commuting terms, default 4)
            problem_name (str): Name/description of the problem
        """
        self.hamiltonian_config = hamiltonian_config
        self.problem_name = problem_name
        self.circuit = None
        
        # Extract configuration
        self.pauli_terms = hamiltonian_config.get('pauli_terms', [])
        self.angle = hamiltonian_config.get('angle', 0.0)
        self.trotter_steps = hamiltonian_config.get('trotter_steps', 4)
        self.num_qubits = hamiltonian_config.get('num_qubits', 2)
        
        # Compute target unitary for distance calculation
        self.target_unitary = self._compute_target_unitary()
    
    def _compute_target_unitary(self):
        """
        Compute the target unitary matrix exp(i * angle * H).
        
        Returns:
            np.ndarray: Target unitary matrix
        """
        from qiskit.quantum_info import SparsePauliOp
        
        # Build Hamiltonian from Pauli terms
        pauli_list = []
        coeffs = []
        for term in self.pauli_terms:
            pauli_list.append(term)
            coeffs.append(self.angle)
        
        # Create SparsePauliOp
        hamiltonian = SparsePauliOp(pauli_list, coeffs=coeffs)
        
        # Compute exp(i * H)
        from scipy.linalg import expm
        h_matrix = hamiltonian.to_matrix()
        unitary = expm(1j * h_matrix)
        
        return unitary
    
    def _apply_basis_mapping(self, circuit, pauli_string, qubits, inverse=False):
        """
        Apply Clifford gates to map Pauli basis to Z.
        
        Args:
            circuit (QuantumCircuit): Circuit to modify
            pauli_string (str): Pauli string like "ZZ", "XX", etc.
            qubits (list): List of qubit indices
            inverse (bool): If True, apply inverse mapping
        """
        if inverse:
            # Apply inverse mapping (reverse order)
            for i in range(len(pauli_string) - 1, -1, -1):
                pauli = pauli_string[i]
                q = qubits[i]
                if pauli == 'X':
                    circuit.h(q)
                elif pauli == 'Y':
                    circuit.h(q)
                    circuit.s(q)
        else:
            # Apply forward mapping
            for i, pauli in enumerate(pauli_string):
                q = qubits[i]
                if pauli == 'X':
                    circuit.h(q)
                elif pauli == 'Y':
                    circuit.sdg(q)
                    circuit.h(q)
                # Z and I require no mapping
    
    def _apply_cnot_ladder(self, circuit, qubits, reverse=False):
        """
        Apply CNOT ladder for parity computation.
        
        Args:
            circuit (QuantumCircuit): Circuit to modify
            qubits (list): List of qubit indices (only non-I qubits)
            reverse (bool): If True, apply inverse ladder
        """
        if len(qubits) < 2:
            return  # No ladder needed for single qubit
        
        if reverse:
            # Reverse CNOT ladder (uncompute)
            for i in range(len(qubits) - 2, -1, -1):
                circuit.cx(qubits[i], qubits[i + 1])
        else:
            # Forward CNOT ladder (compute parity)
            for i in range(len(qubits) - 1):
                circuit.cx(qubits[i], qubits[i + 1])
    
    def _synthesize_rz(self, angle):
        """
        Synthesize Rz(angle) gate exactly if angle is multiple of π/4, else approximate.
        
        For Pauli evolution, we apply Rz(2*angle), so we check if 2*angle is a multiple of π/4.
        If so, uses exact synthesis with T/S/Z gates. Otherwise, falls back to SolovayKitaev approximation.
        
        Args:
            angle (float): Rotation angle (for Rz(2*angle) in Pauli evolution)
        
        Returns:
            QuantumCircuit: Single-qubit circuit implementing Rz(angle) exactly or approximately
        """
        two_angle = 2 * angle
        # Check if 2*angle is multiple of π/4
        pi_over_4 = np.pi / 4
        remainder = (two_angle % pi_over_4)
        # Check if remainder is close to 0 (exact multiple)
        tolerance = 1e-10
        if remainder < tolerance or abs(remainder - pi_over_4) < tolerance:
            # Exact synthesis
            gate_sequence = map_rz_to_clifford_t(two_angle)
            circuit = QuantumCircuit(1)
            for gate_name in gate_sequence:
                if gate_name == 't':
                    circuit.t(0)
                elif gate_name == 'tdg':
                    circuit.tdg(0)
                elif gate_name == 's':
                    circuit.s(0)
                elif gate_name == 'sdg':
                    circuit.sdg(0)
                elif gate_name == 'z':
                    circuit.z(0)
            return circuit
        else:
            # Use SolovayKitaev approximation
            return self._approximate_rz_with_solovay_kitaev(angle)
    
    def _approximate_rz_with_solovay_kitaev(self, angle):
        """
        Approximate Rz(angle) gate using SolovayKitaev.
        
        Args:
            angle (float): Rotation angle
        
        Returns:
            QuantumCircuit: Clifford+T approximation of Rz(angle)
        """
        # Create single-qubit circuit with Rz
        rz_circuit = QuantumCircuit(1)
        rz_circuit.rz(angle, 0)
        
        # Decompose to u3
        decomposed = rz_circuit.decompose(reps=2)
        unrolled = transpile(
            decomposed,
            basis_gates=['u3'],
            optimization_level=0
        )
        
        # Apply SolovayKitaev
        skd = SolovayKitaev(recursion_degree=2)
        pm = PassManager([
            UnrollCustomDefinitions(standard_gates, ['u3']),
            skd,
        ])
        
        try:
            sk_circuit = pm.run(unrolled)
        except Exception:
            # Fallback: use standard transpile
            sk_circuit = transpile(
                unrolled,
                basis_gates=['h', 't', 'tdg', 'cx'],
                optimization_level=1
            )
        
        # Translate to target basis
        final = transpile(
            sk_circuit,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=0
        )
        
        return final
    
    def _synthesize_pauli_evolution(self, pauli_string, angle):
        """
        Synthesize exp(i * angle * P) where P is a Pauli string.
        
        Args:
            pauli_string (str): Pauli string like "ZZ", "XX", "ZI", etc.
            angle (float): Rotation angle
        
        Returns:
            QuantumCircuit: Circuit implementing exp(i * angle * P)
        """
        # Determine number of qubits from Pauli string
        num_qubits = len(pauli_string)
        circuit = QuantumCircuit(num_qubits)
        
        # Find qubits involved (non-I)
        active_qubits = [i for i, p in enumerate(pauli_string) if p != 'I']
        
        if len(active_qubits) == 0:
            # Identity - return empty circuit (or identity)
            return circuit
        
        # Step 1: Basis mapping (map to Z basis)
        self._apply_basis_mapping(circuit, pauli_string, list(range(num_qubits)), inverse=False)
        
        # Step 2: CNOT ladder to compute parity into last active qubit
        if len(active_qubits) > 1:
            self._apply_cnot_ladder(circuit, active_qubits, reverse=False)
        
        # Step 3: Apply Rz(2*angle) on last active qubit
        # This is the only non-Clifford gate (unless angle is special)
        # Use exact synthesis if 2*angle is multiple of π/4, else approximate
        last_qubit = active_qubits[-1]
        rz_approximation = self._synthesize_rz(angle)
        
        # Insert the Rz approximation circuit on the last qubit
        # The rz_approximation is a single-qubit circuit, so we map qubit 0 to last_qubit
        for instruction in rz_approximation.data:
            gate = instruction.operation
            # Get qubit indices from rz_approximation
            if hasattr(instruction.qubits[0], 'index'):
                rz_qubits = [q.index for q in instruction.qubits]
            else:
                rz_qubits = [rz_approximation.find_bit(q)[0] for q in instruction.qubits]
            
            # Map to target qubit (single-qubit gates only)
            if len(rz_qubits) == 1:
                circuit.append(gate, [last_qubit])
            else:
                # Multi-qubit gate - shouldn't happen, but handle gracefully
                circuit.append(gate, [last_qubit] * len(rz_qubits))
        
        # Step 4: Uncompute parity (reverse CNOT ladder)
        if len(active_qubits) > 1:
            self._apply_cnot_ladder(circuit, active_qubits, reverse=True)
        
        # Step 5: Inverse basis mapping
        self._apply_basis_mapping(circuit, pauli_string, list(range(num_qubits)), inverse=True)
        
        return circuit
    
    def synthesize(self):
        """
        Synthesize a Hamiltonian exponential into a Clifford+T circuit.
        
        Implements different strategies based on problem type:
        - Problem 3: Single Pauli term (ZZ)
        - Problem 4-5: Commuting Pauli terms (sequential synthesis)
        - Problem 6: Non-commuting terms (Trotterization)
        """
        # Check if terms commute
        # For simplicity, assume Problems 4-5 commute, Problem 6 doesn't
        terms = self.pauli_terms
        
        if len(terms) == 1:
            # Problem 3: Single term
            circuit = self._synthesize_pauli_evolution(terms[0], self.angle)
        elif self.trotter_steps == 1 or self._check_commuting(terms):
            # Problems 4-5: Commuting terms, synthesize sequentially
            circuit = QuantumCircuit(self.num_qubits)
            for term in terms:
                term_circuit = self._synthesize_pauli_evolution(term, self.angle)
                circuit.compose(term_circuit, inplace=True)
        else:
            # Problem 6: Non-commuting terms, use Trotterization
            dt = self.angle / self.trotter_steps
            circuit = QuantumCircuit(self.num_qubits)
            
            for _ in range(self.trotter_steps):
                for term in terms:
                    term_circuit = self._synthesize_pauli_evolution(term, dt)
                    circuit.compose(term_circuit, inplace=True)
        
        # Convert S/Z gates to T gates
        circuit = convert_s_z_to_t(circuit)
        
        # Final optimization using the same optimization passes as UnitarySolver
        optimized = self._optimize_circuit(circuit)
        
        self.circuit = optimized
    
    def _optimize_circuit(self, circuit):
        """
        Apply optimization passes to reduce gate count.
        Same as UnitarySolver._optimize_circuit.
        """
        opt_pm = PassManager([
            Optimize1qGates(),
            InverseCancellation(),
            CommutativeCancellation(),
            CommutationAnalysis(),
        ])
        
        optimized = opt_pm.run(circuit)
        
        # Convert S/Z gates back to T gates (Optimize1qGates might merge t,t -> s)
        optimized = convert_s_z_to_t(optimized)
        
        # Manual optimization: cancel T·Tdg pairs
        optimized = self._cancel_t_pairs(optimized)
        
        # Final transpile to ensure correct basis
        final = transpile(
            optimized,
            basis_gates=['h', 't', 'tdg', 'cx'],
            optimization_level=1
        )
        
        return final
    
    def _cancel_t_pairs(self, circuit):
        """
        Cancel adjacent T·Tdg and Tdg·T pairs.
        Same as UnitarySolver._cancel_t_pairs.
        """
        new_circuit = QuantumCircuit(circuit.num_qubits)
        last_gates = {}
        
        for instruction in circuit.data:
            gate = instruction.operation
            if hasattr(instruction.qubits[0], 'index'):
                qubits = [q.index for q in instruction.qubits]
            else:
                qubits = [circuit.find_bit(q)[0] for q in instruction.qubits]
            gate_name = gate.name
            
            if len(qubits) == 1:
                q = qubits[0]
                if q in last_gates:
                    last_gate = last_gates[q]
                    if (last_gate == 't' and gate_name == 'tdg') or \
                       (last_gate == 'tdg' and gate_name == 't'):
                        del last_gates[q]
                        continue
                
                if gate_name in ['t', 'tdg']:
                    last_gates[q] = gate_name
                else:
                    if q in last_gates:
                        del last_gates[q]
            
            new_circuit.append(gate, qubits)
        
        return new_circuit
    
    def _check_commuting(self, terms):
        """
        Check if Pauli terms commute.
        
        Simple check: XX, YY, ZZ all commute with each other.
        ZI and IZ commute with each other but not with XX.
        
        Args:
            terms (list): List of Pauli strings
        
        Returns:
            bool: True if all terms commute
        """
        # For Problems 4-5: XX, YY, ZZ all commute
        commuting_sets = [
            {'XX', 'YY', 'ZZ'},  # All commute
            {'ZI', 'IZ'},  # Commute with each other
        ]
        
        # Check if all terms are in the same commuting set
        for comm_set in commuting_sets:
            if all(term in comm_set for term in terms):
                return True
        
        # Check if terms are in different commuting sets that commute
        if len(terms) == 2:
            if terms[0] in {'ZI', 'IZ'} and terms[1] in {'ZI', 'IZ'}:
                return True
        
        # Default: assume non-commuting if not in known commuting sets
        return False


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


def create_hamiltonian_config_3():
    """
    Create Hamiltonian configuration for Problem 3: exp(i*π/7 * Z⊗Z)
    
    Returns:
        dict: Hamiltonian configuration
    """
    return {
        'pauli_terms': ['ZZ'],
        'angle': np.pi / 7,
        'num_qubits': 2,
        'trotter_steps': 1
    }


def create_hamiltonian_config_4():
    """
    Create Hamiltonian configuration for Problem 4: exp(i*theta * (XX + YY))
    Note: Need to check the actual angle from challenge documentation.
    For now, using pi/7 as placeholder.
    
    Returns:
        dict: Hamiltonian configuration
    """
    return {
        'pauli_terms': ['XX', 'YY'],
        'angle': np.pi / 7,  # Update with actual angle from challenge
        'num_qubits': 2,
        'trotter_steps': 1  # Commuting terms
    }


def create_hamiltonian_config_5():
    """
    Create Hamiltonian configuration for Problem 5: exp(i*π/4 * (XX + YY + ZZ))
    
    The angle is π/4 as specified in the challenge documentation (H_2).
    
    Returns:
        dict: Hamiltonian configuration
    """
    return {
        'pauli_terms': ['XX', 'YY', 'ZZ'],
        'angle': np.pi / 4,  # π/4 as specified in challenge documentation
        'num_qubits': 2,
        'trotter_steps': 1  # All terms commute
    }


def create_hamiltonian_config_6():
    """
    Create Hamiltonian configuration for Problem 6: exp(i*theta * (XX + ZI + IZ))
    Note: Need to check the actual angle from challenge documentation.
    Uses Trotterization since XX doesn't commute with ZI/IZ.
    
    Returns:
        dict: Hamiltonian configuration
    """
    return {
        'pauli_terms': ['XX', 'ZI', 'IZ'],
        'angle': np.pi / 7,  # Update with actual angle from challenge
        'num_qubits': 2,
        'trotter_steps': 4  # Trotterization for non-commuting terms
    }


def create_problem_8_unitary():
    """
    Create unitary matrix for Problem 8 (Structured Unitary 1).
    
    TODO: Update this function with the actual matrix from the challenge PDF.
    Currently returns a placeholder (identity matrix).
    
    Returns:
        np.ndarray: 4×4 complex matrix (placeholder)
    """
    # TODO: Replace np.eye(4) with the actual matrix from challenge PDF
    return np.eye(4, dtype=complex)


def create_problem_9_unitary():
    """
    Create unitary matrix for Problem 9 (Structured Unitary 2).
    
    Matrix from challenge documentation:
    0.5 * [[1,1,1,1], [1,i,-1,-i], [1,-1,1,-1], [1,-i,-1,i]]
    
    Returns:
        np.ndarray: 4×4 complex matrix
    """
    return 0.5 * np.array([
        [1, 1, 1, 1],
        [1, 1j, -1, -1j],
        [1, -1, 1, -1],
        [1, -1j, -1, 1j]
    ], dtype=complex)


def create_problem_10_unitary():
    """
    Create random unitary matrix for Problem 10.
    
    Uses qiskit.quantum_info.random_unitary with seed=42.
    
    Returns:
        np.ndarray: 4×4 complex matrix
    """
    from qiskit.quantum_info import random_unitary
    return random_unitary(4, seed=42).data


def main():
    """
    Main execution: solve Problems 1-11.
    """
    # Phases for Problem 11 (Diagonal Unitary) derived from iQuHACK 2026 PDF
    # Inputs 0000 to 1111 in order
    phases_11 = [
        0,              # 0000
        np.pi,          # 0001
        5 * np.pi / 4,  # 0010
        7 * np.pi / 4,  # 0011
        5 * np.pi / 4,  # 0100
        7 * np.pi / 4,  # 0101
        3 * np.pi / 2,  # 0110
        3 * np.pi / 2,  # 0111
        5 * np.pi / 4,  # 1000
        7 * np.pi / 4,  # 1001
        3 * np.pi / 2,  # 1010
        3 * np.pi / 2,  # 1011
        3 * np.pi / 2,  # 1100
        3 * np.pi / 2,  # 1101
        7 * np.pi / 4,  # 1110
        5 * np.pi / 4   # 1111
    ]
    
    # Define problems
    # Format: (problem_name, target_unitary/config, solver_class, extra_config)
    # For UnitarySolver: target_unitary is matrix, extra_config can be {'recursion_degree': N}
    # For HamiltonianSolver: target_unitary is None, extra_config is hamiltonian_config dict
    # For StatePrepSolver: target_unitary is None, extra_config is {'seed': N}
    # For DiagonalSolver: target_unitary is None, extra_config is {'phases': [...]}
    problems = [
        ("Problem 1: Controlled-Y", create_controlled_y(), UnitarySolver, None),
        ("Problem 2: Controlled-Ry(π/7)", create_controlled_ry(np.pi / 7), UnitarySolver, None),
        ("Problem 3: exp(i*π/7 * Z⊗Z)", None, HamiltonianSolver, create_hamiltonian_config_3()),
        ("Problem 4: exp(i*π/7 * (XX+YY))", None, HamiltonianSolver, create_hamiltonian_config_4()),
        ("Problem 5: exp(i*π/7 * (XX+YY+ZZ))", None, HamiltonianSolver, create_hamiltonian_config_5()),
        ("Problem 6: exp(i*π/7 * (XX+ZI+IZ))", None, HamiltonianSolver, create_hamiltonian_config_6()),
        ("Problem 7: State Preparation", None, StatePrepSolver, {'seed': 42}),
        ("Problem 8: Structured Unitary 1", create_problem_8_unitary(), UnitarySolver, None),
        ("Problem 9: Structured Unitary 2", create_problem_9_unitary(), UnitarySolver, None),
        ("Problem 10: Random Unitary", create_problem_10_unitary(), UnitarySolver, {'recursion_degree': 3}),
        ("Problem 11: Diagonal Unitary", None, DiagonalSolver, {'phases': phases_11}),
    ]
    
    print("=" * 60)
    print("Superquantum Challenge Solver")
    print("=" * 60)
    print()
    
    results_summary = []
    
    for problem_name, target_unitary, solver_class, extra_config in problems:
        print(f"Solving {problem_name}...")
        print("-" * 60)
        
        try:
            # Create solver based on type
            if solver_class == HamiltonianSolver:
                solver = solver_class(extra_config, problem_name)
            elif solver_class == StatePrepSolver:
                solver = solver_class(extra_config['seed'], problem_name)
            elif solver_class == DiagonalSolver:
                solver = solver_class(extra_config['phases'], problem_name)
            else:
                # UnitarySolver
                recursion_degree = extra_config.get('recursion_degree', 2) if extra_config else 2
                solver = solver_class(target_unitary, problem_name, recursion_degree=recursion_degree)
            
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
            import traceback
            traceback.print_exc()
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
    for problem_name, target_unitary, solver_class, extra_config in problems:
        try:
            # Create solver based on type
            if solver_class == HamiltonianSolver:
                solver = solver_class(extra_config, problem_name)
            elif solver_class == StatePrepSolver:
                solver = solver_class(extra_config['seed'], problem_name)
            elif solver_class == DiagonalSolver:
                solver = solver_class(extra_config['phases'], problem_name)
            else:
                # UnitarySolver
                recursion_degree = extra_config.get('recursion_degree', 2) if extra_config else 2
                solver = solver_class(target_unitary, problem_name, recursion_degree=recursion_degree)
            
            # Ensure circuit is synthesized
            if solver.circuit is None:
                solver.synthesize()
            
            # Create filename from problem name
            safe_name = problem_name.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace("/", "_").replace("*", "").replace("+", "_")
            qasm_file = os.path.join(output_dir, f"{safe_name}.qasm")
            solver.export_qasm(qasm_file)
            print(f"  {problem_name}: {qasm_file}")
        except Exception as e:
            print(f"  {problem_name}: Failed to save ({str(e)[:50]}...)")
    
    print(f"\nAll outputs saved to directory: {output_dir}/")


if __name__ == "__main__":
    main()
