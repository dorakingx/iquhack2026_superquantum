# Superquantum Challenge - Quantum Circuit Compilation

A Python framework for solving the Superquantum challenge at iQuHACK 2026. This project implements a class-based synthesis pipeline for compiling target unitary matrices into Clifford+T quantum circuits using Qiskit.

## Overview

The Superquantum challenge focuses on quantum circuit compilation with the following objectives:
- Compile specific unitary matrices into quantum circuits
- Use only the gate set: `{H, T, T_dag, CNOT}`
- Minimize **T-count** (number of T and T_dag gates)
- Minimize **Operator Norm Distance** (approximation error)
- Output circuits in OpenQASM 2.0 format

## Features

- **Operator Norm Distance Calculation**: Computes the distance between target unitaries and synthesized circuits with global phase optimization
- **T-count Measurement**: Counts T and T_dag gates in quantum circuits
- **OpenQASM Export**: Exports circuits to OpenQASM 2.0 format
- **Class-based Architecture**: Flexible solver framework supporting different problem types (unitary synthesis, Hamiltonian decomposition)
- **Qiskit Integration**: Uses Qiskit's transpilation and optimization tools

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dorakingx/iquhack2026_superquantum.git
cd iquhack2026_superquantum
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Solver

To solve the implemented problems (Problem 1 and Problem 2):

```bash
python solve_challenge.py
```

This will:
- Synthesize circuits for each problem
- Calculate T-count and operator norm distance
- Display results in a summary format

### Example Output

```
============================================================
Superquantum Challenge Solver
============================================================

Solving Problem 1: Controlled-Y...
------------------------------------------------------------
  T-count: 64518
  Operator Norm Distance: 1.706408e-05

Solving Problem 2: Controlled-Ry(π/7)...
------------------------------------------------------------
  T-count: 101025
  Operator Norm Distance: 1.787285e-05

============================================================
Summary
============================================================
Problem 1: Controlled-Y: T-count=64518, Distance=1.706408e-05
Problem 2: Controlled-Ry(π/7): T-count=101025, Distance=1.787285e-05
```

### Using the Utility Functions

```python
from utils import operator_norm_distance, count_t_gates, export_to_openqasm
from qiskit import QuantumCircuit
import numpy as np

# Create a target unitary
target_unitary = np.array([[1, 0], [0, 1j]], dtype=complex)

# Create a circuit
circuit = QuantumCircuit(1)
circuit.t(0)

# Calculate operator norm distance
distance = operator_norm_distance(target_unitary, circuit)

# Count T gates
t_count = count_t_gates(circuit)

# Export to OpenQASM
qasm_string = export_to_openqasm(circuit)
```

### Using the Solver Classes

```python
from solve_challenge import UnitarySolver, create_controlled_y

# Create a problem solver
target_unitary = create_controlled_y()
solver = UnitarySolver(target_unitary, "Problem 1: Controlled-Y")

# Solve and evaluate
results = solver.solve()
print(f"T-count: {results['t_count']}")
print(f"Distance: {results['distance']}")

# Export circuit to OpenQASM file
solver.export_qasm("problem1.qasm")
```

## Project Structure

```
superquantum/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── utils.py                    # Utility functions for distance calculation, T-count, and OpenQASM export
├── solve_challenge.py          # Main solver script with class-based architecture
├── superquantum_challenge.pdf  # Challenge documentation
└── .gitignore                  # Git ignore file
```

## Key Components

### `utils.py`

Contains three core utility functions:

1. **`operator_norm_distance(target_unitary, circuit, optimize_phase=True)`**
   - Calculates operator norm distance with global phase optimization
   - Formula: $d(U, \tilde{U}) = \min_{\phi} || U - e^{i\phi} \tilde{U} ||_{op}$

2. **`count_t_gates(circuit)`**
   - Counts total T and T_dag gates in a circuit

3. **`export_to_openqasm(circuit, filename=None)`**
   - Exports circuit to OpenQASM 2.0 format

### `solve_challenge.py`

Implements a class-based synthesis pipeline:

- **`ChallengeSolver`**: Base class with common functionality
  - `check_t_count()`: Count T gates
  - `calculate_distance()`: Calculate operator norm distance
  - `export_qasm()`: Export to OpenQASM
  - `solve()`: Main workflow

- **`UnitarySolver`**: For direct unitary matrix synthesis (Problems 1, 2)
  - Uses Qiskit's `unitary()` and `transpile()` functions
  - Transpiles to basis `['h', 't', 'tdg', 'cx']`

- **`HamiltonianSolver`**: For Hamiltonian exponential synthesis (Problem 3+, stub)
  - Will implement Trotterization/KAK decomposition

## Implemented Problems

- **Problem 1: Controlled-Y**
  - Target: Controlled-Y gate (2-qubit unitary)
  - Solver: `UnitarySolver`

- **Problem 2: Controlled-Ry(π/7)**
  - Target: Controlled-Ry gate with angle π/7
  - Solver: `UnitarySolver`

## Dependencies

- `numpy>=1.20.0` - Matrix operations
- `qiskit>=0.45.0` - Quantum circuit synthesis and transpilation
- `scipy>=1.7.0` - Optimization and linear algebra

## License

This project is part of the iQuHACK 2026 Superquantum challenge.

## Contributing

This is a challenge submission repository. For questions or issues related to the challenge, please refer to the challenge documentation (`superquantum_challenge.pdf`).

## Acknowledgments

- iQuHACK 2026 organizers
- Qiskit development team
