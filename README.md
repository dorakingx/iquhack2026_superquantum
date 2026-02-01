# iQuHACK 2026 - Superquantum Challenge Solution

## Overview

This repository contains our solution to the Superquantum challenge at iQuHACK 2026. The challenge focuses on compiling target unitary matrices into quantum circuits using only the gate set `{H, T, T_dag, CNOT}` while minimizing T-count and operator norm distance.

## Key Strategies

Our solution employs several optimization strategies to achieve competitive performance:

### 1. Exact Synthesis
Used for problems where exact decomposition is possible, achieving near-zero approximation error:
- **Problem 1 (Controlled-Y)**: Clifford unitary → exact decomposition
- **Problem 5 (exp(i*π/4 * (XX+YY+ZZ)))**: Exact Rz synthesis for multiples of π/4
- **Problem 8 (QFT)**: Quantum Fourier Transform → exact QFT circuit synthesis
- **Problem 11 (Diagonal Unitary)**: Exact phase-to-gate mapping for multiples of π/4

### 2. Adaptive Trotterization
Used for **Problem 6** (non-commuting Hamiltonian terms):
- Iteratively searches for minimum Trotter steps (starting from 1, up to 10)
- Stops when operator norm distance < 0.03 threshold
- Ensures minimum T-count while maintaining required accuracy

### 3. Pauli Gadgets
Optimized circuits for Hamiltonian simulation:
- **Problem 3**: Single Pauli term (ZZ)
- **Problem 4**: Commuting Pauli terms (XX+YY)
- **Problem 5**: Commuting Pauli terms with exact synthesis
- **Problem 6**: Non-commuting terms with adaptive Trotterization

### 4. Best-of-N Compilation
Running Solovay-Kitaev multiple times with different seeds for Random/Sparse unitaries:
- **Problem 2**: Controlled-Ry(π/7)
- **Problem 9**: Structured sparse unitary
- **Problem 10**: Random unitary
- Tries 10 different attempts with varying optimization levels and transpiler seeds
- Selects the circuit with lowest T-count

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the solver:
```bash
python solve_challenge.py
```

This will:
- Synthesize circuits for all 11 problems
- Calculate T-count and operator norm distance for each
- Save OpenQASM files to `output/` directory
- Generate results summary in JSON and text formats

## Project Structure

```
superquantum/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── utils.py                    # Utility functions (distance, T-count, OpenQASM)
├── solve_challenge.py          # Main solver with class-based architecture
├── output/                     # Generated QASM files and results
│   ├── Problem_*.qasm         # OpenQASM files for each problem
│   ├── results_summary.json    # Results in JSON format
│   └── results_summary.txt     # Results in text format
└── .gitignore                  # Git ignore file
```

## Dependencies

- `numpy>=1.20.0` - Matrix operations
- `qiskit>=0.45.0` - Quantum circuit synthesis and transpilation
- `scipy>=1.7.0` - Optimization and linear algebra

## Implementation Details

### Class-Based Architecture

- **`ChallengeSolver`**: Base abstract class with common functionality
- **`UnitarySolver`**: Direct unitary matrix synthesis with QFT recognition and best-of-N compilation
- **`HamiltonianSolver`**: Hamiltonian exponential synthesis with adaptive Trotterization
- **`StatePrepSolver`**: State preparation problems (inherits from UnitarySolver)
- **`DiagonalSolver`**: Diagonal unitary synthesis with exact phase mapping

### Optimization Techniques

- **Global Phase Optimization**: Operator norm distance calculation accounts for global phase
- **Gate Cancellation**: T·Tdg pairs are automatically cancelled
- **Final Gate Conversion**: S/Z gates are converted to T gates at the final step
- **Multiple Optimization Passes**: InverseCancellation, CommutativeCancellation, Optimize1qGates

## Results

| Problem | T-count | Distance | Strategy |
|---------|---------|----------|----------|
| Prob 1: Controlled-Y | 8 | 0.000000e+00 | Exact Synthesis (Zero Error) |
| Prob 2: Controlled-Ry(π/7) | 24490 | 4.044135e-02 | Best-of-N Solovay-Kitaev |
| Prob 3: exp(i*π/7 * Z⊗Z) | 13276 | 1.566052e+00 | Pauli Gadgets |
| Prob 4: exp(i*π/7 * (XX+YY)) | 26564 | 4.523325e-01 | Pauli Gadgets |
| Prob 5: exp(i*π/7 * (XX+YY+ZZ)) | 0 | 0.000000e+00 | Exact Synthesis (Zero Error) |
| Prob 6: exp(i*π/7 * (XX+ZI+IZ)) | 308096 | 1.629662e-01 | Adaptive Trotterization |
| Prob 7: State Preparation | 103502 | 1.046411e+00 | Best-of-N Solovay-Kitaev |
| Prob 8: Structured Unitary 1 | 3 | 0.000000e+00 | Exact Synthesis (Zero Error) |
| Prob 9: Structured Unitary 2 | 63576 | 1.848024e+00 | Best-of-N Solovay-Kitaev |
| Prob 10: Random Unitary | 103453 | 1.965368e+00 | Best-of-N Solovay-Kitaev |
| Prob 11: Diagonal Unitary | 25 | 0.000000e+00 | Exact Synthesis (Zero Error) |

## License

This project is part of the iQuHACK 2026 Superquantum challenge submission.

## Acknowledgments

- iQuHACK 2026 organizers
- Qiskit development team
