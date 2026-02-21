
## Overview

Benchmarking framework for evaluating dependency-aware scheduling algorithms for parallel nonlinear system solving on multicore architectures.

**Author:** Adel MOUSSAOUI et al.  
**Paper:** Performance Analysis of Dependency Aware Scheduling Algorithms for Parallel Nonlinear System Solving on Multicore Architectures

## Features

- **10 Scheduling Algorithms**: HEFT, ETF, CP-Priority, CPOP, Level-by-Level, Largest-First, Smallest-First, Most-Successors-First, FIFO, Random
- **Multiple DAG Structures**: Balanced, Serial, Mixed, Sparse, Dense, Wide-Shallow, Narrow-Deep
- **Comprehensive Metrics**: Makespan, CPU utilization, speedup, parallel efficiency, scheduling overhead
- **Statistical Analysis**: Multiple trials with mean, standard deviation, min-max ranges
- **Configurable Tests**: No decomposition baseline, sequential execution, multi-core parallel execution
- **C Constant Calibration**: Empirical estimation of computational constant for fsolve complexity modeling

## Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- NetworkX
- psutil (for hardware detection)
- matplotlib (for visualization)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/paralleleqsolver.git
cd paralleleqsolver

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run benchmark with default settings (structure comparison)
python MulticoreSeqScheduler.py

# Results will be written to:
# - comparison_tables.txt (performance metrics)
# - precedence_graphs.txt (graph statistics)

# Calibrate computational constant C for fsolve
python C_estimation.py
```

## Usage

### Basic Configuration

Edit the main block in `MulticoreSeqScheduler.py`:

```python
# Select test scenarios
run_mode = 'structure_comparison'  # Options: 'all', 'quick', 'balanced_only', 
                                    # 'structure_comparison', 'density_comparison',
                                    # 'shape_comparison', or list of scenario names

# Configure tests
test_no_decomposition = False  # Baseline: monolithic system
test_sequential = True         # Sequential: 1 core
test_parallel_cores = [2, 4, 8]  # Parallel: 2, 4, 8 cores

# Number of trials for statistical significance
num_graphs = 10
```

### Available Scenarios

- `small_balanced`: 10 nodes, 4000 total size
- `medium_balanced`: 20 nodes, 20000 total size
- `large_balanced`: 40 nodes, 40000 total size
- `medium_serial`: 20 nodes with deep dependencies
- `medium_mixed`: 20 nodes with mixed parallel/serial structure
- `sparse_graph`: Low connectivity, high parallelism
- `dense_graph`: High connectivity, limited parallelism
- `wide_shallow`: Many parallel paths
- `narrow_deep`: Long critical path
- `quick_test`: Fast execution for testing

### Scheduling Algorithms

1. **CP-Priority**: Critical path tasks prioritized
2. **HEFT**: Heterogeneous Earliest Finish Time
3. **ETF**: Earliest Time First
4. **CPOP**: Critical Path on Processors
5. **Level-by-Level**: Dependency level based
6. **Largest-First**: Largest tasks first
7. **Smallest-First**: Smallest tasks first
8. **Most-Successors-First**: Tasks with most dependents first
9. **FIFO**: First-In-First-Out
10. **Random**: Random ordering

## Output

### Performance Tables

`comparison_tables.txt` contains:
- Baseline performance (no decomposition)
- Sequential execution metrics
- Parallel execution comparison across schedulers
- Statistical analysis (mean, std dev, min-max)
- Speedup calculations
- Scheduler rankings

### Graph Analysis

`precedence_graphs.txt` contains:
- Critical path information
- Node size statistics
- Graph topology metrics
- Validation results
## Computational Constant C Estimation

The `C_estimation.py` tool empirically calibrates the constant \(C\) in the complexity model \(t = C \cdot N^3\) for `scipy.optimize.fsolve`.

### Equation Types Tested

| Equation Type | Mathematical Form |
|--------------|-------------------|
| Simple Polynomial | \(x^2 - 4 = 0\) |
| Trigonometric | \(\sin(x) + x^2 - 2 = 0\) |
| Exponential | \(e^{x/100} - x/50 - 1 = 0\) |
| Cubic Polynomial | \(x^3 - 2x^2 + x - 5 = 0\) |
| Mixed Transcendental | \(\sin(x) + e^{x/100} - x^2/1000 - 2 = 0\) |

### Example Output
================================================================================
FINAL SUMMARY - Recommended C Values
================================================================================
Simple Polynomial (x² - 4 = 0) : C = 9.9985e-10
Trigonometric (sin(x) + x² - 2 = 0) : C = 9.6747e-10
Exponential (exp(x/100) - x/50 - 1 = 0) : C = 9.4805e-10
Cubic Polynomial (x³ - 2x² + x - 5 = 0) : C = 9.6614e-10
Mixed Transcendental : C = 9.5458e-10

## Project Structure

```
paralleleqsolver/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── MulticoreSeqScheduler.py
├── C_estimation.py                  # C constant calibration tool
├── comparison_tables.txt (generated)
├── precedence_graphs.txt (generated)
└── c_estimation_results_*.json (generated) # Calibration results
```

## Performance Metrics

- **Makespan**: Total execution time from start to finish
- **Critical Path Duration**: Theoretical minimum execution time
- **CPU Utilization**: Percentage of CPU resources used
- **Speedup**: Performance improvement vs baseline/sequential
- **Parallel Efficiency**: Speedup divided by number of cores
- **Scheduling Overhead**: Time spent on scheduling decisions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Moussaoui_DependencyAwareScheduling,
  title   = {Performance Analysis of Dependency Aware Scheduling Algorithms for Parallel Nonlinear System Solving on Multicore Architectures},
  author  = {Moussaoui, Adel and Belhadj, Foudil and Gueddoudj, El Yazid and Bouamama, Salim and Boukhari, Nawel},
  journal = {--},
  volume  = {--},
  number  = {--},
  pages   = {--},
  year    = {--},
  publisher = {--},
  address = {M'sila, Algeria},
  email   = {adel.moussaoui@univ-msila.dz},
  note    = {Corresponding author: Adel Moussaoui}
}```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Contact

Adel MOUSSAOUI - [ adel dot moussaoui at univ-msila dot dz]

Project Link: [https://github.com/adelmoussaoui/paralleleqsolver/]

## Acknowledgments

- NetworkX library for graph algorithms
- SciPy for nonlinear system solving
- Multiprocessing for parallel execution
- psutil for hardware detection
- matplotlib for visualization
