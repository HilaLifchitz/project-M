# Miso: Population Genetics Simulation and Analysis

A comprehensive Python package for simulating and analyzing population genetics dynamics, with a focus on Wright-Fisher models, selection effects, and information-theoretic measures.

## Overview

This project implements various population genetics simulations and analysis tools, including:
- Wright-Fisher simulations with and without recombination
- Selection modeling (directional and stabilizing)
- Kullback-Leibler divergence analysis between distributions
- Convergence analysis and autocorrelation studies
- Phenotype distribution analysis

## Main Components

### Core Simulation Engine

#### `QE.py`
The main simulation engine containing:
- **`WrightFisherSimulator`**: Basic Wright-Fisher simulation
- **`WrightFisherSimulator_FreeRecombination`**: Wright-Fisher with free recombination
- **`TraitEvolutionSimulator_directional`**: Trait evolution under directional selection
- **`TraitEvolutionSimulator_stabilizing`**: Trait evolution under stabilizing selection
- Analytical functions for neutral distribution `φ(t)` and selection distribution `ψ(t)`
- Variance computation and normal distribution PDFs

#### `Markov.py`
Contains the `compute_stationary_quiet` function for computing stationary distributions in Markov chain models.

### Analysis Scripts

#### `3.6B_maybe.py`
Main analysis script for comparing phenotype distributions:
- **`kl_bits()`**: Numerically stable KL divergence calculation with symmetric smoothing
- **`kl_pheno_bits()`**: KL divergence between phenotype distributions with proper binning
- **`kl_psi_vs_binomial()`**: Compare empirical selection distribution to theoretical binomial
- **`sample_Z_hist()`**: Sample phenotype histograms from allele frequency distributions
- **`run()`**: Main execution function with parameterized simulations
- **`plot_pheno_distributions()`**: Visualization of phenotype distributions

#### `convergence_analysis_nick.py`
Advanced convergence analysis with multiple simulations:
- **`run_multiple_simulations()`**: Run multiple Wright-Fisher simulations with checkpointing
- **`compute_average_dynamics()`**: Average KL and phenotype dynamics across simulations
- **`compute_autocorrelation()`**: Calculate autocorrelation for time series
- **`analyze_autocorrelation_at_points()`**: Autocorrelation analysis at specific time points
- **`plot_average_dynamics()`**: Visualization of average dynamics with confidence bands
- **`plot_autocorrelation_analysis()`**: Autocorrelation decay analysis plots

#### `kl_comparison_test.py`
Systematic comparison of different KL divergence calculation methods:
- Compares old vs. new KL calculation methods
- Demonstrates effects of smoothing and reference distributions
- Provides detailed explanations of numerical differences

### Testing and Validation

#### `test_simple_neutral_approach.py`
Previous attempt at equilibrium detection (reference for KL divergence values).

#### `quick_80bit_test.py`
Quick validation tests for KL divergence calculations.

## Key Features

### Numerical Stability
- Robust KL divergence calculation with symmetric additive smoothing
- Proper handling of zero probabilities and sparse distributions
- Renormalization to ensure valid probability distributions

### Simulation Capabilities
- **Population size**: Configurable population sizes (N)
- **Loci**: Variable number of loci (l)
- **Selection**: Directional and stabilizing selection parameters
- **Mutation**: Configurable mutation rates
- **Recombination**: Free recombination model
- **Generations**: Long-term evolution tracking

### Analysis Tools
- **KL Divergence**: Multiple calculation methods for different use cases
- **Phenotype Analysis**: Mean, variance, and distribution tracking
- **Autocorrelation**: Memory effects and decay rate analysis
- **Convergence**: Multi-simulation averaging and statistical analysis

### Data Management
- **Checkpointing**: Automatic saving of simulation progress
- **Data Persistence**: NPZ format for efficient storage
- **Reproducibility**: Seed-based random number generation

## Usage Examples

### Basic Simulation
```python
from QE import WrightFisherSimulator_FreeRecombination

# Run a single simulation
sim = WrightFisherSimulator_FreeRecombination(
    N=40, l=1000, s=0.01, mu=0.0005, T=10000, seed=42
)
results = sim.run()
```

### KL Divergence Analysis
```python
from 3.6B_maybe import run

# Run phenotype distribution comparison
run(N=40, l=1000, N_pop=2000, seed=42)
```

### Convergence Analysis
```python
from convergence_analysis_nick import main

# Run comprehensive convergence analysis
main()  # Uses default parameters
```

## Dependencies

- **NumPy**: Numerical computing
- **SciPy**: Statistical functions and optimization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Enhanced statistical graphics

## File Structure

```
Miso/
├── QE.py                          # Core simulation engine
├── Markov.py                      # Markov chain utilities
├── 3.6B_maybe.py                  # Main analysis script
├── convergence_analysis_nick.py   # Convergence analysis
├── kl_comparison_test.py          # KL divergence comparison
├── test_simple_neutral_approach.py # Equilibrium detection
├── quick_80bit_test.py            # Quick validation tests
├── images/                        # Generated plots and figures
└── convergence_results/           # Simulation results and checkpoints
```

## Research Context

This project is designed for population genetics research, particularly focusing on:
- Information-theoretic measures in evolutionary dynamics
- Selection effects on phenotype distributions
- Convergence properties of Wright-Fisher models
- Numerical methods for population genetics simulations

## Notes

- All KL divergence calculations are in bits (log base 2)
- Simulations use checkpointing to prevent data loss
- Results are automatically saved in NPZ format for later analysis
- The project includes comprehensive error handling and progress tracking
