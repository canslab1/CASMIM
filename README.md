# CASMIM — Cellular Automata with Social Mirror Identities Model

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A Python 3 implementation of the **CASMIM** model for simulating SARS transmission dynamics and evaluating public health policy interventions on a small-world epidemiological network.

## Overview

In the early 2000s, SARS outbreaks in cities such as Singapore, Taipei, and Toronto demonstrated how daily-contact social networks and long-distance movement could rapidly amplify disease spread. CASMIM addresses this by combining cellular automata with a "social mirror identity" mechanism: each person owns multiple *agents* (mirrors) scattered across a 500 &times; 500 torus lattice, representing the different social spheres (home, workplace, hospital, etc.) that a single individual participates in daily.

The model integrates:

- **Cellular Automata** on a 2D torus lattice (500 &times; 500) for spatial agent interactions
- **Social Mirror Identities** to represent daily-contact social networks and long-distance movement
- **SEIR+D compartmental model** (Susceptible &rarr; Exposed &rarr; Infective &rarr; Recovered &rarr; Immune &rarr; Susceptible, with Death branch)
- **8 public health policies** (mask wearing, temperature screening, hospitalization, home quarantine, contact reduction, visit restriction, vaccination, medical policy)
- **Contact tracing** via BFS-based algorithms with level-1 and level-2 quarantine
- **Super-spreader modeling** and **age-stratified mortality**

The model was originally developed in Borland C++ Builder (2003-2005) and has been ported to Python 3 with a PySide6 (Qt) GUI.

## Features

- **Interactive lattice visualization** — 500 &times; 500 macro lattice and 100 &times; 100 micro lattice with click-to-navigate
- **Real-time SEIR+D display** — Color-coded agents (sky-blue = susceptible/exposed/immune, red = infective, silver = recovered, black = died)
- **6 chart types** — Accumulative, daily, notifiable, infective, accumulative quarantine, daily quarantine
- **8 configurable policies** — Each policy can be toggled on/off during simulation with adjustable effect and coverage rates
- **Dynamic policy activation** — Policies can be enabled or disabled mid-simulation to study intervention timing
- **Super-spreader modeling** — Configurable probability for super-spreader designation
- **Excel output** — Simulation statistics exported via openpyxl with 4 output sheets (cumulative counts, daily deltas, action log, running averages)
- **Numba JIT acceleration** — Core simulation loop compiled to native code via Numba `@njit`, achieving ~8x speedup over pure Python; fallback to Python path via `CASMIM_NO_NUMBA=1`
- **NumPy-accelerated** — Structure-of-Arrays (SoA) data layout with vectorized operations for population-level computations

## Installation

```bash
git clone https://github.com/canslab1/CASMIM.git
cd CASMIM
pip install -r requirements.txt
```

### Dependencies

| Package | Version |
|---------|---------|
| PySide6 | &ge; 6.5 |
| NumPy | &ge; 1.24 |
| Numba | &ge; 0.60.0 |
| pyqtgraph | &ge; 0.13 |
| openpyxl | &ge; 3.1 |

## Usage

```bash
python main.py
```

This launches the GUI application with:

- **Left panel**: Disease parameters (Setup tab), population parameters (World tab), policy controls (Policy tab) with checkboxes and effect/coverage sliders
- **Center panel**: Macro tab (500 &times; 500 lattice) with real-time color-coded agent states; click to navigate the Micro tab (100 &times; 100 magnified view)
- **Right panel**: 6 pyqtgraph chart widgets displaying accumulative, daily, notifiable, infective, and quarantine statistics
- **Bottom panel**: 9-panel status bar showing coordinates, agent state, identity, current day, and mortality summary

### Controls

| Control | Function |
|---------|----------|
| **Run** | Execute continuous simulation (one day per step) |
| **Stop** | Pause the simulation |
| **Setup** | Initialize (or re-initialize) the population and lattice |
| **Policy checkboxes** | Toggle individual policies on/off during simulation |
| **Vaccine button** | Administer a batch of vaccines to random unvaccinated individuals |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CASMIM_NO_NUMBA=1` | Disable Numba JIT and use the pure Python engine (useful for debugging or when Numba is unavailable) |

## Disease Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exposed_period` | 5 | Average incubation period (days) |
| `symptomatic_period` | 23 | Average symptomatic duration (days) |
| `infective_period` | 3 | Average infectious period (days) |
| `recovered_period` | 7 | Average recovery period (days) |
| `immune_period` | 60 | Average immunity duration (days) |
| `quarantine_period` | 10 | Quarantine/isolation duration (days) |
| `transmission_prob` | 0.05 | Per-contact infection probability |
| `immune_prob` | 0.02 | Natural immunity probability |
| `detect_rate` | 0.9 | Symptom detection rate |
| `super_rate` | 0.0001 | Super-spreader designation probability |
| `mortality_old` | 0.52 | Mortality rate for elderly |
| `mortality_prime` | 0.17 | Mortality rate for prime-age adults |
| `mortality_young` | 0.05 | Mortality rate for young individuals |

## Policy Parameters

| Policy | Effect | Coverage | Description |
|--------|--------|----------|-------------|
| Mask wearing | 0.9 | 0.9 | Reduces transmission probability |
| Temperature screening | 0.9 | 0.9 | Increases detection rate for early isolation |
| Hospital isolation | 0.5 | 0.95 | Isolates infective individuals in hospital |
| Home quarantine | — | 0.81 | Quarantines contacts at home |
| Visit restriction | — | 0.9 | Restricts hospital visitors to reduce nosocomial infection |
| Contact reduction | — | 0.9 | Reduces social contacts (e.g., school/workplace closure) |
| Vaccination | — | count | Administers vaccines to random unvaccinated individuals |
| Medical policy | 0.9 | 0.9 | Applies antiviral treatment to infective individuals |

## Algorithm Overview

CASMIM simulates epidemic dynamics through a daily step cycle:

1. **Population initialization** — 100,000 individuals are created with age-stratified demographics (young/prime/old). Each person is assigned multiple "social mirror" agents distributed across the 500 &times; 500 torus lattice using a Gaussian quota distribution, representing their presence in different social spheres.

2. **Daily traversal** — Each day, all individuals are traversed in a randomly chosen direction (forward or reverse) to eliminate ordering bias. For each person, the engine executes:

3. **State transition** — The SEIR+D state machine advances:
   - **Susceptible**: Agents interact with Moore-neighborhood neighbors; transmission occurs with probability `transmission_prob` (modified by mask and medical policies).
   - **Exposed**: Counter increments; transitions to Infective after `exposed_period` days.
   - **Infective**: May be detected (with probability `detect_rate`) and isolated/quarantined. Transmission to neighbors continues. After `infective_period` days, transitions to Recovered or Died (age-stratified mortality).
   - **Recovered**: After `recovered_period` days, transitions to Immune.
   - **Immune**: After `immune_period` days, returns to Susceptible (unless permanently immunized by vaccine).

4. **Contact tracing** — When an infective individual is detected, BFS-based contact tracing identifies level-1 (direct contacts) and optionally level-2 (contacts of contacts) neighbors for home quarantine.

5. **Super-spreader effect** — Super-spreaders interact with all 8 Moore-neighborhood cells instead of a randomly selected single neighbor, dramatically increasing their transmission reach.

## Implementation Notes

- **Numba JIT compilation**: The core simulation loop (`change_society`) and all subroutines (13 functions total) are compiled to native code via `@nb.njit(cache=True)`, yielding ~8x speedup. BFS contact tracing uses pre-allocated arrays with head/tail pointers instead of Python `deque`.
- **AoS &rarr; SoA conversion**: The original C++ Array-of-Structures (`society.people[i].state`) is converted to Structure-of-Arrays (`people_state[i]`) using NumPy for cache-friendly vectorized operations.
- **Contact tracing**: Changed from recursive DFS (C++) to iterative BFS (Python/Numba) to avoid stack overflow on large populations.
- **Policy vectorization**: All policy applications (except vaccination) use `np.random.random(N) < available` for O(1) per-person Bernoulli trials instead of scalar loops.
- **Incremental rendering**: A dirty-set mechanism (`dirty_pids`) tracks only agents whose state changed each day, avoiding full-lattice repaints.
- **Chart rendering**: Migrated from VCL TChart (C++) to pyqtgraph (Python) for real-time chart updates.

## Project Structure

```
CASMIM/
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package metadata (PEP 621)
├── CITATION.cff                 # Citation metadata
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # MIT License
└── sars_sim/
    ├── __init__.py
    ├── models.py                # Data structures (StateEnum, SimulationParams, SimulationData)
    ├── world.py                 # Lattice management, population initialization, agent distribution
    ├── engine.py                # Core SEIR+D simulation engine, transmission and state transition logic
    ├── engine_numba.py          # Numba JIT-compiled kernels (13 @njit functions, ~8x speedup)
    ├── policies.py              # 8 public health policy implementations (vectorized)
    ├── statistics.py            # Statistics tracking, Excel output (4 sheets)
    └── gui/
        ├── __init__.py
        ├── main_window.py       # Main application window (PySide6)
        ├── lattice_view.py      # Macro/micro lattice visualization (QImage ARGB32)
        ├── charts.py            # 6 pyqtgraph chart widgets
        ├── controls.py          # Parameter, disease, and policy control panels
        └── status_bar.py        # 9-panel status bar
```

## Authors

- **Chung-Yuan Huang** (黃崇源) — Department of Computer Science and Information Engineering, Chang Gung University, Taiwan (gscott@mail.cgu.edu.tw)
- **Chuen-Tsai Sun** — Department of Computer Science, National Yang Ming Chiao Tung University, Taiwan
- **Ji-Lung Hsieh** — Graduate Institute of Journalism, National Taiwan University, Taiwan
- **Yi-Ming Arthur Chen** — Department of Information Management, National Central University, Taiwan
- **Holin Lin** — Department of Sociology, National Taiwan University, Taiwan

## Citation

If you use this software in your research, please cite:

> Huang, C.-Y., Sun, C.-T., Hsieh, J.-L., Chen, Y.-M. A., & Lin, H. (2005). A Novel Small-World Model: Using Social Mirror Identities for Epidemic Simulations. *SIMULATION*, 81(10), 671-699. https://doi.org/10.1177/0037549705061519

See `CITATION.cff` for machine-readable citation metadata.

## References

1. Huang, C.-Y., Sun, C.-T., Hsieh, J.-L., & Lin, H. (2004). Simulating SARS: Small-World Epidemiological Modeling and Public Health Policy Assessments. *Journal of Artificial Societies and Social Simulation*, 7(4), 2. http://jasss.soc.surrey.ac.uk/7/4/2.html

2. Huang, C.-Y., Sun, C.-T., Hsieh, J.-L., Chen, Y.-M. A., & Lin, H. (2005). A Novel Small-World Model: Using Social Mirror Identities for Epidemic Simulations. *SIMULATION*, 81(10), 671-699. https://doi.org/10.1177/0037549705061519

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
