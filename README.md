# CASMIM SARS Epidemic Simulation

A Python 3 implementation of the **Cellular Automata with Social Mirror Identities Model (CASMIM)** for simulating SARS transmission dynamics and evaluating public health policy interventions.

## Overview

This project implements a small-world epidemiological simulation model that combines:

- **Cellular Automata** on a 2D torus lattice (500 x 500) for spatial agent interactions
- **Social Mirror Identities** to represent daily-contact social networks and long-distance movement
- **SEIR+D compartmental model** (Susceptible → Exposed → Infective → Recovered → Immune → Susceptible, with Death branch)
- **8 public health policies** (mask wearing, temperature screening, hospitalization, home quarantine, contact reduction, visit restriction, vaccination, medical policy)
- **Contact tracing** via BFS-based algorithms with level-1 and level-2 quarantine

The model was originally developed in Borland C++ Builder (2003-2005) and has been ported to Python 3 with a PySide6 (Qt) GUI.

## Features

- Interactive 500 x 500 macro lattice view and 100 x 100 micro lattice view with navigation
- Real-time SEIR+D state visualization with color-coded agents
- 6 chart types: accumulative, daily, notifiable, infective, accumulative quarantine, daily quarantine
- Configurable disease parameters (incubation period, infectious period, transmission rate, mortality rate, etc.)
- Dynamic policy activation/deactivation during simulation
- Super-spreader modeling
- Excel output for simulation statistics (via openpyxl)

## Requirements

- Python 3.10+
- PySide6
- NumPy
- pyqtgraph
- openpyxl

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Project Structure

```
├── main.py                      # Entry point
├── sars_sim/
│   ├── __init__.py
│   ├── models.py                # Data structures (StateEnum, SimulationParams, SimulationData)
│   ├── world.py                 # World/lattice management, agent distribution
│   ├── engine.py                # SEIR+D simulation engine, transmission logic
│   ├── policies.py              # 8 public health policy implementations
│   ├── statistics.py            # Statistics tracking, Excel file output
│   └── gui/
│       ├── __init__.py
│       ├── main_window.py       # Main application window
│       ├── lattice_view.py      # Macro/micro lattice visualization
│       ├── charts.py            # 6 pyqtgraph chart widgets
│       ├── controls.py          # Parameter, disease, and policy panels
│       └── status_bar.py        # 9-panel status bar
```

## References

1. Huang, C.-Y., Sun, C.-T., Hsieh, J.-L., & Lin, H. (2004). Simulating SARS: Small-World Epidemiological Modeling and Public Health Policy Assessments. *Journal of Artificial Societies and Social Simulation*, 7(4), 2. http://jasss.soc.surrey.ac.uk/7/4/2.html

2. Huang, C.-Y., Sun, C.-T., Hsieh, J.-L., Chen, Y.-M. A., & Lin, H. (2005). A Novel Small-World Model: Using Social Mirror Identities for Epidemic Simulations. *SIMULATION*, 81(10), 671-699. https://doi.org/10.1177/0037549705061519

## License

MIT License. See [LICENSE](LICENSE) for details.
