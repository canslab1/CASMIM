# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2025

### Changed
- Complete rewrite from Borland C++ Builder to Python 3 with PySide6 (Qt) GUI
- Data layout converted from Array-of-Structures (AoS) to Structure-of-Arrays (SoA) using NumPy for vectorized operations
- Contact tracing algorithm changed from recursive DFS to iterative BFS
- Chart rendering migrated from VCL TChart to pyqtgraph
- Statistics output migrated from custom file I/O to openpyxl (Excel .xlsx)
- GUI framework migrated from VCL (Visual Component Library) to PySide6/Qt

### Added
- Modular Python package structure (`sars_sim/`)
- Separate GUI module (`sars_sim/gui/`) with dedicated widgets for lattice views, charts, controls, and status bar
- NumPy-based vectorized agent generation with normal distribution and fine-tuning

## [1.0.0] - 2003-2005

### Added
- Original implementation in Borland C++ Builder 6
- CASMIM (Cellular Automata with Social Mirror Identities Model) for SARS epidemic simulation
- SEIR+D compartmental model with 8 public health policies
- 500 x 500 macro lattice and 100 x 100 micro lattice visualization
- Super-spreader modeling and contact tracing
- Simulation of SARS outbreaks in Singapore, Taipei, and Toronto
