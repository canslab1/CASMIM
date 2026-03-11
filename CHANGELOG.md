# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## v2.1.0 (2026-03-11)

### Added
- Numba JIT-compiled simulation engine (`engine_numba.py`) with 13 `@nb.njit(cache=True)` functions, achieving ~8x speedup over pure Python
- `CASMIM_NO_NUMBA=1` environment variable to disable Numba and fall back to pure Python engine
- Pre-allocated BFS arrays for contact tracing in Numba kernel (replaces Python `deque`/`set`)
- Dirty-flag tracking array for incremental lattice repaint in Numba path

### Fixed
- Window progressive enlargement during simulation run (changed QLabel SizePolicy from Expanding to Ignored)
- Exposed/Infective radio buttons allowing simultaneous selection (added QButtonGroup)
- Micro lattice view rendering as tiny blue square when switching from Macro tab (removed AlignCenter constraint, added cached base pixmap with resizeEvent rescaling)
- Micro view Attribute field always showing "Susceptible" regardless of disease state (use `state != SUSCEPTIBLE` condition instead of `timer > 0`, matching C++ original)
- English text consistency across all GUI panels (label capitalization, abbreviations, spacing)

### Changed
- UI tab names: "Macro View" → "Macro", "Micro View" → "Micro", "Parameters" → "World", "Policies" → "Policy"
- Chart tab names: "Accumulative" → "Accumulation", "Daily" → "Incidence", "Notifiable" → "Notification", "Infective" → "Infection"
- Button labels: "Initial" → "Setup", "Run" → "Go"
- Swapped Save/Print button order (Save before Print), Stop/Go button order (Stop before Go)
- Label "avg." → "Runs", "Attr." → "Attribute", "day" → "day(s), contact"
- Control bar reorganized into QGroupBox grid layout

### Removed
- Unused progress bar widget and all related dead code

## v2.0.0 (2025-03-10)

Complete rewrite from Borland C++ Builder to Python 3.

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
- Incremental world-color update via dirty-set mechanism (`dirty_pids`) for efficient rendering
- Vectorized policy application using NumPy Bernoulli trials
- 9-panel status bar displaying real-time population counts by state
- CITATION.cff for academic referencing
- pyproject.toml for PEP 621 package metadata

## v1.0.0 (2003-2005)

Original implementation by Chung-Yuan Huang, Chuen-Tsai Sun, Ji-Lung Hsieh, Yi-Ming Arthur Chen, and Holin Lin.

### Added
- Original implementation in Borland C++ Builder 6
- CASMIM (Cellular Automata with Social Mirror Identities Model) for SARS epidemic simulation
- SEIR+D compartmental model with 8 public health policies
- 500 x 500 macro lattice and 100 x 100 micro lattice visualization
- Super-spreader modeling and contact tracing
- Simulation of SARS outbreaks in Singapore, Taipei, and Toronto
- Age-stratified mortality rates (young, prime, old)
- Dynamic policy activation/deactivation during simulation
