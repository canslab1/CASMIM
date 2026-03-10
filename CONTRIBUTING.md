# Contributing to CASMIM

Thank you for your interest in contributing to the CASMIM SARS Epidemic Simulation project!

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/canslab1/CASMIM/issues) page to report bugs or request features.
- When reporting a bug, please include:
  - Python version (`python --version`)
  - Operating system
  - Steps to reproduce the issue
  - Expected vs. actual behavior
  - Relevant error messages or screenshots

### Submitting Changes

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CASMIM.git
   cd CASMIM
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test them.
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of your change"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a **Pull Request** on GitHub.

## Development Setup

```bash
git clone https://github.com/canslab1/CASMIM.git
cd CASMIM
pip install -r requirements.txt
python main.py  # Verify the GUI launches correctly
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints where practical.
- Keep functions focused and reasonably sized.
- Document non-obvious algorithms with comments referencing the paper.
- Keep comments in the language consistent with the surrounding code (Chinese or English).

## Project Architecture

| Module | Responsibility |
|--------|---------------|
| `sars_sim/models.py` | Data structures (StateEnum, SimulationParams, SimulationData) |
| `sars_sim/engine.py` | Core SEIR+D simulation engine, transmission and state transition logic |
| `sars_sim/world.py` | Lattice management, population initialization, agent distribution |
| `sars_sim/policies.py` | 8 public health policy implementations |
| `sars_sim/statistics.py` | Statistics tracking, Excel output |
| `sars_sim/gui/` | PySide6 graphical interface (main window, lattice views, charts, controls, status bar) |

## References

When contributing epidemiological model changes, please reference the original papers:

1. Huang et al. (2004). *JASSS*, 7(4).
2. Huang et al. (2005). *SIMULATION*, 81(10), 671-699.

## Questions?

Feel free to open an issue for any questions about the codebase or contribution process.
