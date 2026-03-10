"""SARS Epidemic Dynamics Simulation -- Entry Point."""

import sys
from PySide6.QtWidgets import QApplication
from sars_sim.gui.main_window import SARSMainWindow


def main():
    app = QApplication(sys.argv)
    window = SARSMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
