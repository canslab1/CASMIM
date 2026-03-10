from PySide6.QtWidgets import QStatusBar, QLabel
from ..models import StateEnum


class SimulationStatusBar(QStatusBar):
    """9-panel status bar showing simulation state info."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create 9 QLabel panels
        self.panels = []
        widths = [60, 60, 80, 80, 80, 80, 60, 100, 300]
        for i, w in enumerate(widths):
            label = QLabel("")
            label.setMinimumWidth(w)
            label.setFrameStyle(QLabel.Shape.Panel | QLabel.Shadow.Sunken)
            self.addPermanentWidget(label)
            self.panels.append(label)

    def update_cell_info(self, x, y, pid, agent_count, state, isolated, quarantined, agent_no, is_home):
        """Update panels 0-6 with cell/person info (on mouse hover)."""
        self.panels[0].setText(f"X = {x}")
        self.panels[1].setText(f"Y = {y}")
        self.panels[2].setText(f"No. = {pid}")
        self.panels[3].setText(f"amount = {agent_count}")

        state_code = self._get_state_code(state, isolated, quarantined)
        self.panels[4].setText(f"state = {state_code}")
        self.panels[5].setText(f"Identity = {agent_no}")
        self.panels[6].setText("Root" if is_home else "Other")

    def update_day_info(self, day):
        """Update panel 7 with current day."""
        self.panels[7].setText(f"day = {day}")

    def update_summary(self, recovered, died):
        """Update panel 8 with summary statistics."""
        if recovered + died > 0:
            mortality = int((died / (recovered + died)) * 100)
            self.panels[8].setText(f"(R) {recovered} person(s), (D) {died} person(s), mortality = {mortality}%")
        else:
            self.panels[8].setText(f"(R) {recovered} person(s), (D) {died} person(s)")

    @staticmethod
    def _get_state_code(state, isolated, quarantined):
        """Return state code string like (S), (E).(I), (I).(Q)."""
        codes = {
            StateEnum.SUSCEPTIBLE: "(S)",
            StateEnum.EXPOSED: "(E)",
            StateEnum.INFECTIVE: "(I)",
            StateEnum.RECOVERED: "(R)",
            StateEnum.IMMUNE: "(M)",
            StateEnum.DIED: "(D)",
        }
        msg = codes.get(state, "(?)")
        if isolated:
            msg += ".(I)"
        elif quarantined:
            msg += ".(Q)"
        return msg
