"""Parameter control panels for the SARS simulation GUI.

Three panel classes implemented as QWidget subclasses.  The original
seven tabs are consolidated into three tabs: Parameters, Disease,
and Policies.  Each panel exposes typed signals so that the main
window can react to user changes without polling.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(text: str, default: int = 0) -> int:
    """Convert *text* to int, returning *default* on failure."""
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return default


def _safe_float(text: str, default: float = 0.0) -> float:
    """Convert *text* to float, returning *default* on failure."""
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return default


def _make_line_edit(default, *, read_only: bool = False, max_width: int = 120) -> QLineEdit:
    """Return a QLineEdit pre-filled with *default* (converted to str)."""
    le = QLineEdit(str(default))
    le.setMaximumWidth(max_width)
    if read_only:
        le.setReadOnly(True)
        le.setEnabled(False)
    return le


def _make_section_label(text: str) -> QLabel:
    """Return a QLabel used as a section heading."""
    return QLabel(text)


def _make_separator() -> QFrame:
    """Return a horizontal line separator."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


# ===================================================================
# 1. ParameterPanel  (consolidates Small World + System + Environment
#                      + Population into a single tab)
# ===================================================================

class ParameterPanel(QWidget):
    """Consolidated parameter panel combining world, population, output,
    and small-world controls into one scrollable tab.

    Field order (as requested):
      World Height, World Width, Max Population, Max Agent,
      Gossip Mode, Old Man Rate, Young Man Rate, Super Rate,
      distribution memo, Output File 1-4, Input as well as Run,
      Degree Count, Calculate, small-world result memo.
    """

    calculate_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Outer layout holds a QScrollArea so the panel can scroll
        # when the window is too small to show everything.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        inner = QWidget()
        # Ensure the inner widget (and all its children) uses the same
        # font as the application so text sizes are consistent with
        # other tabs.
        inner.setFont(self.font())
        layout = QVBoxLayout(inner)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self._height_edit = _make_line_edit(500, read_only=True)
        self._width_edit = _make_line_edit(500, read_only=True)
        self._max_pop_edit = _make_line_edit(100000)
        self._max_agent_edit = _make_line_edit(5)

        self._fixed_radio = QRadioButton("Fixed")
        self._floating_radio = QRadioButton("Floating")
        self._fixed_radio.setChecked(True)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._fixed_radio)
        mode_row.addWidget(self._floating_radio)
        mode_row.addStretch()

        self._old_rate_edit = _make_line_edit(0.2)
        self._young_rate_edit = _make_line_edit(0.3)
        self._super_rate_edit = _make_line_edit(0.0001)

        form.addRow("Height:", self._height_edit)
        form.addRow("Width:", self._width_edit)
        form.addRow("Max Population:", self._max_pop_edit)
        form.addRow("Max Agent:", self._max_agent_edit)
        form.addRow("Gossip Mode:", mode_row)
        form.addRow("Old Man Rate:", self._old_rate_edit)
        form.addRow("Young Man Rate:", self._young_rate_edit)
        form.addRow("Super Rate:", self._super_rate_edit)
        layout.addLayout(form)

        # --- distribution memo (shows agent count distribution) ---
        self._dist_memo = QTextEdit()
        self._dist_memo.setReadOnly(True)
        self._dist_memo.setMaximumHeight(120)
        layout.addWidget(self._dist_memo)

        layout.addWidget(_make_separator())

        self._output_checks: list[QCheckBox] = []
        for i in range(1, 5):
            cb = QCheckBox(f"Output File {i}")
            cb.setChecked(True)
            self._output_checks.append(cb)
            layout.addWidget(cb)

        self._auto_run_check = QCheckBox("Input as well as Run")
        self._auto_run_check.setChecked(False)
        layout.addWidget(self._auto_run_check)

        layout.addWidget(_make_separator())

        degree_form = QFormLayout()
        degree_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        degree_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self._degree_edit = _make_line_edit(100)
        degree_form.addRow("Degree of Separation:", self._degree_edit)
        layout.addLayout(degree_form)

        self._calc_btn = QPushButton("Calculate")
        self._calc_btn.clicked.connect(self._on_calculate)
        layout.addWidget(self._calc_btn)

        self._sw_memo = QTextEdit()
        self._sw_memo.setReadOnly(True)
        layout.addWidget(self._sw_memo)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # -- public API (Population) -----------------------------------------

    def get_params(self) -> dict:
        """Return population-related parameters as a dict."""
        return {
            "max_population": _safe_int(self._max_pop_edit.text(), 100000),
            "max_agent": _safe_int(self._max_agent_edit.text(), 5),
            "gossip_fixed": self._fixed_radio.isChecked(),
            "old_man_rate": _safe_float(self._old_rate_edit.text(), 0.2),
            "young_man_rate": _safe_float(self._young_rate_edit.text(), 0.3),
            "super_rate": _safe_float(self._super_rate_edit.text(), 0.0001),
        }

    def set_distribution_text(self, text: str) -> None:
        """Set the agent distribution memo content."""
        self._dist_memo.setPlainText(text)

    def set_max_population(self, value: int) -> None:
        """Set the Max Population field value."""
        self._max_pop_edit.setText(str(value))

    def set_max_population_readonly(self, readonly: bool) -> None:
        """Set the Max Population field read-only state."""
        self._max_pop_edit.setReadOnly(readonly)
        self._max_pop_edit.setEnabled(not readonly)

    # -- public API (System) ---------------------------------------------

    def get_output_flags(self) -> list[bool]:
        """Return a list of 4 bools corresponding to Output File 1-4."""
        return [cb.isChecked() for cb in self._output_checks]

    def is_auto_run(self) -> bool:
        return self._auto_run_check.isChecked()

    # -- public API (Small World) ----------------------------------------

    def set_result_text(self, text: str) -> None:
        """Set the small-world result memo content."""
        self._sw_memo.setPlainText(text)

    # -- private slots ---------------------------------------------------

    def _on_calculate(self) -> None:
        self.calculate_requested.emit(_safe_int(self._degree_edit.text(), 10))


# ===================================================================
# 5. DiseasePanel
# ===================================================================

class DiseasePanel(QWidget):
    """Disease-specific parameters (periods, probabilities, mortality)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QFormLayout(self)

        self._trans_prob_edit = _make_line_edit(0.05)
        self._immune_prob_edit = _make_line_edit(0.02)
        self._exposed_edit = _make_line_edit(5)
        self._symptomatic_edit = _make_line_edit(23)
        self._infectious_edit = _make_line_edit(3)
        self._recovered_edit = _make_line_edit(7)
        self._antibody_edit = _make_line_edit(60)
        self._detect_rate_edit = _make_line_edit(0.9)
        self._mort_old_edit = _make_line_edit(0.52)
        self._mort_prime_edit = _make_line_edit(0.17)
        self._mort_young_edit = _make_line_edit(0.05)

        layout.addRow("Transmission Probability:", self._trans_prob_edit)
        layout.addRow("Immune Probability:", self._immune_prob_edit)
        layout.addRow("Avg. Exposed Period:", self._exposed_edit)
        layout.addRow("Avg. Symptomatic Period:", self._symptomatic_edit)
        layout.addRow("Avg. Infectious Period:", self._infectious_edit)
        layout.addRow("Avg. Recovered Period:", self._recovered_edit)
        layout.addRow("Avg. Antibody Period:", self._antibody_edit)
        layout.addRow("Detection Rate:", self._detect_rate_edit)
        layout.addRow("Mortality (Old):", self._mort_old_edit)
        layout.addRow("Mortality (Prime):", self._mort_prime_edit)
        layout.addRow("Mortality (Young):", self._mort_young_edit)

    # -- public API ------------------------------------------------------

    def get_params(self) -> dict:
        return {
            "transmission_prob": _safe_float(self._trans_prob_edit.text(), 0.05),
            "immune_prob": _safe_float(self._immune_prob_edit.text(), 0.02),
            "exposed_period": _safe_int(self._exposed_edit.text(), 5),
            "symptomatic_period": _safe_int(self._symptomatic_edit.text(), 23),
            "infective_period": _safe_int(self._infectious_edit.text(), 3),
            "recovered_period": _safe_int(self._recovered_edit.text(), 7),
            "immune_period": _safe_int(self._antibody_edit.text(), 60),
            "detect_rate": _safe_float(self._detect_rate_edit.text(), 0.9),
            "mortality_old": _safe_float(self._mort_old_edit.text(), 0.52),
            "mortality_prime": _safe_float(self._mort_prime_edit.text(), 0.17),
            "mortality_young": _safe_float(self._mort_young_edit.text(), 0.05),
        }


# ===================================================================
# 3. PoliciesPanel  (all 8 public-health policies in one tab)
# ===================================================================

class PoliciesPanel(QWidget):
    """Consolidated policy panel combining all 8 public-health policy
    sections into one scrollable tab.

    Sections (in order):
      Face Mask, Take Temperature, Home Quarantine, Hospital,
      Reduce Contact, Stop Visitant, Vaccine, Medical Policy.
    """

    mask_changed = Signal(bool, float, float)
    temperature_changed = Signal(bool, float, float)
    home_changed = Signal(bool, float)
    home_period_changed = Signal(int)
    trace_changed = Signal(bool)
    level_changed = Signal(bool)          # True  -> level B
    hospital_changed = Signal(bool, float, float)
    contact_changed = Signal(bool, float)
    visit_changed = Signal(bool, float)
    vaccine_requested = Signal(int, bool)
    medical_policy_changed = Signal(bool, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        inner = QWidget()
        inner.setFont(self.font())
        lay = QVBoxLayout(inner)

        # ---- Face Mask ----
        mask_layout = QGridLayout()
        self._mask_check = QCheckBox("Face Mask")
        self._mask_effect_edit = _make_line_edit(0.9)
        self._mask_avail_edit = _make_line_edit(0.9)
        mask_layout.addWidget(self._mask_check, 0, 0, 1, 2)
        mask_layout.addWidget(QLabel("Effect:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        mask_layout.addWidget(self._mask_effect_edit, 1, 1)
        mask_layout.addWidget(QLabel("Available:"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        mask_layout.addWidget(self._mask_avail_edit, 2, 1)
        lay.addLayout(mask_layout)

        self._mask_check.toggled.connect(self._emit_mask)
        self._mask_effect_edit.editingFinished.connect(self._emit_mask)
        self._mask_avail_edit.editingFinished.connect(self._emit_mask)

        # ---- Take Temperature ----
        lay.addWidget(_make_separator())
        temp_layout = QGridLayout()
        self._temp_check = QCheckBox("Take Temperature")
        self._temp_effect_edit = _make_line_edit(0.9)
        self._temp_avail_edit = _make_line_edit(0.9)
        temp_layout.addWidget(self._temp_check, 0, 0, 1, 2)
        temp_layout.addWidget(QLabel("Effect:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        temp_layout.addWidget(self._temp_effect_edit, 1, 1)
        temp_layout.addWidget(QLabel("Available:"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        temp_layout.addWidget(self._temp_avail_edit, 2, 1)
        lay.addLayout(temp_layout)

        self._temp_check.toggled.connect(self._emit_temperature)
        self._temp_effect_edit.editingFinished.connect(self._emit_temperature)
        self._temp_avail_edit.editingFinished.connect(self._emit_temperature)

        # ---- Home Quarantine ----
        lay.addWidget(_make_separator())
        home_layout = QGridLayout()
        self._home_check = QCheckBox("Home Quarantine")
        self._home_avail_edit = _make_line_edit(0.81)
        self._home_period_edit = _make_line_edit(10)
        home_layout.addWidget(self._home_check, 0, 0, 1, 2)
        home_layout.addWidget(QLabel("Available:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        home_layout.addWidget(self._home_avail_edit, 1, 1)
        home_layout.addWidget(QLabel("Period:"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        home_layout.addWidget(self._home_period_edit, 2, 1)

        self._class_a_radio = QRadioButton("Class A")
        self._class_b_radio = QRadioButton("Class B")
        self._class_a_radio.setChecked(True)
        class_row = QHBoxLayout()
        class_row.addWidget(self._class_a_radio)
        class_row.addWidget(self._class_b_radio)
        home_layout.addLayout(class_row, 3, 0, 1, 2)

        self._trace_on_check = QCheckBox("Trace On")
        self._trace_off_check = QCheckBox("Trace Off")
        self._trace_on_check.setChecked(True)
        trace_row = QHBoxLayout()
        trace_row.addWidget(self._trace_on_check)
        trace_row.addWidget(self._trace_off_check)
        home_layout.addLayout(trace_row, 4, 0, 1, 2)
        lay.addLayout(home_layout)

        self._trace_on_check.toggled.connect(self._on_trace_on_toggled)
        self._trace_off_check.toggled.connect(self._on_trace_off_toggled)
        self._home_check.toggled.connect(self._emit_home)
        self._home_avail_edit.editingFinished.connect(self._emit_home)
        self._home_period_edit.editingFinished.connect(self._emit_home_period)
        self._class_a_radio.toggled.connect(self._emit_level)
        self._class_b_radio.toggled.connect(self._emit_level)
        self._trace_on_check.toggled.connect(self._emit_trace)

        # ---- Hospital ----
        lay.addWidget(_make_separator())
        hosp_layout = QGridLayout()
        self._hosp_check = QCheckBox("Hospital")
        self._hosp_check.setChecked(True)
        self._hosp_effect_edit = _make_line_edit(0.5)
        self._hosp_avail_edit = _make_line_edit(0.95)
        hosp_layout.addWidget(self._hosp_check, 0, 0, 1, 2)
        hosp_layout.addWidget(QLabel("Effect:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        hosp_layout.addWidget(self._hosp_effect_edit, 1, 1)
        hosp_layout.addWidget(QLabel("Available:"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        hosp_layout.addWidget(self._hosp_avail_edit, 2, 1)
        lay.addLayout(hosp_layout)

        self._hosp_check.toggled.connect(self._emit_hospital)
        self._hosp_effect_edit.editingFinished.connect(self._emit_hospital)
        self._hosp_avail_edit.editingFinished.connect(self._emit_hospital)

        # ---- Reduce Contact ----
        lay.addWidget(_make_separator())
        contact_layout = QGridLayout()
        self._contact_check = QCheckBox("Reduce Contact")
        self._contact_avail_edit = _make_line_edit(0.9)
        contact_layout.addWidget(self._contact_check, 0, 0, 1, 2)
        contact_layout.addWidget(QLabel("Available:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        contact_layout.addWidget(self._contact_avail_edit, 1, 1)
        lay.addLayout(contact_layout)

        self._contact_check.toggled.connect(self._emit_contact)
        self._contact_avail_edit.editingFinished.connect(self._emit_contact)

        # ---- Stop Visitant ----
        lay.addWidget(_make_separator())
        visit_layout = QGridLayout()
        self._visit_check = QCheckBox("Stop Visitant")
        self._visit_avail_edit = _make_line_edit(0.9)
        visit_layout.addWidget(self._visit_check, 0, 0, 1, 2)
        visit_layout.addWidget(QLabel("Available:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        visit_layout.addWidget(self._visit_avail_edit, 1, 1)
        lay.addLayout(visit_layout)

        self._visit_check.toggled.connect(self._emit_visit)
        self._visit_avail_edit.editingFinished.connect(self._emit_visit)

        # ---- Vaccine ----
        lay.addWidget(_make_separator())
        vac_layout = QGridLayout()
        self._vac_btn = QPushButton("Vaccine")
        self._vac_count_edit = _make_line_edit(2000)
        self._vac_forever_check = QCheckBox("Forever")
        self._vac_period_check = QCheckBox("Period")
        self._vac_period_check.setChecked(True)
        vac_layout.addWidget(self._vac_btn, 0, 0)
        vac_layout.addWidget(self._vac_count_edit, 0, 1)
        vac_layout.addWidget(self._vac_forever_check, 1, 0)
        vac_layout.addWidget(self._vac_period_check, 1, 1)
        lay.addLayout(vac_layout)

        self._vac_forever_check.toggled.connect(self._on_forever_toggled)
        self._vac_period_check.toggled.connect(self._on_period_toggled)
        self._vac_btn.clicked.connect(self._emit_vaccine)

        # ---- Medical Policy ----
        lay.addWidget(_make_separator())
        med_layout = QGridLayout()
        self._med_check = QCheckBox("Medical Policy")
        self._med_effect_edit = _make_line_edit(0.9)
        self._med_avail_edit = _make_line_edit(0.9)
        med_layout.addWidget(self._med_check, 0, 0, 1, 2)
        med_layout.addWidget(QLabel("Effect:"), 1, 0, Qt.AlignRight | Qt.AlignVCenter)
        med_layout.addWidget(self._med_effect_edit, 1, 1)
        med_layout.addWidget(QLabel("Available:"), 2, 0, Qt.AlignRight | Qt.AlignVCenter)
        med_layout.addWidget(self._med_avail_edit, 2, 1)
        lay.addLayout(med_layout)

        self._med_check.toggled.connect(self._emit_medical_policy)
        self._med_effect_edit.editingFinished.connect(self._emit_medical_policy)
        self._med_avail_edit.editingFinished.connect(self._emit_medical_policy)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # -- internal signal emitters ----------------------------------------

    def _emit_mask(self) -> None:
        self.mask_changed.emit(
            self._mask_check.isChecked(),
            _safe_float(self._mask_effect_edit.text(), 0.9),
            _safe_float(self._mask_avail_edit.text(), 0.9),
        )

    def _emit_temperature(self) -> None:
        self.temperature_changed.emit(
            self._temp_check.isChecked(),
            _safe_float(self._temp_effect_edit.text(), 0.9),
            _safe_float(self._temp_avail_edit.text(), 0.9),
        )

    def _emit_home(self) -> None:
        self.home_changed.emit(
            self._home_check.isChecked(),
            _safe_float(self._home_avail_edit.text(), 0.81),
        )

    def _emit_home_period(self) -> None:
        self.home_period_changed.emit(
            _safe_int(self._home_period_edit.text(), 10),
        )

    def _emit_trace(self) -> None:
        self.trace_changed.emit(self._trace_on_check.isChecked())

    def _emit_level(self) -> None:
        self.level_changed.emit(self._class_b_radio.isChecked())

    def _emit_hospital(self) -> None:
        self.hospital_changed.emit(
            self._hosp_check.isChecked(),
            _safe_float(self._hosp_effect_edit.text(), 0.5),
            _safe_float(self._hosp_avail_edit.text(), 0.95),
        )

    def _emit_contact(self) -> None:
        self.contact_changed.emit(
            self._contact_check.isChecked(),
            _safe_float(self._contact_avail_edit.text(), 0.9),
        )

    def _emit_visit(self) -> None:
        self.visit_changed.emit(
            self._visit_check.isChecked(),
            _safe_float(self._visit_avail_edit.text(), 0.9),
        )

    def _emit_vaccine(self) -> None:
        self.vaccine_requested.emit(
            _safe_int(self._vac_count_edit.text(), 2000),
            self._vac_forever_check.isChecked(),
        )

    def _emit_medical_policy(self) -> None:
        self.medical_policy_changed.emit(
            self._med_check.isChecked(),
            _safe_float(self._med_effect_edit.text(), 0.9),
            _safe_float(self._med_avail_edit.text(), 0.9),
        )

    # -- keep trace checkboxes mutually exclusive ------------------------

    def _on_trace_on_toggled(self, checked: bool) -> None:
        if checked:
            self._trace_off_check.setChecked(False)

    def _on_trace_off_toggled(self, checked: bool) -> None:
        if checked:
            self._trace_on_check.setChecked(False)

    # -- keep vaccine checkboxes mutually exclusive ----------------------

    def _on_forever_toggled(self, checked: bool) -> None:
        if checked:
            self._vac_period_check.setChecked(False)

    def _on_period_toggled(self, checked: bool) -> None:
        if checked:
            self._vac_forever_check.setChecked(False)

    # -- public convenience accessors ------------------------------------

    def get_home_period(self) -> int:
        return _safe_int(self._home_period_edit.text(), 10)
