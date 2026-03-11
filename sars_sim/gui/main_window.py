"""Main window for the SARS simulation application.

Ties together all GUI components: lattice views, charts, parameter panels,
and control buttons into a single main window matching the original Delphi
application layout (828x735).
"""

import time
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QTabWidget, QHBoxLayout,
    QVBoxLayout, QPushButton, QLineEdit, QLabel, QRadioButton, QCheckBox,
    QTextEdit, QProgressBar, QApplication, QGroupBox, QSplitter, QMessageBox)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QButtonGroup

from ..models import SimulationParams, SimulationData, MeasureData, StateEnum
from ..world import WorldManager
from ..engine import SimulationEngine
from ..policies import PolicyManager
from ..statistics import StatisticsTracker, FileOutput
from .lattice_view import MacroLatticeView, MicroLatticeView
from .charts import (AccumulativeChart, DailyChart, NotifiableChart,
    InfectiveChart, AccumulativeQuarantineChart, DailyQuarantineChart)
from .controls import (ParameterPanel, DiseasePanel, PoliciesPanel)
from .status_bar import SimulationStatusBar


class SARSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulating SARS: Small-World Epidemiological Modeling "
                            "and Public Health Policy Assessments, Version 2")
        self.resize(1280, 1100)

        # Data objects (created with defaults, updated from GUI before init)
        self.params = SimulationParams()
        self.sim_data = SimulationData(self.params)
        self.measure = MeasureData()

        # Managers
        self.world_mgr = WorldManager(self.params, self.sim_data)
        self.engine = SimulationEngine(self.params, self.sim_data)
        self.policy_mgr = PolicyManager(self.params, self.sim_data)
        self.stats = StatisticsTracker(self.sim_data, self.measure)
        self.file_output = FileOutput()

        # Simulation timer (matches original TTimer)
        self.sim_timer = QTimer(self)
        self.sim_timer.setInterval(100)  # faster than original 1000ms for responsiveness
        self.sim_timer.timeout.connect(self._on_timer_tick)
        self._run_count = 0  # counter for runSimulation

        self.serial = int(time.time())

        self._build_ui()
        self._connect_signals()
        self._disable_components()  # Start with most controls disabled

    # ------------------------------------------------------------------ #
    #                          UI Construction                            #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        """Build the complete GUI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top area: splitter with left tabs and right tabs
        splitter = QSplitter(Qt.Horizontal)

        # Left tab widget (lattice views + charts)
        self.left_tabs = QTabWidget()
        # Create all views and charts
        self.macro_view = MacroLatticeView(self.sim_data)
        self.micro_view = MicroLatticeView(self.sim_data)
        self.chart_accumulative = AccumulativeChart()
        self.chart_daily = DailyChart()
        self.chart_notifiable = NotifiableChart()
        self.chart_infective = InfectiveChart()
        self.chart_acc_quarantine = AccumulativeQuarantineChart()
        self.chart_daily_quarantine = DailyQuarantineChart()

        self.left_tabs.addTab(self.macro_view, "Macro View")
        self.left_tabs.addTab(self.micro_view, "Micro View")
        self.left_tabs.addTab(self.chart_accumulative, "Accumulative")
        self.left_tabs.addTab(self.chart_daily, "Daily")
        self.left_tabs.addTab(self.chart_notifiable, "Notifiable")
        self.left_tabs.addTab(self.chart_infective, "Infective")
        self.left_tabs.addTab(self.chart_acc_quarantine, "Acc. Quarantine")
        self.left_tabs.addTab(self.chart_daily_quarantine, "Daily Quarantine")

        # Right tab widget (parameter panels)
        self.right_tabs = QTabWidget()
        self.panel_parameter = ParameterPanel()
        self.panel_disease = DiseasePanel()
        self.panel_policies = PoliciesPanel()

        self.right_tabs.addTab(self.panel_parameter, "Parameters")
        self.right_tabs.addTab(self.panel_disease, "Disease")
        self.right_tabs.addTab(self.panel_policies, "Policies")
        self.right_tabs.setCurrentIndex(1)  # Disease tab default

        splitter.addWidget(self.left_tabs)
        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        self.right_tabs.setMinimumWidth(350)

        main_layout.addWidget(splitter, stretch=1)

        # Bottom control area
        self._build_control_bar(main_layout)

        # Status bar
        self.status_bar = SimulationStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_control_bar(self, parent_layout):
        """Build the bottom control buttons and fields."""
        # ---- Row 1: Import, Run, Stop controls ----
        row1 = QHBoxLayout()

        self.btn_import = QPushButton("Import")
        self.edit_new_case = QLineEdit("3")
        self.edit_new_case.setMaximumWidth(50)
        self.lbl_person = QLabel("person")
        self.radio_exposed = QRadioButton("Exposed")
        self.radio_exposed.setChecked(True)
        self.radio_infective = QRadioButton("Infective")
        self._case_type_group = QButtonGroup(self)
        self._case_type_group.addButton(self.radio_exposed)
        self._case_type_group.addButton(self.radio_infective)
        self.chk_super = QCheckBox("Super")

        self.btn_run = QPushButton("Run")
        self.btn_stop = QPushButton("Stop")
        self.edit_period = QLineEdit("1")
        self.edit_period.setMaximumWidth(40)
        self.lbl_day = QLabel("day")
        self.edit_gossip = QLineEdit("1")
        self.edit_gossip.setMaximumWidth(40)
        self.lbl_times = QLabel("times")

        for w in [self.btn_import, self.edit_new_case, self.lbl_person,
                   self.radio_exposed, self.radio_infective, self.chk_super,
                   self.btn_run, self.btn_stop, self.edit_period, self.lbl_day,
                   self.edit_gossip, self.lbl_times]:
            row1.addWidget(w)
        row1.addStretch()
        parent_layout.addLayout(row1)

        # ---- Row 2: Initial, Finish, Exit, progress ----
        row2 = QHBoxLayout()
        self.btn_initial = QPushButton("Initial")
        self.btn_finish = QPushButton("Finish")
        self.btn_exit = QPushButton("Exit")
        self.btn_print = QPushButton("Print")
        self.btn_save = QPushButton("Save")
        self.edit_filename = QLineEdit("Accumulative Graph")
        self.edit_filename.setMaximumWidth(150)
        self.lbl_filename = QLabel("File")

        self.edit_avg_count = QLineEdit("0")
        self.edit_avg_count.setMaximumWidth(40)
        self.edit_avg_count.setReadOnly(True)
        self.lbl_avg_count = QLabel("avg.")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)

        self.memo_status = QTextEdit()
        self.memo_status.setMaximumHeight(60)
        self.memo_status.setMaximumWidth(300)
        self.memo_status.setReadOnly(True)

        for w in [self.btn_initial, self.btn_finish, self.btn_exit,
                   self.btn_print, self.btn_save, self.edit_filename, self.lbl_filename,
                   self.edit_avg_count, self.lbl_avg_count,
                   self.progress_bar, self.memo_status]:
            row2.addWidget(w)
        row2.addStretch()
        parent_layout.addLayout(row2)

        # ---- Row 3: People state detail + infected stats ----
        row3 = QHBoxLayout()
        self.lbl_attr = QLabel("Attr.")
        self.edit_people_state = QLineEdit()
        self.edit_people_state.setReadOnly(True)
        self.edit_people_state.setMinimumWidth(300)

        self.lbl_infected = QLabel("Infected:")
        self.edit_infected_normal = QLineEdit("0")
        self.edit_infected_normal.setMaximumWidth(60)
        self.edit_infected_normal.setReadOnly(True)
        self.lbl_normal = QLabel("Normal")
        self.edit_infected_hospital = QLineEdit("0")
        self.edit_infected_hospital.setMaximumWidth(60)
        self.edit_infected_hospital.setReadOnly(True)
        self.lbl_hospital = QLabel("Hospital")
        self.edit_normal_rate = QLineEdit()
        self.edit_normal_rate.setMaximumWidth(60)
        self.edit_normal_rate.setReadOnly(True)
        self.edit_hospital_rate = QLineEdit()
        self.edit_hospital_rate.setMaximumWidth(60)
        self.edit_hospital_rate.setReadOnly(True)

        for w in [self.lbl_attr, self.edit_people_state,
                   self.lbl_infected, self.edit_infected_normal, self.lbl_normal,
                   self.edit_infected_hospital, self.lbl_hospital,
                   self.edit_normal_rate, self.edit_hospital_rate]:
            row3.addWidget(w)
        row3.addStretch()
        parent_layout.addLayout(row3)

    # ------------------------------------------------------------------ #
    #                         Signal Connections                          #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        """Connect all signals between components."""
        # Buttons
        self.btn_initial.clicked.connect(self._on_initial)
        self.btn_import.clicked.connect(self._on_import_cases)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_finish.clicked.connect(self._on_finish)
        self.btn_exit.clicked.connect(self.close)
        self.btn_print.clicked.connect(self._on_print)
        self.btn_save.clicked.connect(self._on_save)

        # Lattice views
        self.macro_view.cell_clicked.connect(self._on_lattice_click)
        self.macro_view.cell_hovered.connect(self._on_lattice_hover)
        self.micro_view.cell_hovered.connect(self._on_lattice_hover)
        self.micro_view.person_detail.connect(self._on_person_detail)

        # Policy panels
        self.panel_policies.mask_changed.connect(self._on_mask_changed)
        self.panel_policies.temperature_changed.connect(self._on_temp_changed)
        self.panel_policies.home_changed.connect(self._on_home_changed)
        self.panel_policies.home_period_changed.connect(self._on_home_period_changed)
        self.panel_policies.hospital_changed.connect(self._on_hospital_changed)
        self.panel_policies.trace_changed.connect(self._on_trace_changed)
        self.panel_policies.level_changed.connect(self._on_level_changed)
        self.panel_policies.contact_changed.connect(self._on_contact_changed)
        self.panel_policies.visit_changed.connect(self._on_visit_changed)
        self.panel_policies.vaccine_requested.connect(self._on_vaccine)
        self.panel_policies.medical_policy_changed.connect(self._on_medical_policy_changed)

        # Small world
        self.panel_parameter.calculate_requested.connect(self._on_small_world)

        # Gossip change
        self.edit_gossip.editingFinished.connect(self._on_gossip_changed)

        # Tab changes -- pause/unpause chart rendering
        self.left_tabs.currentChanged.connect(self._on_left_tab_changed)

    # ------------------------------------------------------------------ #
    #                       Parameter Synchronisation                     #
    # ------------------------------------------------------------------ #

    def _update_params_from_gui(self):
        """Read all GUI fields and update self.params."""
        pop = self.panel_parameter.get_params()
        dis = self.panel_disease.get_params()

        self.params.max_population = pop.get('max_population', self.params.max_population)
        self.params.max_agent = pop.get('max_agent', self.params.max_agent)
        self.params.old_man_rate = pop.get('old_man_rate', self.params.old_man_rate)
        self.params.young_man_rate = pop.get('young_man_rate', self.params.young_man_rate)
        self.params.super_rate = pop.get('super_rate', self.params.super_rate)
        self.params.gossip_fixed = pop.get('gossip_fixed', self.params.gossip_fixed)

        self.params.transmission_prob = dis.get('transmission_prob', self.params.transmission_prob)
        self.params.immune_prob = dis.get('immune_prob', self.params.immune_prob)
        self.params.exposed_period = dis.get('exposed_period', self.params.exposed_period)
        self.params.symptomatic_period = dis.get('symptomatic_period', self.params.symptomatic_period)
        self.params.infective_period = dis.get('infective_period', self.params.infective_period)
        self.params.recovered_period = dis.get('recovered_period', self.params.recovered_period)
        self.params.immune_period = dis.get('immune_period', self.params.immune_period)
        self.params.detect_rate = dis.get('detect_rate', self.params.detect_rate)
        self.params.mortality_old = dis.get('mortality_old', self.params.mortality_old)
        self.params.mortality_prime = dis.get('mortality_prime', self.params.mortality_prime)
        self.params.mortality_young = dis.get('mortality_young', self.params.mortality_young)

        # Gossip from control bar
        try:
            self.params.gossip_steps = int(self.edit_gossip.text())
        except ValueError:
            pass

    # ------------------------------------------------------------------ #
    #                          Action Handlers                            #
    # ------------------------------------------------------------------ #

    def _on_initial(self):
        """Initialize button handler -- sets up a new simulation run."""
        self._update_params_from_gui()

        # Recreate data if population size changed
        if (self.sim_data.N != self.params.max_population or
                self.sim_data.M != self.params.max_agent):
            self.sim_data = SimulationData(self.params)
            self.world_mgr = WorldManager(self.params, self.sim_data)
            self.engine = SimulationEngine(self.params, self.sim_data)
            self.policy_mgr = PolicyManager(self.params, self.sim_data)
            self.stats = StatisticsTracker(self.sim_data, self.measure)
            self.macro_view.data = self.sim_data
            self.micro_view.data = self.sim_data

        # Init output files
        self.serial = int(time.time())
        output_flags = self.panel_parameter.get_output_flags()
        if not self.file_output.init_files(self.serial, output_flags):
            QMessageBox.warning(self, "Error", "Unable to create output directory")
            return

        self.measure.size += 1
        self.edit_avg_count.setText(str(self.measure.size))

        # Reset
        self.sim_data.reset()
        self.sim_data.infected_by_hospital = 0
        self.sim_data.infected_by_normal = 0

        self.stats.backup()

        # Init world and society
        self.world_mgr.init_world()
        self.world_mgr.init_society()

        # Init policies from current checkbox states
        self._apply_all_policies()

        # Generate agents
        self.memo_status.clear()
        self.memo_status.append("Generating agents...")
        QApplication.processEvents()

        dist, actual_pop = self.world_mgr.generate_distributed_agents()

        # Lock Max Population to actual value
        # 設計說明：此處刻意將 max_population 縮減為實際擁有分身的人數
        # (actual_pop)，而非保留原始 N。由於 generate_distributed_agents
        # 透過 rng.choice(N, actual_n, replace=False) 隨機挑選受分配者，
        # 索引 >= actual_pop 的個體雖可能擁有分身，但後續透過 GUI 切換
        # 政策時（policies.py 中以 params.max_population 為上界），這些
        # 個體不會被套用新政策。此為刻意設計：僅對「前 actual_pop 個」
        # 索引的個體動態更新政策，其餘個體維持初始化時的政策狀態不變。
        self.panel_parameter.set_max_population(actual_pop)
        self.panel_parameter.set_max_population_readonly(True)
        self.params.max_population = actual_pop

        # Show distribution and actual population
        total_cells = self.params.max_height * self.params.max_width
        dist_text = f"Lattice cells = {total_cells:,}\n"
        dist_text += f"Actual population = {actual_pop:,}\n\n"
        for i in range(self.params.max_agent):
            dist_text += f"{i + 1} identity: {dist[i]:,}\n"
        self.panel_parameter.set_distribution_text(dist_text)

        # Update charts
        self._clear_charts()
        chart_data = self.stats.update_graphic_data()
        self._update_charts(chart_data)
        self._update_infected_display()
        self.file_output.write_daily(
            self.sim_data.day, self.sim_data.statistic, self.sim_data.old_statistic,
            self.stats.get_state_counts(), self.measure)

        # Draw
        self.macro_view.update_image()
        self.micro_view.update_image()
        self.progress_bar.setMaximum(self.params.max_population)

        self.memo_status.append("O.K.")
        self._enable_components()
        self._show_main_page()

    def _on_import_cases(self):
        """Import new cases."""
        try:
            num = int(self.edit_new_case.text())
        except ValueError:
            return

        as_exposed = self.radio_exposed.isChecked()
        is_super = self.chk_super.isChecked()
        auto_run = self.panel_parameter.is_auto_run()

        if not auto_run:
            self.sim_data.day += 1

        self.stats.backup()

        self.engine.import_cases(num, as_exposed, is_super)

        if auto_run:
            self.engine.change_society()

        chart_data = self.stats.update_graphic_data()
        self._update_charts(chart_data)
        self._update_infected_display()
        self.file_output.write_daily(
            self.sim_data.day, self.sim_data.statistic, self.sim_data.old_statistic,
            self.stats.get_state_counts(), self.measure)

        self.file_output.write_action(
            f"[Action]: Day {self.sim_data.day}, imported {num} "
            f"{'super ' if is_super else ''}{'exposed' if as_exposed else 'infective'} case(s).")

        self._show_result()

    def _on_run(self):
        """Start the simulation timer."""
        self._run_count = 0
        self.sim_timer.start()

    def _on_stop(self):
        """Stop the simulation timer."""
        self.sim_timer.stop()

    def _on_timer_tick(self):
        """Timer callback -- runs one simulation step."""
        try:
            period = int(self.edit_period.text())
        except ValueError:
            period = 1

        if self._run_count == 0:
            self.file_output.write_action(
                f"[Action]: Day {self.sim_data.day}, starting simulation for {period} day(s), "
                f"{'fixed' if self.params.gossip_fixed else 'floating'} gossip "
                f"{self.params.gossip_steps} step(s).")

        if self._run_count < period:
            self.stats.backup()
            self.engine.change_society()
            chart_data = self.stats.update_graphic_data()
            self._update_charts(chart_data)
            self._update_infected_display()
            self.file_output.write_daily(
                self.sim_data.day, self.sim_data.statistic, self.sim_data.old_statistic,
                self.stats.get_state_counts(), self.measure)
            # Update lattice views every day so user can see progression.
            # Use incremental color update for dirty people only.
            self.world_mgr.update_dirty_colors(self.engine.dirty_pids)
            self.macro_view.update_image()
            self.micro_view.update_image()
            self._run_count += 1
        else:
            self._run_count = 0
            self._show_result()
            self.sim_timer.stop()

    def _on_finish(self):
        """Finish simulation, close files."""
        self._disable_components()
        self.file_output.close(self.measure.size)

        # Restore Max Population to editable with default value
        self.panel_parameter.set_max_population(100000)
        self.panel_parameter.set_max_population_readonly(False)

        self.memo_status.append("Finish")

    def _on_print(self):
        """Print current chart (placeholder)."""
        pass  # TODO: implement printing if needed

    def _on_save(self):
        """Save current chart."""
        idx = self.left_tabs.currentIndex()
        name = self.edit_filename.text()
        day = self.sim_data.day
        fname = f"{name}-{day}.png"
        charts = [None, None, self.chart_accumulative, self.chart_daily,
                  self.chart_notifiable, self.chart_infective,
                  self.chart_acc_quarantine, self.chart_daily_quarantine]
        if 2 <= idx < len(charts) and charts[idx]:
            charts[idx].save_image(fname)

    def _on_lattice_click(self, x, y):
        """Switch to micro view centered on clicked cell."""
        self.micro_view.set_center(x, y)
        self.left_tabs.setCurrentIndex(1)

    def _on_lattice_hover(self, x, y, pid):
        """Update status bar on hover."""
        d = self.sim_data
        if 0 <= pid < d.N:
            agent_no = d.world_agent_no[y, x] if 0 <= y < d.H and 0 <= x < d.W else 0
            is_home = bool(d.agent_home[pid, agent_no]) if agent_no >= 0 else False
            self.status_bar.update_cell_info(
                x, y, pid, int(d.people_count[pid]),
                int(d.people_state[pid]),
                bool(d.people_isolated[pid]),
                bool(d.people_quarantined[pid]),
                int(agent_no), is_home)

    def _on_person_detail(self, detail):
        """Show person detail in the edit field."""
        self.edit_people_state.setText(detail)

    # ------------------------------------------------------------------ #
    #                         Policy Handlers                             #
    # ------------------------------------------------------------------ #

    def _on_mask_changed(self, enabled, effect, available):
        self.params.mask_effect = effect
        self.params.mask_available = available
        self.policy_mgr.apply_mask_policy(enabled, available)

    def _on_temp_changed(self, enabled, effect, available):
        self.params.temp_effect = effect
        self.params.temp_available = available
        self.policy_mgr.apply_temperature_policy(enabled, available)

    def _on_home_changed(self, enabled, available):
        self.params.home_available = available
        self.policy_mgr.apply_home_quarantine_policy(enabled, available)

    def _on_home_period_changed(self, period):
        self.params.quarantine_period = period

    def _on_hospital_changed(self, enabled, effect, available):
        self.params.hospital_effect = effect
        self.params.hospital_available = available
        self.policy_mgr.apply_hospital_policy(enabled, available)

    def _on_trace_changed(self, trace_on):
        self.params.trace_on = trace_on
        self.engine.params.trace_on = trace_on

    def _on_level_changed(self, is_level_b):
        self.params.isolated_level_b = is_level_b
        self.engine.params.isolated_level_b = is_level_b

    def _on_contact_changed(self, enabled, available):
        self.params.contact_available = available
        self.policy_mgr.apply_contact_reduction(enabled, available)

    def _on_visit_changed(self, enabled, available):
        self.params.visit_available = available
        self.policy_mgr.apply_visit_restriction(enabled, available)

    def _on_vaccine(self, count, forever):
        self.policy_mgr.apply_vaccine(count, forever)
        self._show_result()

    def _on_medical_policy_changed(self, enabled, effect, available):
        self.params.medical_policy_effect = effect
        self.params.medical_policy_available = available
        self.engine.medical_policy_enabled = enabled
        self.policy_mgr.apply_medical_policy_toggle(enabled)

    def _on_gossip_changed(self):
        try:
            self.params.gossip_steps = int(self.edit_gossip.text())
        except ValueError:
            pass

    def _on_small_world(self, count):
        """Run small world degree calculation using BFS."""
        import random as rng

        d = self.sim_data
        N = self.params.max_population
        H, W = d.H, d.W

        # Guard: agents must be placed first
        if not np.any(d.people_count > 0):
            QMessageBox.warning(self, "Small World",
                                "Please click Initial first to create the society.")
            return

        result_text = ""
        avg_degree = 0.0

        self.panel_parameter.set_result_text("Calculating...")
        QApplication.processEvents()

        for trial in range(count):
            # Pick two distinct random people with agents
            while True:
                id1 = rng.randrange(N)
                if d.people_count[id1] > 0:
                    break
            while True:
                id2 = rng.randrange(N)
                if d.people_count[id2] > 0 and id2 != id1:
                    break

            result_text += f"{trial}:\nID1 = {id1}\n"
            for j in range(d.people_count[id1]):
                result_text += f"(x, y) = {d.agent_loc_x[id1, j]}, {d.agent_loc_y[id1, j]}\n"
            result_text += f"ID2 = {id2}\n"
            for j in range(d.people_count[id2]):
                result_text += f"(x, y) = {d.agent_loc_x[id2, j]}, {d.agent_loc_y[id2, j]}\n"

            # BFS for degree of separation
            relations = set([id1])
            found = False
            degree = 0

            while not found:
                degree += 1
                new_neighbors = set()
                for pid in relations:
                    for k in range(d.people_count[pid]):
                        mid_y = d.agent_loc_y[pid, k]
                        mid_x = d.agent_loc_x[pid, k]
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if dy == 0 and dx == 0:
                                    continue
                                ny = (mid_y + dy + H) % H
                                nx_ = (mid_x + dx + W) % W
                                nid = d.world_people_id[ny, nx_]
                                if nid == id2:
                                    found = True
                                    break
                                new_neighbors.add(int(nid))
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    relations = new_neighbors

            avg_degree += degree
            result_text += f"degree = {degree}\n"

            # Update progress every 10 trials to keep GUI responsive
            if (trial + 1) % 10 == 0:
                self.panel_parameter.set_result_text(
                    f"Calculating... {trial + 1}/{count}\n")
                QApplication.processEvents()

        avg_degree /= count
        result_text += f"\navg. degree = {avg_degree:.2f}\n"
        self.panel_parameter.set_result_text(result_text)

    # ------------------------------------------------------------------ #
    #                         Helper Methods                              #
    # ------------------------------------------------------------------ #

    def _on_left_tab_changed(self, index):
        """Pause rendering on hidden charts, unpause the visible one."""
        charts = {
            2: self.chart_accumulative,
            3: self.chart_daily,
            4: self.chart_notifiable,
            5: self.chart_infective,
            6: self.chart_acc_quarantine,
            7: self.chart_daily_quarantine,
        }
        for tab_idx, chart in charts.items():
            chart.set_render_paused(tab_idx != index)

    def _apply_all_policies(self):
        """Apply all policies from current GUI state. Called during init.

        Mirrors C++ ``initPolicies`` which re-fires all checkbox handlers
        so the new population receives per-person policy flags.
        """
        self.panel_policies._emit_mask()
        self.panel_policies._emit_temperature()
        self.panel_policies._emit_home()
        self.panel_policies._emit_trace()
        self.panel_policies._emit_level()
        self.panel_policies._emit_hospital()
        self.panel_policies._emit_contact()
        self.panel_policies._emit_visit()
        self.panel_policies._emit_medical_policy()

    def _show_result(self):
        """Refresh display after simulation or import."""
        self.progress_bar.setValue(0)
        self.world_mgr.reset_world_colors()
        self.macro_view.update_image()
        self.micro_view.update_image()
        self._show_status()

    def _show_status(self):
        """Show summary in memo and status bar."""
        counts = self.stats.get_state_counts()
        d = self.sim_data

        self.memo_status.clear()
        self.memo_status.append(f"day(s) = {d.day}")
        self.memo_status.append(
            f"(S) = {counts.get(0, 0)}, (E) = {counts.get(1, 0)}, "
            f"(I) = {counts.get(2, 0)}, (R) = {counts.get(3, 0)}, "
            f"(M) = {counts.get(4, 0)}, (D) = {counts.get(5, 0)}")

        self.status_bar.update_day_info(d.day)
        self.status_bar.update_summary(
            int(d.statistic[StateEnum.RECOVERED]),
            int(d.statistic[StateEnum.DIED]))

    def _update_charts(self, chart_data):
        """Update all charts with new data."""
        day = self.sim_data.day

        self.chart_accumulative.add_data_point(day, chart_data['accumulative'])
        self.chart_daily.add_data_point(day, chart_data['daily'])
        self.chart_notifiable.add_data_point(
            day, chart_data['isolated_daily'],
            chart_data['died_daily'],
            chart_data['isolated_avg'])
        self.chart_infective.add_data_point(
            day, chart_data['infected_by_normal'],
            chart_data['infected_by_hospital'])
        self.chart_acc_quarantine.add_data_point(day, chart_data['quarantined_total'])
        self.chart_daily_quarantine.add_data_point(day, chart_data['quarantined_daily'])

    def _update_infected_display(self):
        """Update infected-by stats display."""
        normal = self.sim_data.infected_by_normal
        hospital = self.sim_data.infected_by_hospital
        total = normal + hospital
        self.edit_infected_normal.setText(str(normal))
        self.edit_infected_hospital.setText(str(hospital))
        if total > 0:
            self.edit_normal_rate.setText(f"{normal / total * 100:.1f}%")
            self.edit_hospital_rate.setText(f"{hospital / total * 100:.1f}%")

    def _clear_charts(self):
        """Clear all chart data."""
        self.chart_accumulative.clear()
        self.chart_daily.clear()
        self.chart_notifiable.clear()
        self.chart_infective.clear()
        self.chart_acc_quarantine.clear()
        self.chart_daily_quarantine.clear()

    def _show_main_page(self):
        """Show main tabs (macro view + disease panel)."""
        self.left_tabs.setCurrentIndex(0)
        self.right_tabs.setCurrentIndex(1)

    def _enable_components(self):
        """Enable simulation controls after initialization."""
        self.btn_initial.setEnabled(False)
        self.btn_exit.setEnabled(False)
        self.btn_import.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_finish.setEnabled(True)
        self.btn_print.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.edit_new_case.setEnabled(True)
        self.edit_period.setEnabled(True)
        self.edit_gossip.setEnabled(True)
        self.chk_super.setEnabled(True)
        self.radio_exposed.setEnabled(True)
        self.radio_infective.setEnabled(True)

    def _disable_components(self):
        """Disable simulation controls."""
        self.btn_initial.setEnabled(True)
        self.btn_exit.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_finish.setEnabled(False)
        self.btn_print.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.edit_new_case.setEnabled(False)
        self.edit_period.setEnabled(False)
        self.edit_gossip.setEnabled(False)
        self.chk_super.setEnabled(False)
        self.radio_exposed.setEnabled(False)
        self.radio_infective.setEnabled(False)
        self.memo_status.clear()
