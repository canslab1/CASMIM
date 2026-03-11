"""Chart widgets for the SARS simulation using pyqtgraph.

Provides six chart widget classes, each wrapping a pyqtgraph.PlotWidget
with x-axis labeled "day(s)" and y-axis labeled "person(s)".

Colors match the original C++ Builder application:
  Exposed=Yellow, Infective=Red, Recovered=Blue, Immune=White,
  Died=Gray, Isolated=Green.
"""

import pyqtgraph as pg
from pyqtgraph import exporters
from PySide6.QtWidgets import QWidget, QVBoxLayout


def _setup_large_legend(plot_widget):
    """Configure the legend with a larger, clearer font."""
    legend = plot_widget.addLegend()
    legend.setLabelTextSize('14pt')
    return legend


# Original Delphi TColor constants (BGR) -> RGB tuples
_CLR_YELLOW = (255, 255, 0)
_CLR_RED = (255, 0, 0)
_CLR_BLUE = (0, 0, 255)
_CLR_WHITE = (200, 200, 200)   # Use light gray since white is invisible on white bg
_CLR_GRAY = (128, 128, 128)
_CLR_GREEN = (0, 128, 0)
_CLR_NAVY = (0, 0, 128)


class AccumulativeChart(QWidget):
    """Cumulative counts for Exposed, Infective, Recovered, Immune, Died, Isolated."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'day(s)')
        self.plot_widget.setLabel('left', 'person(s)')
        _setup_large_legend(self.plot_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self._days = []
        self._exposed = []
        self._infective = []
        self._recovered = []
        self._immune = []
        self._died = []
        self._isolated = []
        self._render_paused = False

        self._curve_exposed = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_YELLOW, width=2), name='Exposed',
        )
        self._curve_infective = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_RED, width=2), name='Infective',
        )
        self._curve_recovered = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_BLUE, width=2), name='Recovered',
        )
        self._curve_immune = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_WHITE, width=2), name='Immune',
        )
        self._curve_died = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_GRAY, width=2), name='Died',
        )
        self._curve_isolated = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_GREEN, width=2), name='Isolated',
        )

    def add_data_point(self, day, accumulative_dict):
        """Append a data point for the given *day*.

        Parameters
        ----------
        day : int
            Simulation day number.
        accumulative_dict : dict
            Mapping of StateEnum values (1..6) to cumulative counts.
            1=Exposed, 2=Infective, 3=Recovered, 4=Immune, 5=Died, 6=Isolated.
        """
        self._days.append(day)
        self._exposed.append(accumulative_dict.get(1, 0))
        self._infective.append(accumulative_dict.get(2, 0))
        self._recovered.append(accumulative_dict.get(3, 0))
        self._immune.append(accumulative_dict.get(4, 0))
        self._died.append(accumulative_dict.get(5, 0))
        self._isolated.append(accumulative_dict.get(6, 0))

        if not self._render_paused:
            self._render()

    def _render(self):
        """Push buffered data to pyqtgraph curves."""
        self._curve_exposed.setData(self._days, self._exposed)
        self._curve_infective.setData(self._days, self._infective)
        self._curve_recovered.setData(self._days, self._recovered)
        self._curve_immune.setData(self._days, self._immune)
        self._curve_died.setData(self._days, self._died)
        self._curve_isolated.setData(self._days, self._isolated)

    def set_render_paused(self, paused):
        """Pause or resume rendering.  On unpause, flush buffered data."""
        self._render_paused = paused
        if not paused and self._days:
            self._render()

    def clear(self):
        """Reset all data and clear the plot."""
        self._days.clear()
        self._exposed.clear()
        self._infective.clear()
        self._recovered.clear()
        self._immune.clear()
        self._died.clear()
        self._isolated.clear()

        self._curve_exposed.setData([], [])
        self._curve_infective.setData([], [])
        self._curve_recovered.setData([], [])
        self._curve_immune.setData([], [])
        self._curve_died.setData([], [])
        self._curve_isolated.setData([], [])

    def save_image(self, filepath):
        """Export the chart as an image to *filepath*."""
        exporter = exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(filepath)


class DailyChart(QWidget):
    """Daily deltas: Infective as bars, others as lines.
    Includes Isolated series matching original C++."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'day(s)')
        self.plot_widget.setLabel('left', 'person(s)')
        _setup_large_legend(self.plot_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self._days = []
        self._exposed = []
        self._infective = []
        self._recovered = []
        self._immune = []
        self._died = []
        self._isolated = []
        self._render_paused = False

        # Bar series for infective daily
        self._bar_infective = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush=_CLR_RED, name='Infective',
        )
        self.plot_widget.addItem(self._bar_infective)

        # Line series for the remaining states
        self._curve_exposed = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_YELLOW, width=2), name='Exposed',
        )
        self._curve_recovered = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_BLUE, width=2), name='Recovered',
        )
        self._curve_immune = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_WHITE, width=2), name='Immune',
        )
        self._curve_died = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_GRAY, width=2), name='Died',
        )
        self._curve_isolated = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_GREEN, width=2), name='Isolated',
        )

    def add_data_point(self, day, daily_dict):
        """Append a data point for the given *day*.

        Parameters
        ----------
        day : int
            Simulation day number.
        daily_dict : dict
            Mapping of StateEnum values (1..6) to daily delta counts.
            1=Exposed, 2=Infective, 3=Recovered, 4=Immune, 5=Died, 6=Isolated.
        """
        self._days.append(day)
        self._exposed.append(daily_dict.get(1, 0))
        self._infective.append(daily_dict.get(2, 0))
        self._recovered.append(daily_dict.get(3, 0))
        self._immune.append(daily_dict.get(4, 0))
        self._died.append(daily_dict.get(5, 0))
        self._isolated.append(daily_dict.get(6, 0))

        if not self._render_paused:
            self._render()

    def _render(self):
        """Push buffered data to pyqtgraph curves/bars."""
        self._bar_infective.setOpts(x=self._days, height=self._infective)
        self._curve_exposed.setData(self._days, self._exposed)
        self._curve_recovered.setData(self._days, self._recovered)
        self._curve_immune.setData(self._days, self._immune)
        self._curve_died.setData(self._days, self._died)
        self._curve_isolated.setData(self._days, self._isolated)

    def set_render_paused(self, paused):
        self._render_paused = paused
        if not paused and self._days:
            self._render()

    def clear(self):
        """Reset all data and clear the plot."""
        self._days.clear()
        self._exposed.clear()
        self._infective.clear()
        self._recovered.clear()
        self._immune.clear()
        self._died.clear()
        self._isolated.clear()

        self._bar_infective.setOpts(x=[], height=[])
        self._curve_exposed.setData([], [])
        self._curve_recovered.setData([], [])
        self._curve_immune.setData([], [])
        self._curve_died.setData([], [])
        self._curve_isolated.setData([], [])

    def save_image(self, filepath):
        """Export the chart as an image to *filepath*."""
        exporter = exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(filepath)


class NotifiableChart(QWidget):
    """Notifiable disease chart: Isolated (green) and Died (red) as stacked bars,
    Average as a navy line. Matches original C++ mbStacked bar mode."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'day(s)')
        self.plot_widget.setLabel('left', 'person(s)')
        self.plot_widget.addLegend()

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self._days = []
        self._isolated = []
        self._died = []
        self._avg = []
        self._render_paused = False

        # Stacked bars: Isolated at bottom (green), Died on top (red)
        self._bar_isolated = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush=_CLR_GREEN, name='Isolation',
        )
        self.plot_widget.addItem(self._bar_isolated)

        self._bar_died = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush=_CLR_RED, name='Death',
        )
        self.plot_widget.addItem(self._bar_died)

        # Line series for running average
        self._curve_avg = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_NAVY, width=2), name='Average',
        )

    def add_data_point(self, day, isolated_daily, died_daily, avg):
        self._days.append(day)
        self._isolated.append(isolated_daily)
        self._died.append(died_daily)
        self._avg.append(avg)

        if not self._render_paused:
            self._render()

    def _render(self):
        self._bar_isolated.setOpts(x=self._days, height=self._isolated, width=0.8)
        self._bar_died.setOpts(
            x=self._days, height=self._died, y0=self._isolated, width=0.8,
        )
        self._curve_avg.setData(self._days, self._avg)

    def set_render_paused(self, paused):
        self._render_paused = paused
        if not paused and self._days:
            self._render()

    def clear(self):
        self._days.clear()
        self._isolated.clear()
        self._died.clear()
        self._avg.clear()

        self._bar_isolated.setOpts(x=[], height=[])
        self._bar_died.setOpts(x=[], height=[])
        self._curve_avg.setData([], [])

    def save_image(self, filepath):
        exporter = exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(filepath)


class InfectiveChart(QWidget):
    """Infection source chart: normal contact (red) vs hospital (green) as stacked bars.
    Matches original C++ colors and mbStacked bar mode."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'day(s)')
        self.plot_widget.setLabel('left', 'person(s)')
        self.plot_widget.addLegend()

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self._days = []
        self._normal = []
        self._hospital = []
        self._render_paused = False

        self._bar_normal = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush=_CLR_RED, name='Normal',
        )
        self.plot_widget.addItem(self._bar_normal)

        self._bar_hospital = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush=_CLR_GREEN, name='Hospital',
        )
        self.plot_widget.addItem(self._bar_hospital)

    def add_data_point(self, day, normal_count, hospital_count):
        self._days.append(day)
        self._normal.append(normal_count)
        self._hospital.append(hospital_count)

        if not self._render_paused:
            self._render()

    def _render(self):
        self._bar_normal.setOpts(x=self._days, height=self._normal, width=0.8)
        self._bar_hospital.setOpts(
            x=self._days, height=self._hospital, y0=self._normal, width=0.8,
        )

    def set_render_paused(self, paused):
        self._render_paused = paused
        if not paused and self._days:
            self._render()

    def clear(self):
        self._days.clear()
        self._normal.clear()
        self._hospital.clear()

        self._bar_normal.setOpts(x=[], height=[])
        self._bar_hospital.setOpts(x=[], height=[])

    def save_image(self, filepath):
        exporter = exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(filepath)


class AccumulativeQuarantineChart(QWidget):
    """Cumulative quarantine count as a single red line series."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'day(s)')
        self.plot_widget.setLabel('left', 'person(s)')
        self.plot_widget.addLegend()

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self._days = []
        self._total_quarantined = []
        self._render_paused = False

        self._curve_quarantined = self.plot_widget.plot(
            [], [], pen=pg.mkPen(_CLR_RED, width=2), name='Quarantined',
        )

    def add_data_point(self, day, total_quarantined):
        self._days.append(day)
        self._total_quarantined.append(total_quarantined)
        if not self._render_paused:
            self._render()

    def _render(self):
        self._curve_quarantined.setData(self._days, self._total_quarantined)

    def set_render_paused(self, paused):
        self._render_paused = paused
        if not paused and self._days:
            self._render()

    def clear(self):
        self._days.clear()
        self._total_quarantined.clear()
        self._curve_quarantined.setData([], [])

    def save_image(self, filepath):
        exporter = exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(filepath)


class DailyQuarantineChart(QWidget):
    """Daily quarantine count as a single red bar series."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setLabel('bottom', 'day(s)')
        self.plot_widget.setLabel('left', 'person(s)')
        self.plot_widget.addLegend()

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self._days = []
        self._daily_quarantined = []
        self._render_paused = False

        self._bar_quarantined = pg.BarGraphItem(
            x=[], height=[], width=0.8, brush=_CLR_RED, name='Quarantined',
        )
        self.plot_widget.addItem(self._bar_quarantined)

    def add_data_point(self, day, daily_quarantined):
        self._days.append(day)
        self._daily_quarantined.append(daily_quarantined)
        if not self._render_paused:
            self._render()

    def _render(self):
        self._bar_quarantined.setOpts(
            x=self._days, height=self._daily_quarantined,
        )

    def set_render_paused(self, paused):
        self._render_paused = paused
        if not paused and self._days:
            self._render()

    def clear(self):
        self._days.clear()
        self._daily_quarantined.clear()
        self._bar_quarantined.setOpts(x=[], height=[])

    def save_image(self, filepath):
        exporter = exporters.ImageExporter(self.plot_widget.plotItem)
        exporter.export(filepath)
