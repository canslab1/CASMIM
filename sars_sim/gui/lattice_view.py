"""Lattice visualization widgets for the SARS simulation.

Provides two widgets:
- MacroLatticeView: displays the full 500x500 world lattice.
- MicroLatticeView: displays a magnified 100x100 sub-region.

Both views dynamically resize to fill their parent container while
maintaining a square aspect ratio.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QSizePolicy,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal, QPoint

from sars_sim.models import (
    Colors,
    SimulationData,
    StateEnum,
    AgeEnum,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARTIAL_LATTICE_SIZE = 100
MAX_MOVE_SIZE = 8              # number of power levels (0..7)
MOVE_BASE = 2                  # step = MOVE_BASE ** move_power


# ===================================================================
# MacroLatticeView
# ===================================================================

class MacroLatticeView(QWidget):
    """Displays the full lattice, scaled to fill available space."""

    cell_clicked = Signal(int, int)       # (x, y) of the clicked cell
    cell_hovered = Signal(int, int, int)  # (x, y, peopleID)

    def __init__(self, data: SimulationData, parent=None):
        super().__init__(parent)
        self.data = data
        self._H = data.H
        self._W = data.W

        # Pre-allocated buffers to avoid per-frame allocation.
        self._display = np.empty((self._H, self._W), dtype=np.uint32)
        self._visible = np.empty((self._H, self._W), dtype=bool)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        # QLabel used as the rendering surface.
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update_image(self):
        """Render *world_color* to a QImage, scaled to fill the label."""
        data = self.data
        H, W = self._H, self._W

        ids = data.world_people_id        # (H, W) int32
        nos = data.world_agent_no         # (H, W) int32

        # Cells with id == -1 have no agent; treat as invisible (GRAY).
        occupied = ids >= 0
        safe_ids = np.where(occupied, ids, 0)
        safe_nos = np.where(occupied, nos, 0)

        self._visible[:] = False
        self._visible[occupied] = data.agent_visible[safe_ids[occupied], safe_nos[occupied]]

        np.copyto(self._display, data.world_color, where=self._visible)
        np.copyto(self._display, np.uint32(Colors.GRAY), where=~self._visible)

        # QImage reads from the numpy buffer directly; _display stays alive.
        qimg = QImage(self._display.data, W, H, W * 4, QImage.Format.Format_ARGB32)

        # Scale to fit label, keeping square aspect ratio with nearest-neighbor.
        lw = self._label.width()
        lh = self._label.height()
        if lw > 0 and lh > 0:
            scaled = qimg.scaled(lw, lh,
                                 Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.FastTransformation)
            self._label.setPixmap(QPixmap.fromImage(scaled))

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _map_to_world(self, pos):
        """Map a mouse position to world (x, y), or None if out of range."""
        pm = self._label.pixmap()
        if pm is None or pm.isNull():
            return None
        # The pixmap may be smaller than the label due to aspect ratio.
        pw, ph = pm.width(), pm.height()
        lw, lh = self._label.width(), self._label.height()
        # Offset for centered pixmap inside the label.
        ox = (lw - pw) / 2
        oy = (lh - ph) / 2
        rx = pos.x() - ox
        ry = pos.y() - oy
        if rx < 0 or ry < 0 or rx >= pw or ry >= ph:
            return None
        x = int(rx * self._W / pw)
        y = int(ry * self._H / ph)
        x = max(0, min(x, self._W - 1))
        y = max(0, min(y, self._H - 1))
        return x, y

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            result = self._map_to_world(event.position())
            if result:
                self.cell_clicked.emit(result[0], result[1])

    def mouseMoveEvent(self, event):
        result = self._map_to_world(event.position())
        if result:
            x, y = result
            pid = int(self.data.world_people_id[y, x])
            self.cell_hovered.emit(x, y, pid)


# ===================================================================
# MicroLatticeView
# ===================================================================

class MicroLatticeView(QWidget):
    """Displays a magnified 100x100 sub-region of the lattice.

    Navigation buttons (N / S / E / W) shift the view; the Centre button
    cycles through power-of-two step sizes (1, 2, 4, ... 128).
    """

    cell_hovered = Signal(int, int, int)  # world (x, y, peopleID)
    person_detail = Signal(str)           # detail string for status display

    def __init__(self, data: SimulationData, parent=None):
        super().__init__(parent)
        self.data = data
        self._H = data.H
        self._W = data.W

        self.partial_view = QPoint(self._W // 2, self._H // 2)
        self.move_power = 0  # exponent: step = 2 ** move_power

        # Pre-allocated buffers to avoid per-frame allocation.
        S = PARTIAL_LATTICE_SIZE
        self._display = np.empty((S, S), dtype=np.uint32)
        self._visible = np.empty((S, S), dtype=bool)

        # Cached meshgrid indices — recomputed only when partial_view changes.
        self._cached_center = None      # QPoint or None
        self._wy_grid = None            # (S, S) int array
        self._wx_grid = None            # (S, S) int array

        self.setMouseTracking(True)

        # --- image label ---
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setMouseTracking(True)
        self._label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # --- navigation buttons ---
        self._btn_north = QPushButton("N")
        self._btn_south = QPushButton("S")
        self._btn_west = QPushButton("W")
        self._btn_east = QPushButton("E")
        self._btn_center = QPushButton(self._center_label())

        for btn in (self._btn_north, self._btn_south,
                    self._btn_west, self._btn_east, self._btn_center):
            btn.setFixedSize(40, 28)

        self._btn_north.clicked.connect(self._on_north)
        self._btn_south.clicked.connect(self._on_south)
        self._btn_west.clicked.connect(self._on_west)
        self._btn_east.clicked.connect(self._on_east)
        self._btn_center.clicked.connect(self._on_center_click)

        # --- layout ---
        # Navigation grid (compass-style)
        nav_grid = QGridLayout()
        nav_grid.setSpacing(2)
        nav_grid.addWidget(self._btn_north, 0, 1, Qt.AlignmentFlag.AlignCenter)
        nav_grid.addWidget(self._btn_west, 1, 0, Qt.AlignmentFlag.AlignCenter)
        nav_grid.addWidget(self._btn_center, 1, 1, Qt.AlignmentFlag.AlignCenter)
        nav_grid.addWidget(self._btn_east, 1, 2, Qt.AlignmentFlag.AlignCenter)
        nav_grid.addWidget(self._btn_south, 2, 1, Qt.AlignmentFlag.AlignCenter)

        top_layout = QVBoxLayout(self)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addWidget(self._label, 1, Qt.AlignmentFlag.AlignCenter)
        top_layout.addLayout(nav_grid)
        self.setLayout(top_layout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _center_label(self) -> str:
        step = int(MOVE_BASE ** self.move_power)
        return f"x{step}"

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def update_image(self):
        """Render the 100x100 region around *partial_view*, scaled to fill label."""
        data = self.data
        H, W = self._H, self._W
        S = PARTIAL_LATTICE_SIZE
        half = S // 2  # 50

        # Recompute meshgrid only when the view centre has changed.
        if self._cached_center != self.partial_view:
            self._cached_center = QPoint(self.partial_view)
            start_y = (self.partial_view.y() - half + H) % H
            start_x = (self.partial_view.x() - half + W) % W
            wy = (start_y + np.arange(S)) % H
            wx = (start_x + np.arange(S)) % W
            self._wy_grid, self._wx_grid = np.meshgrid(wy, wx, indexing="ij")

        wy_grid = self._wy_grid
        wx_grid = self._wx_grid

        ids = data.world_people_id[wy_grid, wx_grid]
        nos = data.world_agent_no[wy_grid, wx_grid]

        occupied = ids >= 0
        safe_ids = np.where(occupied, ids, 0)
        safe_nos = np.where(occupied, nos, 0)

        self._visible[:] = False
        self._visible[occupied] = data.agent_visible[safe_ids[occupied], safe_nos[occupied]]

        colors_raw = data.world_color[wy_grid, wx_grid]
        np.copyto(self._display, colors_raw, where=self._visible)
        np.copyto(self._display, np.uint32(Colors.GRAY), where=~self._visible)

        # QImage reads from the numpy buffer directly; _display stays alive.
        qimg = QImage(self._display.data, S, S,
                      S * 4, QImage.Format.Format_ARGB32)

        lw = self._label.width()
        lh = self._label.height()
        if lw > 0 and lh > 0:
            scaled = qimg.scaled(lw, lh,
                                 Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.FastTransformation)
            self._label.setPixmap(QPixmap.fromImage(scaled))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_center(self, x: int, y: int):
        """Set the centre of the partial view and refresh the image."""
        self.partial_view = QPoint(x, y)
        self.update_image()

    # ------------------------------------------------------------------
    # Navigation slots
    # ------------------------------------------------------------------

    def _on_north(self):
        step = int(MOVE_BASE ** self.move_power)
        self.partial_view = QPoint(
            self.partial_view.x(),
            (self.partial_view.y() - step + self._H) % self._H,
        )
        self.update_image()

    def _on_south(self):
        step = int(MOVE_BASE ** self.move_power)
        self.partial_view = QPoint(
            self.partial_view.x(),
            (self.partial_view.y() + step) % self._H,
        )
        self.update_image()

    def _on_west(self):
        step = int(MOVE_BASE ** self.move_power)
        self.partial_view = QPoint(
            (self.partial_view.x() - step + self._W) % self._W,
            self.partial_view.y(),
        )
        self.update_image()

    def _on_east(self):
        step = int(MOVE_BASE ** self.move_power)
        self.partial_view = QPoint(
            (self.partial_view.x() + step) % self._W,
            self.partial_view.y(),
        )
        self.update_image()

    def _on_center_click(self):
        self.move_power = (self.move_power + 1) % MAX_MOVE_SIZE
        self._btn_center.setText(self._center_label())

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _map_to_cell(self, pos):
        """Map mouse position to micro cell index (mx, my), or None."""
        pm = self._label.pixmap()
        if pm is None or pm.isNull():
            return None
        pw, ph = pm.width(), pm.height()
        lw, lh = self._label.width(), self._label.height()
        ox = (lw - pw) / 2
        oy = (lh - ph) / 2
        rx = pos.x() - ox
        ry = pos.y() - oy
        if rx < 0 or ry < 0 or rx >= pw or ry >= ph:
            return None
        mx = int(rx * PARTIAL_LATTICE_SIZE / pw)
        my = int(ry * PARTIAL_LATTICE_SIZE / ph)
        mx = max(0, min(mx, PARTIAL_LATTICE_SIZE - 1))
        my = max(0, min(my, PARTIAL_LATTICE_SIZE - 1))
        return mx, my

    def mouseMoveEvent(self, event):
        result = self._map_to_cell(event.position())
        if result is None:
            return

        mx, my = result
        H, W = self._H, self._W
        half = PARTIAL_LATTICE_SIZE // 2

        wx = (self.partial_view.x() - half + mx + W) % W
        wy = (self.partial_view.y() - half + my + H) % H
        pid = int(self.data.world_people_id[wy, wx])

        self.cell_hovered.emit(wx, wy, pid)

        if pid >= 0:
            detail = self._build_person_detail(pid)
            self.person_detail.emit(detail)

    # ------------------------------------------------------------------
    # Detail string builder
    # ------------------------------------------------------------------

    def _build_person_detail(self, pid: int) -> str:
        """Return a human-readable status string for person *pid*.

        Examples::

            "Immune, Super Spreader, day = 5, Old, Isolated"
            "Susceptible, Young, Free"
            "Susceptible, day = 3, Prime, Home Quarantine, day = 2, Class:1"
        """
        data = self.data

        if pid < 0 or pid >= data.N:
            return ""

        parts: list[str] = []

        # --- immunity / susceptibility ---
        if data.people_immunity[pid]:
            parts.append("Immune")
        else:
            parts.append("Susceptible")

        # --- super spreader ---
        if data.people_super[pid]:
            parts.append("Super Spreader")

        # --- disease timer (days since state change) ---
        timer = int(data.people_timer[pid])
        if timer > 0:
            parts.append(f"day = {timer}")

        # --- age group ---
        age = int(data.people_age[pid])
        if age == AgeEnum.OLD:
            parts.append("Old")
        elif age == AgeEnum.PRIME:
            parts.append("Prime")
        else:
            parts.append("Young")

        # --- isolation / quarantine / free ---
        if data.people_isolated[pid]:
            parts.append("Isolated")
        elif data.people_quarantined[pid]:
            q_count = int(data.people_quarantined_count[pid])
            q_level = int(data.people_quarantined_level[pid])
            q_str = "Home Quarantine"
            if q_count > 0:
                q_str += f", day = {q_count}"
            q_str += f", Class:{q_level}"
            parts.append(q_str)
        else:
            parts.append("Free")

        return ", ".join(parts)
