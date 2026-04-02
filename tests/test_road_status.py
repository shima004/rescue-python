"""Tests for src.utility.road_status.get_passable_edge_pairs.

Each test case covers a distinct geometric scenario and saves a PNG visualisation
to tests/output/ so the geometry can be inspected visually.

Coordinate unit: mm (same as the RRS world model).
AGENT_RADIUS = 500 mm.

Road layout used in most tests (6000 x 6000 square):

    (0,6000)────────────────(6000,6000)
       │                         │
  n1 ← │  ←  left wall (n1)      │ ← right wall (n2) → n2
       │                         │
    (0,0)──────────────────(6000,0)
                  ↑
              bottom wall (n3) – used only in T-junction tests
"""

from __future__ import annotations

import os
from itertools import combinations
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI / headless environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from rcrscore.entities import Blockade, Edge, EntityID, Road
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from src.utility.road_status import (
  AGENT_RADIUS,
  _get_blockade_polygon,
  _get_road_polygon,
  get_passable_edge_pairs,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_road(edge_specs: list) -> Road:
  """Return a Road whose edges are built from (sx, sy, ex, ey, neighbor_id)."""
  road = Road(0)
  edges = [
    Edge(sx, sy, ex, ey, EntityID(nid) if nid is not None else None)
    for sx, sy, ex, ey, nid in edge_specs
  ]
  road.set_edges(edges)
  return road


def make_blockade(bid: int, x0: int, y0: int, x1: int, y1: int) -> Blockade:
  """Return a rectangular Blockade with apexes in CCW order."""
  b = Blockade(bid)
  b.set_apexes([x0, y0, x1, y0, x1, y1, x0, y1])
  return b


def mock_world(blockades: set[Blockade]) -> MagicMock:
  wi = MagicMock()
  wi.get_blockades.return_value = blockades
  return wi


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------


def _poly_coords(poly: Polygon) -> np.ndarray:
  return np.array(poly.exterior.coords)


def visualize(
  title: str,
  road: Road,
  blockades: set[Blockade],
  tested_pairs: list[tuple[Edge, Edge]],
  results: list[bool],
  filename: str,
) -> None:
  """Save a figure with one subplot per tested edge pair."""
  road_poly = _get_road_polygon(road)
  blockade_polys = [_get_blockade_polygon(b) for b in blockades]
  blockade_union = unary_union(blockade_polys) if blockade_polys else Polygon()
  all_edges = road.get_edges() or []
  connected = [e for e in all_edges if e.get_neighbour() is not None]

  n = len(tested_pairs)
  fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
  fig.suptitle(title, fontsize=12)

  for idx, ((e1, e2), passable) in enumerate(zip(tested_pairs, results)):
    ax = axes[0][idx]

    # Road
    if not road_poly.is_empty:
      ax.add_patch(
        MplPolygon(
          _poly_coords(road_poly),
          closed=True,
          facecolor="#cce5ff",
          edgecolor="steelblue",
          linewidth=1.5,
          zorder=1,
        )
      )

    # Blockades
    for bp in blockade_polys:
      if not bp.is_empty:
        ax.add_patch(
          MplPolygon(
            np.array(bp.exterior.coords),
            closed=True,
            facecolor="#ffaaaa",
            edgecolor="red",
            alpha=0.8,
            zorder=2,
          )
        )

    # Corridor and eroded passage
    corridor_pts = [
      (e1.get_start_x(), e1.get_start_y()),
      (e1.get_end_x(), e1.get_end_y()),
      (e2.get_start_x(), e2.get_start_y()),
      (e2.get_end_x(), e2.get_end_y()),
    ]
    corridor = Polygon(corridor_pts).convex_hull
    passage = corridor.intersection(road_poly)
    clear = passage.difference(blockade_union)
    eroded = clear.buffer(-AGENT_RADIUS)

    if not passage.is_empty:
      ax.add_patch(
        MplPolygon(
          np.array(passage.exterior.coords),
          closed=True,
          facecolor="yellow",
          edgecolor="goldenrod",
          alpha=0.3,
          linewidth=1,
          zorder=3,
        )
      )

    # Entry regions near each edge (shows where agent enters/exits)
    for e, color in ((e1, "#66aaff"), (e2, "#ff9900")):
      entry = passage.intersection(
        LineString(
          [(e.get_start_x(), e.get_start_y()), (e.get_end_x(), e.get_end_y())]
        ).buffer(AGENT_RADIUS)
      )
      if not entry.is_empty:
        ax.add_patch(
          MplPolygon(
            np.array(entry.exterior.coords),
            closed=True,
            facecolor=color,
            alpha=0.25,
            zorder=4,
          )
        )

    for geom in list(eroded.geoms) if hasattr(eroded, "geoms") else [eroded]:
      if not geom.is_empty:
        ax.add_patch(
          MplPolygon(
            np.array(geom.exterior.coords),
            closed=True,
            facecolor="#00cc66",
            edgecolor="darkgreen",
            alpha=0.5,
            linewidth=1,
            zorder=5,
          )
        )

    # Connected edges
    for e in connected:
      is_tested = e in (e1, e2)
      color = "#ff6600" if is_tested else "#888888"
      lw = 4 if is_tested else 1.5
      ax.plot(
        [e.get_start_x(), e.get_end_x()],
        [e.get_start_y(), e.get_end_y()],
        color=color,
        linewidth=lw,
        solid_capstyle="round",
        zorder=6,
      )
      mx = (e.get_start_x() + e.get_end_x()) / 2
      my = (e.get_start_y() + e.get_end_y()) / 2
      neighbour = e.get_neighbour()
      ax.text(
        mx,
        my,
        f"n{neighbour.get_value()}" if neighbour is not None else "?",
        fontsize=8,
        ha="center",
        va="center",
        color=color,
        fontweight="bold",
        zorder=7,
      )

    result_str = "PASSABLE" if passable else "BLOCKED"
    result_color = "darkgreen" if passable else "crimson"
    n1 = e1.get_neighbour().get_value()
    n2 = e2.get_neighbour().get_value()
    ax.set_title(f"n{n1} ↔ n{n2} : {result_str}", color=result_color, fontsize=11)

    if not road_poly.is_empty:
      minx, miny, maxx, maxy = road_poly.bounds
      pad = max(maxx - minx, maxy - miny) * 0.08
      ax.set_xlim(minx - pad, maxx + pad)
      ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.legend(
      handles=[
        mpatches.Patch(facecolor="#cce5ff", edgecolor="steelblue", label="Road"),
        mpatches.Patch(facecolor="#ffaaaa", edgecolor="red", label="Blockade"),
        mpatches.Patch(facecolor="yellow", alpha=0.5, label="Corridor"),
        mpatches.Patch(facecolor="#66aaff", alpha=0.4, label="Entry e1"),
        mpatches.Patch(facecolor="#ff9900", alpha=0.4, label="Entry e2"),
        mpatches.Patch(
          facecolor="#00cc66", alpha=0.6, label=f"Eroded (r={AGENT_RADIUS})"
        ),
      ],
      fontsize=7,
      loc="upper right",
    )

  plt.tight_layout()
  out = os.path.join(OUTPUT_DIR, filename)
  plt.savefig(out, bbox_inches="tight", dpi=120)
  plt.close(fig)
  print(f"\n  [viz] {out}")


# ---------------------------------------------------------------------------
# Shared road definition (6000 x 6000 square, edges on left / right walls)
#
#  edge order determines polygon vertex order for get_apexes():
#    bottom (no neighbor) → right (n2) → top (no neighbor) → left (n1)
# ---------------------------------------------------------------------------

SQUARE_EDGES = [
  (0, 0, 6000, 0, None),  # bottom wall
  (6000, 0, 6000, 6000, 2),  # right wall  → neighbor 2
  (6000, 6000, 0, 6000, None),  # top wall
  (0, 6000, 0, 0, 1),  # left wall   → neighbor 1
]


def _get_connected(road: Road) -> list[Edge]:
  return [e for e in (road.get_edges() or []) if e.get_neighbour() is not None]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoBlockades:
  """Early-return path: road.get_blockades() is None → all pairs passable."""

  def test_all_pairs_passable(self) -> None:
    road = make_road(SQUARE_EDGES)
    road.set_blockades(None)  # Trigger early-return path
    wi = mock_world(set())

    result = get_passable_edge_pairs(road, wi)

    n1, n2 = EntityID(1), EntityID(2)
    assert (n1, n2) in result or (n2, n1) in result

    e1, e2 = _get_connected(road)
    visualize(
      "No blockades – all pairs passable (early return)",
      road,
      set(),
      [(e1, e2)],
      [True],
      "01_no_blockades.png",
    )


class TestGeometryPassable:
  """Blockades exist but leave enough room for the agent."""

  def test_corner_blockade(self) -> None:
    """Small blockade tucked in one corner; corridor clear width >> AGENT_RADIUS."""
    road = make_road(SQUARE_EDGES)
    road.set_blockades([EntityID(99)])
    b = make_blockade(99, 0, 0, 1500, 1500)  # 1500x1500 corner piece
    wi = mock_world({b})

    result = get_passable_edge_pairs(road, wi)
    assert len(result) == 1

    e1, e2 = _get_connected(road)
    visualize(
      "Corner blockade – passable",
      road,
      {b},
      [(e1, e2)],
      [True],
      "02_corner_blockade_passable.png",
    )

  def test_partial_blockade_sufficient_gap(self) -> None:
    """Blockade spans part of the road height; remaining gap > 2 * AGENT_RADIUS."""
    road = make_road(SQUARE_EDGES)
    road.set_blockades([EntityID(99)])
    # Blockade leaves 1200 mm gap at top (6000 - 4800 = 1200 > 2*500 = 1000)
    b = make_blockade(99, 2500, 0, 3500, 4800)
    wi = mock_world({b})

    result = get_passable_edge_pairs(road, wi)
    assert len(result) == 1

    e1, e2 = _get_connected(road)
    visualize(
      "Partial blockade, gap 1200 mm > 2×AGENT_RADIUS – passable",
      road,
      {b},
      [(e1, e2)],
      [True],
      "03_partial_blockade_sufficient_gap.png",
    )


class TestGeometryBlocked:
  """Blockades that make one or all passages impassable."""

  def test_full_height_blockade(self) -> None:
    """Blockade spans the full road height, severing the corridor.

    This validates the connectivity check: each half of the eroded corridor
    is non-empty on its own, but no single component touches both edges.
    """
    road = make_road(SQUARE_EDGES)
    road.set_blockades([EntityID(99)])
    b = make_blockade(99, 2800, -100, 3200, 6100)  # full-height wall
    wi = mock_world({b})

    result = get_passable_edge_pairs(road, wi)
    assert len(result) == 0

    e1, e2 = _get_connected(road)
    visualize(
      "Full-height blockade severs corridor – BLOCKED",
      road,
      {b},
      [(e1, e2)],
      [False],
      "04_full_height_blockade_blocked.png",
    )

  def test_partial_blockade_insufficient_gap(self) -> None:
    """Blockade leaves only 900 mm gap < 2 * AGENT_RADIUS (1000 mm)."""
    road = make_road(SQUARE_EDGES)
    road.set_blockades([EntityID(99)])
    # Gap at top: 6000 - 5100 = 900 < 1000
    b = make_blockade(99, 2500, 0, 3500, 5100)
    wi = mock_world({b})

    result = get_passable_edge_pairs(road, wi)
    assert len(result) == 0

    e1, e2 = _get_connected(road)
    visualize(
      "Partial blockade, gap 900 mm < 2×AGENT_RADIUS – BLOCKED",
      road,
      {b},
      [(e1, e2)],
      [False],
      "05_partial_blockade_insufficient_gap.png",
    )

  def test_two_blockades_together_seal_corridor(self) -> None:
    """Two blockades each covering half the road height combine to seal it."""
    road = make_road(SQUARE_EDGES)
    road.set_blockades([EntityID(1), EntityID(2)])
    # b1 covers lower half, b2 covers upper half, with 400 mm overlap in the middle
    b1 = make_blockade(1, 2500, 0, 3500, 3400)
    b2 = make_blockade(2, 2500, 3000, 3500, 6100)
    wi = mock_world({b1, b2})

    result = get_passable_edge_pairs(road, wi)
    assert len(result) == 0

    e1, e2 = _get_connected(road)
    visualize(
      "Two blockades together seal corridor – BLOCKED",
      road,
      {b1, b2},
      [(e1, e2)],
      [False],
      "06_two_blockades_sealed.png",
    )


class TestTJunction:
  """Three-way intersection: one road area with three connected edges."""

  # Square road with an extra connected edge at the bottom-centre.
  #  bottom-left (no neighbor) | bottom-center (n3) | bottom-right (no neighbor)
  #  right wall (n2) | top wall (no neighbor) | left wall (n1)
  T_EDGES = [
    (0, 0, 2000, 0, None),  # bottom-left
    (2000, 0, 4000, 0, 3),  # bottom-centre → neighbor 3
    (4000, 0, 6000, 0, None),  # bottom-right
    (6000, 0, 6000, 6000, 2),  # right wall → neighbor 2
    (6000, 6000, 0, 6000, None),  # top wall
    (0, 6000, 0, 0, 1),  # left wall  → neighbor 1
  ]

  def test_all_pairs_passable_no_blockades(self) -> None:
    road = make_road(self.T_EDGES)
    road.set_blockades(None)
    wi = mock_world(set())

    result = get_passable_edge_pairs(road, wi)
    assert len(result) == 3  # (n1,n2), (n1,n3), (n2,n3)

    edges = _get_connected(road)
    pairs = list(combinations(edges, 2))
    visualize(
      "T-junction, no blockades – all 3 pairs passable",
      road,
      set(),
      pairs,
      [True, True, True],
      "07_t_junction_no_blockades.png",
    )

  def test_one_passage_blocked(self) -> None:
    """Vertical wall blocks n1 ↔ n2 but leaves n1 ↔ n3 and n2 ↔ n3 open."""
    road = make_road(self.T_EDGES)
    road.set_blockades([EntityID(99)])
    # Wall running right down the middle of the road, sealing left from right
    b = make_blockade(99, 2800, -100, 3200, 6100)
    wi = mock_world({b})

    result = get_passable_edge_pairs(road, wi)
    pairs_found = {(a.get_value(), b_id.get_value()) for a, b_id in result}

    # n1 ↔ n2 should be blocked; n1 ↔ n3 and n2 ↔ n3 should still be passable
    assert (1, 2) not in pairs_found and (2, 1) not in pairs_found
    assert (1, 3) in pairs_found or (3, 1) in pairs_found
    assert (2, 3) in pairs_found or (3, 2) in pairs_found

    edges = _get_connected(road)
    pairs = list(combinations(edges, 2))
    # Determine passability for each pair for visualisation
    pair_results = [
      (e1.get_neighbour(), e2.get_neighbour()) in result
      or (e2.get_neighbour(), e1.get_neighbour()) in result
      for e1, e2 in pairs
    ]
    visualize(
      "T-junction, centre wall – n1↔n2 blocked, n1↔n3 and n2↔n3 passable",
      road,
      {b},
      pairs,
      pair_results,
      "08_t_junction_one_blocked.png",
    )
