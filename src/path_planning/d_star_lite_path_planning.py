from __future__ import annotations

# D* Lite incremental path planning algorithm.
# Searches backwards from goal to start and supports replanning when costs change.
# Reference: Koenig & Likhachev, "D* Lite", AAAI 2002.

import heapq
import itertools

from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.path_planning import (
  PathPlanning,
)
from rcrscore.entities import Area, Building, Entity, EntityID, Road

_INF = float("inf")


class DStarLitePathPlanning(PathPlanning):
  def __init__(
    self,
    agent_info: AgentInfo,
    world_info: WorldInfo,
    scenario_info: ScenarioInfo,
    module_manager: ModuleManager,
    develop_data: DevelopData,
  ) -> None:
    super().__init__(
      agent_info, world_info, scenario_info, module_manager, develop_data
    )
    entities: list[Entity] = self._world_info.get_entities_of_types([Building, Road])
    self._graph: dict[EntityID, set[EntityID]] = {}
    for entity in entities:
      if isinstance(entity, Area):
        self._graph[entity.get_entity_id()] = set(
          neighbor for neighbor in entity.get_neighbors() if neighbor != EntityID(0)
        )

    # D* Lite persistent state across get_path() calls.
    # Reused as long as the goal does not change.
    self._g: dict[EntityID, float] = {}
    self._rhs: dict[EntityID, float] = {}
    self._heap: list = []
    self._valid_key: dict[EntityID, tuple[float, float]] = {}
    self._counter = itertools.count()
    # km compensates for heuristic inconsistency when the start position moves.
    self._km: float = 0.0
    self._s_goal: EntityID | None = None
    # s_last tracks the previous start to compute the km increment on next call.
    self._s_last: EntityID | None = None

  # ------------------------------------------------------------------
  # Cost functions
  # ------------------------------------------------------------------

  def _h(self, a: EntityID, b: EntityID) -> float:
    """Heuristic: straight-line distance between two nodes."""
    return self._world_info.get_distance(a, b)

  def _c(self, a: EntityID, b: EntityID) -> float:
    """Edge traversal cost; returns inf if a and b are not adjacent."""
    if b in self._graph.get(a, set()):
      return self._world_info.get_distance(a, b)
    return _INF

  # ------------------------------------------------------------------
  # D* Lite core helpers
  # ------------------------------------------------------------------

  def _calculate_key(self, s: EntityID, s_start: EntityID) -> tuple[float, float]:
    # Two-component key: primary sorts by f-value, secondary by g/rhs minimum.
    min_val = min(self._g[s], self._rhs[s])
    return (min_val + self._h(s_start, s) + self._km, min_val)

  def _update_vertex(self, u: EntityID, s_start: EntityID) -> None:
    # Remove u from the open list, then re-insert if locally inconsistent.
    self._valid_key.pop(u, None)
    if self._g[u] != self._rhs[u]:
      key = self._calculate_key(u, s_start)
      self._valid_key[u] = key
      heapq.heappush(self._heap, (key, next(self._counter), u))

  def _top_key(self) -> tuple[float, float] | None:
    # Discard stale entries and return the minimum valid key.
    while self._heap:
      key, _, u = self._heap[0]
      if self._valid_key.get(u) == key:
        return key
      heapq.heappop(self._heap)
    return None

  def _initialize(self, s_goal: EntityID, s_start: EntityID) -> None:
    """Reset all state for a new goal."""
    self._g = {node: _INF for node in self._graph}
    self._rhs = {node: _INF for node in self._graph}
    self._rhs[s_goal] = 0.0
    self._km = 0.0
    self._heap = []
    self._valid_key = {}
    self._counter = itertools.count()
    self._s_goal = s_goal
    self._s_last = s_start

    init_key = self._calculate_key(s_goal, s_start)
    self._valid_key[s_goal] = init_key
    heapq.heappush(self._heap, (init_key, next(self._counter), s_goal))

  def _compute_shortest_path(self, s_start: EntityID, s_goal: EntityID) -> None:
    """Expand nodes until s_start is locally consistent and optimally resolved."""
    while True:
      tk = self._top_key()
      if tk is None:
        break
      if not (
        tk < self._calculate_key(s_start, s_start)
        or self._rhs[s_start] != self._g[s_start]
      ):
        break

      key, _, u = heapq.heappop(self._heap)
      if self._valid_key.get(u) != key:
        continue  # Stale entry; skip.

      k_new = self._calculate_key(u, s_start)
      if key < k_new:
        # Key grew due to start movement; re-insert with corrected key.
        self._valid_key[u] = k_new
        heapq.heappush(self._heap, (k_new, next(self._counter), u))
      elif self._g[u] > self._rhs[u]:
        # Locally overconsistent: accept rhs as g, propagate to predecessors.
        self._g[u] = self._rhs[u]
        self._valid_key.pop(u, None)
        for s in self._graph.get(u, set()):
          if s != s_goal:
            new_rhs = self._c(s, u) + self._g[u]
            if new_rhs < self._rhs[s]:
              self._rhs[s] = new_rhs
          self._update_vertex(s, s_start)
      else:
        # Locally underconsistent: invalidate g, recompute rhs for affected nodes.
        g_old = self._g[u]
        self._g[u] = _INF
        self._valid_key.pop(u, None)
        for s in self._graph.get(u, set()) | {u}:
          if self._rhs[s] == self._c(s, u) + g_old:
            if s != s_goal:
              self._rhs[s] = min(
                (self._c(s, sp) + self._g[sp] for sp in self._graph.get(s, set())),
                default=_INF,
              )
          self._update_vertex(s, s_start)

  # ------------------------------------------------------------------
  # PathPlanning interface
  # ------------------------------------------------------------------

  def get_path(
    self, from_entity_id: EntityID, to_entity_id: EntityID
  ) -> list[EntityID]:
    s_start = from_entity_id
    s_goal = to_entity_id

    if s_start == s_goal:
      return [s_start]
    if s_start not in self._graph or s_goal not in self._graph:
      return []

    if s_goal != self._s_goal:
      # Goal changed: full reinitialization required.
      self._initialize(s_goal, s_start)
    else:
      # Same goal: accumulate heuristic shift caused by start movement so that
      # previously computed keys remain valid lower bounds.
      if self._s_last is not None and self._s_last != s_start:
        self._km += self._h(self._s_last, s_start)
      self._s_last = s_start

    self._compute_shortest_path(s_start, s_goal)

    if self._g[s_start] == _INF:
      return []

    # Reconstruct path by greedy descent along minimum-cost successors.
    path = [s_start]
    current = s_start
    visited: set[EntityID] = {s_start}

    while current != s_goal:
      neighbors = self._graph.get(current, set())
      next_node = min(
        (n for n in neighbors if n in self._g),
        key=lambda n: self._c(current, n) + self._g[n],
        default=None,
      )
      if next_node is None or next_node in visited:
        return []
      path.append(next_node)
      visited.add(next_node)
      current = next_node

    return path

  def get_path_to_multiple_destinations(
    self, from_entity_id: EntityID, destination_entity_ids: set[EntityID]
  ) -> list[EntityID]:
    """Return the shortest path from from_entity_id to any of the destinations."""
    best_path: list[EntityID] = []
    best_dist = _INF
    for dest in destination_entity_ids:
      path = self.get_path(from_entity_id, dest)
      if not path:
        continue
      dist = sum(self._c(path[i], path[i + 1]) for i in range(len(path) - 1))
      if dist < best_dist:
        best_dist = dist
        best_path = path
    return best_path

  def get_distance(self, from_entity_id: EntityID, to_entity_id: EntityID) -> float:
    path = self.get_path(from_entity_id, to_entity_id)
    return sum(self._c(path[i], path[i + 1]) for i in range(len(path) - 1))

  def calculate(self) -> DStarLitePathPlanning:
    return self
