import math
import threading
from typing import cast

from adf_core_python.core.agent.action.common.action_move import ActionMove
from adf_core_python.core.agent.action.police.action_clear import ActionClear
from adf_core_python.core.agent.action.police.action_clear_area import ActionClearArea
from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo, ScenarioInfoKeys
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.agent.precompute.precompute_data import PrecomputeData
from adf_core_python.core.component.action.extend_action import ExtendAction
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from adf_core_python.core.logger.logger import get_agent_logger
from rcrscore.entities import (
  Blockade,
  Building,
  Edge,
  EntityID,
  Road,
)
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union

from src.utility.road_status import (
  get_blockade_polygon,
  get_impassable_edges,
  get_road_polygon,
)

AGENT_MOVE_DISTANCE = 42000
AGENT_RADIUS = 500
DEBUG = False


class ExtendActionClear(ExtendAction):
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

    self.logger = get_agent_logger(__name__, self.agent_info)
    self.previous_position: Point | None = None
    self.previous_action: ActionMove | ActionClearArea | None = None

    self._path_planning: PathPlanning = cast(
      PathPlanning,
      module_manager.get_module(
        "ExtendActionClear.PathPlanning",
        "adf_core_python.implement.module.algorithm.a_star_path_planning.AStarPathPlanning",
      ),
    )

  def set_target_entity_id(self, target_entity_id: EntityID) -> ExtendAction:
    self._target_entity_id = None
    target_entity = self.world_info.get_entity(target_entity_id)
    if target_entity is not None:
      if isinstance(target_entity, Road):
        self._target_entity_id = target_entity_id
      elif isinstance(target_entity, Blockade):
        self._target_entity_id = target_entity.get_position()
      elif isinstance(target_entity, Building):
        self._target_entity_id = target_entity_id
    return self

  def calculate(self) -> ExtendAction:
    if self._target_entity_id is not None:
      self.result = self._create_clear_action()

    self.previous_action = self.result  # type: ignore
    self.previous_position = self._get_my_position()

    return self

  def _create_clear_action(
    self,
  ) -> ActionMove | ActionClearArea | ActionClear | None:
    target_road_entity: Road = self._get_target_road_entity()
    agent_position: Point = self._get_my_position()
    clear_distance: float = self.scenario_info.get_value(
      ScenarioInfoKeys.CLEAR_REPAIR_DISTANCE, 8000.0
    )
    clear_rad: float = self.scenario_info.get_value(
      ScenarioInfoKeys.CLEAR_REPAIR_RAD, 1000.0
    )
    # Amount of blockade area removed per clear action
    clear_rate: float = (
      self.scenario_info.get_value(ScenarioInfoKeys.CLEAR_REPAIR_RATE, 1000.0)
      * 1000000
      * 0.9
    )

    blockades = self.world_info.get_blockades(target_road_entity)
    if not blockades:
      return None
    blockade_union = unary_union([get_blockade_polygon(b) for b in blockades])
    if blockade_union.is_empty:
      return None
    impassable_edges = get_impassable_edges(target_road_entity, self.world_info)

    # 前のアクションがMOVEで、かつ前の位置からあまり動けていない（＝クリアが必要な瓦礫に近づいているのに動けていない）場合は、クリアを優先する
    is_stack = False
    if (
      isinstance(self.previous_action, ActionMove)
      and self.previous_position is not None
    ):
      dist_moved = agent_position.distance(self.previous_position)
      if dist_moved < AGENT_MOVE_DISTANCE * 0.05:
        self.logger.debug(
          f"Previous action was MOVE but agent only moved {dist_moved:.1f} units, which is less than half of expected {AGENT_MOVE_DISTANCE}. Prioritizing clear action."
        )
        is_stack = True

    # 自分の位置の半径AGENT_RADIUS内に瓦礫がある場合は、まずはその瓦礫をクリアすることを優先する
    agent_position_blockades = self.world_info.get_blockades(
      self.world_info.get_entity_position_entity(self.agent_info.get_entity_id())  # type: ignore
    )
    if agent_position_blockades:
      agent_blockade_union = unary_union(
        [get_blockade_polygon(b) for b in agent_position_blockades]
      )
      if agent_blockade_union.is_empty:
        return None
      if agent_blockade_union.distance(agent_position) < AGENT_RADIUS:
        self.logger.debug(
          f"Agent is within {AGENT_RADIUS} of blockade at its position, prioritizing clearing it."
        )

        if self.agent_info.get_time() < 20:
          # もっとも近い瓦礫を対象にする
          nearest_blockade = min(
            agent_position_blockades,
            key=lambda b: agent_position.distance(get_blockade_polygon(b)),
          )
          return ActionClear(nearest_blockade)
        else:
          blockade_union = agent_blockade_union
          impassable_edges = get_impassable_edges(
            self.world_info.get_entity_position_entity(self.agent_info.get_entity_id()),  # type: ignore
            self.world_info,
          )
          self._target_entity_id = self.agent_info.get_position_entity_id()

    # For each candidate target point (agent→blockade centroid direction, at clear_distance),
    # compute how much blockade area the clear rectangle would cover.
    # The clear rectangle is the line from agent to target buffered by clear_rad.
    best_clear_point: Point | None = None
    best_clear_area: float = -1.0

    # Candidate targets: one per decomposed blockade piece, aimed from the agent.
    geoms = (
      list(blockade_union.geoms)
      if hasattr(blockade_union, "geoms")
      else [blockade_union]
    )
    candidate_sweeps: list[
      tuple[Point, float]
    ] = []  # (target, covered_area) for debug plot
    for geom in geoms:
      centroid = geom.centroid
      dx = centroid.x - agent_position.x
      dy = centroid.y - agent_position.y
      dist = math.hypot(dx, dy)
      if dist == 0:
        continue
      # Target point: agent position + unit vector * clear_distance
      target = Point(
        agent_position.x + dx / dist * clear_distance,
        agent_position.y + dy / dist * clear_distance,
      )
      sweep = LineString([agent_position, target]).buffer(
        clear_rad, cap_style=2
      )  # Square end cap for more consistent coverage area
      covered_area = sweep.intersection(blockade_union).area
      candidate_sweeps.append((target, covered_area))
      if covered_area > best_clear_area:
        best_clear_area = covered_area
        best_clear_point = target

    # If the best candidate covers enough blockade area, clear it.
    # if best_clear_point is not None and best_clear_area >= clear_rate:
    if best_clear_point is not None and best_clear_area > 0:
      # 対象となっている瓦礫とbest_clear_areaの割合を求める
      total_blockade_area = blockade_union.area
      blockade_clear_rate = (
        best_clear_area / total_blockade_area * 100 if total_blockade_area > 0 else 0
      )
      self.logger.debug(
        f"Best clear candidate covers {best_clear_area:.1f} area, which is {blockade_clear_rate:.1f}% of the total blockade area. clear_rate threshold is {clear_rate:.1f}."
      )
      if (
        best_clear_area > clear_rate or blockade_clear_rate > 20 or is_stack
      ):  # 20%以上をクリアできれば実行
        self.logger.debug(
          f"CLEAR_AREA at ({best_clear_point.x:.0f}, {best_clear_point.y:.0f}), "
          f"covered area: {best_clear_area:.1f} >= clear_rate: {clear_rate:.1f}"
        )
        if DEBUG:
          threading.Thread(
            target=self._debug_plot,
            args=(
              get_road_polygon(target_road_entity),
              blockade_union,
              agent_position,
              clear_distance,
              clear_rad,
              candidate_sweeps,
              best_clear_point,
              None,
              None,
              [],
              impassable_edges,
              self.agent_info.get_entity_id(),
            ),
            daemon=True,
          ).start()
        return ActionClearArea(
          position_x=int(best_clear_point.x), position_y=int(best_clear_point.y)
        )

    # No position currently satisfies clear_rate.
    # Find the best reachable move target.
    agent_position_entity_id = self.agent_info.get_position_entity_id()
    if agent_position_entity_id is None:
      return None
    assert self._target_entity_id is not None
    path = self._path_planning.get_path(
      from_entity_id=agent_position_entity_id,
      to_entity_id=self._target_entity_id,
    )

    # Build the reachable road polygon: union of all road polygons along the path.
    road_polygons = []
    for eid in path:
      entity = self.world_info.get_entity(eid)
      if isinstance(entity, Road):
        road_polygons.append(get_road_polygon(entity))
    reachable_roads = (
      unary_union(road_polygons)
      if road_polygons
      else get_road_polygon(target_road_entity)
    )

    # Subtract blockades buffered by AGENT_RADIUS — these are physically impassable.
    passable = reachable_roads.difference(blockade_union.buffer(AGENT_RADIUS))

    # Find the connected component that contains the agent's current position.
    passable_geoms = list(passable.geoms) if hasattr(passable, "geoms") else [passable]
    agent_component = next(
      (g for g in passable_geoms if g.distance(agent_position) < AGENT_RADIUS),
      passable_geoms[0] if passable_geoms else None,
    )

    if agent_component is None or agent_component.is_empty:
      return None

    # Candidate stand points: points on the agent_component boundary that face the
    # blockade (i.e. the boundary segment adjacent to the buffered blockade).
    # Sample evenly along this "facing boundary" to get discrete candidates.
    facing_boundary = agent_component.boundary.intersection(
      blockade_union.buffer(AGENT_RADIUS)
    )

    sample_count = 16
    if facing_boundary.is_empty:
      # No direct contact — fall back to the nearest point on the boundary.
      facing_boundary = nearest_points(blockade_union, agent_component.boundary)[1]
      stand_candidates: list[Point] = [facing_boundary]
    else:
      boundary_len = facing_boundary.length
      step = boundary_len / sample_count if boundary_len > 0 else 0
      stand_candidates = (
        [facing_boundary.interpolate(step * i) for i in range(sample_count + 1)]
        if step > 0
        else [facing_boundary.centroid]
      )

    best_move_target: Point | None = None
    best_move_coverage: float = -1.0
    # (stand, aim_target, coverage) for each candidate — used for debug visualisation
    move_candidates: list[tuple[Point, Point, float]] = []
    for stand in stand_candidates:
      # Aim toward the nearest point on the blockade from this stand position.
      aim = nearest_points(stand, blockade_union)[1]
      dx = aim.x - stand.x
      dy = aim.y - stand.y
      dist = math.hypot(dx, dy)
      if dist == 0:
        continue
      stand_target = Point(
        stand.x + dx / dist * clear_distance,
        stand.y + dy / dist * clear_distance,
      )
      sweep = LineString([stand, stand_target]).buffer(clear_rad, cap_style=2)
      coverage = sweep.intersection(blockade_union).area
      move_candidates.append((stand, stand_target, coverage))
      if coverage > best_move_coverage:
        best_move_coverage = coverage
        best_move_target = stand

    if best_move_target is None:
      return None

    move_target = best_move_target

    self.logger.debug(
      f"MOVE to ({move_target.x:.0f}, {move_target.y:.0f}: {path}) to reach blockade, "
      f"best covered area was {best_clear_area:.1f} < clear_rate: {clear_rate:.1f}, "
      f"expected coverage from move target: {best_move_coverage:.1f}"
    )
    if DEBUG:
      threading.Thread(
        target=self._debug_plot,
        args=(
          get_road_polygon(target_road_entity),
          blockade_union,
          agent_position,
          clear_distance,
          clear_rad,
          candidate_sweeps,
          None,
          move_target,
          agent_component,
          move_candidates,
          impassable_edges,
          self.agent_info.get_entity_id(),
        ),
        daemon=True,
      ).start()
    return ActionMove(
      path=path,
      destination_x=int(move_target.x),
      destination_y=int(move_target.y),
    )

  @staticmethod
  def _debug_plot(
    road_polygon: Polygon,
    blockade_union: Polygon,
    agent_position: Point,
    clear_distance: float,
    clear_rad: float,
    candidate_sweeps: list[tuple[Point, float]],
    clear_point: Point | None,
    move_target: Point | None,
    agent_component: Polygon | None,
    move_candidates: list[tuple[Point, Point, float]],
    impassable_edges: set[Edge],
    id: EntityID | None = None,
  ) -> None:
    """Save a debug plot of the clear action candidates to debug_clear.png."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    # Use Figure/FigureCanvasAgg directly to avoid the pyplot global state,
    # which is not thread-safe in Python 3.14t (free-threaded build).
    fig = Figure(figsize=(8, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Reachable component (agent can physically reach here)
    if agent_component is not None and not agent_component.is_empty:
      comp_geoms = (
        list(agent_component.geoms)
        if hasattr(agent_component, "geoms")
        else [agent_component]
      )
      for i, cg in enumerate(comp_geoms):
        if isinstance(cg, Polygon) and not cg.is_empty:
          cx, cy = cg.exterior.xy
          ax.fill(
            cx,
            cy,
            color="lightblue",
            alpha=0.3,
            label="Reachable area" if i == 0 else None,
          )

    # Road outline
    if not road_polygon.is_empty:
      rx, ry = road_polygon.exterior.xy
      ax.plot(rx, ry, color="black", linewidth=1.5, label="Road")

    # Blockade union (may be multi-polygon)
    geoms = (
      list(blockade_union.geoms)
      if hasattr(blockade_union, "geoms")
      else [blockade_union]
    )
    for i, g in enumerate(geoms):
      if isinstance(g, Polygon) and not g.is_empty:
        bx, by = g.exterior.xy
        ax.fill(bx, by, color="red", alpha=0.4, label="Blockade" if i == 0 else None)
        ax.plot(bx, by, color="red", linewidth=0.8)

    # All candidate sweeps (grey)
    for target, covered_area in candidate_sweeps:
      sweep = LineString([agent_position, target]).buffer(clear_rad, cap_style=2)
      if isinstance(sweep, Polygon) and not sweep.is_empty:
        sx, sy = sweep.exterior.xy
        ax.fill(sx, sy, color="grey", alpha=0.15)
        ax.plot(sx, sy, color="grey", linewidth=0.5)
      ax.annotate(
        f"{covered_area:.0f}",
        xy=(target.x, target.y),
        fontsize=6,
        color="grey",
        ha="center",
      )

    # Best clear sweep (cyan) or move target (orange)
    if clear_point is not None:
      best_sweep = LineString([agent_position, clear_point]).buffer(
        clear_rad, cap_style=2
      )
      if isinstance(best_sweep, Polygon) and not best_sweep.is_empty:
        sx, sy = best_sweep.exterior.xy
        ax.fill(sx, sy, color="cyan", alpha=0.4, label="Best sweep")
        ax.plot(sx, sy, color="cyan", linewidth=1.2)
      ax.scatter(
        clear_point.x, clear_point.y, color="cyan", zorder=5, label="CLEAR target"
      )

    # Move candidates: each stand point, its sweep, and coverage score
    best_cov = max((cov for _, _, cov in move_candidates), default=-1.0)
    for stand, aim, cov in move_candidates:
      is_best = cov == best_cov
      color = "orange" if is_best else "yellow"
      sweep = LineString([stand, aim]).buffer(clear_rad, cap_style=2)
      if isinstance(sweep, Polygon) and not sweep.is_empty:
        sx, sy = sweep.exterior.xy
        ax.fill(sx, sy, color=color, alpha=0.25)
        ax.plot(sx, sy, color=color, linewidth=0.6)
      ax.scatter(stand.x, stand.y, color=color, marker="s", s=30, zorder=4)
      ax.annotate(
        f"{cov:.0f}",
        xy=(stand.x, stand.y),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=6,
        color=color,
        ha="center",
      )

    if move_target is not None:
      ax.scatter(
        move_target.x,
        move_target.y,
        color="orange",
        zorder=6,
        marker="^",
        s=100,
        label="MOVE target (best)",
      )
      ax.annotate(
        "MOVE",
        xy=(move_target.x, move_target.y),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=7,
        color="orange",
        ha="center",
      )

    if impassable_edges:
      for edge in impassable_edges:
        ex = [edge.get_start_x(), edge.get_end_x()]
        ey = [edge.get_start_y(), edge.get_end_y()]
        ax.plot(ex, ey, color="magenta", linewidth=2, label="Impassable edge")

    # Agent position
    ax.scatter(
      agent_position.x, agent_position.y, color="green", zorder=6, s=80, label="Agent"
    )
    # Clear distance circle
    circle = Circle(
      (agent_position.x, agent_position.y),
      clear_distance,
      color="green",
      fill=False,
      linestyle="--",
      linewidth=0.8,
      label=f"clear_distance={clear_distance:.0f}",
    )
    ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("ExtendActionClear debug")
    fig.tight_layout()
    filename = f"debug_clear_{id}.png"
    fig.savefig(filename, dpi=120)
    fig.clf()

  def _get_target_road_entity(self) -> Road:
    if self._target_entity_id is None:
      self.logger.warning("Target entity ID is not set. Cannot get target road entity.")
      raise ValueError("Target entity ID is not set.")
    target_entity = self.world_info.get_entity(self._target_entity_id)
    if target_entity is None or not isinstance(target_entity, Road):
      self.logger.warning(
        f"Target entity ID {self._target_entity_id} is not a valid Road. Cannot get target road entity."
      )
      raise ValueError(f"Invalid target entity ID {self._target_entity_id}.")
    return target_entity

  def _get_my_position(self) -> Point:
    agent = self.agent_info.get_myself()
    if agent is None:
      self.logger.warning(
        "Agent information is not available. Cannot get agent position."
      )
      raise ValueError("Agent information is not available.")
    agent_position_x, agent_position_y = agent.get_location()
    if agent_position_x is None or agent_position_y is None:
      self.logger.warning("Agent position is not available. Cannot get agent position.")
      raise ValueError("Agent position is not available.")
    return Point(agent_position_x, agent_position_y)

  def _get_edge_midpoint(self, edge: Edge) -> Point:
    start_x, start_y = edge.get_start_x(), edge.get_start_y()
    end_x, end_y = edge.get_end_x(), edge.get_end_y()
    return Point((start_x + end_x) / 2, (start_y + end_y) / 2)

  def precompute(self, precompute_data: PrecomputeData) -> ExtendAction:
    super().precompute(precompute_data)
    if self.get_count_precompute() > 1:
      return self
    self._path_planning.precompute(precompute_data)
    return self

  def resume(self, precompute_data: PrecomputeData) -> ExtendAction:
    super().resume(precompute_data)
    if self.get_count_resume() > 1:
      return self
    self._path_planning.resume(precompute_data)
    return self

  def prepare(self) -> ExtendAction:
    super().prepare()
    if self.get_count_prepare() > 1:
      return self
    self._path_planning.prepare()
    return self

  def update_info(self, message_manager: MessageManager) -> ExtendAction:
    super().update_info(message_manager)
    if self.get_count_update_info() > 1:
      return self
    self._path_planning.update_info(message_manager)
    return self
