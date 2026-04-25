# Refuge selection module based on M/M/c queueing model.
# Provides build_refuge_info() to snapshot the world state, and
# select_refuge() to pick the best destination for a transported civilian.

from __future__ import annotations

import math

from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from rcrscore.entities import Civilian, EntityID, Human, Refuge

HUMAN_MAX_HP = 10_000


class RefugeInfo:
  """Snapshot of a refuge's capacity and the injuries of civilians inside it."""

  def __init__(self, refuge: Refuge) -> None:
    self._bed_capacity: int = refuge.get_bed_capacity() or 1
    # Maps civilian ID → current damage value
    self._civilian_damages: dict[EntityID, int] = {}

  def get_bed_capacity(self) -> int:
    return self._bed_capacity

  def count_civilians(self) -> int:
    return len(self._civilian_damages)

  def total_civilian_damage(self) -> int:
    return sum(self._civilian_damages.values())

  def mean_civilian_damage(self) -> float:
    count = self.count_civilians()
    if count == 0:
      return 0.0
    return self.total_civilian_damage() / count

  def _add_civilian(self, civilian_id: EntityID, damage: int) -> None:
    self._civilian_damages[civilian_id] = damage


def build_refuge_info(world_info: WorldInfo) -> dict[EntityID, RefugeInfo]:
  """Build a fresh refuge snapshot from the current world model.

  Returns a dict mapping refuge EntityID → RefugeInfo, with each refuge's
  injured civilians already recorded. The caller is responsible for storing
  and refreshing this dict each tick.
  """
  refuges: dict[EntityID, RefugeInfo] = {}

  for refuge_id in world_info.get_entity_ids_of_types([Refuge]):
    entity = world_info.get_entity(refuge_id)
    if isinstance(entity, Refuge):
      refuges[refuge_id] = RefugeInfo(entity)

  for civilian_id in world_info.get_entity_ids_of_types([Civilian]):
    entity = world_info.get_entity(civilian_id)
    if not isinstance(entity, Civilian):
      continue
    position = entity.get_position()
    if position not in refuges:
      continue
    hp = entity.get_hp()
    # Skip healthy civilians (HP at maximum means no injury)
    if hp is None or hp >= HUMAN_MAX_HP:
      continue
    damage = entity.get_damage()
    if damage is None:
      continue
    refuges[position]._add_civilian(civilian_id, damage)

  return refuges


def select_refuge(
  transporting_human: Human,
  refuges: dict[EntityID, RefugeInfo],
  agent_info: AgentInfo,
  path_planning: PathPlanning,
) -> EntityID | None:
  """Return the refuge with the lowest expected time-to-bed using M/M/c model.

  For each refuge, the cost is max(travel_time, wait_time): the civilian
  cannot get a bed before the ambulance arrives AND before a bed is free.
  Returns None if refuges is empty.
  """
  if not refuges:
    return None

  refuge_costs: dict[EntityID, float] = {
    refuge_id: max(
      _estimate_travel_time(refuge_id, agent_info, path_planning),
      _estimate_wait_time(refuge_info, agent_info),
    )
    for refuge_id, refuge_info in refuges.items()
  }

  return min(refuge_costs, key=lambda k: refuge_costs[k])


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _estimate_travel_time(
  refuge_id: EntityID, agent_info: AgentInfo, path_planning: PathPlanning
) -> float:
  """Estimate travel time as the number of hops in the planned path."""
  from_position = agent_info.get_position_entity_id()
  if from_position is None:
    return float("inf")
  path = path_planning.get_path(from_position, refuge_id)
  return float(len(path)) if path else float("inf")


def _estimate_wait_time(refuge_info: RefugeInfo, agent_info: AgentInfo) -> float:
  """Estimate queue wait time at the refuge using the M/M/c formula."""
  if refuge_info.count_civilians() == 0:
    return 0.0

  # Number of servers (beds)
  c = max(refuge_info.get_bed_capacity(), 1)

  current_time = agent_info.get_time()
  if current_time is None or current_time == 0:
    return 0.0

  # Average arrival rate: civilians observed / elapsed time
  lambda_ = refuge_info.count_civilians() / current_time

  mean_damage = refuge_info.mean_civilian_damage()
  if mean_damage <= 0:
    return 0.0

  # Average service rate: 1 / mean service time (proxied by mean damage)
  mu = 1.0 / mean_damage
  # Traffic intensity per server
  rho = lambda_ / (c * mu)

  # M/M/c is only stable when ρ < 1
  if rho >= 1.0:
    return float("inf")

  pi_zero = _calc_empty_queue_probability(c, rho)
  return _calc_average_wait_time(c, lambda_, rho, pi_zero)


def _calc_empty_queue_probability(c: int, rho: float) -> float:
  """Compute P0 — the probability that all servers are idle (Erlang-C).

  Uses log-space arithmetic to avoid OverflowError when c is large.
  """
  log_crho = math.log(c * rho)
  # log((c*rho)^k / k!) for k in 0..c-1
  log_terms = [k * log_crho - math.lgamma(k + 1) for k in range(c)]
  # log((c*rho)^c / c! / (1-rho))
  log_terms.append(c * log_crho - math.lgamma(c + 1) - math.log(1.0 - rho))
  max_log = max(log_terms)
  log_total = max_log + math.log(sum(math.exp(lt - max_log) for lt in log_terms))
  return math.exp(-log_total)


def _calc_average_wait_time(
  c: int, lambda_: float, rho: float, pi_zero: float
) -> float:
  """Compute Wq — the average time a customer waits in the queue.

  Uses log-space arithmetic to avoid OverflowError when c is large.
  """
  log_wq = (
    math.log(pi_zero)
    + math.log(rho)
    + c * math.log(c * rho)
    - math.lgamma(c + 1)
    - math.log(lambda_)
    - 2.0 * math.log(1.0 - rho)
  )
  return math.exp(log_wq)
