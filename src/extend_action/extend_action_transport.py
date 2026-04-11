from typing import cast

from adf_core_python.core.agent.action.ambulance.action_load import ActionLoad
from adf_core_python.core.agent.action.ambulance.action_unload import ActionUnload
from adf_core_python.core.agent.action.common.action_move import ActionMove
from adf_core_python.core.agent.action.common.action_rest import ActionRest
from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.agent.precompute.precompute_data import PrecomputeData
from adf_core_python.core.component.action.extend_action import ExtendAction
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from adf_core_python.core.logger.logger import get_agent_logger
from rcrscore.entities import (
  AmbulanceTeam,
  Area,
  EntityID,
  Human,
  Refuge,
)

from src.utility.agent_status import (
  is_alived,
  is_buried,
  is_damaged,
  is_transporting_by_another_ambulance,
)
from src.utility.refuge_selection import RefugeInfo, build_refuge_info, select_refuge


class ExtendActionTransport(ExtendAction):
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
    self._target_entity_id: EntityID | None = None
    self._logger = get_agent_logger(
      f"{self.__class__.__module__}.{self.__class__.__qualname__}",
      self.agent_info,
    )

    self._path_planning: PathPlanning = cast(
      PathPlanning,
      self.module_manager.get_module(
        "ExtendActionTransport.PathPlanning",
        "adf_core_python.implement.module.algorithm.a_star_path_planning.AStarPathPlanning",
      ),
    )
    self._refuges: dict[EntityID, RefugeInfo] = {}

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
    self._refuges = build_refuge_info(self.world_info)
    return self

  def set_target_entity_id(self, target_entity_id: EntityID) -> ExtendAction:
    entity = self.world_info.get_entity(target_entity_id)
    if isinstance(entity, Human) or isinstance(entity, Area):
      self._target_entity_id = target_entity_id
    else:
      self._target_entity_id = None
    return self

  def calculate(self) -> ExtendAction:
    self._result = None
    agent = cast(AmbulanceTeam, self.agent_info.get_myself())

    transporting_human = self.agent_info.some_one_on_board()
    if transporting_human is not None:
      self.result = self._calc_transporting_human_action(transporting_human)
      if self.result is not None:
        return self

    if self._target_entity_id is not None:
      self.result = self._calc_rescue(
        agent, self._path_planning, self._target_entity_id
      )

    return self

  def _calc_transporting_human_action(
    self,
    transport_human: Human,
  ) -> ActionMove | ActionUnload | ActionRest | None:
    if not is_alived(transport_human) or not is_damaged(transport_human):
      return ActionUnload()

    if (
      self._target_entity_id is not None
      and transport_human.get_entity_id() != self._target_entity_id
    ):
      return ActionUnload()

    agent_position_entity_id = self.agent_info.get_position_entity_id()
    if agent_position_entity_id is None:
      return None
    agent_position_entity = self.world_info.get_entity(agent_position_entity_id)
    if agent_position_entity is None:
      return None
    if isinstance(agent_position_entity, Refuge):
      return ActionUnload()

    path = self._get_best_refuge_path(agent_position_entity_id, transport_human)
    if len(path) > 0:
      return ActionMove(path)

    return None

  def _calc_rescue(
    self,
    agent: AmbulanceTeam,
    path_planning: PathPlanning,
    target_entity_id: EntityID,
  ) -> ActionMove | ActionLoad | None:
    target_entity = self.world_info.get_entity(target_entity_id)
    if target_entity is None:
      return None

    agent_position = agent.get_position()
    if agent_position is None:
      return None

    if isinstance(target_entity, Human):
      if (
        not is_alived(target_entity)
        or not is_damaged(target_entity)
        or is_buried(target_entity)
        or is_transporting_by_another_ambulance(
          target_entity, self.world_info, agent.get_entity_id()
        )
      ):
        return None

      target_position = target_entity.get_position()
      if target_position is None:
        return None

      if agent_position == target_position:
        return ActionLoad(target_entity.get_entity_id())
      else:
        path = path_planning.get_path(agent_position, target_position)
        if len(path) > 0:
          return ActionMove(path)

    if isinstance(target_entity, Area):
      path = path_planning.get_path(agent_position, target_entity.get_entity_id())
      if len(path) > 0:
        return ActionMove(path)

    return None

  def _get_best_refuge_path(
    self, from_position_entity_id: EntityID, transporting_human: Human
  ) -> list[EntityID]:
    refuge_id = select_refuge(
      transporting_human, self._refuges, self.agent_info, self._path_planning
    )
    if refuge_id is None:
      return []
    path = self._path_planning.get_path(from_position_entity_id, refuge_id)
    return path if path else []
