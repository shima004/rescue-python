from enum import Enum

from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo, ScenarioInfoKeys
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.action.extend_action import ExtendAction
from rcrscore.entities import (
  Blockade,
  Building,
  EntityID,
  Road,
)

from src.utility.road_status import get_blockade_polygon, get_road_polygon


class ActionType(Enum):
  CLEAR_AREA = "clear_area"
  MOVE = "move"


AGENT_MOVE_DISTANCE = 42000


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
    return self

  def _create_clear_strategy(
    self, target_entity_id: EntityID
  ) -> list[tuple[ActionType, tuple[float, float]]]:
    target_road_entity = self.world_info.get_entity(target_entity_id)
    if target_road_entity is None or not isinstance(target_road_entity, Road):
      return []
    agent = self.agent_info.get_myself()
    if agent is None:
      return []
    agent_position_x, agent_position_y = agent.get_location()
    if agent_position_x is None or agent_position_y is None:
      return []

    road_polygon = get_road_polygon(target_road_entity)
    blockades = self.world_info.get_blockades(target_road_entity)
    blockade_polygons = [get_blockade_polygon(blockade) for blockade in blockades]

    # 指定された座標とエージェントの座標からなる方向ベクトルをclear_distanceの長さに伸ばし、それに対して直角方向にclear_radの長さのベクトルを加算した矩形が瓦礫の除去範囲となり、エージェントに近い瓦礫の部分から面積がclear_amountの分だけ減少する
    clear_distance: float = self.scenario_info.get_value(
      ScenarioInfoKeys.CLEAR_REPAIR_DISTANCE, 8000.0
    )
    clear_rad: float = self.scenario_info.get_value(
      ScenarioInfoKeys.CLEAR_REPAIR_RAD, 1000.0
    )
    clear_amount: float = self.scenario_info.get_value(
      ScenarioInfoKeys.CLEAR_REPAIR_RATE, 20
    )

    return []
