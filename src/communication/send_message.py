from __future__ import annotations

from typing import TYPE_CHECKING

from adf_core_python.core.agent.communication.standard.bundle.information.message_road import (
  MessageRoad,
)
from adf_core_python.core.agent.communication.standard.bundle.standard_message_priority import (
  StandardMessagePriority,
)
from adf_core_python.core.component.module.abstract_module import AbstractModule
from rcrscore.entities import Road

if TYPE_CHECKING:
  from adf_core_python.core.agent.communication.message_manager import MessageManager
  from adf_core_python.core.agent.develop.develop_data import DevelopData
  from adf_core_python.core.agent.info.agent_info import AgentInfo
  from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
  from adf_core_python.core.agent.info.world_info import WorldInfo
  from adf_core_python.core.agent.module.module_manager import ModuleManager
  from adf_core_python.core.agent.precompute.precompute_data import PrecomputeData


class SendMessage(AbstractModule):
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

  def _send_my_position(self, message_manager: MessageManager) -> None:
    my_position_entity_id = self._agent_info.get_position_entity_id()
    if my_position_entity_id is None:
      return None
    my_position_entity = self._world_info.get_entity(my_position_entity_id)
    if my_position_entity is None:
      return None
    if not isinstance(my_position_entity, Road):
      return None
    my_entity_id = self._agent_info.get_entity_id()
    if my_entity_id is None:
      return None

    message_road = MessageRoad(
      is_wireless_message=True,
      sender_entity_id=my_entity_id,
      priority=StandardMessagePriority.HIGH,
      road=my_position_entity,
      is_passable=True,
      is_send_blockade_location=False,
      blockade=None,
    )
    message_manager.add_message(message_road)

  def _read_message(self, message_manager: MessageManager) -> None:
    messages = message_manager.get_received_message_list()
    for message in messages:
      if isinstance(message, MessageRoad):
        self._logger.info(
          f"Received MessageRoad: sender_entity_id={message.get_sender_entity_id()}, "
          f"road={message.get_road_entity_id()}, is_passable={message.get_is_passable()}"
        )

  def update_info(self, message_manager: MessageManager) -> SendMessage:
    super().update_info(message_manager)
    self._send_my_position(message_manager)
    self._read_message(message_manager)

    return self

  def calculate(self) -> SendMessage:
    return self

  def precompute(self, precompute_data: PrecomputeData) -> SendMessage:
    super().precompute(precompute_data)
    return self

  def resume(self, precompute_data: PrecomputeData) -> SendMessage:
    super().resume(precompute_data)
    return self

  def prepare(self) -> SendMessage:
    super().prepare()
    return self
