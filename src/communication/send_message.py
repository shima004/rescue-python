from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from adf_core_python.core.agent.communication.standard.bundle.information.message_ambulance_team import (
  MessageAmbulanceTeam,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_civilian import (
  MessageCivilian,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_fire_brigade import (
  MessageFireBrigade,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_police_force import (
  MessagePoliceForce,
)
from adf_core_python.core.agent.communication.standard.bundle.information.message_road import (
  MessageRoad,
)
from adf_core_python.core.agent.communication.standard.bundle.standard_message_priority import (
  StandardMessagePriority,
)
from adf_core_python.core.component.module.abstract_module import AbstractModule
from rcrscore.entities import (
  AmbulanceTeam,
  Civilian,
  EntityID,
  FireBrigade,
  Human,
  PoliceForce,
  Road,
)

from src.utility.agent_status import (
  is_alived,
  is_buried,
  is_damaged,
  is_transported_to_refuge,
)
from src.utility.road_status import has_impassable_edge_pair

if TYPE_CHECKING:
  from adf_core_python.core.agent.communication.message_manager import MessageManager
  from adf_core_python.core.agent.develop.develop_data import DevelopData
  from adf_core_python.core.agent.info.agent_info import AgentInfo
  from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
  from adf_core_python.core.agent.info.world_info import WorldInfo
  from adf_core_python.core.agent.module.module_manager import ModuleManager
  from adf_core_python.core.agent.precompute.precompute_data import PrecomputeData


@dataclass
class MessageTTL:
  ttl: int
  id: int
  message: (
    MessageAmbulanceTeam
    | MessageFireBrigade
    | MessagePoliceForce
    | MessageCivilian
    | MessageRoad
  )


def decrease_message_ttl(message_list: list[MessageTTL]) -> None:
  for message_ttl in message_list:
    message_ttl.ttl -= 1


def has_same_id_message(
  message_list: list[MessageTTL],
  id: int,
) -> bool:
  for message_ttl in message_list:
    if message_ttl.id == id:
      return True
  return False


def add_messages_to_manager(
  message_manager: MessageManager,
  message_list: list[MessageTTL],
  logger,
) -> None:
  for message_ttl in message_list:
    if message_ttl.ttl > 0:
      message_manager.add_message(message_ttl.message)


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

    self.message_list: list[MessageTTL] = []
    self.already_sent_message_buried_agent_entity_ids: set[EntityID] = set()
    self.already_sent_message_damaged_agent_entity_ids: set[EntityID] = set()
    self.already_sent_message_passable_road_entity_ids: set[EntityID] = set()

  def _send_messages(self, message_manager: MessageManager) -> None:
    entity = self._agent_info.get_myself()
    if not isinstance(entity, Human):
      return

    messages = self._create_buried_agent_message()
    self.message_list.extend(messages)
    messages = self._create_damaged_agent_message()
    self.message_list.extend(messages)
    messages = self._create_passable_road_message()
    self.message_list.extend(messages)

    add_messages_to_manager(message_manager, self.message_list, self._logger)
    decrease_message_ttl(self.message_list)

  def _create_buried_agent_message(self) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if not isinstance(entity, Human):
        continue
      if (
        is_buried(entity)
        and entity_id not in self.already_sent_message_buried_agent_entity_ids
      ):
        messages.append(
          MessageTTL(
            ttl=2,
            id=entity_id.get_value(),
            message=self._create_agent_message(entity, is_wireless=True),
          )
        )
        self.already_sent_message_buried_agent_entity_ids.add(entity_id)
    return messages

  def _create_damaged_agent_message(self) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if not isinstance(entity, Civilian):
        continue
      if (
        is_damaged(entity)
        and is_alived(entity)
        and not is_buried(entity)
        and not is_transported_to_refuge(entity, self._world_info)
        and entity_id not in self.already_sent_message_damaged_agent_entity_ids
      ):
        messages.append(
          MessageTTL(
            ttl=2,
            id=entity_id.get_value(),
            message=self._create_agent_message(entity, is_wireless=True),
          )
        )
        self.already_sent_message_damaged_agent_entity_ids.add(entity_id)
    return messages

  def _create_passable_road_message(self) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if not isinstance(entity, Road):
        continue
      if (
        not has_impassable_edge_pair(entity, self._world_info)
        and entity_id not in self.already_sent_message_passable_road_entity_ids
      ):
        messages.append(
          MessageTTL(
            ttl=2,
            id=entity_id.get_value(),
            message=self._create_road_message(entity, is_wireless=True),
          )
        )
        self.already_sent_message_passable_road_entity_ids.add(entity_id)
    return messages

  def _read_message(self, message_manager: MessageManager) -> None:
    messages = message_manager.get_received_message_list()
    for message in messages:
      if isinstance(message, MessageAmbulanceTeam):
        id = message.get_ambulance_team_entity_id()
        if id is not None:
          self.already_sent_message_buried_agent_entity_ids.add(id)
      elif isinstance(message, MessageFireBrigade):
        id = message.get_fire_brigade_entity_id()
        if id is not None:
          self.already_sent_message_buried_agent_entity_ids.add(id)
      elif isinstance(message, MessagePoliceForce):
        id = message.get_police_force_entity_id()
        if id is not None:
          self.already_sent_message_buried_agent_entity_ids.add(id)
      elif isinstance(message, MessageCivilian):
        id = message.get_civilian_entity_id()
        buriedness = message.get_civilian_buriedness()
        if id is not None and buriedness is not None and buriedness > 0:
          self.already_sent_message_buried_agent_entity_ids.add(id)
        damage = message.get_civilian_damage()
        if (
          id is not None
          and damage is not None
          and damage > 0
          and buriedness is not None
          and buriedness == 0
        ):
          self.already_sent_message_damaged_agent_entity_ids.add(id)
      elif isinstance(message, MessageRoad):
        id = message.get_road_entity_id()
        is_passable = message.get_is_passable()
        if id is not None and is_passable is not None and is_passable:
          self.already_sent_message_passable_road_entity_ids.add(id)

  def _create_agent_message(
    self, entity: Human, is_wireless: bool
  ) -> MessageAmbulanceTeam | MessageFireBrigade | MessagePoliceForce | MessageCivilian:
    if isinstance(entity, AmbulanceTeam):
      return MessageAmbulanceTeam(
        is_wireless_message=is_wireless,
        ambulance_team=entity,
        action=0,
        target_entity_id=None,  # ty:ignore[invalid-argument-type]
        priority=StandardMessagePriority.HIGH,
        sender_entity_id=entity.get_entity_id(),
      )
    elif isinstance(entity, FireBrigade):
      return MessageFireBrigade(
        is_wireless_message=is_wireless,
        fire_brigade=entity,
        action=0,
        target_entity_id=None,  # ty:ignore[invalid-argument-type]
        priority=StandardMessagePriority.HIGH,
        sender_entity_id=entity.get_entity_id(),
      )
    elif isinstance(entity, PoliceForce):
      return MessagePoliceForce(
        is_wireless_message=is_wireless,
        police_force=entity,
        action=0,
        target_entity_id=None,  # ty:ignore[invalid-argument-type]
        priority=StandardMessagePriority.HIGH,
        sender_entity_id=entity.get_entity_id(),
      )
    elif isinstance(entity, Civilian):
      return MessageCivilian(
        is_wireless_message=is_wireless,
        civilian=entity,
        priority=StandardMessagePriority.HIGH,
        sender_entity_id=entity.get_entity_id(),
      )
    else:
      raise ValueError("Unknown agent type")

  def _create_road_message(self, entity: Road, is_wireless: bool) -> MessageRoad:
    return MessageRoad(
      is_wireless_message=is_wireless,
      road=entity,
      is_send_blockade_location=True,
      is_passable=True,
      blockade=None,
      priority=StandardMessagePriority.NORMAL,
      sender_entity_id=entity.get_entity_id(),
    )

  def update_info(self, message_manager: MessageManager) -> SendMessage:
    super().update_info(message_manager)
    self._read_message(message_manager)
    self._send_messages(message_manager)

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
