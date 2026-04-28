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
from adf_core_python.core.agent.info.scenario_info import ScenarioInfoKeys
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
  is_transporting,
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
    self.already_sent_message_stuck_road_entity_ids: set[EntityID] = set()

  def _get_simulation_phase(self) -> int:
    current_time = self._agent_info.get_time()
    max_steps = self._scenario_info.get_value(ScenarioInfoKeys.KERNEL_TIMESTEPS, 300)
    ratio = current_time / max_steps if max_steps > 0 else 0.0
    if ratio < 0.3:
      return 1
    elif ratio < 0.5:
      return 2
    elif ratio < 0.7:
      return 3
    else:
      return 4

  def _send_messages(self, message_manager: MessageManager) -> None:
    entity = self._agent_info.get_myself()
    if not isinstance(entity, Human):
      return

    phase = self._get_simulation_phase()
    self._logger.debug(
      "Communication phase", phase=phase, time=self._agent_info.get_time()
    )

    if phase == 1:
      self.message_list.extend(
        self._create_stuck_road_message(StandardMessagePriority.HIGH)
      )
      self.message_list.extend(
        self._create_buried_agent_message(
          rescue_priority=StandardMessagePriority.NORMAL,
          civilian_priority=None,
        )
      )
      self.message_list.extend(
        self._create_damaged_agent_message(StandardMessagePriority.LOW)
      )
      self.message_list.extend(
        self._create_passable_road_message(StandardMessagePriority.LOW)
      )
    elif phase == 2:
      self.message_list.extend(
        self._create_buried_agent_message(
          rescue_priority=StandardMessagePriority.HIGH,
          civilian_priority=StandardMessagePriority.NORMAL,
        )
      )
      self.message_list.extend(
        self._create_damaged_agent_message(StandardMessagePriority.NORMAL)
      )
      self.message_list.extend(
        self._create_passable_road_message(StandardMessagePriority.NORMAL)
      )
    elif phase == 3:
      self.message_list.extend(
        self._create_buried_agent_message(
          rescue_priority=StandardMessagePriority.NORMAL,
          civilian_priority=StandardMessagePriority.HIGH,
        )
      )
      self.message_list.extend(
        self._create_damaged_agent_message(StandardMessagePriority.HIGH)
      )
      self.message_list.extend(
        self._create_passable_road_message(StandardMessagePriority.LOW)
      )
    else:  # phase == 4
      self.message_list.extend(
        self._create_buried_agent_message(
          rescue_priority=StandardMessagePriority.LOW,
          civilian_priority=StandardMessagePriority.HIGH,
        )
      )
      self.message_list.extend(
        self._create_damaged_agent_message(StandardMessagePriority.HIGH)
      )
      self.message_list.extend(
        self._create_passable_road_message(StandardMessagePriority.LOW)
      )

    add_messages_to_manager(message_manager, self.message_list)
    decrease_message_ttl(self.message_list)

  def _create_buried_agent_message(
    self,
    rescue_priority: StandardMessagePriority | None,
    civilian_priority: StandardMessagePriority | None,
  ) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if not isinstance(entity, Human):
        continue
      if (
        is_buried(entity)
        and entity_id not in self.already_sent_message_buried_agent_entity_ids
      ):
        priority = (
          civilian_priority if isinstance(entity, Civilian) else rescue_priority
        )
        if priority is None:
          continue
        messages.append(
          MessageTTL(
            ttl=2,
            id=entity_id.get_value(),
            message=self._create_agent_message(
              entity, is_wireless=True, priority=priority
            ),
          )
        )
        self._logger.debug(
          "Send buried agent message",
          entity_id=entity_id.get_value(),
          priority=priority.name,
        )
        self.already_sent_message_buried_agent_entity_ids.add(entity_id)
    return messages

  def _create_damaged_agent_message(
    self, priority: StandardMessagePriority
  ) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if not isinstance(entity, Civilian):
        continue
      if (
        is_damaged(entity)
        and is_alived(entity)
        and not is_buried(entity)
        and not is_transporting(entity, self._world_info)
        and not is_transported_to_refuge(entity, self._world_info)
        and entity_id not in self.already_sent_message_damaged_agent_entity_ids
      ):
        messages.append(
          MessageTTL(
            ttl=2,
            id=entity_id.get_value(),
            message=self._create_agent_message(
              entity, is_wireless=True, priority=priority
            ),
          )
        )
        self._logger.debug(
          "Send damaged civilian message",
          entity_id=entity_id.get_value(),
          priority=priority.name,
        )
        self.already_sent_message_damaged_agent_entity_ids.add(entity_id)
    return messages

  def _create_passable_road_message(
    self, priority: StandardMessagePriority
  ) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if not isinstance(entity, Road):
        continue
      if (
        entity_id not in self.already_sent_message_passable_road_entity_ids
        and not has_impassable_edge_pair(entity, self._world_info)
      ):
        messages.append(
          MessageTTL(
            ttl=2,
            id=entity_id.get_value(),
            message=self._create_road_message(
              entity, is_wireless=True, priority=priority
            ),
          )
        )
        self._logger.debug(
          "Send passable road message",
          entity_id=entity_id.get_value(),
          priority=priority.name,
        )
        self.already_sent_message_passable_road_entity_ids.add(entity_id)
    return messages

  def _create_stuck_road_message(
    self, priority: StandardMessagePriority
  ) -> list[MessageTTL]:
    messages: list[MessageTTL] = []
    entity = self._agent_info.get_myself()
    if not isinstance(entity, (FireBrigade, AmbulanceTeam)):
      return messages
    position_id = entity.get_position()
    if position_id is None:
      return messages
    road = self._world_info.get_entity(position_id)
    if not isinstance(road, Road):
      return messages
    if (
      position_id not in self.already_sent_message_stuck_road_entity_ids
      and has_impassable_edge_pair(road, self._world_info)
    ):
      messages.append(
        MessageTTL(
          ttl=2,
          id=position_id.get_value(),
          message=MessageRoad(
            is_wireless_message=True,
            road=road,
            is_send_blockade_location=True,
            is_passable=False,
            blockade=None,
            priority=priority,
            sender_entity_id=road.get_entity_id(),
          ),
        )
      )
      self._logger.debug(
        "Send stuck road message",
        road_id=position_id.get_value(),
        priority=priority.name,
      )
      self.already_sent_message_stuck_road_entity_ids.add(position_id)
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
    self, entity: Human, is_wireless: bool, priority: StandardMessagePriority
  ) -> MessageAmbulanceTeam | MessageFireBrigade | MessagePoliceForce | MessageCivilian:
    if isinstance(entity, AmbulanceTeam):
      return MessageAmbulanceTeam(
        is_wireless_message=is_wireless,
        ambulance_team=entity,
        action=0,
        target_entity_id=None,  # ty:ignore[invalid-argument-type]
        priority=priority,
        sender_entity_id=entity.get_entity_id(),
      )
    elif isinstance(entity, FireBrigade):
      return MessageFireBrigade(
        is_wireless_message=is_wireless,
        fire_brigade=entity,
        action=0,
        target_entity_id=None,  # ty:ignore[invalid-argument-type]
        priority=priority,
        sender_entity_id=entity.get_entity_id(),
      )
    elif isinstance(entity, PoliceForce):
      return MessagePoliceForce(
        is_wireless_message=is_wireless,
        police_force=entity,
        action=0,
        target_entity_id=None,  # ty:ignore[invalid-argument-type]
        priority=priority,
        sender_entity_id=entity.get_entity_id(),
      )
    elif isinstance(entity, Civilian):
      return MessageCivilian(
        is_wireless_message=is_wireless,
        civilian=entity,
        priority=priority,
        sender_entity_id=entity.get_entity_id(),
      )
    else:
      raise ValueError("Unknown agent type")

  def _create_road_message(
    self, entity: Road, is_wireless: bool, priority: StandardMessagePriority
  ) -> MessageRoad:
    return MessageRoad(
      is_wireless_message=is_wireless,
      road=entity,
      is_send_blockade_location=True,
      is_passable=True,
      blockade=None,
      priority=priority,
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
