from typing import cast

from adf_core_python.core.agent.communication.message_manager import MessageManager
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
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.complex.human_detector import HumanDetector
from rcrscore.entities import (
  AmbulanceTeam,
  Civilian,
  Entity,
  EntityID,
  FireBrigade,
  Human,
  PoliceForce,
)

from src.utility.agent_status import (
  is_alived,
  is_buried,
  is_ghost_human,
  is_transported_to_refuge,
  is_transporting,
)
from src.utility.message_collection import (
  MessageHumanInfo,
  clear_percepted_human_messages,
  increment_step_ago_for_human_messages,
)
from src.utility.message_to_object import message_to_object_of_human


class FireDetector(HumanDetector):
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

    self._target_human: Human | None = None
    self._invalid_human_entity_ids: set[EntityID] = set()
    self._human_buriedness_messages: list[MessageHumanInfo] = []

    self._clustering: Clustering = cast(
      Clustering,
      module_manager.get_module(
        "FireDetector.Clustering",
        "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
      ),
    )
    self.register_sub_module(self._clustering)

  def update_info(self, message_manager: MessageManager) -> FireDetector:
    super().update_info(message_manager)

    self._human_buriedness_messages = increment_step_ago_for_human_messages(
      self._human_buriedness_messages
    )

    for message in message_manager.get_received_message_list():
      if isinstance(
        message,
        MessageCivilian
        | MessageFireBrigade
        | MessageAmbulanceTeam
        | MessagePoliceForce,
      ):
        human = message_to_object_of_human(message)
        if human is None:
          continue
        if (not is_buried(human)) or (human.get_position() is None):
          continue
        self._human_buriedness_messages.append(
          MessageHumanInfo(
            human=human,
            step_ago=0,
          )
        )

    self._human_buriedness_messages = clear_percepted_human_messages(
      self._human_buriedness_messages, self._world_info
    )

    return self

  def get_target_entity_id(self) -> EntityID | None:
    if self._target_human is not None:
      entity = self._world_info.get_entity(self._target_human.get_entity_id())
      if entity is None:
        positon = self._target_human.get_position()
        if positon is not None:
          return positon
      if isinstance(entity, Human):
        self._target_human = entity
      return self._target_human.get_entity_id()
    return None

  def calculate(self) -> HumanDetector:
    if self._target_human is not None:
      if not self._is_valid_human(self._target_human):
        self._target_human = None
      elif is_ghost_human(self._target_human, self._world_info, self._agent_info):
        self._logger.info(
          f"Detect ghost human: {self._target_human.get_entity_id().get_value()}"
        )
        self._invalid_human_entity_ids.add(self._target_human.get_entity_id())
        self._target_human = None

    if self._target_human is None:
      self._target_human = self._select_target()

    return self

  def _select_target(self) -> Human | None:
    cluster_index: int = self._clustering.get_cluster_index(
      self._agent_info.get_entity_id()
    )
    cluster_entities: list[Entity] = self._clustering.get_cluster_entities(
      cluster_index
    )

    cluster_valid_humans: list[Human] = [
      entity
      for entity in cluster_entities
      if isinstance(entity, Human)
      and self._is_valid_human(entity)
      and entity.get_entity_id() not in self._invalid_human_entity_ids
    ]

    cluster_entity_ids = set(entity.get_entity_id() for entity in cluster_entities)
    for message in self._human_buriedness_messages:
      if message.human.get_entity_id() in self._invalid_human_entity_ids:
        continue
      if message.human.get_position() in cluster_entity_ids:
        cluster_valid_humans.append(message.human)

    if len(cluster_valid_humans) != 0:
      return self._get_best_human(cluster_valid_humans)

    world_valid_humans: list[Human] = [
      entity
      for entity in self._world_info.get_entities_of_types([Civilian])
      if isinstance(entity, Human)
      and self._is_valid_human(entity)
      and entity.get_entity_id() not in self._invalid_human_entity_ids
    ]
    if len(world_valid_humans) != 0:
      return self._get_best_human(world_valid_humans)

    return None

  def _get_human_type_cost(self, human: Human) -> int:
    match human:
      case FireBrigade():
        return 0
      case PoliceForce():
        return 1
      case AmbulanceTeam():
        return 2
      case Civilian():
        return 3
      case _:
        return 4

  def _get_distance_cost(self, human: Human) -> float:
    human_position_entity_id = human.get_position()
    if human_position_entity_id is None:
      return float("inf")

    distance = self._world_info.get_distance(
      self._agent_info.get_entity_id(), human_position_entity_id
    )
    if distance is None:
      return float("inf")

    return distance

  def _get_buriedness_cost(self, human: Human) -> int:
    buriedness = human.get_buriedness()
    if buriedness is None:
      return 2**20
    return buriedness

  def _get_entity_id_cost(self, human: Human) -> int:
    return human.get_entity_id().get_value()

  def _get_best_human(self, humans: list[Human]) -> Human:
    if len(humans) == 0:
      raise Exception("Human list is empty")
    return sorted(
      humans,
      key=lambda human: (
        self._get_human_type_cost(human),
        self._get_distance_cost(human),
        self._get_buriedness_cost(human),
        self._get_entity_id_cost(human),
      ),
    )[0]

  def _is_valid_human(self, human: Human) -> bool:
    return (
      is_alived(human)
      and is_buried(human)
      and not is_transporting(human, self._world_info)
      and not is_transported_to_refuge(human, self._world_info)
    )
