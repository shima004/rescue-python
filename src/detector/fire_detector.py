from typing import cast

from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.complex.human_detector import HumanDetector
from rcrscore.entities import Civilian, Entity, EntityID, Human

from src.utility.agent_status import (
  is_alived,
  is_buried,
  is_ghost_human,
  is_transported_to_refuge,
  is_transporting,
)


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

    self._clustering: Clustering = cast(
      Clustering,
      module_manager.get_module(
        "FireDetector.Clustering",
        "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
      ),
    )
    self.register_sub_module(self._clustering)

  def get_target_entity_id(self) -> EntityID | None:
    if self._target_human is not None:
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
    if len(cluster_valid_humans) != 0:
      return self._get_nearest_human(cluster_valid_humans)

    world_valid_humans: list[Human] = [
      entity
      for entity in self._world_info.get_entities_of_types([Civilian])
      if isinstance(entity, Human)
      and self._is_valid_human(entity)
      and entity.get_entity_id() not in self._invalid_human_entity_ids
    ]
    if len(world_valid_humans) != 0:
      return self._get_nearest_human(world_valid_humans)

    return None

  def _get_nearest_human(self, humans: list[Human]) -> Human:
    nearest_human = humans[0]
    nearest_distance = self._world_info.get_distance(
      self._agent_info.get_entity_id(),
      nearest_human.get_entity_id(),
    )
    for human in humans:
      distance = self._world_info.get_distance(
        self._agent_info.get_entity_id(),
        human.get_entity_id(),
      )
      if distance < nearest_distance:
        nearest_distance = distance
        nearest_human = human
    return nearest_human

  def _is_valid_human(self, human: Human) -> bool:
    return (
      is_alived(human)
      and is_buried(human)
      and not is_transporting(human, self._world_info)
      and not is_transported_to_refuge(human, self._world_info)
    )
