from typing import cast

from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from adf_core_python.core.component.module.complex.search import Search
from rcrscore.entities import Building, Entity, EntityID, Refuge

from src.communication.send_message import SendMessage


class AmbulanceSearch(Search):
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

    self._unreached_building_ids: set[EntityID] = set()
    self._perceived_buildings: set[EntityID] = set()
    self._result: EntityID | None = None

    self._clustering: Clustering = cast(
      Clustering,
      module_manager.get_module(
        "AmbulanceSearch.Clustering",
        "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
      ),
    )

    self._path_planning: PathPlanning = cast(
      PathPlanning,
      module_manager.get_module(
        "AmbulanceSearch.PathPlanning",
        "adf_core_python.implement.module.algorithm.a_star_path_planning.AStarPathPlanning",
      ),
    )

    self._send_message: SendMessage = cast(
      SendMessage,
      module_manager.get_module(
        "AmbulanceSearch.SendMessage",
        "src.communication.send_message.SendMessage",
      ),
    )

    self.register_sub_module(self._clustering)
    self.register_sub_module(self._path_planning)
    self.register_sub_module(self._send_message)

  def get_target_entity_id(self) -> EntityID | None:
    return self._result

  def update_info(self, message_manager: MessageManager) -> Search:
    super().update_info(message_manager)
    if self.get_count_update_info() > 1:
      return self

    self._logger.debug(
      f"unreached_building_ids: {[str(id) for id in self._unreached_building_ids]}"
    )

    return self

  def calculate(self) -> Search:
    self._update_search_targets()
    if len(self._unreached_building_ids) == 0:
      self._unreached_building_ids = self._refresh_search_targets()
    self._result = self._get_search_target()
    return self

  def _refresh_search_targets(self) -> set[EntityID]:
    cluster_index: int = self._clustering.get_cluster_index(
      self._agent_info.get_entity_id()
    )
    cluster_entities: list[Entity] = self._clustering.get_cluster_entities(
      cluster_index
    )
    building_entity_ids: set[EntityID] = {
      entity.get_entity_id()
      for entity in cluster_entities
      if isinstance(entity, Building) and not isinstance(entity, Refuge)
    }

    self._logger.info(
      f"Cluster {cluster_index}: {len(cluster_entities)} entities, "
      f"{len(building_entity_ids)} building targets"
    )

    for entity_id in self._perceived_buildings:
      entity = self._world_info.get_entity(entity_id)
      if isinstance(entity, Building) and not isinstance(entity, Refuge):
        if not self._is_broken_building(entity_id):
          building_entity_ids.discard(entity_id)

    self._logger.info(
      f"After filtering non-broken buildings: {len(building_entity_ids)} building targets"
    )

    return building_entity_ids

  def _update_search_targets(self) -> None:
    searched_building_id = self._agent_info.get_position_entity_id()
    if searched_building_id is not None:
      self._unreached_building_ids.discard(searched_building_id)

    for entity_id in self._world_info.get_change_set().get_changed_entities():
      entity = self._world_info.get_entity(entity_id)
      if isinstance(entity, Building) and not isinstance(entity, Refuge):
        self._perceived_buildings.add(entity_id)
        if not self._is_broken_building(entity_id):
          self._unreached_building_ids.discard(entity_id)

  def _get_search_target(self) -> EntityID | None:
    nearest_building_id: EntityID | None = None
    nearest_distance: float | None = None
    for building_id in self._unreached_building_ids:
      distance = self._world_info.get_distance(
        self._agent_info.get_entity_id(), building_id
      )
      if nearest_distance is None or distance < nearest_distance:
        nearest_building_id = building_id
        nearest_distance = distance

    return nearest_building_id

  def _is_broken_building(self, entity_id: EntityID) -> bool:
    entity = self._world_info.get_entity(entity_id)
    if isinstance(entity, Building) and not isinstance(entity, Refuge):
      brokenness = entity.get_brokenness()
      if brokenness is not None and brokenness > 0:
        return True
    return False
