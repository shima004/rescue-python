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
from rcrscore.entities import Edge, Entity, EntityID, Road

from src.communication.send_message import SendMessage


class PoliceSearch(Search):
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

    self._unreached_targets: dict[EntityID, set[Edge]] = {}
    self._result: EntityID | None = None

    self._clustering: Clustering = cast(
      Clustering,
      module_manager.get_module(
        "PoliceSearch.Clustering",
        "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
      ),
    )

    self._path_planning: PathPlanning = cast(
      PathPlanning,
      module_manager.get_module(
        "PoliceSearch.PathPlanning",
        "adf_core_python.implement.module.algorithm.a_star_path_planning.AStarPathPlanning",
      ),
    )

    self._send_message: SendMessage = cast(
      SendMessage,
      module_manager.get_module(
        "PoliceSearch.SendMessage",
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

    return self

  def calculate(self) -> Search:
    self._update_search_targets()
    if len(self._unreached_targets) == 0:
      self._unreached_targets = self._refresh_search_targets()
    self._result = self._get_search_targets()
    return self

  def _refresh_search_targets(self) -> dict[EntityID, set[Edge]]:
    cluster_index: int = self._clustering.get_cluster_index(
      self._agent_info.get_entity_id()
    )
    cluster_entities: list[Entity] = self._clustering.get_cluster_entities(
      cluster_index
    )
    road_entities: set[Road] = {
      entity for entity in cluster_entities if isinstance(entity, Road)
    }

    passable_edges: dict[EntityID, set[Edge]] = {}
    for road in road_entities:
      if edges := road.get_edges():
        for edge in edges:
          if edge.get_neighbour() is not None:
            passable_edges.setdefault(road.get_entity_id(), set()).add(edge)

    return passable_edges

  def _update_search_targets(self) -> None:
    searched_building_id = self._agent_info.get_position_entity_id()
    if searched_building_id is not None:
      self._unreached_targets.pop(searched_building_id, None)

  def _get_search_targets(self) -> EntityID | None:
    nearest_target_id: EntityID | None = None
    nearest_distance: float | None = None
    for target_id in self._unreached_targets:
      distance = self._world_info.get_distance(
        self._agent_info.get_entity_id(), target_id
      )
      if nearest_distance is None or distance < nearest_distance:
        nearest_target_id = target_id
        nearest_distance = distance
    return nearest_target_id
