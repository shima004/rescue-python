from typing import cast

from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from adf_core_python.core.component.module.complex.road_detector import RoadDetector
from rcrscore.entities import Entity, EntityID, Road

from src.utility.road_status import get_impassable_edge_pairs, has_impassable_edge_pair


class PoliceDetector(RoadDetector):
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

    self._target_road_entity_id: EntityID | None = None

    self._path_planning: PathPlanning = cast(
      PathPlanning,
      module_manager.get_module(
        "PoliceDetector.PathPlanning",
        "adf_core_python.implement.module.algorithm.a_star_path_planning.AStarPathPlanning",
      ),
    )
    self._clustering: Clustering = cast(
      Clustering,
      module_manager.get_module(
        "PoliceDetector.Clustering",
        "adf_core_python.implement.module.algorithm.k_means_clustering.KMeansClustering",
      ),
    )

    self.register_sub_module(self._path_planning)
    self.register_sub_module(self._clustering)

  def get_target_entity_id(self) -> EntityID | None:
    return self._target_road_entity_id

  def calculate(self) -> RoadDetector:
    if self._target_road_entity_id is not None:
      target_road = self._world_info.get_entity(self._target_road_entity_id)
      assert isinstance(target_road, Road)
      if not self._is_valid_road(target_road):
        self._target_road_entity_id = None

    if self._target_road_entity_id is None:
      target_road = self._get_target_road()
      if target_road is not None:
        self._target_road_entity_id = target_road.get_entity_id()

    self._logger.info(
      f"""
      Detect target road: {self._target_road_entity_id.get_value() if self._target_road_entity_id is not None else None}
      Impassible edge pairs: {get_impassable_edge_pairs(target_road, self._world_info) if target_road is not None else None}
      """
    )

    return self

  def _get_target_road(self) -> Road | None:
    cluster_index: int = self._clustering.get_cluster_index(
      self._agent_info.get_entity_id()
    )
    cluster_entities: list[Entity] = self._clustering.get_cluster_entities(
      cluster_index
    )

    cluster_valid_road: list[Road] = [
      entity
      for entity in cluster_entities
      if isinstance(entity, Road) and self._is_valid_road(entity)
    ]
    if len(cluster_valid_road) != 0:
      return self._get_nearest_road(cluster_valid_road)

    world_valid_roads: list[Road] = [
      entity
      for entity in self._world_info.get_entities_of_types([Road])
      if isinstance(entity, Road) and self._is_valid_road(entity)
    ]
    if len(world_valid_roads) != 0:
      return self._get_nearest_road(world_valid_roads)

    return None

  def _get_nearest_road(self, roads: list[Road]) -> Road:
    nearest_road = roads[0]
    nearest_distance = self._world_info.get_distance(
      self._agent_info.get_entity_id(),
      nearest_road.get_entity_id(),
    )
    for road in roads:
      distance = self._world_info.get_distance(
        self._agent_info.get_entity_id(),
        road.get_entity_id(),
      )
      if distance < nearest_distance:
        nearest_distance = distance
        nearest_road = road
    return nearest_road

  def _is_valid_road(self, road: Road) -> bool:
    return has_impassable_edge_pair(road, self._world_info)
