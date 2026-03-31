import numpy as np
from adf_core_python.core.agent.develop.develop_data import DevelopData
from adf_core_python.core.agent.info.agent_info import AgentInfo
from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
from adf_core_python.core.agent.info.world_info import WorldInfo
from adf_core_python.core.agent.module.module_manager import ModuleManager
from adf_core_python.core.agent.precompute.precompute_data import PrecomputeData
from adf_core_python.core.component.module.algorithm.clustering import Clustering
from rcrscore.entities import (
  AmbulanceCenter,
  Building,
  Entity,
  EntityID,
  FireStation,
  GasStation,
  Hydrant,
  PoliceOffice,
  Refuge,
  Road,
)
from rcrscore.urn import EntityURN
from sklearn.cluster import KMeans


class KMeansPPClustering(Clustering):
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
    myself = agent_info.get_myself()
    if myself is None:
      raise RuntimeError("Could not get agent entity")
    match myself.get_urn():
      case EntityURN.AMBULANCE_TEAM:
        self._cluster_number = int(
          scenario_info.get_value(
            "scenario.agents.at",
            1,
          )
        )
      case EntityURN.POLICE_FORCE:
        self._cluster_number = int(
          scenario_info.get_value(
            "scenario.agents.pf",
            1,
          )
        )
      case EntityURN.FIRE_BRIGADE:
        self._cluster_number = int(
          scenario_info.get_value(
            "scenario.agents.fb",
            1,
          )
        )
      case _:
        self._cluster_number = 1

    sorted_entities = sorted(
      world_info.get_entities_of_types(
        [
          myself.__class__,
        ]
      ),
      key=lambda entity: entity.get_entity_id().get_value(),
    )
    self.entity_cluster_indices = {
      entity.get_entity_id(): idx for idx, entity in enumerate(sorted_entities)
    }

    self.cluster_entities: list[list[Entity]] = []
    self.entities: list[Entity] = world_info.get_entities_of_types(
      [
        AmbulanceCenter,
        FireStation,
        GasStation,
        Hydrant,
        PoliceOffice,
        Refuge,
        Road,
        Building,
      ]
    )

  def calculate(self) -> Clustering:
    return self

  def precompute(self, precompute_data: PrecomputeData) -> Clustering:
    cluster_entities = self.create_cluster(self._cluster_number, self.entities)
    precompute_data.write_json_data(
      {
        "cluster_entities": [
          [entity.get_entity_id().get_value() for entity in cluster]
          for cluster in cluster_entities
        ]
      },
      self.__class__.__name__,
    )
    self._logger.info(
      f"Precomputed clusters: {len(cluster_entities)} clusters created and saved"
    )
    return self

  def resume(self, precompute_data: PrecomputeData) -> Clustering:
    data = precompute_data.read_json_data(self.__class__.__name__)
    self.cluster_entities = [
      [
        entity
        for entity_id in cluster
        if (entity := self._world_info.get_entity(EntityID(entity_id))) is not None
      ]
      for cluster in data["cluster_entities"]
    ]
    self._logger.info(f"Resumed clusters: {len(self.cluster_entities)} clusters loaded")
    return self

  def get_cluster_number(self) -> int:
    return self._cluster_number

  def get_cluster_index(self, entity_id: EntityID) -> int:
    return self.entity_cluster_indices.get(entity_id, 0)

  def get_cluster_entities(self, cluster_index: int) -> list[Entity]:
    if cluster_index >= len(self.cluster_entities):
      return []
    return self.cluster_entities[cluster_index]

  def get_cluster_entity_ids(self, cluster_index: int) -> list[EntityID]:
    if cluster_index >= len(self.cluster_entities):
      return []
    return [entity.get_entity_id() for entity in self.cluster_entities[cluster_index]]

  def prepare(self) -> Clustering:
    super().prepare()
    if self.get_count_prepare() > 1:
      return self
    self.cluster_entities = self.create_cluster(self._cluster_number, self.entities)
    self._logger.info(
      f"Prepared clusters: {len(self.cluster_entities)} clusters created"
    )
    return self

  def create_cluster(
    self, cluster_number: int, entities: list[Entity]
  ) -> list[list[Entity]]:
    valid_entities: list[Entity] = []
    positions: list[list[int]] = []
    for entity in entities:
      x, y = entity.get_location()
      if x is None or y is None:
        continue
      valid_entities.append(entity)
      positions.append([x, y])

    if not valid_entities:
      return [[] for _ in range(cluster_number)]

    effective_cluster_number = min(cluster_number, len(valid_entities))
    kmeans = KMeans(
      n_clusters=effective_cluster_number,
      init="k-means++",
      random_state=0,
    )
    kmeans.fit(np.asarray(positions))

    clusters: list[list[Entity]] = [[] for _ in range(cluster_number)]
    labels = kmeans.labels_
    if labels is None:
      return clusters

    for entity, label in zip(valid_entities, labels):
      clusters[label].append(entity)

    return clusters
