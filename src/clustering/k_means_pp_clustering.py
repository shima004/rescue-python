import numpy as np
from adf_core_python.core.agent.communication.message_manager import MessageManager
from adf_core_python.core.agent.communication.standard.bundle.information.message_ambulance_team import (
  MessageAmbulanceTeam,
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
    self.cluster_centroids: list[tuple[float, float]] = []
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
    self._deleted_agent_entity_ids: set[EntityID] = set()

  def calculate(self) -> Clustering:
    return self

  def update_info(self, message_manager: MessageManager) -> Clustering:
    super().update_info(message_manager)
    messages = message_manager.get_received_message_list()
    for message in messages:
      sender_id = None
      buriedness = None
      if isinstance(message, MessageAmbulanceTeam):
        sender_id = message.get_sender_entity_id()
        buriedness = message.get_ambulance_team_buriedness()
      elif isinstance(message, MessagePoliceForce):
        sender_id = message.get_sender_entity_id()
        buriedness = message.get_police_force_buriedness()
      elif isinstance(message, MessageFireBrigade):
        sender_id = message.get_sender_entity_id()
        buriedness = message.get_fire_brigade_buriedness()
      else:
        continue
      if (
        sender_id is not None
        and buriedness is not None
        and buriedness > 0
        and sender_id not in self._deleted_agent_entity_ids
      ):
        cluster_index = self.get_cluster_index(sender_id)
        if cluster_index != -1:
          self._logger.debug(
            f"Received buried message from {sender_id.get_value()}, buriedness={
              buriedness
            }, deleting cluster {cluster_index}"
          )
          self._delete_cluster(cluster_index)
          self._deleted_agent_entity_ids.add(sender_id)
          self._logger.debug(f"Deleted cluster {self.entity_cluster_indices}")

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
    self._logger.debug(
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
    self.cluster_centroids = [
      tuple(
        np.mean(
          [
            entity.get_location()
            for entity in cluster
            if entity.get_location() is not None
          ],
          axis=0,
        )
      )
      for cluster in self.cluster_entities
    ]
    self._logger.debug(
      f"Resumed clusters: {len(self.cluster_entities)} clusters loaded"
    )
    return self

  def get_cluster_number(self) -> int:
    return self._cluster_number

  def get_cluster_index(self, entity_id: EntityID) -> int:
    return self.entity_cluster_indices.get(entity_id, -1)

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
    self._logger.debug(
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
    self.cluster_centroids = [tuple(centroid) for centroid in kmeans.cluster_centers_]
    labels = kmeans.labels_
    if labels is None:
      return clusters

    for entity, label in zip(valid_entities, labels):
      clusters[label].append(entity)

    return clusters

  # 指定されたクラスタを削除し、最近傍のクラスタに再割り当てする関数
  def _delete_cluster(self, cluster_index: int) -> None:
    if cluster_index >= len(self.cluster_entities):
      return

    # 削除するクラスタのエンティティとセントロイドを取得
    deleted_cluster_entities = self.cluster_entities[cluster_index]
    deleted_centroid = self.cluster_centroids[cluster_index]

    # 削除前に最近傍クラスタを探す（削除後はインデックスがずれるため）
    reassign_to = self._find_nearest_other_cluster(cluster_index, deleted_centroid)

    # クラスタとセントロイドを削除
    del self.cluster_entities[cluster_index]
    del self.cluster_centroids[cluster_index]

    # reassign_to は削除前のインデックスなので、cluster_index より大きければ補正
    if reassign_to is not None and reassign_to > cluster_index:
      reassign_to -= 1

    # エージェントのクラスタインデックスを更新:
    # - 削除されたクラスタのエージェントは最近傍クラスタへ再割り当て（復帰時に備える）
    # - cluster_index より大きいインデックスはずれを補正して -1
    self.entity_cluster_indices = {
      eid: (
        reassign_to
        if idx == cluster_index and reassign_to is not None
        else (idx - 1 if idx > cluster_index else idx)
      )
      for eid, idx in self.entity_cluster_indices.items()
      if not (idx == cluster_index and reassign_to is None)
    }

    # 削除されたクラスタのエンティティを最近傍のクラスタに再割り当て
    for entity in deleted_cluster_entities:
      nearest_cluster_index = self._find_nearest_cluster(entity)
      if nearest_cluster_index is not None:
        self.cluster_entities[nearest_cluster_index].append(entity)

  def _find_nearest_other_cluster(
    self, exclude_index: int, centroid: tuple[float, float]
  ) -> int | None:
    """exclude_index を除いたクラスタの中で centroid に最も近いクラスタのインデックスを返す"""
    min_distance = float("inf")
    nearest = None
    for i, c in enumerate(self.cluster_centroids):
      if i == exclude_index:
        continue
      distance = np.linalg.norm(np.array(centroid) - np.array(c))
      if distance < min_distance:
        min_distance = distance
        nearest = i
    return nearest

  def _find_nearest_cluster(self, entity: Entity) -> int | None:
    x, y = entity.get_location()
    if x is None or y is None:
      return None

    min_distance = float("inf")
    nearest_cluster_index = None
    for i, centroid in enumerate(self.cluster_centroids):
      distance = np.linalg.norm(np.array([x, y]) - np.array(centroid))
      if distance < min_distance:
        min_distance = distance
        nearest_cluster_index = i
    return nearest_cluster_index
