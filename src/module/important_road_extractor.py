"""対象の活動地域で重要道路を算出するためのモジュール。

全建物から最寄り避難所への搬送経路上の道路通過頻度を集計し、
頻度が閾値以上の道路と全避難所の隣接道路を重要道路として抽出します。
このモジュールの計算は事前計算で実行されます。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from adf_core_python.core.component.module.abstract_module import AbstractModule
from adf_core_python.core.component.module.algorithm.path_planning import PathPlanning
from rcrscore.entities import Building, EntityID, Refuge
from rcrscore.urn import EntityURN

if TYPE_CHECKING:
  from adf_core_python.core.agent.develop.develop_data import DevelopData
  from adf_core_python.core.agent.info.agent_info import AgentInfo
  from adf_core_python.core.agent.info.scenario_info import ScenarioInfo
  from adf_core_python.core.agent.info.world_info import WorldInfo
  from adf_core_python.core.agent.module.module_manager import ModuleManager
  from adf_core_python.core.agent.precompute.precompute_data import PrecomputeData

_MODULE_NAME_PATH_PLANNING = "ImportantRoadExtractor.PathPlanning"
_DEFAULT_PATH_PLANNING = (
  "adf_core_python.implement.module.algorithm.a_star_path_planning.AStarPathPlanning"
)

_DEVELOP_DATA_MIN_FREQUENCY = "ImportantRoadExtractor.minFrequency"

# kobe1.gml（建物757棟）で閾値10が有効だった実績に基づく比率。
# 未設定時はこの比率を建物数に掛けて動的に閾値を決定する。
_FREQUENCY_RATIO = 10 / 757


class ImportantRoadExtractor(AbstractModule):
  """全建物→最寄り避難所の搬送経路から重要道路を抽出するモジュール。

  頻度 >= minFrequency の道路と全避難所の隣接道路を重要道路とします。

  minFrequency は develop.json で明示指定できます。未指定の場合は
  建物数 × _FREQUENCY_RATIO（kobe1実績ベース）を切り上げた値を使用します。

  パラメータは develop.json で設定可能:
  - ``ImportantRoadExtractor.minFrequency``: 重要道路と判定する最小通過建物数
    （未設定時: 建物数に応じて自動計算）
  """

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
    self._path_planning: PathPlanning = cast(
      PathPlanning,
      module_manager.get_module(
        _MODULE_NAME_PATH_PLANNING,
        _DEFAULT_PATH_PLANNING,
      ),
    )
    self.register_sub_module(self._path_planning)

    # None = 未指定（calculate() で建物数から動的に決定）
    self._min_frequency: int | None = develop_data.get_value(
      _DEVELOP_DATA_MIN_FREQUENCY, None
    )
    self._important_road_ids: set[EntityID] = set()

  def calculate(self) -> ImportantRoadExtractor:
    """重要道路を計算する。"""
    if self._important_road_ids:
      return self

    refuges = self._world_info.get_entities_of_types([Refuge])
    buildings = self._world_info.get_entities_of_types([Building])

    if not refuges:
      self._logger.warning("No refuges found; no important roads will be extracted.")
      return self

    # 全建物から最寄り避難所への経路を計算し、道路通過頻度を集計
    freq: dict[EntityID, int] = {}
    skipped = 0
    for building in buildings:
      bid = building.get_entity_id()
      nearest_refuge = min(
        refuges,
        key=lambda r: self._world_info.get_distance(bid, r.get_entity_id()),
      )
      path = self._path_planning.get_path(bid, nearest_refuge.get_entity_id())
      if not path:
        skipped += 1
        continue
      for eid in path:
        entity = self._world_info.get_entity(eid)
        if entity is not None and entity.get_urn() == EntityURN.ROAD:
          freq[eid] = freq.get(eid, 0) + 1

    self._logger.info(
      f"FrequencyCount: {len(buildings)} buildings, {skipped} unreachable, "
      f"{len(freq)} roads used"
    )
    self._log_frequency_distribution(freq)

    # 閾値: 明示指定があればそれを使い、なければ建物数から自動計算
    if self._min_frequency is not None:
      threshold = self._min_frequency
    else:
      threshold = max(1, round(len(buildings) * _FREQUENCY_RATIO))
      self._logger.info(
        f"MinFrequency: auto={threshold} (buildings={len(buildings)}, ratio={_FREQUENCY_RATIO:.4f})"
      )

    # 頻度閾値以上の道路を重要道路として選択
    self._important_road_ids = {
      rid for rid, cnt in freq.items() if cnt >= threshold
    }
    self._logger.info(
      f"FrequencyFilter: threshold={threshold}, "
      f"selected={len(self._important_road_ids)}/{len(freq)}"
    )

    # 全避難所の隣接道路を明示的に追加（頻度に関わらず入口を保証）
    refuge_adj_added = 0
    for refuge in refuges:
      for nbr_id in refuge.get_neighbors():
        entity = self._world_info.get_entity(nbr_id)
        if entity is not None and entity.get_urn() == EntityURN.ROAD:
          if nbr_id not in self._important_road_ids:
            self._important_road_ids.add(nbr_id)
            refuge_adj_added += 1
    if refuge_adj_added:
      self._logger.info(
        f"RefugeAdjacent: added {refuge_adj_added} roads below threshold"
      )

    self._logger.info(f"TotalImportantRoads: {len(self._important_road_ids)}")
    return self

  def _log_frequency_distribution(self, freq: dict[EntityID, int]) -> None:
    """頻度分布をログ出力する（閾値チューニング用）。"""
    if not freq:
      return
    thresholds = [1, 5, 10, 20, 50, 100]
    dist = {t: sum(1 for c in freq.values() if c >= t) for t in thresholds}
    self._logger.info(f"FreqDist(roads>=N): {dist}")

  def precompute(self, precompute_data: PrecomputeData) -> ImportantRoadExtractor:
    super().precompute(precompute_data)
    if self.get_count_precompute() > 1:
      return self
    self.calculate()
    precompute_data.write_json_data(
      {
        "important_roads": [eid.get_value() for eid in self._important_road_ids],
      },
      self.__class__.__name__,
    )
    return self

  def resume(self, precompute_data: PrecomputeData) -> ImportantRoadExtractor:
    super().resume(precompute_data)
    if self.get_count_resume() > 1:
      return self
    try:
      data = precompute_data.read_json_data(self.__class__.__name__)
    except FileNotFoundError:
      self._logger.warning("PrecomputeData not found, falling back to calculate()")
      self.calculate()
      return self
    self._important_road_ids = {EntityID(eid) for eid in data["important_roads"]}
    return self

  def prepare(self) -> ImportantRoadExtractor:
    super().prepare()
    if self.get_count_prepare() > 1:
      return self
    self.calculate()
    return self

  def get_important_road_ids(self) -> set[EntityID]:
    """重要道路の EntityID セットを返す。"""
    return self._important_road_ids
