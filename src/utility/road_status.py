from itertools import combinations

from adf_core_python.core.agent.info.world_info import WorldInfo
from rcrscore.entities import Blockade, Edge, EntityID, Road
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

AGENT_RADIUS = 500


def get_passable_edge_pairs(
  road: Road, world_info: WorldInfo
) -> set[tuple[EntityID, EntityID]]:
  edges = road.get_edges()
  if edges is None:
    return set()
  connected_edges: set[Edge] = set(
    edge for edge in edges if edge.get_neighbour() is not None
  )
  if not connected_edges:
    return set()

  passable_edge_pairs: set[tuple[EntityID, EntityID]] = set()
  edge_pairs = combinations(connected_edges, 2)
  blockades = road.get_blockades()
  if blockades is None:
    for edge1, edge2 in edge_pairs:
      passable_edge_pairs.add((edge1.get_neighbour(), edge2.get_neighbour()))
    return passable_edge_pairs

  road_polygon = get_road_polygon(road)
  blockade_union = unary_union(
    [get_blockade_polygon(b) for b in world_info.get_blockades(road)]
  )

  for edge1, edge2 in edge_pairs:
    if _is_passable(edge1, edge2, road_polygon, blockade_union):
      passable_edge_pairs.add((edge1.get_neighbour(), edge2.get_neighbour()))
  return passable_edge_pairs


def get_impassable_edge_pairs(
  road: Road, world_info: WorldInfo
) -> set[tuple[EntityID, EntityID]]:
  edges = road.get_edges()
  if edges is None:
    return set()
  connected_edges: set[Edge] = set(
    edge for edge in edges if edge.get_neighbour() is not None
  )
  if not connected_edges:
    return set()

  impassable_edge_pairs: set[tuple[EntityID, EntityID]] = set()
  edge_pairs = combinations(connected_edges, 2)
  blockades = road.get_blockades()
  if blockades is None:
    return impassable_edge_pairs

  road_polygon = get_road_polygon(road)
  blockade_union = unary_union(
    [get_blockade_polygon(b) for b in world_info.get_blockades(road)]
  )

  for edge1, edge2 in edge_pairs:
    if not _is_passable(edge1, edge2, road_polygon, blockade_union):
      impassable_edge_pairs.add((edge1.get_neighbour(), edge2.get_neighbour()))
  return impassable_edge_pairs


def has_impassable_edge_pair(road: Road, world_info: WorldInfo) -> bool:
  return len(get_impassable_edge_pairs(road, world_info)) > 0


def get_road_polygon(road: Road) -> Polygon:
  apexes = road.get_apexes()
  if not apexes:
    return Polygon()
  # buffer(0) normalises self-intersecting rings that can arise from
  # non-standard vertex orderings in the RRS world model.
  return Polygon([(apexes[i], apexes[i + 1]) for i in range(0, len(apexes), 2)]).buffer(
    0
  )


def get_blockade_polygon(blockade: Blockade) -> Polygon:
  apexes = blockade.get_apexes()
  if not apexes:
    return Polygon()
  return Polygon([(apexes[i], apexes[i + 1]) for i in range(0, len(apexes), 2)]).buffer(
    0
  )


def _is_passable(
  e1: Edge, e2: Edge, road_polygon: Polygon, blockade_union: Polygon
) -> bool:
  # Build a corridor polygon as the convex hull of the four edge endpoints.
  corridor_pts = [
    (e1.get_start_x(), e1.get_start_y()),
    (e1.get_end_x(), e1.get_end_y()),
    (e2.get_start_x(), e2.get_start_y()),
    (e2.get_end_x(), e2.get_end_y()),
  ]
  corridor = Polygon(corridor_pts).convex_hull

  # Clip the corridor to the actual road area.
  passage = corridor.intersection(road_polygon)

  # Remove blockade-occupied area, then erode by agent radius.
  clear = passage.difference(blockade_union)
  eroded = clear.buffer(-AGENT_RADIUS)

  if eroded.is_empty:
    return False

  # Verify connectivity: at least one eroded component must be reachable from
  # both edges. A blockade may split the corridor into disconnected pieces even
  # if each piece is individually non-empty after erosion.
  entry1 = passage.intersection(
    LineString(
      [(e1.get_start_x(), e1.get_start_y()), (e1.get_end_x(), e1.get_end_y())]
    ).buffer(AGENT_RADIUS)
  )
  entry2 = passage.intersection(
    LineString(
      [(e2.get_start_x(), e2.get_start_y()), (e2.get_end_x(), e2.get_end_y())]
    ).buffer(AGENT_RADIUS)
  )
  geoms = list(eroded.geoms) if hasattr(eroded, "geoms") else [eroded]
  return any(g.intersects(entry1) and g.intersects(entry2) for g in geoms)
