"""GML マップ上に重要道路を赤でハイライトするスクリプト。

使い方（rescue-python/ ルートから）:
    uv run python tests/test_important_road_visualization.py [GML_PATH] [JSON_PATH] [-o OUTPUT_PATH]

例:
    # デフォルト（kobe1.gml）
    uv run python tests/test_important_road_visualization.py

    # 別マップを指定
    uv run python tests/test_important_road_visualization.py tests/gmls/sf1.gml
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
GML_PATH = ROOT / "tests/gmls/kobe1.gml"
JSON_PATH = ROOT / "precompute/police_force/ImportantRoadExtractor.json"
OUTPUT_PATH = ROOT / "tests/output/kobe1_important_roads.png"

# GML 名前空間
NS = {
    "rcr": "urn:roborescue:map:gml",
    "gml": "http://www.opengis.net/gml",
    "xlink": "http://www.w3.org/1999/xlink",
}

# ---------------------------------------------------------------------------
# GML パース
# ---------------------------------------------------------------------------


def parse_gml(path: Path) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, tuple[str, str]],
    dict[str, list[tuple[str, str]]],
    dict[str, list[tuple[str, str]]],
]:
    """GMLを解析してノード・エッジ・道路・建物の辞書を返す。

    Returns:
        nodes:     {node_id: (x, y)}
        edges:     {edge_id: (minus_node_id, plus_node_id)}
        roads:     {road_id: [(edge_id, orientation), ...]}
        buildings: {building_id: [(edge_id, orientation), ...]}
    """
    print(f"GML をパース中: {path}")
    tree = ET.parse(path)
    root = tree.getroot()

    # ノード座標
    nodes: dict[str, tuple[float, float]] = {}
    for node in root.findall(".//gml:Node", NS):
        nid = node.get("{http://www.opengis.net/gml}id")
        coords_el = node.find(".//gml:coordinates", NS)
        if nid is None or coords_el is None or coords_el.text is None:
            continue
        x, y = map(float, coords_el.text.strip().split(","))
        nodes[nid] = (x, y)

    # エッジ（始点ノード "-", 終点ノード "+"）
    edges: dict[str, tuple[str, str]] = {}
    for edge in root.findall(".//gml:Edge", NS):
        eid = edge.get("{http://www.opengis.net/gml}id")
        if eid is None:
            continue
        minus_node = plus_node = None
        for dn in edge.findall("gml:directedNode", NS):
            orientation = dn.get("orientation")
            href = dn.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
            if orientation == "-":
                minus_node = href
            elif orientation == "+":
                plus_node = href
        if minus_node and plus_node:
            edges[eid] = (minus_node, plus_node)

    # 道路（directedEdge のリスト）
    roads: dict[str, list[tuple[str, str]]] = {}
    for road in root.findall(".//rcr:road", NS):
        rid = road.get("{http://www.opengis.net/gml}id")
        if rid is None:
            continue
        face_edges = []
        for de in road.findall(".//gml:directedEdge", NS):
            orientation = de.get("orientation", "+")
            href = de.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
            if href:
                face_edges.append((href, orientation))
        roads[rid] = face_edges

    # 建物
    buildings: dict[str, list[tuple[str, str]]] = {}
    for building in root.findall(".//rcr:building", NS):
        bid = building.get("{http://www.opengis.net/gml}id")
        if bid is None:
            continue
        face_edges = []
        for de in building.findall(".//gml:directedEdge", NS):
            orientation = de.get("orientation", "+")
            href = de.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
            if href:
                face_edges.append((href, orientation))
        buildings[bid] = face_edges

    print(
        f"  ノード: {len(nodes)}, エッジ: {len(edges)}, "
        f"道路: {len(roads)}, 建物: {len(buildings)}"
    )
    return nodes, edges, roads, buildings


# ---------------------------------------------------------------------------
# ポリゴン組み立て
# ---------------------------------------------------------------------------


def build_polygon(
    face_edges: list[tuple[str, str]],
    edges: dict[str, tuple[str, str]],
    nodes: dict[str, tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """directedEdge リストから多角形の頂点列を組み立てる。

    orientation="+"  → edge の minus ノード（始点）→ plus ノード（終点）
    orientation="-"  → edge の plus ノード（終点）→ minus ノード（始点）
    """
    vertices: list[tuple[float, float]] = []
    for edge_id, orientation in face_edges:
        edge = edges.get(edge_id)
        if edge is None:
            return None
        minus_id, plus_id = edge
        if orientation == "+":
            start_id = minus_id
        else:
            start_id = plus_id
        coord = nodes.get(start_id)
        if coord is None:
            return None
        vertices.append(coord)
    return vertices if len(vertices) >= 3 else None


# ---------------------------------------------------------------------------
# 描画
# ---------------------------------------------------------------------------


def render(
    nodes: dict[str, tuple[float, float]],
    edges: dict[str, tuple[str, str]],
    roads: dict[str, list[tuple[str, str]]],
    buildings: dict[str, list[tuple[str, str]]],
    important_ids: set[int],
    output_path: Path,
    gml_name: str = "",
) -> None:
    print("ポリゴンを組み立て中...")
    normal_patches: list[MplPolygon] = []
    important_patches: list[MplPolygon] = []
    building_patches: list[MplPolygon] = []
    skipped = 0

    for rid, face_edges in roads.items():
        verts = build_polygon(face_edges, edges, nodes)
        if verts is None:
            skipped += 1
            continue
        patch = MplPolygon(verts, closed=True)
        if int(rid) in important_ids:
            important_patches.append(patch)
        else:
            normal_patches.append(patch)

    for _, face_edges in buildings.items():
        verts = build_polygon(face_edges, edges, nodes)
        if verts is None:
            continue
        building_patches.append(MplPolygon(verts, closed=True))

    print(
        f"  normal: {len(normal_patches)}, important: {len(important_patches)}, "
        f"buildings: {len(building_patches)}, skipped: {skipped}"
    )

    print("描画中...")
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    ax.set_aspect("equal")
    total = len(normal_patches) + len(important_patches)
    ax.set_title(
        f"{gml_name} — Important roads: {len(important_patches)} / {total}",
        fontsize=14,
    )

    if building_patches:
        col = PatchCollection(building_patches, facecolor="#aec6e8", edgecolor="none", alpha=0.25)
        ax.add_collection(col)

    if normal_patches:
        col = PatchCollection(normal_patches, facecolor="#cccccc", edgecolor="#999999", linewidth=0.2, alpha=0.5)
        ax.add_collection(col)

    if important_patches:
        col = PatchCollection(important_patches, facecolor="#e63946", edgecolor="#800000", linewidth=0.4, alpha=0.8)
        ax.add_collection(col)

    ax.autoscale_view()

    legend_handles = [
        mpatches.Patch(facecolor="#e63946", edgecolor="#800000", label=f"Important road ({len(important_patches)})"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#999999", label=f"Normal road ({len(normal_patches)})"),
        mpatches.Patch(facecolor="#aec6e8", edgecolor="none", label=f"Building ({len(building_patches)})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存: {output_path}")


# ---------------------------------------------------------------------------
# pytest テスト（JSON がなければ skip）
# ---------------------------------------------------------------------------


def test_important_road_visualization() -> None:
    """precompute 済みの重要道路を kobe1.gml 上に描画して PNG を保存する。

    precompute/police_force/ImportantRoadExtractor.json が存在しない場合は skip。
    出力: tests/check_important_road/output/kobe1_important_roads.png
    """
    import pytest

    if not JSON_PATH.exists():
        pytest.skip(f"Precompute data not found: {JSON_PATH}")

    with JSON_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    important_ids: set[int] = set(data["important_roads"])

    nodes, edges, roads, buildings = parse_gml(GML_PATH)
    output_path = ROOT / "tests/output/kobe1_important_roads.png"
    render(nodes, edges, roads, buildings, important_ids, output_path, gml_name=GML_PATH.name)

    assert output_path.exists()


# ---------------------------------------------------------------------------
# スクリプトとして直接実行
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize important roads on a GML map.")
    parser.add_argument(
        "gml",
        nargs="?",
        type=Path,
        default=GML_PATH,
        help="Path to the GML map file (default: kobe1.gml)",
    )
    parser.add_argument(
        "json",
        nargs="?",
        type=Path,
        default=JSON_PATH,
        help="Path to the ImportantRoadExtractor JSON file (default: police_force precompute)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: output/<gml_stem>_important_roads.png)",
    )
    args = parser.parse_args()

    gml_path: Path = args.gml
    json_path: Path = args.json
    output_path: Path = args.output or (
        ROOT / "tests/output" / f"{gml_path.stem}_important_roads.png"
    )

    if not json_path.exists():
        raise FileNotFoundError(f"Precompute data not found: {json_path}")

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    important_ids: set[int] = set(data["important_roads"])
    print(f"  重要道路数: {len(important_ids)}")

    nodes, edges, roads, buildings = parse_gml(gml_path)
    render(nodes, edges, roads, buildings, important_ids, output_path, gml_name=gml_path.name)


if __name__ == "__main__":
    main()
