"""
indoor_routing.py — Multi-floor indoor routing service.
Supports: ground, floor1, floor2, floor3, floor4
"""
import pandas as pd
import networkx as nx
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'static', 'data')

_graphs = {}
_nodes  = {}

FLOOR_FILES = {
    'ground': ('ground_nodes.csv',  'ground_edges.csv'),
    'floor1': ('floor1_nodes.csv',  'floor1_edges.csv'),
    'floor2': ('floor2_nodes.csv',  'floor2_edges.csv'),
    'floor3': ('floor3_nodes.csv',  'floor3_edges.csv'),
    'floor4': ('floor4_nodes.csv',  'floor4_edges.csv'),
}

FLOOR_LABELS = {
    'ground': 'Ground Floor',
    'floor1': 'Floor 1',
    'floor2': 'Floor 2',
    'floor3': 'Floor 3',
    'floor4': 'Floor 4',
}


def load_graph(floor: str):
    if floor not in FLOOR_FILES:
        raise ValueError(f'Unknown floor: {floor}')
    nodes_file, edges_file = FLOOR_FILES[floor]
    nodes_df = pd.read_csv(os.path.join(DATA_DIR, nodes_file))
    edges_df = pd.read_csv(os.path.join(DATA_DIR, edges_file))
    graph = nx.Graph()
    for _, row in nodes_df.iterrows():
        graph.add_node(int(row['id']), name=row['name'], x=float(row['x']), y=float(row['y']))
    for _, row in edges_df.iterrows():
        graph.add_edge(int(row['source']), int(row['target']), weight=float(row['distance']))
    _graphs[floor] = graph
    _nodes[floor]  = nodes_df
    print(f"[IndoorRouting] {floor}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


def _ensure(floor: str):
    if floor not in _graphs:
        load_graph(floor)


def get_indoor_locations(floor: str = 'ground') -> list:
    _ensure(floor)
    return _nodes[floor][['id', 'name', 'x', 'y']].to_dict(orient='records')


def find_node_by_name(name: str, floor: str) -> int:
    _ensure(floor)
    df = _nodes[floor]
    match = df[df['name'].str.lower() == name.lower()]
    return None if match.empty else int(match.iloc[0]['id'])


def get_indoor_route(start_name: str, end_name: str, floor: str = 'ground') -> dict:
    _ensure(floor)
    graph = _graphs[floor]
    start_id = find_node_by_name(start_name, floor)
    end_id   = find_node_by_name(end_name,   floor)
    if start_id is None:
        return {'error': f'Start location "{start_name}" not found'}
    if end_id is None:
        return {'error': f'Destination "{end_name}" not found'}
    if start_id == end_id:
        return {'error': 'Start and destination are the same'}
    try:
        path_ids   = nx.dijkstra_path(graph, start_id, end_id, weight='weight')
        total_dist = nx.dijkstra_path_length(graph, start_id, end_id, weight='weight')
        path_coords = []
        for nid in path_ids:
            node = graph.nodes[nid]
            path_coords.append({'id': nid, 'name': node['name'], 'x': node['x'], 'y': node['y']})
        return {'path': path_coords, 'total_distance': round(total_dist, 1), 'steps': len(path_ids)}
    except nx.NetworkXNoPath:
        return {'error': 'No path found between these locations'}


# Pre-load all floors on import
for _fl in FLOOR_FILES:
    load_graph(_fl)
