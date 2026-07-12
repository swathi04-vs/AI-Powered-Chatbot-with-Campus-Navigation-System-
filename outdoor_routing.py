"""
outdoor_routing.py  —  GeoJSON-based campus outdoor routing.

The GeoJSON LineStrings were drawn with gaps between segments that
logically connect. We fix this in three passes:

  Pass 1 – union-find merges coords within SNAP_M (5 m).
  Pass 2 – build the spine chain: all nodes that lie on the main
            east-west campus road (lat ≈ 12.87499) are sorted by
            longitude and chained in order. Wrong bridge-edges that
            skip spine nodes are removed and replaced with correct
            adjacent-spine edges.
  Pass 3 – remaining isolated components are bridged to the main
            component by shortest gap.
"""

import json, math, os
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOJSON  = os.path.join(BASE_DIR, 'static', 'data', 'campus.geojson')

SNAP_M       = 5.0      # merge near-coincident raw coords
SPINE_LAT    = 12.874991  # lat of the main horizontal campus road
SPINE_LAT_M  = 18.0    # a node is "on the spine" if within this many metres vertically
SPINE_LNG_MIN = 74.9386
SPINE_LNG_MAX = 74.9406
BRIDGE_M     = 120.0    # max gap allowed in final bridging pass

_graph     = None
_node_c    = None
_buildings = []


# ── helpers ──────────────────────────────────────────────────
def _hav(a, b):        # a, b = [lng, lat]
    R = 6_371_000
    la1, lo1 = math.radians(a[1]), math.radians(a[0])
    la2, lo2 = math.radians(b[1]), math.radians(b[0])
    d = (math.sin((la2-la1)/2)**2
         + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2)
    return 2 * R * math.asin(math.sqrt(d))

def _lat_m(lat):
    return abs(lat - SPINE_LAT) * 111_320

def _centroid(ring):
    n = len(ring)
    return [sum(c[0] for c in ring)/n, sum(c[1] for c in ring)/n]


# ── graph construction ────────────────────────────────────────
def _build():
    global _graph, _node_c, _buildings

    with open(GEOJSON, encoding='utf-8') as f:
        gj = json.load(f)

    segs = [feat['geometry']['coordinates']
            for feat in gj['features']
            if feat['geometry']['type'] == 'LineString']

    all_raw = []
    for s in segs:
        all_raw.extend(s)
    N = len(all_raw)

    # ── Pass 1: union-find snap (5 m) ────────────────────────
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(N):
        for j in range(i+1, N):
            if _hav(all_raw[i], all_raw[j]) <= SNAP_M:
                union(i, j)

    class_nid = {}; nid_ctr = [0]

    def get_nid(i):
        r = find(i)
        if r not in class_nid:
            class_nid[r] = nid_ctr[0]; nid_ctr[0] += 1
        return class_nid[r]

    acc = {}
    for i, c in enumerate(all_raw):
        r = find(i)
        if r not in acc: acc[r] = [0.0, 0.0, 0]
        acc[r][0] += c[0]; acc[r][1] += c[1]; acc[r][2] += 1
    for r in acc:
        get_nid(r)

    node_c = {class_nid[r]: [s/cnt, lt/cnt]
              for r, (s, lt, cnt) in acc.items()}

    # Build graph from segments
    G = nx.Graph()
    for nid_, c in node_c.items():
        G.add_node(nid_, lng=c[0], lat=c[1])

    offset = 0
    for pts in segs:
        for i in range(len(pts)-1):
            na = get_nid(offset+i)
            nb = get_nid(offset+i+1)
            if na != nb:
                w = _hav(node_c[na], node_c[nb])
                if not G.has_edge(na, nb):
                    G.add_edge(na, nb, weight=w)
        offset += len(pts)

    # ── Pass 2: correct spine chain ───────────────────────────
    # Collect all spine-level nodes (on the main east-west road)
    spine_nodes = sorted(
        [n for n in G.nodes()
         if (_lat_m(node_c[n][1]) <= SPINE_LAT_M
             and SPINE_LNG_MIN <= node_c[n][0] <= SPINE_LNG_MAX)],
        key=lambda n: -node_c[n][0]   # east → west
    )

    # Remove any existing edges that SKIP over spine nodes
    # (bridge edges that jump from node A to node C when B lies between them)
    to_remove = []
    for a, b in list(G.edges()):
        if a not in spine_nodes or b not in spine_nodes:
            continue
        ia = spine_nodes.index(a)
        ib = spine_nodes.index(b)
        if abs(ia - ib) > 1:   # skips at least one spine node
            to_remove.append((a, b))
    for a, b in to_remove:
        G.remove_edge(a, b)

    # Add chain edges between consecutive spine nodes
    for i in range(len(spine_nodes)-1):
        a, b = spine_nodes[i], spine_nodes[i+1]
        if not G.has_edge(a, b):
            w = _hav(node_c[a], node_c[b])
            G.add_edge(a, b, weight=w, spine_chain=True)

    # Connect non-spine nodes that are near the spine to their correct
    # nearest spine neighbour (e.g. east-side stubs that are 1 node off)
    for n in list(G.nodes()):
        if n in spine_nodes:
            continue
        c = node_c[n]
        if not (SPINE_LNG_MIN <= c[0] <= SPINE_LNG_MAX):
            continue
        if _lat_m(c[1]) > SPINE_LAT_M:
            continue
        # Near-spine but not in spine_nodes — snap to nearest spine node
        nearest = min(spine_nodes, key=lambda s: _hav(node_c[s], c))
        d = _hav(node_c[nearest], c)
        if not G.has_edge(n, nearest):
            G.add_edge(n, nearest, weight=d, spine_snap=True)

    # ── Pass 3: bridge remaining isolated components ──────────
    for _ in range(80):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(comps) == 1:
            break
        main = set(comps[0])
        bridged_any = False
        for sc in comps[1:]:
            best_d, bp = float('inf'), None
            for sn in sc:
                for mn in main:
                    d = _hav(node_c[sn], node_c[mn])
                    if d < best_d:
                        best_d, bp = d, (sn, mn)
            if bp and best_d <= BRIDGE_M:
                G.add_edge(bp[0], bp[1], weight=best_d, bridged=True)
                main |= set(sc)
                bridged_any = True
        if not bridged_any:
            break

    # ── Snap buildings to nearest graph node ──────────────────
    all_nids = list(G.nodes())
    buildings = []

    for feat in gj['features']:
        geom  = feat['geometry']
        props = feat.get('properties') or {}
        name  = (props.get('name') or props.get('name ') or '').strip()
        if not name:
            continue
        if geom['type'] == 'Point':
            c = geom['coordinates']
            buildings.append({'name': name, 'lat': c[1], 'lng': c[0]})
        elif geom['type'] == 'Polygon':
            cen = _centroid(geom['coordinates'][0])
            buildings.append({'name': name, 'lat': cen[1], 'lng': cen[0]})

    def snap(b, exclude=None):
        bc = [b['lng'], b['lat']]
        candidates = [n for n in all_nids if n != exclude]
        return min(candidates, key=lambda n: _hav(node_c[n], bc))

    # Assign snap nodes; handle buildings that share the same nearest node.
    # Library Block and Automobile Lab are adjacent polygons that both snap
    # to the same pathway node — give each its own distinct closest node.
    LIB_NAME  = 'Library Block'
    AUTO_NAME = 'Automobile Lab'
    lib_b  = next((b for b in buildings if b['name'] == LIB_NAME),  None)
    auto_b = next((b for b in buildings if b['name'] == AUTO_NAME), None)

    if lib_b and auto_b:
        # Automobile Lab is closer to node45 — let it keep the nearest node.
        auto_b['snap_node'] = snap(auto_b)
        # Library Block must use a DIFFERENT node so routes are distinct.
        lib_b['snap_node']  = snap(lib_b, exclude=auto_b['snap_node'])
        assigned = {lib_b['name'], auto_b['name']}
    else:
        assigned = set()

    for b in buildings:
        if b['name'] not in assigned:
            b['snap_node'] = snap(b)

    _graph     = G
    _node_c    = node_c
    _buildings = buildings

    print(f"[OutdoorRouting] {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, "
          f"{nx.number_connected_components(G)} component(s), "
          f"{len(buildings)} buildings")


def _ensure():
    if _graph is None:
        _build()


# ── Public API ────────────────────────────────────────────────

def get_outdoor_locations():
    _ensure()
    seen, out = set(), []
    for b in _buildings:
        if b['name'] not in seen:
            seen.add(b['name'])
            out.append({'name': b['name'], 'lat': b['lat'], 'lng': b['lng']})
    return sorted(out, key=lambda x: x['name'])


def _find_b(name):
    nl = name.strip().lower()
    for b in _buildings:
        if b['name'].lower() == nl:
            return b
    return None


def get_outdoor_route(start_name, end_name):
    _ensure()
    sb = _find_b(start_name)
    eb = _find_b(end_name)
    if not sb:
        return {'error': f'Start "{start_name}" not found'}
    if not eb:
        return {'error': f'Destination "{end_name}" not found'}
    if sb['name'] == eb['name']:
        return {'error': 'Start and destination are the same'}
    try:
        ids     = nx.dijkstra_path(
                      _graph, sb['snap_node'], eb['snap_node'], weight='weight')
        total_m = nx.dijkstra_path_length(
                      _graph, sb['snap_node'], eb['snap_node'], weight='weight')
        coords  = [{'lat': _node_c[n][1], 'lng': _node_c[n][0]}
                   for n in ids]
        return {'path': coords, 'total_distance': round(total_m),
                'steps': len(coords), 'start': sb['name'], 'end': eb['name']}
    except nx.NetworkXNoPath:
        return {'error': 'No path between these locations'}
    except Exception as e:
        return {'error': str(e)}


def get_outdoor_route_from_gps(lat, lng, end_name):
    _ensure()
    gps  = [lng, lat]
    snap = min(_graph.nodes(),
               key=lambda n: _hav(_node_c[n], gps))
    eb   = _find_b(end_name)
    if not eb:
        return {'error': f'Destination "{end_name}" not found'}
    try:
        ids     = nx.dijkstra_path(
                      _graph, snap, eb['snap_node'], weight='weight')
        total_m = nx.dijkstra_path_length(
                      _graph, snap, eb['snap_node'], weight='weight')
        coords  = [{'lat': _node_c[n][1], 'lng': _node_c[n][0]}
                   for n in ids]
        return {'path': coords, 'total_distance': round(total_m),
                'steps': len(coords), 'start': 'Your Location', 'end': eb['name']}
    except nx.NetworkXNoPath:
        return {'error': 'No path found'}


_build()
