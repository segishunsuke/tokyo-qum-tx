# -*- coding: utf-8 -*-
"""
マルチモーダル最短経路探索・交通量配分プログラム【旅客版・一般道路／高速道路 分離ネットワーク】

主要モード別スキム版は search_passenger_separated_modeskim.py。

【修正履歴】
- settings に rail_network_file を追加（反実仮想分析用．省略時は rail_network_{year}.gpkg）。
- use_walk=1（道路を徒歩として扱う）のときも高速道路ネットワークを読み込んでおり、
  徒歩の利用者が高速道路を自動車の速度・費用で通行できてしまっていた
  （コメントでは「高速道路は徒歩で走れないので use_highway は無効」としていた）。
  → use_walk=1 のときは use_highway の値にかかわらず高速道路を読み込まないよう修正
    （統合道路ネットワーク版で高速道路リンクを封鎖しているのと同じ扱い）。
"""
import os
import csv
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from numba import njit
import warnings

# 不要な警告を非表示にし、コンソール出力をクリーンに保つ
warnings.filterwarnings("ignore")

# ==========================================================
# 1. Numba内蔵 超高速バイナリヒープ (優先度付きキュー)
# ==========================================================
# ダイクストラ法のボトルネックとなる「最もコストの低いノードの取り出し」を
# C言語並みの速度で行うための自作ヒープ構造。PythonのheapqはNumba内では使えないため必須。
@njit
def heap_push(heap_keys, heap_nodes, node_to_heap, n_items, node, key):
    i = n_items
    heap_keys[i] = key
    heap_nodes[i] = node
    node_to_heap[node] = i
    while i > 0:
        parent = (i - 1) >> 1
        if heap_keys[i] < heap_keys[parent]:
            tk = heap_keys[i]; heap_keys[i] = heap_keys[parent]; heap_keys[parent] = tk
            tn = heap_nodes[i]; heap_nodes[i] = heap_nodes[parent]; heap_nodes[parent] = tn
            node_to_heap[heap_nodes[i]] = i
            node_to_heap[heap_nodes[parent]] = parent
            i = parent
        else:
            break
    return n_items + 1

@njit
def heap_pop(heap_keys, heap_nodes, node_to_heap, n_items):
    ret_node = heap_nodes[0]
    ret_key = heap_keys[0]
    node_to_heap[ret_node] = -1
    n_items -= 1
    if n_items > 0:
        heap_keys[0] = heap_keys[n_items]
        heap_nodes[0] = heap_nodes[n_items]
        node_to_heap[heap_nodes[0]] = 0
        i = 0
        while True:
            left = (i << 1) + 1
            right = left + 1
            smallest = i
            if left < n_items and heap_keys[left] < heap_keys[smallest]: smallest = left
            if right < n_items and heap_keys[right] < heap_keys[smallest]: smallest = right
            if smallest != i:
                tk = heap_keys[i]; heap_keys[i] = heap_keys[smallest]; heap_keys[smallest] = tk
                tn = heap_nodes[i]; heap_nodes[i] = heap_nodes[smallest]; heap_nodes[smallest] = tn
                node_to_heap[heap_nodes[i]] = i
                node_to_heap[heap_nodes[smallest]] = smallest
                i = smallest
            else:
                break
    return ret_node, ret_key, n_items

@njit
def heap_decrease_key(heap_keys, heap_nodes, node_to_heap, node, new_key):
    i = node_to_heap[node]
    if i != -1 and new_key < heap_keys[i]:
        heap_keys[i] = new_key
        while i > 0:
            parent = (i - 1) >> 1
            if heap_keys[i] < heap_keys[parent]:
                tk = heap_keys[i]; heap_keys[i] = heap_keys[parent]; heap_keys[parent] = tk
                tn = heap_nodes[i]; heap_nodes[i] = heap_nodes[parent]; heap_nodes[parent] = tn
                node_to_heap[heap_nodes[i]] = i
                node_to_heap[heap_nodes[parent]] = parent
                i = parent
            else:
                break

# ==========================================================
# 2. 超高速ダイクストラ & バックトレース配分 (Numba化)
# ==========================================================
@njit
def gc_dist(lon1, lat1, lon2, lat2):
    """大円距離の計算 (アクセスリンク生成時などに使用)"""
    pi_d_180 = np.pi / 180.0
    phi1, phi2 = lat1 * pi_d_180, lat2 * pi_d_180
    dphi, dlam = (lat2 - lat1) * pi_d_180, (lon2 - lon1) * pi_d_180
    a = np.sin(dphi * 0.5)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam * 0.5)**2
    
    # Numba内では配列以外に np.clip が使えない仕様のため、標準の min/max で丸める
    a = min(1.0, max(0.0, a))
    return 6371.0087714 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

@njit
def run_dijkstra_and_assign(o_node, target_nodes, od_volumes, n_nodes, 
                            head, to, cost, head_from,
                            labels, edge_pointers, heap_keys, heap_nodes, node_to_heap, 
                            link_volumes, link_idx_map):
    """
    1つの出発地セントロイド(O)からすべての到着地セントロイド(D)への最短経路を
    一度に探索し（One-to-Many）、確定した経路を逆順に辿って交通量を各リンクに加算する。
    出発側セントロイドは流出リンクのみ、到着側セントロイドは流入リンクのみを持つため、
    セントロイド通り抜けを禁止する特殊処理は不要である。
    """
    # 探索用配列の初期化
    labels.fill(np.inf)
    edge_pointers.fill(-1)
    node_to_heap.fill(-1)
    
    labels[o_node] = 0.0
    n_items = heap_push(heap_keys, heap_nodes, node_to_heap, 0, o_node, 0.0)
    
    # 1. Dijkstra 探索 (全ノードに対する最短費用を確定)
    while n_items > 0:
        u, d_val, n_items = heap_pop(heap_keys, heap_nodes, node_to_heap, n_items)
        if d_val > labels[u]: continue
        
# CSR形式配列 (head, to) を使って、ノードuから出るリンクを高速に走査
        start_edge = head[u]
        end_edge = head[u + 1]
        
        for e in range(start_edge, end_edge):
            v = to[e]
            w = cost[e]
            new_dist = d_val + w
            
            # より安い経路が見つかれば更新
            if new_dist < labels[v]:
                labels[v] = new_dist
                edge_pointers[v] = e # どこから来たかを記録 (バックトレース用)
                if node_to_heap[v] != -1:
                    heap_decrease_key(heap_keys, heap_nodes, node_to_heap, v, new_dist)
                else:
                    n_items = heap_push(heap_keys, heap_nodes, node_to_heap, n_items, v, new_dist)

    # 2. バックトレースと交通量の配分 (All-or-Nothing配分)
    for i in range(len(target_nodes)):
        d_node = target_nodes[i]
        vol = od_volumes[i]
        
        # 交通量がゼロ、または到達不可能な目的地はスキップ
        if vol <= 0.0 or labels[d_node] == np.inf:
            continue
            
        curr = d_node
        # 目的地から出発地へ向かって、辿ってきたリンクを逆行する
        while curr != o_node:
            e = edge_pointers[curr]
            if e == -1: break
            
            orig_link_id = link_idx_map[e]
            link_volumes[orig_link_id] += vol # リンクに交通量を加算
            curr = head_from[e]


# ==========================================================
# 3. ネットワーク統合エンジン
# ==========================================================
class TrafficAssigner:
    def __init__(self, settings_file):
        self.settings = self._load_settings(settings_file)
        self.p        = self._build_params()   # 検証済みパラメータ (SimpleNamespace)

        self.nodes             = []
        self.links             = []
        self.current_global_id = 0
        self.local_to_global   = {}  # (mode_name, orig_node_id) -> global_id のマッピング

        self.road_kdtree         = None
        self.road_global_ids     = []
        self.origin_centroid_global_ids = []
        self.destination_centroid_global_ids = []

    def _load_settings(self, file_path):
        """settings.csv を読み込み、数値に変換できるものは変換する（raw 辞書を返す）"""
        s = {}
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    val = row[1].strip()
                    try:
                        s[row[0].strip()] = float(val)
                    except ValueError:
                        s[row[0].strip()] = val  # 数値変換できない場合は文字列のまま保持
        return s

    def _build_params(self):
        """
        settings.csv の生辞書を検証して SimpleNamespace に展開する。

        【設計方針】
        - 必須キーが欠損している場合は KeyError を即時 raise する
          （`.get()` のデフォルト値に頼って静かに誤動作するより明示的なエラーが望ましい）
        - use_xxx フラグは int(1/0) として読む
        - bbox は空欄を許容する唯一のオプションパラメータ
        """
        from types import SimpleNamespace
        s = self.settings

        def req(key):
            """必須キーを取得。なければ KeyError を raise する"""
            if key not in s:
                raise KeyError(f"settings に必須パラメータ '{key}' が見つかりません。")
            return s[key]

        p = SimpleNamespace(
            # --- 年度・基本フラグ ---
            init_year        = int(req('init_year')),
            term_year        = int(req('term_year')),
            interval         = int(req('interval')),
            use_walk         = int(req('use_walk')),
            use_highway      = int(req('use_highway')),
            use_bus          = int(req('use_bus')),
            use_highway_bus  = int(req('use_highway_bus')),
            use_rail         = int(req('use_rail')),
            use_sea_route    = int(req('use_sea_route')),
            use_air          = int(req('use_air')),

            # --- アクセスリンク接続パラメータ ---
            n_acc_min        = int(req('n_access_min')),
            n_acc_max        = int(req('n_access_max')),
            acc_max_dist_km  = float(req('access_max_dist_km')),

            # --- コストパラメータ ---
            speed_access_kmh   = float(req('speed_access_kmh')),
            speed_centroid_connector_kmh = float(req('speed_centroid_connector_kmh')),
            speed_walking_kmh  = float(req('speed_walking_kmh')),  # 徒歩速度 [km/h]
            vot_per_min        = float(req('vot_per_min')),
            vot_road_per_min   = float(req('vot_road_per_min')),
            vod_road_per_km    = float(req('vod_road_per_km')),
            vod_highway_per_km = float(req('vod_highway_per_km')),
        )

        # 複合パラメータ
        # use_walk=1 の場合: 道路は徒歩移動のみ → 車両コスト(vot_road)はゼロとして扱う
        # use_walk=0 の場合: 通常通り vot_road_per_min を加算
        p.vot_total = p.vot_per_min + (0.0 if p.use_walk == 1 else p.vot_road_per_min)

        # --- BBox（空欄を許容するオプションパラメータ）---
        try:
            bvals = [float(s['bbox_min_lon']), float(s['bbox_min_lat']),
                     float(s['bbox_max_lon']), float(s['bbox_max_lat'])]
            p.bbox = tuple(bvals)
        except (KeyError, ValueError, TypeError):
            p.bbox = None  # 未設定または空欄の場合は全域を対象にする

        return p

    def _load_gpkg(self, filepath, layer):
        # 【修正】使用するモードのファイルが無い場合に，黙って空のネットワークとして扱うと
        #         そのモードが無い状態で費用が計算されてしまうため，エラーで止める
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"入力ファイルが見つかりません: {filepath}")
        return gpd.read_file(filepath, layer=layer, bbox=self.p.bbox)


    def load_road_network(self):
        """
        ベースとなる一般道路ネットワーク（local_road_network.gpkg）の読み込みとKDTreeの構築。

        【ファイル仕様】
        - ノード属性: node_id
        - リンク属性: link_id, from_node, to_node, distance_km, time_min
        """
        print("Loading Local Road Network...")
        # ファイル名を local_road_network.gpkg に変更
        gdf_n = self._load_gpkg('./input/local_road_network.gpkg', 'nodes')
        gdf_l = self._load_gpkg('./input/local_road_network.gpkg', 'links')
        if gdf_n.empty: return
        
        road_coords = []
        for _, row in gdf_n.iterrows():
            g_id = self.current_global_id
            self.local_to_global[('road', row.get('node_id', row.name))] = g_id
            self.nodes.append({'global_id': g_id, 'lon': row.geometry.x, 'lat': row.geometry.y,
                               'mode': 1, 'name': 'road', 'orig_id': row.get('node_id', row.name)})
            road_coords.append((row.geometry.x, row.geometry.y))
            self.road_global_ids.append(g_id)
            self.current_global_id += 1
            
        print("  Building KDTree for access links...")
        # 各交通モードの停留所・駅・港などから最寄りの一般道路ノードへ
        # アクセスリンクを引くための空間インデックス
        self.road_kdtree = cKDTree(road_coords)

        p         = self.p
        # -------------------------------------------------------
        # use_walk=1 の場合:
        #   道路ネットワークを「徒歩のみ」として扱う。
        #   - リンクの time_min（自動車前提）は使わず、
        #     distance_km / speed_walking_kmh * 60 で再計算する
        #   - 車両コスト(vot_road)はゼロ（_build_params で vot_total に反映済み）
        #   - 高速道路は徒歩で走れないので use_highway は無効
        #     （build_full_network で高速道路ネットワークを読み込まない）
        # use_walk=0 の場合:
        #   通常の自動車ネットワークとして扱う。time_min をそのまま使用。
        # -------------------------------------------------------
        use_walk      = p.use_walk
        walk_speed    = p.speed_walking_kmh
        vot_total     = p.vot_total
        vod           = p.vod_road_per_km
        
        for _, row in gdf_l.iterrows():
            u = self.local_to_global.get(('road', row['from_node']))
            v = self.local_to_global.get(('road', row['to_node']))
            if u is not None and v is not None:
                dist_km  = float(row['distance_km'])
                if use_walk == 1:
                    # 徒歩速度で時間を再計算（高速道路は build_full_network で読み込まない）
                    time_min = dist_km / walk_speed * 60.0
                else:
                    time_min = float(row['time_min'])
                cost = time_min * vot_total + dist_km * vod
                self.links.append({'from': u, 'to': v, 'cost': cost, 'type': 1,
                                   'orig_id': row['link_id'], 'name': ''})

    def load_mode_network(self, mode_name, type_id, filename,
                          access_board_type, access_alight_type,
                          connect_condition,
                          cost_func_internal, cost_func_board, cost_func_alight):
        """
        各交通機関（高速道路・バス・高速バス・鉄道・海路・航空）のネットワークを読み込み、
        一般道路ネットワークと結ぶ汎用メソッド。

        【引数】
        mode_name          : モード識別文字列（local_to_global マップのキーに使用）
        type_id            : リンク種別番号（highway=2, bus=3, highway_bus=4, rail=5, sea=6, air=7）
        filename           : input フォルダ内の gpkg ファイル名
        access_board_type  : 乗車アクセスリンクの type 番号（例: 31）
        access_alight_type : 降車アクセスリンクの type 番号（例: 32）
        connect_condition  : (node_row) -> bool
                             Trueを返すノードのみ一般道路と接続する
                             （例: 鉄道は is_station==true のみ、バスは常に True）
        cost_func_internal : (link_row) -> float
                             モード内リンクの一般化費用を返す関数
        cost_func_board    : (node_row, dist_km) -> float
                             【設計統一】ノード行とアクセス距離を受け取り乗車アクセスコストを返す。
                             初乗り運賃・待ち時間はノード属性から読むため、
                             settings.csv への依存を排除している。
        cost_func_alight   : (node_row, dist_km) -> float
                             降車アクセスコストを返す関数（通常はアクセスコストのみ）
        """
        print(f"Loading {mode_name.capitalize()} Network...")
        gdf_n = self._load_gpkg(f'./input/{filename}', 'nodes')
        gdf_l = self._load_gpkg(f'./input/{filename}', 'links')
        if gdf_n.empty: return

        s = self.settings
        p               = self.p
        n_acc_min       = p.n_acc_min
        n_acc_max       = p.n_acc_max
        acc_max_dist_km = p.acc_max_dist_km

        for _, row in gdf_n.iterrows():
            g_id = self.current_global_id
            self.local_to_global[(mode_name, row.get('node_id', row.name))] = g_id
            
            mode_lon, mode_lat = row.geometry.x, row.geometry.y
            self.nodes.append({'global_id': g_id, 'lon': mode_lon, 'lat': mode_lat,
                               'mode': type_id, 'name': row.get('name', ''),
                               'orig_id': row.get('node_id', row.name)})
            self.current_global_id += 1

            # connect_condition が False のノード（乗換専用ノード等）はスキップ
            if not connect_condition(row):
                continue

            # 最大 n_acc_max 個の候補ノードを一括取得
            dists_approx, idxs = self.road_kdtree.query((mode_lon, mode_lat), k=n_acc_max)
            if n_acc_max == 1:
                dists_approx, idxs = [dists_approx], [idxs]
            
            for rank, (_, idx) in enumerate(zip(dists_approx, idxs)):
                r_g_id  = self.road_global_ids[idx]
                r_node  = self.nodes[r_g_id]
                dist_km = gc_dist(mode_lon, mode_lat, r_node['lon'], r_node['lat'])
                
                # n_acc_min 本目までは距離制限なし（孤立地域でも最低限接続を保証）
                # それ以上は acc_max_dist_km 以内の場合のみ接続
                if rank >= n_acc_min and dist_km > acc_max_dist_km:
                    continue
                
                # ノード属性（初乗り運賃・待ち時間）を cost_func に渡してアクセスコストを計算
                c_board  = cost_func_board(row, dist_km)
                c_alight = cost_func_alight(row, dist_km)
                
                self.links.append({'from': r_g_id, 'to': g_id,   'cost': c_board,  'type': access_board_type,  'orig_id': -1, 'name': 'Access'})
                self.links.append({'from': g_id,   'to': r_g_id, 'cost': c_alight, 'type': access_alight_type, 'orig_id': -1, 'name': 'Access'})

        for _, row in gdf_l.iterrows():
            u = self.local_to_global.get((mode_name, row['from_node']))
            v = self.local_to_global.get((mode_name, row['to_node']))
            if u is not None and v is not None:
                cost = cost_func_internal(row)
                name = row.get('name', row.get('operator', row.get('route_name', '')))
                self.links.append({'from': u, 'to': v, 'cost': cost, 'type': type_id,
                                   'orig_id': row.get('link_id', -1), 'name': name})

    def load_centroids(self):
        """
        出発側・到着側セントロイドを分け、一般道路相当の費用を持つコネクタで接続する。

        出発側は centroid_O -> road のみ、到着側は road -> centroid_D のみとする。
        この有向構造により、セントロイド通り抜けを構造的に排除する。
        コネクタ費用には一般道路と同じ vot_total と vod_road_per_km を適用する。
        """
        print("Loading Centroids (separate origin/destination nodes)...")
        gdf_c = self._load_gpkg('./input/centroids.gpkg', 'centroids')
        if gdf_c.empty:
            return

        p = self.p
        if p.speed_centroid_connector_kmh <= 0.0:
            raise ValueError("speed_centroid_connector_kmh は正の値にしてください。")

        for _, row in gdf_c.iterrows():
            c_lon, c_lat = row.geometry.x, row.geometry.y
            orig_id = row.get('node_id', row.name)
            name = row.get('name', '')

            o_g_id = self.current_global_id
            self.nodes.append({
                'global_id': o_g_id, 'lon': c_lon, 'lat': c_lat,
                'mode': 98, 'name': name, 'orig_id': orig_id
            })
            self.origin_centroid_global_ids.append(o_g_id)
            self.current_global_id += 1

            d_g_id = self.current_global_id
            self.nodes.append({
                'global_id': d_g_id, 'lon': c_lon, 'lat': c_lat,
                'mode': 99, 'name': name, 'orig_id': orig_id
            })
            self.destination_centroid_global_ids.append(d_g_id)
            self.current_global_id += 1

            dists_approx, idxs = self.road_kdtree.query(
                (c_lon, c_lat), k=p.n_acc_max
            )
            if p.n_acc_max == 1:
                dists_approx, idxs = [dists_approx], [idxs]

            for rank, (_, idx) in enumerate(zip(dists_approx, idxs)):
                r_g_id = self.road_global_ids[idx]
                r_node = self.nodes[r_g_id]
                dist_km = gc_dist(c_lon, c_lat, r_node['lon'], r_node['lat'])

                if rank >= p.n_acc_min and dist_km > p.acc_max_dist_km:
                    continue

                time_min = dist_km / p.speed_centroid_connector_kmh * 60.0
                connector_cost = (
                    time_min * p.vot_total
                    + dist_km * p.vod_road_per_km
                )

                self.links.append({
                    'from': o_g_id, 'to': r_g_id,
                    'cost': connector_cost, 'type': 98,
                    'orig_id': -1, 'name': 'CentroidOrigin'
                })
                self.links.append({
                    'from': r_g_id, 'to': d_g_id,
                    'cost': connector_cost, 'type': 99,
                    'orig_id': -1, 'name': 'CentroidDestination'
                })

    def build_full_network(self, year):
        """
        year: 分析対象年。ネットワークファイル名（rail_network_{year}.gpkg 等）に使用。
        """
        self.load_road_network()
        p         = self.p
        vot       = p.vot_per_min        # 旅客の時間価値 [円/分]
        vot_total = p.vot_total          # 旅客VOT + アクセス移動VOT（車両費用含む）
        vod       = p.vod_road_per_km    # 一般道路の距離コスト [円/km]
        speed_acc = p.speed_access_kmh   # アクセス移動速度 [km/h]（徒歩・車など settings で柔軟に設定）

        # ----------------------------------------------------------
        # 高速道路 (type 2)
        # 接続条件: is_ic == true のノードのみ（インターチェンジ）
        # 初乗り料金: ノード属性 initial_toll_car
        # 区間料金:   リンク属性 toll_car
        # 【修正】use_walk=1（徒歩）のときは高速道路を使えないので読み込まない
        #         （修正前は徒歩でも高速道路を自動車の速度・費用で通行できていた）
        # ----------------------------------------------------------
        if p.use_highway == 1 and p.use_walk == 0:
            c_cond = lambda n: bool(n.get('is_ic', False))
            c_int  = lambda r: (r['time_min'] * vot_total
                                + r['distance_km'] * p.vod_highway_per_km
                                + float(r.get('toll_car', 0.0)))
            # 乗車: アクセスコスト + 初乗り料金（ノード属性）
            c_brd  = lambda n, d: ((d / speed_acc * 60) * vot_total
                                   + d * vod
                                   + float(n.get('initial_toll_car', 0.0)))
            # 降車: アクセスコストのみ
            c_alg  = lambda n, d: (d / speed_acc * 60) * vot_total + d * vod
            self.load_mode_network('highway', 2, f'highway_network_{year}.gpkg',
                                   21, 22, c_cond, c_int, c_brd, c_alg)

        # ----------------------------------------------------------
        # 路線バス (type 3)
        # 接続条件: すべてのノード（全ノードがバス停）
        # 初乗り運賃: ノード属性 initial_fare
        # 待ち時間:   ノード属性 waiting_time_min
        # 区間運賃:   リンク属性 fare
        # ----------------------------------------------------------
        if p.use_bus == 1:
            c_cond = lambda n: True  # 全ノードが接続対象
            c_int  = lambda r: r['time_min'] * vot + float(r.get('fare', 0.0))
            c_brd  = lambda n, d: ((d / speed_acc * 60) * vot_total
                                   + d * vod
                                   + float(n.get('waiting_time_min', 0.0)) * vot
                                   + float(n.get('initial_fare', 0.0)))
            c_alg  = lambda n, d: (d / speed_acc * 60) * vot_total + d * vod
            self.load_mode_network('bus', 3, f'bus_network_{year}.gpkg',
                                   31, 32, c_cond, c_int, c_brd, c_alg)

        # ----------------------------------------------------------
        # 高速バス (type 4)
        # 接続条件: すべてのノード（全ノードがバス停）
        # 初乗り運賃: ノード属性 initial_fare
        # 待ち時間:   ノード属性 waiting_time_min
        # 区間運賃:   リンク属性 fare
        # ※路線バスと同じ構造だが VOT は旅客 vot のみ（バス車内コストに道路VOTは含まない）
        # ----------------------------------------------------------
        if p.use_highway_bus == 1:
            c_cond = lambda n: True
            c_int  = lambda r: r['time_min'] * vot + float(r.get('fare', 0.0))
            c_brd  = lambda n, d: ((d / speed_acc * 60) * vot_total
                                   + d * vod
                                   + float(n.get('waiting_time_min', 0.0)) * vot
                                   + float(n.get('initial_fare', 0.0)))
            c_alg  = lambda n, d: (d / speed_acc * 60) * vot_total + d * vod
            self.load_mode_network('highway_bus', 4, f'highway_bus_network_{year}.gpkg',
                                   41, 42, c_cond, c_int, c_brd, c_alg)

        # ----------------------------------------------------------
        # 鉄道 (type 5)
        # 接続条件: is_station == true のノードのみ（乗換専用ノードを除外）
        # 初乗り運賃: ノード属性 initial_fare
        # 待ち時間:   ノード属性 waiting_time_min
        # 区間運賃:   リンク属性 fare
        # ※ shinkansen 特例は廃止（特急料金は fare に含まれる）
        # ----------------------------------------------------------
        if p.use_rail == 1:
            c_cond = lambda n: bool(n.get('is_station', False))
            c_int  = lambda r: r['time_min'] * vot + float(r.get('fare', 0.0))
            c_brd  = lambda n, d: ((d / speed_acc * 60) * vot_total
                                   + d * vod
                                   + float(n.get('waiting_time_min', 0.0)) * vot
                                   + float(n.get('initial_fare', 0.0)))
            c_alg  = lambda n, d: (d / speed_acc * 60) * vot_total + d * vod
            # 【追加】settings の rail_network_file で鉄道ネットワークのファイルを指定できる
            #   （反実仮想：TX を削除した rail_network_2015_noTX.gpkg など．{year} は年次に置き換え）
            rail_file = str(self.settings.get('rail_network_file', '')).strip() or 'rail_network_{year}.gpkg'
            rail_file = rail_file.replace('{year}', str(year))
            if not os.path.exists(f'./input/{rail_file}'):
                raise FileNotFoundError(f'./input/{rail_file} がありません。')
            print(f"  鉄道ネットワーク: {rail_file}")
            self.load_mode_network('rail', 5, rail_file,
                                   51, 52, c_cond, c_int, c_brd, c_alg)

        # ----------------------------------------------------------
        # 海路 (type 6)
        # 接続条件: is_transfer == false かつ is_passenger == true
        # 初乗り運賃: ノード属性 initial_fare_passenger
        # 待ち時間:   ノード属性 waiting_time_passenger
        # 区間運賃:   リンク属性 fare_passenger
        # 旅客不可リンク (is_passenger == false) は大ペナルティで実質通行不可
        # ----------------------------------------------------------
        if p.use_sea_route == 1:
            c_cond = lambda n: (not bool(n.get('is_transfer', True))
                                and bool(n.get('is_passenger', False)))
            def c_int_sea(r):
                if not r.get('is_passenger', False):
                    return 1e8
                return r['time_min'] * vot + float(r.get('fare_passenger', 0.0))
            c_brd  = lambda n, d: ((d / speed_acc * 60) * vot_total
                                   + d * vod
                                   + float(n.get('waiting_time_passenger', 0.0)) * vot
                                   + float(n.get('initial_fare_passenger', 0.0)))
            c_alg  = lambda n, d: (d / speed_acc * 60) * vot_total + d * vod
            self.load_mode_network('sea', 6, 'sea_network.gpkg',
                                   61, 62, c_cond, c_int_sea, c_brd, c_alg)

        # ----------------------------------------------------------
        # 航空 (type 7)
        # 接続条件: is_transfer == false かつ is_passenger == true
        # 初乗り運賃: ノード属性 initial_fare_passenger
        # 待ち時間:   ノード属性 waiting_time_min_passenger
        # 区間運賃:   リンク属性 fare_passenger
        # 旅客不可リンク (is_passenger == false) は大ペナルティで実質通行不可
        # ----------------------------------------------------------
        if p.use_air == 1:
            c_cond = lambda n: (not bool(n.get('is_transfer', True))
                                and bool(n.get('is_passenger', False)))
            def c_int_air(r):
                if not r.get('is_passenger', False):
                    return 1e8
                return r['time_min'] * vot + float(r.get('fare_passenger', 0.0))
            c_brd  = lambda n, d: ((d / speed_acc * 60) * vot_total
                                   + d * vod
                                   + float(n.get('waiting_time_min_passenger', 0.0)) * vot
                                   + float(n.get('initial_fare_passenger', 0.0)))
            c_alg  = lambda n, d: (d / speed_acc * 60) * vot_total + d * vod
            self.load_mode_network('air', 7, f'air_network_{year}.gpkg',
                                   71, 72, c_cond, c_int_air, c_brd, c_alg)

        self.load_centroids()

    def compile_for_numba(self):
        """
        Numbaでのダイクストラ探索を極限まで高速化するため、
        Pythonのリスト構造を1次元配列を用いたCSR(Compressed Sparse Row)形式のグラフデータに変換する。
        """
        print("Compiling network to CSR format for Numba...")
        n_nodes = len(self.nodes)
        n_links = len(self.links)
        
        # from_node を基準にソートしておく
        self.links.sort(key=lambda x: x['from'])
        
        head = np.zeros(n_nodes + 1, dtype=np.int32)
        to = np.zeros(n_links, dtype=np.int32)
        cost = np.zeros(n_links, dtype=np.float64)
        head_from = np.zeros(n_links, dtype=np.int32)
        
        curr_head = 0
        for i, link in enumerate(self.links):
            u = link['from']
            while curr_head <= u:
                head[curr_head] = i
                curr_head += 1
            to[i] = link['to']
            cost[i] = link['cost']
            head_from[i] = u
            
        while curr_head <= n_nodes:
            head[curr_head] = n_links
            curr_head += 1
            
        return n_nodes, head, to, cost, head_from

    def assign_traffic(self, od_csv_path=None, year=None):
        """
        year: 分析対象年。省略時は settings の init_year を使用。
              __main__ のループから明示的に渡すことで、1つの settings ファイルのまま
              複数年を順番に処理できる。
        """
        if year is None:
            year = self.p.init_year
        self.build_full_network(year)
        n_nodes, head, to, cost, head_from = self.compile_for_numba()
        
        n_c = len(self.origin_centroid_global_ids)
        print(f"  セントロイド数: {n_c}")
        if n_c == 0:
            raise RuntimeError("セントロイドが読み込まれていません（./input/centroids.gpkg と bbox を確認）。")
        if n_c != len(self.destination_centroid_global_ids):
            raise RuntimeError("出発側と到着側のセントロイド数が一致しません。")
        origin_nodes = np.array(self.origin_centroid_global_ids, dtype=np.int32)
        target_nodes = np.array(self.destination_centroid_global_ids, dtype=np.int32)
        od_matrix = np.zeros((n_c, n_c), dtype=np.float64)

        # === 【バグ修正】リスト形式OD表の読み込み ===
        if od_csv_path is not None and os.path.exists(od_csv_path):
            print(f"Loading OD Matrix from '{od_csv_path}'...")
            
            # 【重要】IDが数値で読み込まれた場合の照合エラーを防ぐため、キーを文字列にキャストして保持
            # 【修正】ID の表記ゆれ（0 / 0.0 / '0'）を吸収して照合する
            def _nid(x):
                try:
                    return str(int(float(x)))
                except (TypeError, ValueError):
                    return str(x).strip()
            orig_to_idx = {_nid(self.nodes[g_id]['orig_id']): i for i, g_id in enumerate(self.origin_centroid_global_ids)}
            
            df_od = pd.read_csv(od_csv_path)
            
            # ヘッダ名に依存せず、左から1列目(O), 2列目(D), 3列目(交通量)として扱う
            n_skip, vol_skip = 0, 0.0
            for o_raw, d_raw, vol in zip(df_od.iloc[:, 0], df_od.iloc[:, 1], df_od.iloc[:, 2].astype(float)):
                o_orig, d_orig = _nid(o_raw), _nid(d_raw)
                # ネットワーク上に存在する起点・終点の場合のみ交通量をセット
                if o_orig in orig_to_idx and d_orig in orig_to_idx:
                    od_matrix[orig_to_idx[o_orig], orig_to_idx[d_orig]] += vol
                else:
                    n_skip += 1
                    vol_skip += vol
            if n_skip:
                print(f"  ※ セントロイドと照合できない OD：{n_skip:,} 行（交通量 {vol_skip:,.1f}）")
            if od_matrix.sum() <= 0.0:
                raise RuntimeError("OD 表の交通量がすべて 0 です（OD 表の ID と centroids.gpkg の node_id を確認）。")
                    
            print(f"  Loaded actual OD volumes. Total Volume: {od_matrix.sum():.2f}")
        else:
            print("Warning: No valid OD CSV provided. Using fallback mock OD (1.0 for all pairs).")
            od_matrix = np.ones((n_c, n_c), dtype=np.float64)
            np.fill_diagonal(od_matrix, 0.0)
        # ====================================================

        print("Allocating memory for assignment...")
        labels = np.empty(n_nodes, dtype=np.float64)
        edge_pointers = np.empty(n_nodes, dtype=np.int32)
        heap_keys = np.empty(n_nodes, dtype=np.float64)
        heap_nodes = np.empty(n_nodes, dtype=np.int32)
        node_to_heap = np.empty(n_nodes, dtype=np.int32)
        
        link_volumes = np.zeros(len(self.links), dtype=np.float64)
        min_cost_matrix = np.zeros((n_c, n_c), dtype=np.float64)

        print("Running Dijkstra & Assignment for all Origins...")
        for i, o_node in enumerate(origin_nodes):
            if i % 100 == 0: print(f"  Processed {i}/{len(target_nodes)} origins...")
            od_row = od_matrix[i]
            
            run_dijkstra_and_assign(
                o_node, target_nodes, od_row, n_nodes,
                head, to, cost, head_from,
                labels, edge_pointers, heap_keys, heap_nodes, node_to_heap,
                link_volumes, np.arange(len(self.links), dtype=np.int32)
            )

            # 最小旅行費用(インピーダンス)を行列に抽出
            min_cost_matrix[i, :] = labels[target_nodes]

        print("Assignment complete. Exporting results...")
        # 出力ファイル名のサフィックス: _passenger_{year}
        # 【修正】run_label（例：pt, car）を付け，徒歩＋公共交通と自動車の結果が上書きされないようにする
        label = str(self.settings.get('run_label', '')).strip()
        suffix = f"_passenger_{label}_{year}" if label else f"_passenger_{year}"
        self._export_results(link_volumes, suffix)

        # 最小旅行費用のリスト出力
        print("Exporting OD Minimum Cost List...")
        with open(f'./output/od_min_cost{suffix}.csv', 'w', encoding='utf-8-sig', newline='') as fout:
            fout.write("o_id,d_id,cost\n")
            for i in range(n_c):
                o_id = self.nodes[self.origin_centroid_global_ids[i]]['orig_id']
                for j in range(n_c):
                    d_id = self.nodes[self.destination_centroid_global_ids[j]]['orig_id']
                    fout.write(f"{o_id},{d_id},{min_cost_matrix[i, j]}\n")
                    
        print("All processes completed successfully.")

    def _export_results(self, link_volumes, suffix=''):
        """
        各モード別の交通量配分結果をCSV・GPKGに分割して出力する。
        suffix: ファイル名に付加するサフィックス（例: '_passenger_2020'）
        """
        os.makedirs("./output", exist_ok=True)
        station_vols   = {}
        access_geom    = []
        access_records = []

        # 全モード統一フォーマット: ORIG_ID, VOLUME
        files = {
            1: csv.writer(open(f'./output/result_local_road{suffix}.csv',    'w', newline='', encoding='utf-8-sig')),
            2: csv.writer(open(f'./output/result_highway{suffix}.csv',       'w', newline='', encoding='utf-8-sig')),
            3: csv.writer(open(f'./output/result_bus{suffix}.csv',           'w', newline='', encoding='utf-8-sig')),
            4: csv.writer(open(f'./output/result_highway_bus{suffix}.csv',   'w', newline='', encoding='utf-8-sig')),
            5: csv.writer(open(f'./output/result_rail{suffix}.csv',          'w', newline='', encoding='utf-8-sig')),
            6: csv.writer(open(f'./output/result_sea{suffix}.csv',           'w', newline='', encoding='utf-8-sig')),
            7: csv.writer(open(f'./output/result_air{suffix}.csv',           'w', newline='', encoding='utf-8-sig')),
        }
        for w in files.values():
            w.writerow(["ORIG_ID", "VOLUME"])

        for i, link in enumerate(self.links):
            vol    = link_volumes[i]
            if vol <= 0: continue

            l_type = link['type']
            if l_type in files:
                files[l_type].writerow([link['orig_id'], vol])

            elif isinstance(l_type, int) and 20 <= l_type < 80:
                u, v = self.nodes[link['from']], self.nodes[link['to']]
                access_geom.append(LineString([(u['lon'], u['lat']), (v['lon'], v['lat'])]))
                access_records.append({'link_type': l_type, 'volume': vol})

                is_boarding   = (l_type % 10 == 1)
                mode_node_idx = link['to'] if is_boarding else link['from']
                mode_str      = {2: "Highway", 3: "Bus", 4: "HighwayBus",
                                 5: "Rail", 6: "Sea", 7: "Air"}[l_type // 10]

                if mode_node_idx not in station_vols:
                    n = self.nodes[mode_node_idx]
                    station_vols[mode_node_idx] = [n['lon'], n['lat'], 0.0, 0.0, mode_str]

                if is_boarding: station_vols[mode_node_idx][2] += vol
                else:           station_vols[mode_node_idx][3] += vol

        if access_records:
            gdf_acc = gpd.GeoDataFrame(access_records, geometry=access_geom, crs="EPSG:4326")
            gdf_acc.to_file(f'./output/result_access{suffix}.gpkg', layer='access', driver='GPKG')

        with open(f'./output/result_stations{suffix}.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(["NODE_ID", "ORIG_ID", "NAME", "LON", "LAT", "BOARDING", "ALIGHTING", "TOTAL", "MODE"])
            for n_idx, data in station_vols.items():
                n     = self.nodes[n_idx]
                total = data[2] + data[3]
                w.writerow([n_idx, n['orig_id'], n['name'], data[0], data[1], data[2], data[3], total, data[4]])

if __name__ == '__main__':
    # settings から年度範囲を読み込み、init_year から term_year まで interval 刻みで実行。
    # term_year も必ず実行する（range は term_year+1 まで）。
    # 各年で独立したオブジェクトを生成することで、年をまたいだ状態の混入を防ぐ。
    # 【追加】設定ファイルをコマンドライン引数で指定できるようにする
    #   python search_passenger_separated.py ./input/settings_passenger_pt.csv
    import sys
    SETTINGS = sys.argv[1] if len(sys.argv) > 1 else './input/settings_passenger.csv'
    OD_CSV   = sys.argv[2] if len(sys.argv) > 2 else './input/od_passenger.csv'

    # 年度範囲の取得（settings のみ読んで p を構築）
    _eng = TrafficAssigner(SETTINGS)
    _p = _eng.p
    # 【追加】settings に years（例：1995;2015;2018）があればその年次を実行する
    yrs = str(_eng.settings.get('years', '')).strip()
    if yrs:
        years = [int(float(y)) for y in yrs.replace(',', ';').split(';') if y.strip()]
    else:
        years = list(range(_p.init_year, _p.term_year + 1, _p.interval))
        if _p.term_year not in years:
            years.append(_p.term_year)   # interval の割り切れに関わらず term_year を必ず含める
    print(f"実行する年次: {years}，ラベル: {_eng.settings.get('run_label', '')}")
    os.makedirs('./output', exist_ok=True)

    for year in years:
        print(f"\n{'='*60}")
        print(f"  Processing year: {year}")
        print(f"{'='*60}")
        engine = TrafficAssigner(SETTINGS)
        engine.assign_traffic(OD_CSV, year=year)