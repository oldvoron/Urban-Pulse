import pandas as pd
import numpy as np
import geopandas as gpd
import h3
import osmnx as ox
from math import log2


def poi_category_distribution(poi_df: pd.DataFrame) -> pd.Series:
    if poi_df.empty or "category" not in poi_df.columns:
        return pd.Series(dtype=int)
    return poi_df["category"].value_counts().head(15)


def _resolve_heights(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Extract numeric building heights from OSM columns, with fallbacks."""
    DEFAULT_HEIGHT = 6.0  # 2-storey assumption

    # 1. Try 'height'
    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce")
        valid = h.dropna()
        valid = valid[valid.between(1, 300)]
        if len(valid) > 0:
            pct = round(len(valid) / len(gdf) * 100)
            print(f"[morphology] height column: {pct}% valid")
            return h.fillna(DEFAULT_HEIGHT).clip(1, 300)

    # 2. Try 'building:levels'
    for col in ["building:levels", "levels"]:
        if col in gdf.columns:
            lvl = pd.to_numeric(gdf[col], errors="coerce")
            valid = lvl.dropna()
            valid = valid[valid.between(1, 80)]
            if len(valid) > 0:
                pct = round(len(valid) / len(gdf) * 100)
                print(f"[morphology] {col} fallback: {pct}% valid → ×3.5m")
                return (lvl.fillna(2.0).clip(1, 80) * 3.5)

    # 3. Try 'building:height'
    if "building:height" in gdf.columns:
        h = pd.to_numeric(gdf["building:height"], errors="coerce")
        valid = h.dropna()
        if len(valid) > 0:
            print("[morphology] building:height fallback used")
            return h.fillna(DEFAULT_HEIGHT).clip(1, 300)

    print(f"[morphology] No height data — using default {DEFAULT_HEIGHT}m for all buildings")
    return pd.Series(DEFAULT_HEIGHT, index=gdf.index)


def building_height_stats(buildings_gdf: gpd.GeoDataFrame) -> dict:
    if buildings_gdf is None or buildings_gdf.empty:
        return {}

    heights = _resolve_heights(buildings_gdf)
    heights = heights[heights <= 200]

    if heights.empty:
        return {}

    return {
        "median": float(heights.median()),
        "mean": float(heights.mean()),
        "std": float(heights.std()),
        "p25": float(heights.quantile(0.25)),
        "p75": float(heights.quantile(0.75)),
        "p95": float(heights.quantile(0.95)),
    }


def land_use_diversity(poi_df: pd.DataFrame) -> pd.DataFrame:
    if poi_df.empty or "category" not in poi_df.columns:
        return pd.DataFrame(columns=["h3_cell", "lat", "lon", "diversity_score"])

    df = poi_df.dropna(subset=["lat", "lon", "category"]).copy()

    df["h3_cell"] = df.apply(
        lambda row: h3.latlng_to_cell(row["lat"], row["lon"], 9), axis=1
    )

    def shannon_entropy(categories):
        counts = categories.value_counts()
        probs = counts / counts.sum()
        return -sum(p * log2(p) for p in probs if p > 0)

    grouped = df.groupby("h3_cell")["category"].apply(shannon_entropy).reset_index()
    grouped.columns = ["h3_cell", "diversity_score"]

    def cell_center(cell):
        lat, lon = h3.cell_to_latlng(cell)
        return pd.Series({"lat": lat, "lon": lon})

    centers = grouped["h3_cell"].apply(cell_center)
    result = pd.concat([grouped, centers], axis=1)

    return result[["h3_cell", "lat", "lon", "diversity_score"]]


def street_network_stats(graph) -> dict:
    if graph is None:
        return {}

    try:
        stats = ox.stats.basic_stats(graph)

        result = {
            "avg_degree": float(stats.get("avg_degree", 0)),
            "avg_street_length": float(stats.get("street_length_avg", 0)),
            "intersection_count": int(stats.get("intersection_count", 0)),
            "circuity_avg": float(stats.get("circuity_avg", 1.0)),
        }

        # Orientation entropy
        try:
            _G = None
            try:
                _G = ox.add_edge_bearings(graph)
            except AttributeError:
                _G = ox.bearing.add_edge_bearings(graph)
            edges_b = ox.graph_to_gdfs(_G, nodes=False)
            if "bearing" in edges_b.columns:
                angles = edges_b["bearing"].dropna() % 180
                hist, _ = np.histogram(angles, bins=36, range=(0, 180))
                total = hist.sum()
                probs = hist / total if total > 0 else hist.astype(float)
                entropy_val = -sum(p * log2(p) for p in probs if p > 0)
                result["orientation_entropy"] = float(entropy_val)
                result["orientation_histogram"] = hist.tolist()
            else:
                result["orientation_entropy"] = 0.0
                result["orientation_histogram"] = [0] * 36
        except Exception as e:
            print(f"[street_network_stats] Orientation error: {e}")
            result["orientation_entropy"] = 0.0
            result["orientation_histogram"] = [0] * 36

        # Dead-end ratio
        try:
            degree_vals = [d for _, d in graph.degree()]
            dead_ends = sum(1 for d in degree_vals if d == 1)
            result["dead_end_ratio"] = dead_ends / len(degree_vals) if degree_vals else 0.0
        except Exception:
            result["dead_end_ratio"] = 0.0

        # Street type composition
        try:
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
            if "highway" in edges_gdf.columns:
                _types = ["primary", "secondary", "tertiary", "residential"]

                def _normalize_highway(h):
                    if isinstance(h, list):
                        h = h[0] if h else ""
                    if not isinstance(h, str):
                        return "other"
                    for t in _types:
                        if t in h:
                            return t
                    return "other"

                edges_gdf["highway_type"] = edges_gdf["highway"].apply(_normalize_highway)
                type_counts = edges_gdf["highway_type"].value_counts()
                total = type_counts.sum()
                result["street_type_pct"] = (type_counts / total * 100).to_dict()
            else:
                result["street_type_pct"] = {}
        except Exception as e:
            print(f"[street_network_stats] Street type error: {e}")
            result["street_type_pct"] = {}

        # Block size distribution via polygonize
        try:
            from shapely.ops import polygonize
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
            edges_proj = edges_gdf.to_crs("EPSG:3857")
            from shapely.ops import unary_union
            line_union = unary_union(edges_proj.geometry.values)
            blocks = [b for b in polygonize(line_union) if b.area > 200]
            if blocks:
                block_areas = [b.area for b in blocks]
                result["block_size_median"] = float(np.median(block_areas))
                result["block_size_mean"] = float(np.mean(block_areas))
                result["block_count"] = len(block_areas)
            else:
                result["block_size_median"] = 0.0
                result["block_size_mean"] = 0.0
                result["block_count"] = 0
        except Exception as e:
            print(f"[street_network_stats] Block size error: {e}")
            result["block_size_median"] = 0.0
            result["block_size_mean"] = 0.0
            result["block_count"] = 0

        # Betweenness centrality for top 10% nodes
        try:
            import networkx as nx
            k_sample = min(100, len(graph.nodes))
            bc = nx.betweenness_centrality(graph, normalized=True, endpoints=False, k=k_sample)
            bc_vals = sorted(bc.values(), reverse=True)
            top10 = bc_vals[:max(1, len(bc_vals) // 10)]
            result["betweenness_centrality_top10_mean"] = float(np.mean(top10))
        except Exception as e:
            print(f"[street_network_stats] Betweenness error: {e}")
            result["betweenness_centrality_top10_mean"] = 0.0

        return result

    except Exception as e:
        print(f"[street_network_stats] Error: {e}")
        return {}


def compute_morphological_index(buildings_gdf: gpd.GeoDataFrame, graph) -> dict:
    empty_result = {
        "typologies": pd.Series(dtype=str),
        "typology_counts": pd.Series(dtype=int),
        "hex_metrics": gpd.GeoDataFrame(),
    }

    if buildings_gdf is None or buildings_gdf.empty:
        return empty_result

    buildings = buildings_gdf.copy().reset_index(drop=True)

    if buildings.crs is None:
        buildings = buildings.set_crs("EPSG:4326")

    # Filter to polygon geometries only
    poly_mask = buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    buildings = buildings[poly_mask].reset_index(drop=True)

    if buildings.empty:
        return empty_result

    buildings_m = buildings.to_crs("EPSG:3857")
    buildings_m["footprint_area"] = buildings_m.geometry.area
    buildings_m["perimeter"] = buildings_m.geometry.length
    buildings_m["ratio"] = buildings_m["perimeter"] / buildings_m["footprint_area"].clip(lower=1.0)

    # Use robust height resolver with fallbacks
    buildings_wgs_for_height = buildings_gdf.copy().reset_index(drop=True)
    buildings_wgs_for_height = buildings_wgs_for_height[
        buildings_wgs_for_height.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].reset_index(drop=True)
    heights = _resolve_heights(buildings_wgs_for_height)
    buildings_m["height_num"] = heights.values.clip(0.0, 300.0)

    # Building typology classification
    def _classify(row):
        area = row["footprint_area"]
        ratio = row["ratio"]
        height = row["height_num"]
        if area > 300 and height > 20:
            return "tower"
        if area > 500 and ratio < 0.15:
            return "block"
        if 80 <= area <= 300 and ratio < 0.18:
            return "terraced"
        if area < 200 and ratio > 0.2:
            return "detached"
        # Semi-detached: moderate area and ratio, between detached and terraced
        if 100 <= area <= 250 and 0.15 <= ratio <= 0.25:
            return "semi-detached"
        return "unknown"

    buildings_m["typology"] = buildings_m.apply(_classify, axis=1)

    # Centroids in WGS84 for H3 assignment
    centroids_wgs = buildings_m.to_crs("EPSG:4326").geometry.centroid
    buildings_m["lat"] = centroids_wgs.y
    buildings_m["lon"] = centroids_wgs.x
    buildings_m = buildings_m.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if buildings_m.empty:
        return empty_result

    buildings_m["h3_cell"] = buildings_m.apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 9), axis=1
    )

    # Estimate floors from height
    buildings_m["floors"] = (buildings_m["height_num"] / 3.5).clip(lower=1.0)

    # H3 resolution 9 average area in m²
    try:
        hex_area_m2 = h3.average_hexagon_area(9, unit="m^2")
    except Exception:
        try:
            hex_area_m2 = h3.hex_area(9, unit="m^2")
        except Exception:
            hex_area_m2 = 105332.51  # hardcoded fallback for H3 res 9

    grp = buildings_m.groupby("h3_cell")

    hex_far = grp.apply(
        lambda g: (g["footprint_area"] * g["floors"]).sum() / hex_area_m2
    ).reset_index()
    hex_far.columns = ["h3_cell", "FAR"]

    hex_bcr = grp.apply(
        lambda g: g["footprint_area"].sum() / hex_area_m2
    ).reset_index()
    hex_bcr.columns = ["h3_cell", "BCR"]

    hex_height = grp["height_num"].median().reset_index()
    hex_height.columns = ["h3_cell", "median_height"]

    # Street density per hex cell
    street_density_map = {}
    if graph is not None:
        try:
            edges = ox.graph_to_gdfs(graph, nodes=False)
            edges_wgs = edges.to_crs("EPSG:4326")
            edge_pts = edges_wgs.geometry.centroid
            edge_h3 = edge_pts.apply(
                lambda p: h3.latlng_to_cell(p.y, p.x, 9) if p is not None else None
            ).dropna()
            street_density_map = edge_h3.value_counts().to_dict()
        except Exception as e:
            print(f"[compute_morphological_index] Street density error: {e}")

    hex_metrics = hex_far.merge(hex_bcr, on="h3_cell").merge(hex_height, on="h3_cell")
    hex_metrics["street_density"] = hex_metrics["h3_cell"].map(street_density_map).fillna(0.0)

    # Min-max normalization helper
    def _minmax(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    hex_metrics["FAR_norm"] = _minmax(hex_metrics["FAR"])
    hex_metrics["BCR_norm"] = _minmax(hex_metrics["BCR"])
    hex_metrics["height_norm"] = _minmax(hex_metrics["median_height"])

    hex_metrics["CMI"] = (
        0.4 * hex_metrics["FAR_norm"]
        + 0.3 * hex_metrics["height_norm"]
        + 0.3 * hex_metrics["BCR_norm"]
    )

    # KMeans morphotype clustering
    _label_order = ["dense_urban", "historic_core", "mixed_mid", "industrial", "suburban"]
    features_cols = ["FAR", "BCR", "median_height", "street_density"]
    features = hex_metrics[features_cols].fillna(0.0)

    if len(features) >= 5:
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(features)
            hex_metrics["_cluster_id"] = cluster_ids

            # Sort clusters by mean FAR descending → assign labels
            cluster_far_mean = (
                hex_metrics.groupby("_cluster_id")["FAR"].mean()
                .sort_values(ascending=False)
            )
            remap = {
                int(old_id): _label_order[rank]
                for rank, old_id in enumerate(cluster_far_mean.index)
            }
            hex_metrics["cluster"] = hex_metrics["_cluster_id"].map(remap)
            hex_metrics = hex_metrics.drop(columns=["_cluster_id"])
        except Exception as e:
            print(f"[compute_morphological_index] KMeans error: {e}")
            hex_metrics["cluster"] = "suburban"
    else:
        hex_metrics["cluster"] = "suburban"

    # Add lat/lon centroid coordinates for each hex cell
    def _cell_center(cell):
        lat, lon = h3.cell_to_latlng(cell)
        return pd.Series({"lat": lat, "lon": lon})

    centers = hex_metrics["h3_cell"].apply(_cell_center)
    hex_metrics = pd.concat([hex_metrics, centers], axis=1)

    hex_gdf = gpd.GeoDataFrame(
        hex_metrics,
        geometry=gpd.points_from_xy(hex_metrics["lon"], hex_metrics["lat"]),
        crs="EPSG:4326",
    )

    return {
        "typologies": buildings_m["typology"],
        "typology_counts": buildings_m["typology"].value_counts(),
        "hex_metrics": hex_gdf,
    }


def landuse_composition(landuse_gdf: gpd.GeoDataFrame) -> dict:
    if landuse_gdf is None or landuse_gdf.empty or "landuse_class" not in landuse_gdf.columns:
        return {}

    lu = landuse_gdf.copy()
    if lu.crs is None:
        lu = lu.set_crs("EPSG:4326")
    lu_m = lu.to_crs("EPSG:3857")
    lu_m = lu_m[lu_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    lu_m["area"] = lu_m.geometry.area

    total_area = lu_m["area"].sum()
    if total_area == 0:
        return {}

    class_areas = lu_m.groupby("landuse_class")["area"].sum()
    percentages = (class_areas / total_area * 100).to_dict()

    green_classes = ["park", "forest"]
    green_area = sum(class_areas.get(c, 0.0) for c in green_classes)
    green_ratio = float(green_area / total_area)

    water_proximity_index = None
    water_geoms = lu_m[lu_m["landuse_class"] == "water"].geometry
    if not water_geoms.empty:
        try:
            water_union = water_geoms.unary_union
            non_water = lu_m[lu_m["landuse_class"] != "water"]
            if not non_water.empty:
                distances = non_water.geometry.centroid.distance(water_union)
                water_proximity_index = float(distances.mean())
        except Exception:
            pass

    return {
        "percentages": percentages,
        "green_space_ratio": green_ratio,
        "water_proximity_index": water_proximity_index,
        "total_area_m2": float(total_area),
    }


def _derive_hex_cells(buildings_gdf: gpd.GeoDataFrame):
    """Return DataFrame with h3_cell, lat, lon derived from building centroids."""
    if buildings_gdf is None or buildings_gdf.empty:
        return pd.DataFrame(columns=["h3_cell", "lat", "lon"])
    b = buildings_gdf.copy().reset_index(drop=True)
    if b.crs is None:
        b = b.set_crs("EPSG:4326")
    poly_mask = b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    b = b[poly_mask].reset_index(drop=True)
    if b.empty:
        return pd.DataFrame(columns=["h3_cell", "lat", "lon"])
    centroids = b.to_crs("EPSG:4326").geometry.centroid
    b["lat"] = centroids.y
    b["lon"] = centroids.x
    b = b.dropna(subset=["lat", "lon"])
    b["h3_cell"] = b.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 9), axis=1)
    unique = b.drop_duplicates("h3_cell")

    def _center(cell):
        lat, lon = h3.cell_to_latlng(cell)
        return pd.Series({"lat": lat, "lon": lon})

    centers = unique["h3_cell"].apply(_center)
    return pd.DataFrame({"h3_cell": unique["h3_cell"].values,
                         "lat": centers["lat"].values,
                         "lon": centers["lon"].values})


def transport_accessibility_index(
    buildings_gdf: gpd.GeoDataFrame,
    transit_stops_gdf: gpd.GeoDataFrame,
    graph,
) -> gpd.GeoDataFrame:
    hex_df = _derive_hex_cells(buildings_gdf)
    if hex_df.empty:
        return gpd.GeoDataFrame()

    hex_gdf = gpd.GeoDataFrame(
        hex_df,
        geometry=gpd.points_from_xy(hex_df["lon"], hex_df["lat"]),
        crs="EPSG:4326",
    )
    hex_m = hex_gdf.to_crs("EPSG:3857")
    hex_coords = np.array(list(zip(hex_m.geometry.x, hex_m.geometry.y)))
    n = len(hex_df)

    # Transit score: stops within 400 m (Euclidean in projected CRS)
    transit_scores = np.zeros(n)
    if transit_stops_gdf is not None and not transit_stops_gdf.empty:
        try:
            from scipy.spatial import cKDTree
            ts = transit_stops_gdf.copy()
            if ts.crs is None:
                ts = ts.set_crs("EPSG:4326")
            ts_m = ts.to_crs("EPSG:3857")
            ts_pts = ts_m[ts_m.geometry.geom_type == "Point"]
            if not ts_pts.empty:
                stop_coords = np.array(list(zip(ts_pts.geometry.x, ts_pts.geometry.y)))
                tree = cKDTree(stop_coords)
                counts = tree.query_ball_point(hex_coords, r=400)
                transit_scores = np.array([len(c) for c in counts], dtype=float)
        except Exception as e:
            print(f"[transport_accessibility_index] Transit score error: {e}")

    hex_df = hex_df.copy()
    hex_df["transit_score"] = transit_scores

    # Road hierarchy entropy + cycling coverage per hex cell
    road_entropy = np.zeros(n)
    cycling_coverage = np.zeros(n)
    cell_to_idx = {cell: i for i, cell in enumerate(hex_df["h3_cell"])}

    try:
        hex_area_m2 = h3.average_hexagon_area(9, unit="m^2")
    except Exception:
        try:
            hex_area_m2 = h3.hex_area(9, unit="m^2")
        except Exception:
            hex_area_m2 = 105332.51
    hex_area_km2 = hex_area_m2 / 1e6

    if graph is not None:
        try:
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False)
            edges_wgs = edges_gdf.to_crs("EPSG:4326")
            edge_centroids = edges_wgs.geometry.centroid
            _types = ["motorway", "primary", "secondary", "tertiary", "residential", "cycleway", "footway"]

            def _norm_hw(h):
                if isinstance(h, list):
                    h = h[0] if h else ""
                if not isinstance(h, str):
                    return "other"
                for t in _types:
                    if t in h:
                        return t
                return "other"

            hw_col = edges_gdf["highway"].apply(_norm_hw) if "highway" in edges_gdf.columns else pd.Series("other", index=edges_gdf.index)
            edge_h3 = edge_centroids.apply(lambda p: h3.latlng_to_cell(p.y, p.x, 9) if p else None)

            edges_meta = pd.DataFrame({
                "h3_cell": edge_h3.values,
                "highway_norm": hw_col.values,
            }).dropna(subset=["h3_cell"])

            for cell, grp in edges_meta.groupby("h3_cell"):
                if cell in cell_to_idx:
                    counts = grp["highway_norm"].value_counts()
                    probs = counts / counts.sum()
                    entropy = -sum(p * log2(p) for p in probs if p > 0)
                    road_entropy[cell_to_idx[cell]] = entropy

            # Cycling coverage
            cyc_mask = hw_col.isin(["cycleway"])
            edges_proj = edges_gdf.to_crs("EPSG:3857")
            cyc_proj = edges_proj[cyc_mask.values].copy()
            if not cyc_proj.empty:
                cyc_proj["length"] = cyc_proj.geometry.length
                cyc_wgs = cyc_proj.to_crs("EPSG:4326")
                cyc_h3 = cyc_wgs.geometry.centroid.apply(
                    lambda p: h3.latlng_to_cell(p.y, p.x, 9) if p else None
                )
                cyc_proj["h3_cell"] = cyc_h3.values
                cyc_proj = cyc_proj.dropna(subset=["h3_cell"])
                for cell, grp in cyc_proj.groupby("h3_cell"):
                    if cell in cell_to_idx:
                        cycling_coverage[cell_to_idx[cell]] = grp["length"].sum() / max(hex_area_km2, 1e-6)
        except Exception as e:
            print(f"[transport_accessibility_index] Graph metrics error: {e}")

    hex_df["road_hierarchy_mix"] = road_entropy
    hex_df["cycling_coverage"] = cycling_coverage

    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - mn) / (mx - mn)

    hex_df["transit_score_norm"] = _minmax(hex_df["transit_score"].values)
    hex_df["cycling_norm"] = _minmax(hex_df["cycling_coverage"].values)
    hex_df["road_entropy_norm"] = _minmax(hex_df["road_hierarchy_mix"].values)
    hex_df["transport_index"] = (
        0.5 * hex_df["transit_score_norm"]
        + 0.3 * hex_df["cycling_norm"]
        + 0.2 * hex_df["road_entropy_norm"]
    )

    return gpd.GeoDataFrame(
        hex_df,
        geometry=gpd.points_from_xy(hex_df["lon"], hex_df["lat"]),
        crs="EPSG:4326",
    )


def nature_accessibility_index(
    buildings_gdf: gpd.GeoDataFrame,
    green_spaces_gdf: gpd.GeoDataFrame,
    water_bodies_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    hex_df = _derive_hex_cells(buildings_gdf)
    if hex_df.empty:
        return gpd.GeoDataFrame()

    hex_gdf = gpd.GeoDataFrame(
        hex_df,
        geometry=gpd.points_from_xy(hex_df["lon"], hex_df["lat"]),
        crs="EPSG:4326",
    )
    hex_m = hex_gdf.to_crs("EPSG:3857")
    hex_coords = np.array(list(zip(hex_m.geometry.x, hex_m.geometry.y)))
    n = len(hex_df)

    green_ratio = np.zeros(n)
    park_access = np.zeros(n)

    if green_spaces_gdf is not None and not green_spaces_gdf.empty:
        try:
            gs = green_spaces_gdf.copy()
            if gs.crs is None:
                gs = gs.set_crs("EPSG:4326")
            gs_m = gs.to_crs("EPSG:3857")
            gs_m = gs_m[gs_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            if not gs_m.empty:
                sindex = gs_m.sindex
                buf_radius = 300
                for i, row in hex_m.iterrows():
                    buf = row.geometry.buffer(buf_radius)
                    cands = list(sindex.intersection(buf.bounds))
                    if cands:
                        nearby = gs_m.iloc[cands]
                        inter_area = nearby.geometry.intersection(buf).area.sum()
                        green_ratio[i] = inter_area / buf.area

                for i, row in hex_m.iterrows():
                    buf500 = row.geometry.buffer(500)
                    cands = list(sindex.intersection(buf500.bounds))
                    count = sum(1 for c in cands if gs_m.iloc[c].geometry.intersects(buf500))
                    park_access[i] = float(count)
        except Exception as e:
            print(f"[nature_accessibility_index] Green space error: {e}")

    water_proximity = np.zeros(n)
    flood_risk_tier = np.full(n, "low", dtype=object)

    if water_bodies_gdf is not None and not water_bodies_gdf.empty:
        try:
            from scipy.spatial import cKDTree
            wb = water_bodies_gdf.copy()
            if wb.crs is None:
                wb = wb.set_crs("EPSG:4326")
            wb_m = wb.to_crs("EPSG:3857")

            # Use centroids/representative points for proximity
            wb_pts = wb_m.geometry.representative_point()
            w_coords = np.array(list(zip(wb_pts.x, wb_pts.y)))
            if len(w_coords) > 0:
                tree = cKDTree(w_coords)
                dists, _ = tree.query(hex_coords)
                water_proximity = 1.0 / (1.0 + dists)

                # Flood risk: distance to waterway lines (rivers/streams)
                lines_mask = wb_m.geometry.geom_type.isin(["LineString", "MultiLineString"])
                wl_m = wb_m[lines_mask]
                if not wl_m.empty:
                    wl_pts = wl_m.geometry.representative_point()
                    wl_coords = np.array(list(zip(wl_pts.x, wl_pts.y)))
                    wl_tree = cKDTree(wl_coords)
                    wl_dists, _ = wl_tree.query(hex_coords)
                    flood_risk_tier = np.where(wl_dists <= 50, "high",
                                     np.where(wl_dists <= 200, "medium", "low"))
                else:
                    flood_risk_tier = np.where(dists <= 50, "high",
                                     np.where(dists <= 200, "medium", "low"))
        except Exception as e:
            print(f"[nature_accessibility_index] Water error: {e}")

    hex_df = hex_df.copy()
    hex_df["green_space_ratio"] = green_ratio
    hex_df["park_access_score"] = park_access
    hex_df["water_proximity"] = water_proximity
    hex_df["flood_risk_tier"] = flood_risk_tier

    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - mn) / (mx - mn)

    hex_df["nature_index"] = (
        0.4 * _minmax(green_ratio)
        + 0.4 * _minmax(park_access)
        + 0.2 * water_proximity
    )

    return gpd.GeoDataFrame(
        hex_df,
        geometry=gpd.points_from_xy(hex_df["lon"], hex_df["lat"]),
        crs="EPSG:4326",
    )


def landuse_transport_crossref(
    landuse_gdf: gpd.GeoDataFrame,
    hex_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if landuse_gdf is None or landuse_gdf.empty:
        return pd.DataFrame()
    if hex_gdf is None or hex_gdf.empty:
        return pd.DataFrame()
    if "h3_cell" not in hex_gdf.columns:
        return pd.DataFrame()
    if "landuse_class" not in landuse_gdf.columns:
        return pd.DataFrame()

    try:
        lu = landuse_gdf.copy()
        if lu.crs is None:
            lu = lu.set_crs("EPSG:4326")

        hx = hex_gdf.copy()
        if hx.crs is None:
            hx = hx.set_crs("EPSG:4326")

        hx_m = hx.to_crs("EPSG:3857")
        lu_m = lu.to_crs("EPSG:3857")

        lu_polys = lu_m[lu_m.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if lu_polys.empty:
            return pd.DataFrame()

        joined = gpd.sjoin(
            hx_m[["h3_cell", "geometry",
                  *(c for c in ["transport_index", "nature_index"] if c in hx_m.columns)]],
            lu_polys[["geometry", "landuse_class"]],
            how="left",
            predicate="within",
        )

        agg_dict = {"h3_cell": "count"}
        if "transport_index" in joined.columns:
            agg_dict["transport_index"] = "mean"
        if "nature_index" in joined.columns:
            agg_dict["nature_index"] = "mean"

        result = joined.groupby("landuse_class").agg(agg_dict).reset_index()
        result = result.rename(columns={
            "h3_cell": "hex_count",
            "transport_index": "mean_transport",
            "nature_index": "mean_nature",
        })

        hex_area_ha = 105332.51 / 10000
        result["area_ha"] = result["hex_count"] * hex_area_ha

        if "mean_transport" not in result.columns:
            result["mean_transport"] = 0.0
        if "mean_nature" not in result.columns:
            result["mean_nature"] = 0.0

        return result[result["hex_count"] > 0].reset_index(drop=True)

    except Exception as e:
        print(f"[landuse_transport_crossref] Error: {e}")
        return pd.DataFrame()


# ── Shared helper ─────────────────────────────────────────────────────────────

def _get_norm(df: pd.DataFrame, col: str, default: float = 0.5) -> pd.Series:
    """Return min-max normalised column from df, filling missing/NaN with default."""
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").fillna(default)
    else:
        s = pd.Series(default, index=df.index, dtype=float)
    mn, mx = float(s.min()), float(s.max())
    if mx == mn:
        return pd.Series(default, index=df.index, dtype=float)
    return (s - mn) / (mx - mn)


def _flood_series(df: pd.DataFrame) -> pd.Series:
    """Return per-row flood stress 0/0.5/1 from 'flood_risk_tier' or 'flood_stress'."""
    if "flood_stress" in df.columns:
        return pd.to_numeric(df["flood_stress"], errors="coerce").fillna(0.0)
    if "flood_risk_tier" in df.columns:
        return df["flood_risk_tier"].map({"high": 1.0, "medium": 0.5, "low": 0.0}).fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


# ── Module 1: Urban Stress Index ──────────────────────────────────────────────

def compute_urban_stress_index(merged_hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add urban_stress, stress_level, and 5 component columns to a copy of merged_hex."""
    if merged_hex_gdf is None or merged_hex_gdf.empty or "h3_cell" not in merged_hex_gdf.columns:
        return gpd.GeoDataFrame() if merged_hex_gdf is None or merged_hex_gdf.empty else merged_hex_gdf.copy()

    df = merged_hex_gdf.copy()

    density_stress  = _get_norm(df, "CMI")
    green_deficit   = 1.0 - _get_norm(df, "nature_index")
    transit_deficit = 1.0 - _get_norm(df, "transport_index")
    # Prefer terrain-derived flood risk if available
    if "terrain_flood_risk" in df.columns:
        flood_stress = pd.to_numeric(df["terrain_flood_risk"], errors="coerce").fillna(0.3)
    else:
        flood_stress = _flood_series(df)
    mono_stress     = 1.0 - _get_norm(df, "diversity_score")

    df["density_stress"]  = density_stress.values
    df["green_deficit"]   = green_deficit.values
    df["transit_deficit"] = transit_deficit.values
    df["flood_stress"]    = flood_stress.values
    df["mono_stress"]     = mono_stress.values

    urban_stress = (
        0.25 * density_stress
        + 0.25 * green_deficit
        + 0.25 * transit_deficit
        + 0.15 * flood_stress
        + 0.10 * mono_stress
    )
    df["urban_stress"] = urban_stress.clip(0.0, 1.0).values

    try:
        df["stress_level"] = pd.cut(
            df["urban_stress"],
            bins=[-np.inf, 0.3, 0.5, 0.7, np.inf],
            labels=["low", "moderate", "high", "critical"],
        ).astype(str)
    except Exception:
        df["stress_level"] = "moderate"

    return df


# ── Module 3: Urban Fabric Typology ───────────────────────────────────────────

_FABRIC_LABELS = {
    "compact_vibrant":   "Historic Mixed Core",
    "compact_mixed":     "Dense Residential",
    "compact_low_mix":   "Dense Mono",
    "compact_mono":      "Tower Block",
    "medium_vibrant":    "Active Mid-Rise",
    "medium_mixed":      "Typical Urban",
    "medium_low_mix":    "Mid-Rise Residential",
    "medium_mono":       "Office District",
    "low_vibrant":       "Active Low-Rise",
    "low_mixed":         "Mixed Suburb",
    "low_low_mix":       "Residential Suburb",
    "low_mono":          "Dormitory",
    "sprawl_vibrant":    "Commercial Strip",
    "sprawl_mixed":      "Peri-Urban Mixed",
    "sprawl_low_mix":    "Peri-Urban",
    "sprawl_mono":       "Fringe / Industrial",
}


def compute_fabric_typology(merged_hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify each hex into one of 16 urban fabric types from FAR × land-use mix."""
    if merged_hex_gdf is None or merged_hex_gdf.empty or "h3_cell" not in merged_hex_gdf.columns:
        return gpd.GeoDataFrame() if merged_hex_gdf is None or merged_hex_gdf.empty else merged_hex_gdf.copy()

    df = merged_hex_gdf.copy()
    far_norm       = _get_norm(df, "FAR").clip(0.0, 1.0)
    diversity_norm = _get_norm(df, "diversity_score").clip(0.0, 1.0)

    try:
        compactness = pd.cut(
            far_norm,
            bins=[0.0, 0.25, 0.5, 0.75, 1.001],
            labels=["sprawl", "low", "medium", "compact"],
            include_lowest=True,
        ).astype(str).replace("nan", "low")
    except Exception:
        compactness = pd.Series("low", index=df.index)

    try:
        mix = pd.cut(
            diversity_norm,
            bins=[0.0, 0.25, 0.5, 0.75, 1.001],
            labels=["mono", "low_mix", "mixed", "vibrant"],
            include_lowest=True,
        ).astype(str).replace("nan", "low_mix")
    except Exception:
        mix = pd.Series("low_mix", index=df.index)

    df["compactness"]    = compactness.values
    df["mix"]            = mix.values
    df["fabric_type"]    = df["compactness"] + "_" + df["mix"]
    df["planning_label"] = df["fabric_type"].map(_FABRIC_LABELS).fillna("Mixed Urban")

    return df


# ── Module 4: Temporal Vulnerability ─────────────────────────────────────────

def compute_temporal_vulnerability(merged_hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add vulnerability_score and vuln_class to a copy of merged_hex."""
    if merged_hex_gdf is None or merged_hex_gdf.empty or "h3_cell" not in merged_hex_gdf.columns:
        return gpd.GeoDataFrame() if merged_hex_gdf is None or merged_hex_gdf.empty else merged_hex_gdf.copy()

    df = merged_hex_gdf.copy()

    # Prefer terrain-derived flood risk if available
    if "terrain_flood_risk" in df.columns:
        flood_stress = pd.to_numeric(df["terrain_flood_risk"], errors="coerce").fillna(0.3)
    else:
        flood_stress = _flood_series(df)
    density_stress  = _get_norm(df, "CMI")
    isolation       = 1.0 - _get_norm(df, "transport_index")
    water_proximity = _get_norm(df, "water_proximity", default=0.0)

    vulnerability = (
        0.35 * flood_stress
        + 0.30 * density_stress
        + 0.20 * isolation
        + 0.15 * water_proximity
    ).clip(0.0, 1.0)

    df["vulnerability_score"] = vulnerability.values

    try:
        df["vuln_class"] = pd.cut(
            df["vulnerability_score"],
            bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
            labels=["resilient", "moderate", "high", "extreme"],
        ).astype(str)
    except Exception:
        df["vuln_class"] = "moderate"

    return df


# ── Module 5: Segregation Proxy ───────────────────────────────────────────────

def compute_segregation_proxy(merged_hex_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add segregation_score and seg_class to a copy of merged_hex."""
    if merged_hex_gdf is None or merged_hex_gdf.empty or "h3_cell" not in merged_hex_gdf.columns:
        return gpd.GeoDataFrame() if merged_hex_gdf is None or merged_hex_gdf.empty else merged_hex_gdf.copy()

    df = merged_hex_gdf.copy()

    diversity_norm  = _get_norm(df, "diversity_score")
    transport_norm  = _get_norm(df, "transport_index")
    far_norm        = _get_norm(df, "FAR")

    mono_function    = 1.0 - diversity_norm
    transit_isolation = 1.0 - transport_norm
    density_isolation = ((far_norm > 0.7) & (diversity_norm < 0.3)).astype(float)

    segregation = (
        0.40 * mono_function
        + 0.35 * transit_isolation
        + 0.25 * density_isolation
    ).clip(0.0, 1.0)

    df["segregation_score"] = segregation.values

    try:
        df["seg_class"] = pd.cut(
            df["segregation_score"],
            bins=[-np.inf, 0.25, 0.45, 0.65, np.inf],
            labels=["integrated", "mixed", "at_risk", "isolated"],
        ).astype(str)
    except Exception:
        df["seg_class"] = "mixed"

    return df


# ── Terrain helpers ───────────────────────────────────────────────────────────

def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalise a series, returning 0.5 when all values are equal."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def sample_terrain_for_hexes(hex_gdf: gpd.GeoDataFrame, terrain_data: dict) -> gpd.GeoDataFrame:
    """
    Sample Open-Meteo elevation grid at hex centroids.
    Derives slope and TWI from the grid using np.gradient; no rasterio or srtm needed.
    """
    if hex_gdf is None or hex_gdf.empty:
        return hex_gdf if hex_gdf is not None else gpd.GeoDataFrame()

    gdf = hex_gdf.copy()

    # Ensure lat/lon columns
    if "lat" not in gdf.columns or "lon" not in gdf.columns:
        if "h3_cell" in gdf.columns:
            centers = gdf["h3_cell"].apply(
                lambda c: pd.Series(dict(zip(("lat", "lon"), h3.cell_to_latlng(c))))
            )
            gdf["lat"] = centers["lat"].values
            gdf["lon"] = centers["lon"].values
        else:
            for col, val in [("elevation_m", np.nan), ("slope_deg", np.nan),
                             ("twi", np.nan), ("terrain_flood_risk", 0.5)]:
                gdf[col] = val
            return gdf

    elevation_grid = (terrain_data or {}).get("elevation_grid")
    grid_lats      = (terrain_data or {}).get("lats")
    grid_lons      = (terrain_data or {}).get("lons")

    n_hex = len(gdf)

    if elevation_grid is None or grid_lats is None:
        for col, val in [("elevation_m", np.nan), ("slope_deg", np.nan),
                         ("twi", np.nan), ("terrain_flood_risk", 0.5)]:
            gdf[col] = val
        return gdf

    # Fill NaN in grid
    elev_flat = np.array(elevation_grid, dtype=float)
    med_e = float(np.nanmedian(elev_flat[~np.isnan(elev_flat)])) if np.any(~np.isnan(elev_flat)) else 0.0
    elev_flat = np.where(np.isnan(elev_flat), med_e, elev_flat)

    # Reshape to 2D for gradient-based slope / TWI
    n_pts     = len(elev_flat)
    grid_side = int(round(n_pts ** 0.5))
    if grid_side * grid_side == n_pts:
        elev_2d = elev_flat.reshape(grid_side, grid_side)
        lat_2d  = np.array(grid_lats, dtype=float).reshape(grid_side, grid_side)
        lon_2d  = np.array(grid_lons, dtype=float).reshape(grid_side, grid_side)

        lat_center = float(lat_2d.mean())
        lat_sp_m = max(abs(lat_2d[-1, 0] - lat_2d[0, 0]) / max(grid_side - 1, 1) * 111320, 100.0)
        lon_sp_m = max(
            abs(lon_2d[0, -1] - lon_2d[0, 0]) / max(grid_side - 1, 1)
            * 111320 * abs(np.cos(np.radians(lat_center))),
            100.0,
        )

        dy, dx = np.gradient(elev_2d, lat_sp_m, lon_sp_m)
        slope_2d = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

        tan_s = np.tan(np.radians(slope_2d))
        tan_s[tan_s < 0.001] = 0.001
        cell_area = lat_sp_m * lon_sp_m
        twi_2d = np.clip(np.log(cell_area / tan_s), 0, 20)

        slope_flat = slope_2d.flatten()
        twi_flat   = twi_2d.flatten()
    else:
        slope_flat = np.full(n_pts, 5.0)
        twi_flat   = np.full(n_pts, 6.0)

    # Nearest-neighbour sampling with scipy cKDTree
    try:
        from scipy.spatial import cKDTree
        grid_coords = np.column_stack([np.array(grid_lats, dtype=float),
                                       np.array(grid_lons, dtype=float)])
        hex_coords  = np.column_stack([gdf["lat"].values.astype(float),
                                       gdf["lon"].values.astype(float)])
        _, idx = cKDTree(grid_coords).query(hex_coords)

        e_arr = elev_flat[idx]
        s_arr = slope_flat[idx]
        t_arr = twi_flat[idx]
    except Exception as ex:
        print(f"[sample_terrain_for_hexes] cKDTree sampling error: {ex}")
        e_arr = np.full(n_hex, med_e)
        s_arr = np.full(n_hex, 5.0)
        t_arr = np.full(n_hex, 6.0)

    gdf["elevation_m"] = e_arr
    gdf["slope_deg"]   = s_arr
    gdf["twi"]         = t_arr

    elev_risk  = 1.0 - normalize_series(pd.Series(e_arr, dtype=float)).values
    twi_risk   = normalize_series(pd.Series(t_arr, dtype=float)).values
    slope_risk = 1.0 - normalize_series(pd.Series(s_arr, dtype=float)).values
    gdf["terrain_flood_risk"] = np.clip(
        0.40 * elev_risk + 0.35 * twi_risk + 0.25 * slope_risk, 0.0, 1.0
    )
    return gdf


# ── Fallback hex-grid generators ─────────────────────────────────────────────

def create_hex_grid_from_bbox(bbox, resolution: int = 9) -> gpd.GeoDataFrame:
    """Create H3 hex grid covering bbox — used when building data is unavailable."""
    if bbox is None:
        return gpd.GeoDataFrame()
    min_lon, min_lat, max_lon, max_lat = bbox
    try:
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat], [max_lon, min_lat],
                [max_lon, max_lat], [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }
        cells = list(h3.geo_to_cells(polygon, res=resolution))
        if not cells:
            return gpd.GeoDataFrame()
        lats = [h3.cell_to_latlng(c)[0] for c in cells]
        lons = [h3.cell_to_latlng(c)[1] for c in cells]
        print(f"[hex_grid] bbox fallback: {len(cells)} cells at res={resolution}")
        return gpd.GeoDataFrame(
            {
                "h3_cell": cells, "lat": lats, "lon": lons,
                "FAR": 0.1, "BCR": 0.1, "CMI": 0.1,
                "median_height": 6.0, "street_density": 0.0,
                "cluster": "suburban",
            },
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326",
        )
    except Exception as e:
        print(f"[hex_grid] bbox fallback failed: {e}")
        return gpd.GeoDataFrame()


def create_hex_grid_from_pois(poi_df: pd.DataFrame, resolution: int = 9) -> gpd.GeoDataFrame:
    """Create H3 hex grid from POI centroids — always works when Overture data loads."""
    if poi_df is None or poi_df.empty:
        return gpd.GeoDataFrame()
    try:
        df = poi_df.dropna(subset=["lat", "lon"]).copy()
        df["h3_cell"] = df.apply(
            lambda r: h3.latlng_to_cell(r["lat"], r["lon"], resolution), axis=1
        )

        agg = df.groupby("h3_cell").agg(
            poi_count=("category", "count"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        ).reset_index()

        def _shannon(cats):
            p = cats.value_counts(normalize=True)
            return float(-(p * np.log(p + 1e-10)).sum())

        div = df.groupby("h3_cell")["category"].apply(_shannon).reset_index()
        div.columns = ["h3_cell", "diversity_score"]

        result = agg.merge(div, on="h3_cell", how="left")
        result["FAR"] = 0.1
        result["BCR"] = 0.1
        result["CMI"] = 0.1
        result["median_height"] = 6.0
        result["street_density"] = 0.0
        result["cluster"] = "suburban"
        print(f"[hex_grid] POI fallback: {len(result)} cells at res={resolution}")
        return gpd.GeoDataFrame(
            result,
            geometry=gpd.points_from_xy(result["lon"], result["lat"]),
            crs="EPSG:4326",
        )
    except Exception as e:
        print(f"[hex_grid] POI fallback failed: {e}")
        return gpd.GeoDataFrame()


# ── District Scoring ──────────────────────────────────────────────────────────

def compute_district_scores(merged_hex_gdf: gpd.GeoDataFrame, admin_boundaries: dict) -> pd.DataFrame:
    """
    For each district, compute mean scores and assign A/B/C/D grades.
    Returns DataFrame: district_name | transport_grade | nature_grade | … | overall_grade
    """
    grades = []
    for level in ['admin_9', 'admin_10']:
        gdf = admin_boundaries.get(level) if admin_boundaries else None
        if gdf is None or gdf.empty:
            continue

        hex_points = merged_hex_gdf.copy()
        if 'lon' not in hex_points.columns or 'lat' not in hex_points.columns:
            continue
        hex_points['geometry'] = gpd.points_from_xy(hex_points['lon'], hex_points['lat'])
        hex_points = gpd.GeoDataFrame(hex_points, crs='EPSG:4326')

        try:
            joined = gpd.sjoin(hex_points, gdf[['admin_name', 'geometry']], how='left', predicate='within')
        except Exception:
            continue

        def score_to_grade(score, higher_is_better=True):
            if not higher_is_better:
                score = 1 - score
            if score >= 0.75:
                return 'A'
            elif score >= 0.55:
                return 'B'
            elif score >= 0.35:
                return 'C'
            return 'D'

        for district in joined['admin_name'].dropna().unique():
            district_hexes = joined[joined['admin_name'] == district]
            if len(district_hexes) < 2:
                continue

            row = {'district_name': district, 'hex_count': len(district_hexes)}

            if 'transport_index' in district_hexes.columns:
                mean_t = float(district_hexes['transport_index'].mean())
                row['transport_score'] = round(mean_t, 3)
                row['transport_grade'] = score_to_grade(mean_t)

            if 'nature_index' in district_hexes.columns:
                mean_n = float(district_hexes['nature_index'].mean())
                row['nature_score'] = round(mean_n, 3)
                row['nature_grade'] = score_to_grade(mean_n)

            if 'CMI' in district_hexes.columns:
                mean_m = float(district_hexes['CMI'].mean())
                row['morphology_score'] = round(mean_m, 3)
                row['morphology_grade'] = score_to_grade(mean_m)

            if 'urban_stress' in district_hexes.columns:
                mean_s = float(district_hexes['urban_stress'].mean())
                row['stress_score'] = round(mean_s, 3)
                row['stress_grade'] = score_to_grade(mean_s, higher_is_better=False)

            if 'terrain_flood_risk' in district_hexes.columns:
                mean_f = float(district_hexes['terrain_flood_risk'].mean())
                row['flood_score'] = round(mean_f, 3)
                row['flood_grade'] = score_to_grade(mean_f, higher_is_better=False)

            score_vals = [v for k, v in row.items() if k.endswith('_score') and isinstance(v, float)]
            if score_vals:
                row['overall_grade'] = score_to_grade(sum(score_vals) / len(score_vals))

            grades.append(row)

    return pd.DataFrame(grades) if grades else pd.DataFrame()


# ── Urban Heat Island Proxy ───────────────────────────────────────────────────

def compute_heat_island_proxy(merged_hex_gdf: gpd.GeoDataFrame, climate_data: dict) -> gpd.GeoDataFrame:
    """Proxy UHI: areas with high BCR + low green_ratio = warmer."""
    if 'BCR' not in merged_hex_gdf.columns:
        return merged_hex_gdf

    df = merged_hex_gdf.copy()
    base_temp = climate_data.get('temp_max_avg', 20)
    bcr_norm = df['BCR'].fillna(0).clip(0, 1)
    green_ratio = (
        df['green_space_ratio'].fillna(0)
        if 'green_space_ratio' in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df['uhi_delta'] = (bcr_norm * 3.0) - (green_ratio * 2.0)
    df['estimated_temp'] = base_temp + df['uhi_delta']
    return df
