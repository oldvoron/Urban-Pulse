import osmnx as ox
import pandas as pd
import geopandas as gpd
import time


def configure_osmnx():
    """Configure osmnx with retry-friendly settings."""
    ox.settings.timeout = 180
    ox.settings.max_query_area_size = 25_000_000_000
    ox.settings.overpass_rate_limit = False
    ox.settings.overpass_url = "https://overpass-api.de/api/interpreter"


configure_osmnx()


def osmnx_fetch_with_retry(fetch_func, max_retries=3, delay=3):
    """Retry osmnx fetch with exponential backoff on connection errors (max ~9s total wait)."""
    for attempt in range(max_retries):
        try:
            result = fetch_func()
            time.sleep(2)
            return result
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ('Connection refused', 'Max retries', 'timeout', 'Timeout')):
                if attempt < max_retries - 1:
                    wait = delay * (attempt + 1)
                    print(f"[OSM] Connection failed, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
            raise e
    return None


# ── Overture Maps ─────────────────────────────────────────────────────────────

def fetch_overture_pois(city_name: str, bbox=None, overture_limit: int = 15000) -> pd.DataFrame:
    """
    Fetch POIs from Overture Maps using the overturemaps Python package (primary)
    with DuckDB S3 as fallback. Returns DataFrame with [name, category, lat, lon].
    """
    if bbox is None:
        try:
            b = ox.geocode_to_gdf(city_name).total_bounds  # minx, miny, maxx, maxy
            bbox = (b[0], b[1], b[2], b[3])
        except Exception as e:
            print(f"[overture] geocode failed: {e}")
            return pd.DataFrame(columns=["name", "category", "lat", "lon"])

    min_lon, min_lat, max_lon, max_lat = bbox

    # Primary: overturemaps package
    try:
        import overturemaps
        reader = overturemaps.record_batch_reader("place", bbox=(min_lon, min_lat, max_lon, max_lat))
        table = reader.read_all()
        df = table.to_pandas()

        if df.empty:
            raise ValueError("empty result")

        if "names" in df.columns:
            df["name"] = df["names"].apply(
                lambda x: x.get("primary", "") if isinstance(x, dict) else ""
            )
        else:
            df["name"] = ""

        if "categories" in df.columns:
            df["category"] = df["categories"].apply(
                lambda x: x.get("primary", "unknown") if isinstance(x, dict) else "unknown"
            )
        else:
            df["category"] = "unknown"

        if "geometry" in df.columns:
            def _xy(g):
                if hasattr(g, "x"):
                    return g.x, g.y
                if isinstance(g, dict):
                    coords = g.get("coordinates", [None, None])
                    return coords[0], coords[1]
                try:
                    import shapely.wkb
                    pt = shapely.wkb.loads(bytes(g)) if isinstance(g, (bytes, bytearray, memoryview)) else g
                    return pt.x, pt.y
                except Exception:
                    return None, None

            df[["lon", "lat"]] = df["geometry"].apply(lambda g: pd.Series(_xy(g)))

        df = df.dropna(subset=["lat", "lon"])
        if len(df) > overture_limit:
            df = df.head(overture_limit)
        print(f"[overture] package: {len(df)} places")
        return df[["name", "category", "lat", "lon"]]

    except Exception as e:
        print(f"[overture] package failed: {e}")

    # Fallback: DuckDB S3 (best-effort, path may be stale)
    try:
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")
        q = f"""
            SELECT
                names.primary AS name,
                categories.primary AS category,
                ST_Y(ST_GeomFromWKB(geometry)) AS lat,
                ST_X(ST_GeomFromWKB(geometry)) AS lon
            FROM read_parquet(
                's3://overturemaps-us-west-2/release/2025-05-21.0/theme=places/type=place/*',
                hive_partitioning=1
            )
            WHERE bbox.minx BETWEEN {min_lon} AND {max_lon}
              AND bbox.miny BETWEEN {min_lat} AND {max_lat}
            LIMIT {overture_limit}
        """
        df = con.execute(q).df()
        con.close()
        df = df.dropna(subset=["lat", "lon"])
        print(f"[overture] DuckDB S3: {len(df)} places")
        return df[["name", "category", "lat", "lon"]]
    except Exception as e:
        print(f"[overture] DuckDB S3 fallback failed: {e}")

    return pd.DataFrame(columns=["name", "category", "lat", "lon"])


# ── OSM helpers ───────────────────────────────────────────────────────────────
# osmnx 2.x: bbox = (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat)

def _osm_features(city_name: str, bbox, tags: dict) -> gpd.GeoDataFrame:
    """Call features_from_bbox (when bbox given) or features_from_place, with retry."""
    if bbox is not None:
        result = osmnx_fetch_with_retry(lambda: ox.features_from_bbox(bbox=bbox, tags=tags))
    else:
        result = osmnx_fetch_with_retry(lambda: ox.features_from_place(city_name, tags=tags))
    return result if result is not None else gpd.GeoDataFrame()


# ── Street network & buildings ────────────────────────────────────────────────

def fetch_osm_data(city_name: str, bbox=None, network_dist: int = 5000) -> dict:
    """Return {'graph': G|None, 'buildings': GeoDataFrame|None}."""
    ox.settings.timeout = 180
    ox.settings.max_query_area_size = 25_000_000_000

    result = {"graph": None, "buildings": None}

    # Street network
    try:
        if bbox is not None:
            G = osmnx_fetch_with_retry(lambda: ox.graph_from_bbox(bbox=bbox, network_type="walk"))
        else:
            G = osmnx_fetch_with_retry(lambda: ox.graph_from_place(city_name, network_type="walk"))
        if G is None:
            raise ValueError("graph fetch returned None after retries")
        result["graph"] = G
        print(f"[OSM] Street network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    except Exception as e:
        print(f"[OSM] Street network failed: {e}")
        try:
            loc = ox.geocode(city_name)
            G = osmnx_fetch_with_retry(lambda: ox.graph_from_point(loc, dist=network_dist, network_type="walk"))
            if G is not None:
                result["graph"] = G
                print(f"[OSM] Fallback point network: {len(G.nodes)} nodes")
        except Exception as e2:
            print(f"[OSM] Fallback network failed: {e2}")

    # Buildings — try progressively relaxed tag sets
    for tags in [
        {"building": True},
        {"building": ["yes", "house", "residential", "apartments",
                      "commercial", "retail", "office", "industrial"]},
        {"building": "yes"},
    ]:
        try:
            gdf = _osm_features(city_name, bbox, tags)
            if not gdf.empty:
                result["buildings"] = gdf
                print(f"[OSM] {len(gdf)} buildings (tags: {list(tags.keys())})")
                break
        except Exception as e:
            print(f"[OSM] Building attempt failed ({list(tags.keys())}): {e}")

    return result


# ── Transport data ────────────────────────────────────────────────────────────

def fetch_transport_data(city_name: str, bbox=None,
                         road_tags: list = None) -> dict:
    ox.settings.timeout = 180
    if road_tags is None:
        road_tags = ['primary', 'secondary', 'tertiary']
    result = {
        "roads": gpd.GeoDataFrame(),
        "transit_stops": gpd.GeoDataFrame(),
        "cycling": gpd.GeoDataFrame(),
    }

    # Roads — filtered by config road_tags to control data volume
    try:
        roads = _osm_features(city_name, bbox,
                              {"highway": road_tags})
        roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        if "highway" not in roads.columns:
            roads["highway"] = "other"
        result["roads"] = roads[["geometry", "highway"]].copy()
        print(f"[transport] {len(result['roads'])} road segments")
    except Exception as e:
        print(f"[transport] Roads error: {e}")

    # Transit stops — each tag set independently
    transit_gdfs = []
    for tags in [
        {"highway": "bus_stop"},
        {"railway": ["station", "halt", "tram_stop", "subway_entrance"]},
        {"public_transport": "stop_position"},
    ]:
        try:
            gdf = _osm_features(city_name, bbox, tags)
            if not gdf.empty:
                pts = gdf[gdf.geometry.geom_type == "Point"][["geometry"]].copy()
                if not pts.empty:
                    transit_gdfs.append(pts)
        except Exception:
            pass
    if transit_gdfs:
        stops = gpd.GeoDataFrame(pd.concat(transit_gdfs, ignore_index=True), crs="EPSG:4326")
        stops = stops.drop_duplicates(subset=["geometry"])
        stops["lat"] = stops.geometry.y
        stops["lon"] = stops.geometry.x
        result["transit_stops"] = stops
        print(f"[transport] {len(stops)} transit stops")

    # Cycling infrastructure
    cycling_gdfs = []
    for tags in [{"highway": "cycleway"}, {"bicycle": "designated"}]:
        try:
            gdf = _osm_features(city_name, bbox, tags)
            if not gdf.empty:
                lines = gdf[gdf.geometry.geom_type.isin(
                    ["LineString", "MultiLineString"])][["geometry"]].copy()
                if not lines.empty:
                    cycling_gdfs.append(lines)
        except Exception:
            pass
    if cycling_gdfs:
        result["cycling"] = gpd.GeoDataFrame(
            pd.concat(cycling_gdfs, ignore_index=True), crs="EPSG:4326"
        )

    return result


# ── Nature & green space ──────────────────────────────────────────────────────

def fetch_nature_data(city_name: str, bbox=None) -> dict:
    ox.settings.timeout = 180
    result = {
        "green_spaces": gpd.GeoDataFrame(),
        "water_bodies": gpd.GeoDataFrame(),
        "flood_risk_proxy": gpd.GeoDataFrame(),
    }
    poly_types = {"Polygon", "MultiPolygon"}

    # Green spaces
    green_gdfs = []
    for tags in [
        {"leisure": ["park", "garden", "nature_reserve", "pitch", "playground"]},
        {"landuse": ["forest", "grass", "meadow", "recreation_ground", "village_green"]},
        {"natural": ["wood", "scrub", "heath", "grassland"]},
    ]:
        try:
            gdf = _osm_features(city_name, bbox, tags)
            if not gdf.empty:
                polys = gdf[gdf.geometry.geom_type.isin(poly_types)][["geometry"]].copy()
                if not polys.empty:
                    green_gdfs.append(polys)
        except Exception:
            pass
    if green_gdfs:
        result["green_spaces"] = gpd.GeoDataFrame(
            pd.concat(green_gdfs, ignore_index=True), crs="EPSG:4326"
        )
        print(f"[nature] {len(result['green_spaces'])} green space polygons")

    # Water bodies
    water_gdfs = []
    for tags in [
        {"natural": "water"},
        {"waterway": ["river", "stream", "canal"]},
        {"landuse": "reservoir"},
    ]:
        try:
            gdf = _osm_features(city_name, bbox, tags)
            if not gdf.empty:
                water_gdfs.append(gdf[["geometry"]].copy())
        except Exception:
            pass
    if water_gdfs:
        result["water_bodies"] = gpd.GeoDataFrame(
            pd.concat(water_gdfs, ignore_index=True), crs="EPSG:4326"
        )

    # Flood risk proxy from waterways
    try:
        waterways = _osm_features(city_name, bbox, {"waterway": ["river", "stream"]})
        if not waterways.empty:
            wl = waterways[waterways.geometry.geom_type.isin(
                ["LineString", "MultiLineString"])].copy()
            if not wl.empty:
                wl_m = wl.to_crs("EPSG:3857")
                union_line = wl_m.geometry.unary_union
                flood_rows = []
                for tier, dist in [("high", 50), ("medium", 100), ("low", 200)]:
                    buf = union_line.buffer(dist)
                    flood_rows.append({
                        "geometry": gpd.GeoSeries([buf], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0],
                        "flood_tier": tier,
                        "buffer_m": dist,
                    })
                result["flood_risk_proxy"] = gpd.GeoDataFrame(flood_rows, crs="EPSG:4326")
    except Exception as e:
        print(f"[nature] Flood risk proxy error: {e}")

    return result


# ── Land use ──────────────────────────────────────────────────────────────────

def fetch_osm_landuse(city_name: str, bbox=None) -> gpd.GeoDataFrame:
    ox.settings.timeout = 180
    poly_types = {"Polygon", "MultiPolygon"}
    gdfs = []

    _landuse_classes = {
        "residential", "commercial", "industrial", "retail",
        "park", "forest", "water", "recreation_ground", "meadow", "farmland",
    }

    # Broad landuse sweep
    for tags in [{"landuse": list(_landuse_classes)}, {"landuse": True}]:
        try:
            lu = _osm_features(city_name, bbox, tags)
            if not lu.empty and "landuse" in lu.columns:
                lu = lu[lu.geometry.geom_type.isin(poly_types)].copy()
                lu["landuse_class"] = lu["landuse"].where(
                    lu["landuse"].isin(_landuse_classes), "other"
                )
                gdfs.append(lu[["geometry", "landuse_class"]])
                print(f"[landuse] {len(lu)} landuse polygons")
                break
        except Exception as e:
            print(f"[landuse] landuse error: {e}")

    # Leisure
    _leisure_map = {"park": "park", "garden": "park", "pitch": "park",
                    "playground": "park", "nature_reserve": "forest"}
    try:
        lei = _osm_features(city_name, bbox, {"leisure": list(_leisure_map.keys())})
        if not lei.empty and "leisure" in lei.columns:
            lei = lei[lei.geometry.geom_type.isin(poly_types)].copy()
            lei["landuse_class"] = lei["leisure"].map(_leisure_map).fillna("park")
            gdfs.append(lei[["geometry", "landuse_class"]])
    except Exception as e:
        print(f"[landuse] leisure error: {e}")

    # Natural
    _natural_map = {"water": "water", "wood": "forest", "scrub": "forest",
                    "wetland": "forest", "grassland": "park"}
    try:
        nat = _osm_features(city_name, bbox, {"natural": list(_natural_map.keys())})
        if not nat.empty and "natural" in nat.columns:
            nat = nat[nat.geometry.geom_type.isin(poly_types)].copy()
            nat["landuse_class"] = nat["natural"].map(_natural_map).fillna("other")
            gdfs.append(nat[["geometry", "landuse_class"]])
    except Exception as e:
        print(f"[landuse] natural error: {e}")

    if not gdfs:
        return gpd.GeoDataFrame(columns=["geometry", "landuse_class"], crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")


# ── Terrain (elevation) ───────────────────────────────────────────────────────

def fetch_terrain_data(city_name: str, bbox: tuple, terrain_grid: int = 20) -> dict:
    """
    Fetch elevation from Open-Meteo over a terrain_grid×terrain_grid grid within bbox.
    bbox = (min_lon, min_lat, max_lon, max_lat)
    """
    import requests
    import numpy as np

    if bbox is None or len(bbox) != 4:
        return {"elevation_grid": None}

    min_lon, min_lat, max_lon, max_lat = bbox
    lat_steps = np.linspace(min_lat, max_lat, terrain_grid)
    lon_steps = np.linspace(min_lon, max_lon, terrain_grid)
    grid_lats, grid_lons = np.meshgrid(lat_steps, lon_steps)
    grid_lats = grid_lats.flatten()
    grid_lons = grid_lons.flatten()

    all_elevations = []
    for i in range(0, len(grid_lats), 100):
        batch_lats = grid_lats[i:i + 100]
        batch_lons = grid_lons[i:i + 100]
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/elevation",
                params={
                    "latitude":  ",".join(map(str, batch_lats.round(5))),
                    "longitude": ",".join(map(str, batch_lons.round(5))),
                },
                timeout=15,
            )
            all_elevations.extend(r.json().get("elevation", [None] * len(batch_lats)))
        except Exception as e:
            print(f"[terrain] batch {i} failed: {e}")
            all_elevations.extend([None] * len(batch_lats))

    elevs = np.array([e if e is not None else np.nan for e in all_elevations], dtype=float)
    if (~np.isnan(elevs)).sum() < 10:
        return {"elevation_grid": None}

    return {
        "elevation_grid": elevs,
        "lats": grid_lats,
        "lons": grid_lons,
        "elev_min": float(np.nanmin(elevs)),
        "elev_max": float(np.nanmax(elevs)),
        "elev_mean": float(np.nanmean(elevs)),
    }


# ── Admin boundaries ──────────────────────────────────────────────────────────

def get_city_polygon(admin_boundaries: dict):
    """Return a single Shapely polygon/multipolygon for the city boundary, or None."""
    if admin_boundaries is None:
        return None
    city_gdf = admin_boundaries.get("city")
    if city_gdf is None or city_gdf.empty:
        return None
    try:
        return city_gdf.union_all()
    except AttributeError:
        # geopandas < 1.0 used unary_union
        return city_gdf.unary_union


def fetch_admin_boundaries(city_name: str) -> dict:
    """Fetch city + district boundaries via OSM admin_level tags."""
    result = {}
    try:
        result["city"] = osmnx_fetch_with_retry(lambda: ox.geocode_to_gdf(city_name))
    except Exception:
        result["city"] = None

    for admin_level in [9, 10]:
        try:
            gdf = osmnx_fetch_with_retry(
                lambda al=admin_level: ox.features_from_place(
                    city_name,
                    tags={"boundary": "administrative", "admin_level": str(al)},
                )
            )
            if gdf is None:
                result[f"admin_{admin_level}"] = None
                continue
            if not gdf.empty:
                gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
                if not gdf.empty:
                    name_col = "name" if "name" in gdf.columns else gdf.columns[0]
                    gdf["admin_name"] = gdf[name_col].fillna(f"Zone_{admin_level}")
                    result[f"admin_{admin_level}"] = gdf[["admin_name", "geometry"]].copy()
                    continue
            result[f"admin_{admin_level}"] = None
        except Exception:
            result[f"admin_{admin_level}"] = None

    district_names = []
    for level in ["admin_9", "admin_10"]:
        lvl = result.get(level)
        if lvl is not None and not lvl.empty:
            district_names.extend(lvl["admin_name"].tolist())
    result["district_names"] = list(set(district_names))
    result["has_districts"] = len(district_names) > 0
    return result


# ── Climate ───────────────────────────────────────────────────────────────────

def fetch_climate_data(city_center_lat: float, city_center_lon: float) -> dict:
    """7-day forecast from Open-Meteo (no API key)."""
    import requests
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": city_center_lat,
                "longitude": city_center_lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                         "et0_fao_evapotranspiration",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=10,
        )
        daily = r.json().get("daily", {})
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        return {
            "temp_max_avg": sum(tmax) / max(len(tmax), 1),
            "temp_min_avg": sum(tmin) / max(len(tmin), 1),
            "precip_7day": sum(daily.get("precipitation_sum", [])),
            "raw_daily": daily,
            "source": "Open-Meteo 7-day forecast",
        }
    except Exception as e:
        return {"error": str(e)}


# ── Population (best-effort) ──────────────────────────────────────────────────

def fetch_population_grid(bbox) -> dict:
    """Best-effort WorldPop estimate. Returns {} on any failure."""
    import requests
    if bbox is None:
        return {}
    min_lon, min_lat, max_lon, max_lat = bbox
    try:
        r = requests.get(
            "https://api.worldpop.org/v1/wopr/pointestimate",
            params={
                "iso3": "AUTO", "ver": "1",
                "lat": (min_lat + max_lat) / 2,
                "lon": (min_lon + max_lon) / 2,
            },
            timeout=10,
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}
