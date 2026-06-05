import duckdb
import osmnx as ox
import pandas as pd
import geopandas as gpd


def _fetch_overture_pois_duckdb(min_lon, min_lat, max_lon, max_lat) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-west-2';")

    query = f"""
    SELECT
        names.primary AS name,
        categories.primary AS category,
        ST_Y(ST_GeomFromWKB(geometry)) AS lat,
        ST_X(ST_GeomFromWKB(geometry)) AS lon
    FROM read_parquet(
        's3://overturemaps-us-west-2/release/2025-05-21.0/theme=places/type=place/*',
        hive_partitioning=1
    )
    WHERE
        bbox.minx BETWEEN {min_lon} AND {max_lon}
        AND bbox.miny BETWEEN {min_lat} AND {max_lat}
    LIMIT 5000
    """

    df = con.execute(query).df()
    con.close()
    return df


def _fetch_overture_pois_package(min_lon, min_lat, max_lon, max_lat) -> pd.DataFrame:
    import overturemaps
    reader = overturemaps.record_batch_reader("place", bbox=(min_lon, min_lat, max_lon, max_lat))
    df = reader.read_all().to_pandas()
    if "names" in df.columns:
        df["name"] = df["names"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )
    if "categories" in df.columns:
        df["category"] = df["categories"].apply(
            lambda x: x.get("primary") if isinstance(x, dict) else None
        )
    if "geometry" in df.columns:
        import shapely.wkb
        def _extract_coords(geom):
            try:
                point = shapely.wkb.loads(bytes(geom)) if isinstance(geom, (bytes, bytearray, memoryview)) else geom
                return point.y, point.x
            except Exception:
                return None, None
        df[["lat", "lon"]] = df["geometry"].apply(lambda g: pd.Series(_extract_coords(g)))
    return df


def fetch_overture_pois(city_name: str) -> pd.DataFrame:
    try:
        geocode = ox.geocode_to_gdf(city_name)
        bounds = geocode.total_bounds  # minx, miny, maxx, maxy
        min_lon, min_lat, max_lon, max_lat = bounds

        try:
            df = _fetch_overture_pois_duckdb(min_lon, min_lat, max_lon, max_lat)
        except Exception as e:
            print(f"[fetch_overture_pois] DuckDB S3 failed, retrying with overturemaps package: {e}")
            df = _fetch_overture_pois_package(min_lon, min_lat, max_lon, max_lat)

        df = df.dropna(subset=["lat", "lon"])
        return df[["name", "category", "lat", "lon"]]

    except Exception as e:
        print(f"[fetch_overture_pois] Error: {e}")
        return pd.DataFrame(columns=["name", "category", "lat", "lon"])


def fetch_osm_data(city_name: str) -> tuple:
    try:
        graph = ox.graph_from_place(city_name, network_type="walk")
    except Exception as e:
        print(f"[fetch_osm_data] Graph error: {e}")
        graph = None

    try:
        buildings = ox.features_from_place(city_name, tags={"building": True})
        keep_cols = ["geometry"]
        if "height" in buildings.columns:
            keep_cols.append("height")
        else:
            buildings["height"] = None
            keep_cols.append("height")
        buildings = buildings[keep_cols].copy()
    except Exception as e:
        print(f"[fetch_osm_data] Buildings error: {e}")
        buildings = gpd.GeoDataFrame(columns=["geometry", "height"])

    return graph, buildings


def fetch_osm_landuse(city_name: str) -> gpd.GeoDataFrame:
    gdfs = []
    poly_types = {"Polygon", "MultiPolygon"}

    _landuse_classes = {
        "residential", "commercial", "industrial", "retail",
        "park", "forest", "water",
    }

    try:
        lu_gdf = ox.features_from_place(city_name, tags={"landuse": True})
        if not lu_gdf.empty and "landuse" in lu_gdf.columns:
            lu_gdf = lu_gdf[lu_gdf.geometry.geom_type.isin(poly_types)].copy()
            lu_gdf["landuse_class"] = lu_gdf["landuse"].where(
                lu_gdf["landuse"].isin(_landuse_classes), "other"
            )
            gdfs.append(lu_gdf[["geometry", "landuse_class"]])
    except Exception as e:
        print(f"[fetch_osm_landuse] landuse error: {e}")

    _natural_map = {"water": "water", "wood": "forest", "scrub": "forest"}
    try:
        nat_gdf = ox.features_from_place(city_name, tags={"natural": list(_natural_map.keys())})
        if not nat_gdf.empty and "natural" in nat_gdf.columns:
            nat_gdf = nat_gdf[nat_gdf.geometry.geom_type.isin(poly_types)].copy()
            nat_gdf["landuse_class"] = nat_gdf["natural"].map(_natural_map)
            nat_gdf = nat_gdf.dropna(subset=["landuse_class"])
            gdfs.append(nat_gdf[["geometry", "landuse_class"]])
    except Exception as e:
        print(f"[fetch_osm_landuse] natural error: {e}")

    _leisure_map = {"park": "park", "garden": "park", "pitch": "park"}
    try:
        lei_gdf = ox.features_from_place(city_name, tags={"leisure": list(_leisure_map.keys())})
        if not lei_gdf.empty and "leisure" in lei_gdf.columns:
            lei_gdf = lei_gdf[lei_gdf.geometry.geom_type.isin(poly_types)].copy()
            lei_gdf["landuse_class"] = lei_gdf["leisure"].map(_leisure_map)
            lei_gdf = lei_gdf.dropna(subset=["landuse_class"])
            gdfs.append(lei_gdf[["geometry", "landuse_class"]])
    except Exception as e:
        print(f"[fetch_osm_landuse] leisure error: {e}")

    if not gdfs:
        return gpd.GeoDataFrame(columns=["geometry", "landuse_class"], crs="EPSG:4326")

    merged = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
    )
    return merged
