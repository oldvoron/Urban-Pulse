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


def building_height_stats(buildings_gdf: gpd.GeoDataFrame) -> dict:
    if buildings_gdf.empty or "height" not in buildings_gdf.columns:
        return {}

    heights = pd.to_numeric(buildings_gdf["height"], errors="coerce")
    heights = heights.dropna()
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

        avg_degree = stats.get("avg_degree", 0)
        avg_street_length = stats.get("street_length_avg", 0)
        intersection_count = stats.get("intersection_count", 0)
        circuity_avg = stats.get("circuity_avg", 1.0)

        return {
            "avg_degree": float(avg_degree),
            "avg_street_length": float(avg_street_length),
            "intersection_count": int(intersection_count),
            "circuity_avg": float(circuity_avg),
        }
    except Exception as e:
        print(f"[street_network_stats] Error: {e}")
        return {}
