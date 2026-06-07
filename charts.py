import math

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import Polygon

MORPHOTYPE_COLORS = {
    "historic_core": "#8B1A1A",
    "dense_urban":   "#D4691E",
    "mixed_mid":     "#DAA520",
    "suburban":      "#2E8B57",
    "industrial":    "#4682B4",
}

_LANDUSE_COLORS = {
    "residential": "#E8D5B7", "commercial": "#FFB347",
    "industrial":  "#A9A9A9", "retail":     "#FFA07A",
    "park":        "#2E8B57", "forest":     "#228B22",
    "water":       "#4169E1", "other":      "#D3D3D3",
}

_EMPTY_FIG_HEIGHT = 450

# Shared font settings applied to every chart
_FONT       = dict(family="Arial, sans-serif", size=13)
_TITLE_FONT = dict(size=16, family="Arial, sans-serif")


def _empty(title: str, height: int = _EMPTY_FIG_HEIGHT) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_white", height=height,
                      font=_FONT, title_font=_TITLE_FONT)
    return fig


# ── H3 choropleth helpers ─────────────────────────────────────────────────────

def _compute_zoom(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
    lat_range = max(abs(lat_max - lat_min), 0.001)
    lon_range = max(abs(lon_max - lon_min), 0.001)
    zoom = math.log2(360.0 / max(lat_range, lon_range)) - 1.0
    return float(max(2.0, min(16.0, zoom)))


def compute_zoom_level(gdf) -> float:
    """Step-based zoom from a GDF's total_bounds — consistent across all maps."""
    try:
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        lat_range = max(bounds[3] - bounds[1], 0.001)
        lon_range = max(bounds[2] - bounds[0], 0.001)
        max_range = max(lat_range, lon_range)
        if max_range > 1.0:   return 9.0
        elif max_range > 0.5: return 10.0
        elif max_range > 0.2: return 11.0
        elif max_range > 0.1: return 12.0
        else:                 return 13.0
    except Exception:
        return 11.0


def hex_to_polygon_gdf(hex_df, city_polygon=None) -> gpd.GeoDataFrame:
    """Convert H3 cell IDs in any DataFrame to Shapely Polygon geometries."""
    if hex_df is None or (hasattr(hex_df, "empty") and hex_df.empty):
        return gpd.GeoDataFrame()
    if "h3_cell" not in hex_df.columns:
        return gpd.GeoDataFrame()
    rows = []
    for _, row in hex_df.iterrows():
        try:
            boundary = h3.cell_to_boundary(str(row["h3_cell"]))  # [(lat, lon), …]
            poly = Polygon([(lon, lat) for lat, lon in boundary])
            if poly.is_valid:
                d = {k: v for k, v in row.to_dict().items() if k != "geometry"}
                d["geometry"] = poly
                rows.append(d)
        except Exception:
            pass
    if not rows:
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    if city_polygon is not None:
        try:
            city_gdf = gpd.GeoDataFrame(geometry=[city_polygon], crs="EPSG:4326")
            gdf = gpd.clip(gdf, city_gdf)
        except Exception:
            pass
    return gdf


def _choropleth_hex(
    hex_df,
    color_col: str,
    title: str,
    colorscale=None,
    color_discrete_map: dict = None,
    mapbox_style: str = "carto-positron",
    hover_cols: list = None,
    height: int = 450,
    city_polygon=None,
) -> go.Figure:
    """Build a choropleth_mapbox figure from any DataFrame with an h3_cell column."""
    if hex_df is None or (hasattr(hex_df, "empty") and hex_df.empty):
        return _empty(f"No data for: {title}", height)
    if "h3_cell" not in hex_df.columns:
        return _empty(f"Missing h3_cell column for: {title}", height)
    if color_col not in hex_df.columns:
        return _empty(f"Missing column '{color_col}' for: {title}", height)

    poly_gdf = hex_to_polygon_gdf(hex_df, city_polygon=city_polygon)
    if poly_gdf.empty:
        return _empty(f"Hex polygon conversion failed for: {title}", height)

    poly_gdf = poly_gdf.copy()
    poly_gdf["geojson_id"] = poly_gdf["h3_cell"].astype(str)

    if color_discrete_map is not None:
        poly_gdf[color_col] = poly_gdf[color_col].fillna("unknown").astype(str)

    bounds = poly_gdf.geometry.total_bounds  # [minx, miny, maxx, maxy]
    lon_min, lat_min, lon_max, lat_max = bounds
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0
    zoom = compute_zoom_level(poly_gdf)

    # Build GeoJSON from full poly_gdf (needs geometry)
    geojson = poly_gdf.__geo_interface__

    # Subset DataFrame for Plotly (drop geometry to avoid serialisation issues)
    keep_cols = list({"geojson_id", color_col} | set(hover_cols or []))
    keep_cols = [c for c in keep_cols if c in poly_gdf.columns]
    plot_df = poly_gdf[keep_cols].copy()

    px_kwargs: dict = dict(
        geojson=geojson,
        locations="geojson_id",
        featureidkey="properties.geojson_id",
        color=color_col,
        mapbox_style=mapbox_style,
        zoom=zoom,
        center={"lat": center_lat, "lon": center_lon},
        opacity=0.45,
    )
    if colorscale is not None and color_discrete_map is None:
        px_kwargs["color_continuous_scale"] = colorscale
    if color_discrete_map is not None:
        px_kwargs["color_discrete_map"] = color_discrete_map
    if hover_cols:
        px_kwargs["hover_data"] = {c: True for c in hover_cols if c in plot_df.columns}

    try:
        fig = px.choropleth_mapbox(plot_df, **px_kwargs)
    except Exception as exc:
        try:
            new_kw = {k.replace("mapbox_style", "map_style"): v for k, v in px_kwargs.items()}
            fig = px.choropleth_map(plot_df, **new_kw)
        except Exception:
            return _empty(f"Choropleth render failed ({exc})", height)

    # Explicit mapbox override ensures zoom/centre are always correct
    fig.update_layout(
        title=title,
        title_font=_TITLE_FONT,
        font=_FONT,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        template="plotly_white",
        height=height,
        mapbox=dict(style=mapbox_style, zoom=zoom,
                    center={"lat": center_lat, "lon": center_lon}),
    )
    return fig


# ── Non-map charts (unchanged) ────────────────────────────────────────────────

def chart_poi_distribution(category_series: pd.Series) -> go.Figure:
    if category_series.empty:
        return _empty("No POI data available")
    df = category_series.reset_index()
    df.columns = ["category", "count"]
    df["category"] = df["category"].str.replace("_", " ").str.title()
    total = df["count"].sum()
    threshold = total * 0.01
    main = df[df["count"] >= threshold].copy()
    other_count = int(df[df["count"] < threshold]["count"].sum())
    if other_count > 0:
        main = pd.concat([main, pd.DataFrame([{"category": "Other", "count": other_count}])],
                         ignore_index=True)
    main = main.sort_values("count")
    n = len(main)
    _teal = ["#1a6b8a", "#1e7da0", "#268fb5", "#2ea3ca", "#40b5d8",
             "#5ec3e0", "#80d3e8", "#a8d8ea"]
    colors = [_teal[int(i / max(n - 1, 1) * (len(_teal) - 1))] for i in range(n)]
    fig = go.Figure(go.Bar(
        x=main["count"], y=main["category"], orientation="h",
        marker_color=colors, text=main["count"], textposition="outside",
    ))
    fig.update_layout(
        title="Top POI Categories", xaxis_title="Count", yaxis_title="Category",
        template="plotly_white", height=450, margin=dict(l=10, r=40, t=50, b=40),
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_building_heights(buildings_gdf: gpd.GeoDataFrame) -> go.Figure:
    if buildings_gdf.empty or "height" not in buildings_gdf.columns:
        return _empty("No building height data available", 700)
    heights = pd.to_numeric(buildings_gdf["height"], errors="coerce").dropna()
    heights = heights[heights <= 200]
    if heights.empty:
        return _empty("No valid building heights found", 700)
    median_h = float(heights.median())
    p75_h = float(heights.quantile(0.75))
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Height Distribution", "Height vs Footprint Area"),
                        vertical_spacing=0.12)
    fig.add_trace(go.Histogram(x=heights, nbinsx=40, marker_color="#2196F3", opacity=0.8,
                               name="Count"), row=1, col=1)
    try:
        b = buildings_gdf.copy()
        if b.crs is None:
            b = b.set_crs("EPSG:4326")
        b_m = b.to_crs("EPSG:3857")
        area = b_m.geometry.area
        h_vals = pd.to_numeric(b_m["height"], errors="coerce")
        mask = h_vals.notna() & (h_vals <= 200) & (area > 0)
        h_s, a_s = h_vals[mask], area[mask]
        if not h_s.empty:
            fig.add_trace(go.Scatter(
                x=a_s, y=h_s, mode="markers",
                marker=dict(size=3, color=h_s, colorscale="Viridis", opacity=0.4,
                            showscale=True, colorbar=dict(title="Height (m)", x=1.02)),
                hovertemplate="Area: %{x:.0f} m²<br>Height: %{y:.1f} m<extra></extra>",
                name="Buildings",
            ), row=2, col=1)
    except Exception:
        pass
    fig.add_vline(x=median_h, line_dash="dash", line_color="#E91E63",
                  annotation_text=f"Median: {median_h:.1f}m", annotation_position="top right",
                  row=1, col=1)
    fig.add_vline(x=p75_h, line_dash="dot", line_color="#FF9800",
                  annotation_text=f"P75: {p75_h:.1f}m", annotation_position="top left",
                  row=1, col=1)
    fig.update_xaxes(title_text="Height (m)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Footprint Area (m²)", row=2, col=1)
    fig.update_yaxes(title_text="Height (m)", row=2, col=1)
    fig.update_layout(
        title="Building Height Analysis", template="plotly_white", height=700,
        showlegend=False, font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_street_network_radar(stats_dict: dict, city_name: str) -> go.Figure:
    if not stats_dict:
        return _empty("No street network data available", 400)
    connectivity       = min(stats_dict.get("avg_degree", 0) / 6.0, 1.0)
    block_size_score   = max(0.0, 1.0 - (stats_dict.get("avg_street_length", 200) / 400.0))
    intersection_dens  = min(stats_dict.get("intersection_count", 0) / 5000.0, 1.0)
    network_efficiency = max(0.0, 1.0 - (stats_dict.get("circuity_avg", 1.0) - 1.0))
    dead_end_score     = max(0.0, 1.0 - stats_dict.get("dead_end_ratio", 0.0))
    entropy_score      = min(stats_dict.get("orientation_entropy", 0.0) / 3.5, 1.0)
    pct_major          = stats_dict.get("pct_major_roads", 0.0)
    hierarchy_score    = min(pct_major / 0.3, 1.0) if pct_major else 0.3
    categories = ["Connectivity", "Block Size Score", "Intersection Density", "Network Efficiency",
                  "No Dead-Ends", "Orientation Entropy", "Street Hierarchy"]
    values = [connectivity, block_size_score, intersection_dens, network_efficiency,
              dead_end_score, entropy_score, hierarchy_score]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=categories + [categories[0]],
        fill="toself", fillcolor="rgba(33,150,243,0.25)",
        line=dict(color="#2196F3", width=2), name=city_name,
    ))
    fig.update_layout(
        title=f"Street Network Profile — {city_name}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white", height=400, showlegend=False,
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_landuse_composition(landuse_dict: dict) -> go.Figure:
    percentages = (landuse_dict or {}).get("percentages", {})
    if not percentages:
        return _empty("No land use data available", 400)
    labels = list(percentages.keys())
    values = list(percentages.values())
    marker_colors = [_LANDUSE_COLORS.get(lbl, "#CCCCCC") for lbl in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=marker_colors), textinfo="label+percent",
    ))
    green_pct = sum(percentages.get(k, 0) for k in ["park", "forest"])
    if green_pct >= 15:
        bench_text = f"✓ EU green target met ({green_pct:.1f}%)"
        bench_color = "#27AE60"
    else:
        bench_text = f"✗ Below EU 15% green target ({green_pct:.1f}%)"
        bench_color = "#E74C3C"
    fig.update_layout(
        title="Land Use Composition", template="plotly_white", height=400,
        font=_FONT, title_font=_TITLE_FONT,
        annotations=[go.layout.Annotation(
            text=bench_text, xref="paper", yref="paper",
            x=0.5, y=-0.08, showarrow=False,
            font=dict(size=12, color=bench_color),
        )],
    )
    return fig


def chart_street_orientation(orientation_entropy: float, orientation_histogram: list) -> go.Figure:
    if not orientation_histogram or all(v == 0 for v in orientation_histogram):
        return _empty("No orientation data available")
    n_bins = len(orientation_histogram)
    bin_deg = 180.0 / n_bins
    theta_half = [i * bin_deg + bin_deg / 2 for i in range(n_bins)]
    theta_full = theta_half + [t + 180.0 for t in theta_half]
    r_full = list(orientation_histogram) + list(orientation_histogram)
    max_r = max(r_full) if max(r_full) > 0 else 1
    bar_colors = [f"rgba(33,104,195,{0.4 + 0.6 * v / max_r})" for v in r_full]
    fig = go.Figure(go.Barpolar(
        r=r_full, theta=theta_full, width=[bin_deg] * len(theta_full),
        marker=dict(color=bar_colors), opacity=0.85,
    ))
    fig.update_layout(
        title=f"Street Orientation Distribution  (Entropy: {orientation_entropy:.2f})",
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        template="plotly_white", height=450,
    )
    return fig


# ── Hex choropleth maps ───────────────────────────────────────────────────────

def chart_diversity_heatmap(diversity_df: pd.DataFrame) -> go.Figure:
    if diversity_df is None or diversity_df.empty or "h3_cell" not in diversity_df.columns:
        return _empty("No diversity data available")
    return _choropleth_hex(
        diversity_df, "diversity_score", "Land Use Diversity (H3 Hex Grid)",
        colorscale="Viridis", hover_cols=["diversity_score"],
    )


def chart_far_heatmap(far_gdf) -> go.Figure:
    if far_gdf is None or far_gdf.empty or "FAR" not in far_gdf.columns:
        return _empty("No FAR data available", 500)
    hover = [c for c in ["FAR", "BCR", "CMI", "cluster"] if c in far_gdf.columns]
    return _choropleth_hex(
        far_gdf, "FAR", "Floor Area Ratio by Hex Cell",
        colorscale="Plasma", hover_cols=hover, height=500,
    )


def chart_morphotype_clusters(hex_gdf_with_clusters) -> go.Figure:
    if hex_gdf_with_clusters is None or hex_gdf_with_clusters.empty:
        return _empty("No morphotype data available", 500)
    if "cluster" not in hex_gdf_with_clusters.columns:
        return _empty("No cluster column available", 500)
    hover = [c for c in ["cluster", "FAR", "CMI"] if c in hex_gdf_with_clusters.columns]
    return _choropleth_hex(
        hex_gdf_with_clusters, "cluster", "Urban Morphotype Clusters",
        color_discrete_map=MORPHOTYPE_COLORS, hover_cols=hover, height=500,
    )


def chart_transport_accessibility(transport_hex_gdf) -> go.Figure:
    if transport_hex_gdf is None or transport_hex_gdf.empty or "transport_index" not in transport_hex_gdf.columns:
        return _empty("No transport accessibility data available")
    hover = [c for c in ["transport_index", "transit_score", "cycling_coverage"]
             if c in transport_hex_gdf.columns]
    return _choropleth_hex(
        transport_hex_gdf, "transport_index", "Transport Accessibility Index",
        colorscale="RdYlGn", hover_cols=hover, height=500,
    )


def chart_green_space_access(nature_hex_gdf) -> go.Figure:
    if nature_hex_gdf is None or nature_hex_gdf.empty or "green_space_ratio" not in nature_hex_gdf.columns:
        return _empty("No green space data available")
    hover = [c for c in ["green_space_ratio", "park_access_score"] if c in nature_hex_gdf.columns]
    return _choropleth_hex(
        nature_hex_gdf, "green_space_ratio", "Green Space Access by Hex Cell",
        colorscale="Greens", hover_cols=hover,
    )


def chart_flood_risk_zones(nature_hex_gdf) -> go.Figure:
    if nature_hex_gdf is None or nature_hex_gdf.empty or "flood_risk_tier" not in nature_hex_gdf.columns:
        return _empty("No flood risk data available")
    return _choropleth_hex(
        nature_hex_gdf, "flood_risk_tier", "Flood Risk Zones (Waterway Proximity Proxy)",
        color_discrete_map={"high": "#E74C3C", "medium": "#F39C12", "low": "#27AE60"},
        mapbox_style="carto-darkmatter", hover_cols=["flood_risk_tier"],
    )


# ── Transport non-map charts ──────────────────────────────────────────────────

def chart_road_hierarchy(road_type_counts: dict) -> go.Figure:
    if not road_type_counts:
        return _empty("No road hierarchy data available")
    _group_map = {
        "motorway": "Major", "primary": "Major",
        "secondary": "Connector", "tertiary": "Connector",
        "residential": "Local", "footway": "Local",
        "cycleway": "Active", "other": "Local",
    }
    _group_colors = {"Major": "#E74C3C", "Connector": "#F39C12", "Local": "#27AE60", "Active": "#3498DB"}

    def _norm(h):
        if isinstance(h, list): h = h[0] if h else "other"
        h = str(h).lower()
        for k in _group_map:
            if k in h: return k
        return "other"

    normalized = {}
    for raw_type, count in road_type_counts.items():
        norm = _norm(raw_type)
        normalized[norm] = normalized.get(norm, 0) + int(count)

    group_totals = {}
    for rt, count in normalized.items():
        grp = _group_map.get(rt, "Local")
        group_totals[grp] = group_totals.get(grp, 0) + count

    labels, parents, values, colors = [], [], [], []
    for grp, total in group_totals.items():
        labels.append(grp); parents.append(""); values.append(total)
        colors.append(_group_colors.get(grp, "#95A5A6"))
    for rt, count in normalized.items():
        grp = _group_map.get(rt, "Local")
        labels.append(rt.title()); parents.append(grp); values.append(count)
        colors.append(_group_colors.get(grp, "#95A5A6"))

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors), branchvalues="total",
    ))
    fig.update_layout(title="Road Hierarchy Composition", template="plotly_white", height=450)
    return fig


def chart_transit_heatmap(transit_stops_gdf) -> go.Figure:
    if transit_stops_gdf is None or transit_stops_gdf.empty:
        return _empty("No transit stop data available")
    if "lat" not in transit_stops_gdf.columns:
        if hasattr(transit_stops_gdf, "geometry"):
            pts = transit_stops_gdf[transit_stops_gdf.geometry.geom_type == "Point"]
            if pts.empty: return _empty("No transit stop data available")
            lat, lon = pts.geometry.y, pts.geometry.x
        else:
            return _empty("No transit stop data available")
    else:
        lat, lon = transit_stops_gdf["lat"], transit_stops_gdf["lon"]
    center_lat, center_lon = float(lat.mean()), float(lon.mean())
    try:
        fig = go.Figure(go.Densitymap(
            lat=lat, lon=lon, z=[1] * len(lat),
            radius=25, colorscale="Hot", showscale=False, opacity=0.7,
        ))
    except Exception:
        fig = go.Figure(go.Scattermap(
            lat=lat, lon=lon, mode="markers",
            marker=dict(size=6, color="#FF4500", opacity=0.6),
        ))
    fig.update_layout(
        title="Transit Stop Density",
        map=dict(style="carto-darkmatter", center=dict(lat=center_lat, lon=center_lon), zoom=12),
        template="plotly_white", height=450, margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ── Nature non-map charts ─────────────────────────────────────────────────────

def chart_nature_radar(nature_metrics_dict: dict) -> go.Figure:
    if not nature_metrics_dict:
        return _empty("No nature metrics available", 400)
    axes = ["Green Ratio", "Park Access", "Water Proximity", "Flood Safety"]
    vals = [
        float(nature_metrics_dict.get("green_ratio", 0)),
        float(nature_metrics_dict.get("park_access", 0)),
        float(nature_metrics_dict.get("water_proximity", 0)),
        float(nature_metrics_dict.get("flood_safety", 0)),
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=axes + [axes[0]],
        fill="toself", fillcolor="rgba(39,174,96,0.3)",
        line=dict(color="#27AE60", width=2),
    ))
    fig.update_layout(
        title="Nature Access Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white", height=400, showlegend=False,
    )
    return fig


# ── Cross-reference diamond heatmaps ─────────────────────────────────────────

def _quintile_crosstab(hex_gdf, col_x: str, col_y: str, value_col: str = None):
    df = hex_gdf[[col_x, col_y] + ([value_col] if value_col else [])].dropna()
    if len(df) < 5:
        return None, None
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    try:
        df["x_q"] = pd.qcut(df[col_x], 5, labels=q_labels, duplicates="drop")
        df["y_q"] = pd.qcut(df[col_y], 5, labels=q_labels, duplicates="drop")
    except Exception:
        return None, None
    df = df.dropna(subset=["x_q", "y_q"])
    if df.empty:
        return None, None
    if value_col:
        cross_tab = df.pivot_table(index="y_q", columns="x_q", values=value_col, aggfunc="median", fill_value=0)
    else:
        cross_tab = pd.crosstab(df["y_q"], df["x_q"])
    total = cross_tab.values.sum() if value_col is None else 1
    if value_col is None:
        text = [[f"{v / total * 100:.1f}%" if v > 0 else "" for v in row] for row in cross_tab.values]
    else:
        text = [[f"{v:.2f}" if v > 0 else "" for v in row] for row in cross_tab.values]
    return cross_tab, text


def chart_cross_heatmap_morph_transport(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty:
        return _empty("No merged hex data available", 450)
    if "CMI" not in hex_gdf.columns or "transport_index" not in hex_gdf.columns:
        return _empty("Need CMI and transport_index columns", 450)
    cross_tab, text = _quintile_crosstab(hex_gdf, "CMI", "transport_index")
    if cross_tab is None:
        return _empty("Insufficient data for cross-analysis", 450)
    col_sums = cross_tab.values.sum(axis=0)
    row_sums = cross_tab.values.sum(axis=1)
    x_labels = [str(c) for c in cross_tab.columns]
    y_labels = [str(r) for r in cross_tab.index]
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.82, 0.18],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )
    fig.add_trace(go.Heatmap(
        z=cross_tab.values, x=x_labels, y=y_labels,
        colorscale="Viridis", text=text, texttemplate="%{text}", textfont=dict(size=11),
        showscale=True, colorbar=dict(title="Count", x=1.02),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=row_sums, y=y_labels, orientation="h", marker_color="#7B68EE", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=x_labels, y=col_sums, marker_color="#7B68EE", showlegend=False,
    ), row=2, col=1)
    fig.update_layout(
        title="Morphology × Transport: Urban Structure Matrix",
        template="plotly_white", height=450, margin=dict(l=80, r=60, t=60, b=20),
        font=_FONT, title_font=_TITLE_FONT,
    )
    fig.update_xaxes(title_text="CMI Quintile →", row=2, col=1)
    fig.update_yaxes(title_text="Transport Quintile →", row=1, col=1)
    return fig


def chart_cross_heatmap_nature_morph(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty:
        return _empty("No merged hex data available", 450)
    if "green_space_ratio" not in hex_gdf.columns or "FAR" not in hex_gdf.columns:
        return _empty("Need green_space_ratio and FAR columns", 450)
    cross_tab, text = _quintile_crosstab(hex_gdf, "green_space_ratio", "FAR", value_col="CMI")
    if cross_tab is None:
        return _empty("Insufficient data for cross-analysis", 450)
    col_sums = cross_tab.values.sum(axis=0)
    row_sums = cross_tab.values.sum(axis=1)
    x_labels = [str(c) for c in cross_tab.columns]
    y_labels = [str(r) for r in cross_tab.index]
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.82, 0.18],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )
    fig.add_trace(go.Heatmap(
        z=cross_tab.values, x=x_labels, y=y_labels,
        colorscale="RdYlGn", text=text, texttemplate="%{text}", textfont=dict(size=11),
        showscale=True, colorbar=dict(title="Median CMI", x=1.02),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=row_sums, y=y_labels, orientation="h", marker_color="#27AE60", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=x_labels, y=col_sums, marker_color="#27AE60", showlegend=False,
    ), row=2, col=1)
    fig.update_layout(
        title="Nature × Density: Environmental Quality Matrix",
        template="plotly_white", height=450, margin=dict(l=80, r=60, t=60, b=20),
        font=_FONT, title_font=_TITLE_FONT,
    )
    fig.update_xaxes(title_text="Green Space Ratio Quintile →", row=2, col=1)
    fig.update_yaxes(title_text="FAR Quintile →", row=1, col=1)
    return fig


def chart_cross_heatmap_transport_nature(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty:
        return _empty("No merged hex data available", 450)
    if "transit_score" not in hex_gdf.columns and "transport_index" not in hex_gdf.columns:
        return _empty("Need transit_score or transport_index column", 450)
    if "nature_index" not in hex_gdf.columns:
        return _empty("Need nature_index column", 450)
    x_col = "transit_score" if "transit_score" in hex_gdf.columns else "transport_index"
    cross_tab, text = _quintile_crosstab(hex_gdf, x_col, "nature_index")
    if cross_tab is None:
        return _empty("Insufficient data for cross-analysis", 450)
    col_sums = cross_tab.values.sum(axis=0)
    row_sums = cross_tab.values.sum(axis=1)
    x_labels = [str(c) for c in cross_tab.columns]
    y_labels = [str(r) for r in cross_tab.index]
    n_cols, n_rows = len(cross_tab.columns), len(cross_tab.index)
    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.82, 0.18], row_heights=[0.82, 0.18],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )
    fig.add_trace(go.Heatmap(
        z=cross_tab.values, x=x_labels, y=y_labels,
        colorscale="Blues", text=text, texttemplate="%{text}", textfont=dict(size=11),
        showscale=True, colorbar=dict(title="Count", x=1.02),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=row_sums, y=y_labels, orientation="h", marker_color="#3498DB", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=x_labels, y=col_sums, marker_color="#3498DB", showlegend=False,
    ), row=2, col=1)
    fig.add_annotation(x=n_cols-1, y=n_rows-1, text="Transit-Rich<br>Green Zones", showarrow=False,
                       font=dict(color="#27AE60", size=10, family="Arial Black"), xref="x", yref="y")
    fig.add_annotation(x=0, y=0, text="Urban<br>Stress Zones", showarrow=False,
                       font=dict(color="#E74C3C", size=10, family="Arial Black"), xref="x", yref="y")
    fig.update_layout(
        title="Transit × Nature: Livability Matrix",
        template="plotly_white", height=450, margin=dict(l=80, r=60, t=60, b=20),
        font=_FONT, title_font=_TITLE_FONT,
    )
    fig.update_xaxes(title_text="Transit Score Quintile →", row=2, col=1)
    fig.update_yaxes(title_text="Nature Index Quintile →", row=1, col=1)
    return fig


# ── Additional analytical charts ─────────────────────────────────────────────

def chart_15min_city_score(poi_df, transit_stops_gdf, green_spaces_gdf,
                           city_center_lat: float, city_center_lon: float) -> go.Figure:
    WALK_M = 1250
    if poi_df is None or poi_df.empty:
        return _empty("No POI data for 15-min city score")
    min_lat, max_lat = float(poi_df["lat"].min()), float(poi_df["lat"].max())
    min_lon, max_lon = float(poi_df["lon"].min()), float(poi_df["lon"].max())
    lat_pts, lon_pts = np.linspace(min_lat, max_lat, 10), np.linspace(min_lon, max_lon, 10)
    lats, lons = np.meshgrid(lat_pts, lon_pts)
    sample_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons.flatten(), lats.flatten()), crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    sample_coords = np.array(list(zip(sample_gdf.geometry.x, sample_gdf.geometry.y)))

    def _access_score(fac_coords):
        if len(fac_coords) == 0: return 0.0
        try:
            from scipy.spatial import cKDTree
            dists, _ = cKDTree(fac_coords).query(sample_coords)
            return float((dists <= WALK_M).mean() * 100)
        except Exception:
            return 0.0

    scores = {}
    _svc = {"Food & Drink": ["food_and_beverage", "restaurant", "cafe", "bakery", "bar"],
            "Healthcare": ["health_and_medical", "hospital", "pharmacy", "clinic"],
            "Education": ["education", "school", "university", "kindergarten"]}
    if "category" in poi_df.columns:
        for label, kws in _svc.items():
            mask = poi_df["category"].fillna("").str.lower().apply(lambda c: any(k in c for k in kws))
            sub = poi_df[mask].dropna(subset=["lat", "lon"])
            if not sub.empty:
                pm = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sub["lon"], sub["lat"]),
                                      crs="EPSG:4326").to_crs("EPSG:3857")
                scores[label] = _access_score(np.array(list(zip(pm.geometry.x, pm.geometry.y))))
            else:
                scores[label] = 0.0
    else:
        for label in _svc: scores[label] = 0.0

    if transit_stops_gdf is not None and not transit_stops_gdf.empty:
        ts = transit_stops_gdf.copy()
        if ts.crs is None: ts = ts.set_crs("EPSG:4326")
        ts_m = ts.to_crs("EPSG:3857")
        ts_pts = ts_m[ts_m.geometry.geom_type == "Point"]
        scores["Public Transit"] = _access_score(
            np.array(list(zip(ts_pts.geometry.x, ts_pts.geometry.y)))) if not ts_pts.empty else 0.0
    else:
        scores["Public Transit"] = 0.0

    if green_spaces_gdf is not None and not green_spaces_gdf.empty:
        gs = green_spaces_gdf.copy()
        if gs.crs is None: gs = gs.set_crs("EPSG:4326")
        rpts = gs.to_crs("EPSG:3857").geometry.representative_point()
        scores["Park / Green Space"] = _access_score(np.array(list(zip(rpts.x, rpts.y))))
    else:
        scores["Park / Green Space"] = 0.0

    labels, values = list(scores.keys()), [scores[k] for k in scores]
    bar_colors = ["#27AE60" if v >= 70 else "#F39C12" if v >= 40 else "#E74C3C" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=bar_colors,
                           text=[f"{v:.0f}%" for v in values], textposition="outside"))
    fig.add_vline(x=40, line_dash="dot", line_color="orange", annotation_text="40%")
    fig.add_vline(x=70, line_dash="dot", line_color="green",  annotation_text="70%")
    fig.update_layout(
        title="15-Minute City Score by Service Type",
        xaxis_title="% of city area within 15-min walk", xaxis=dict(range=[0, 110]),
        template="plotly_white", height=450, margin=dict(l=10, r=60, t=50, b=40),
    )
    return fig


def chart_urban_quality_index(hex_gdf_merged) -> go.Figure:
    if hex_gdf_merged is None or hex_gdf_merged.empty:
        return _empty("No merged hex data for urban quality index")
    if not {"transport_index", "nature_index"}.issubset(hex_gdf_merged.columns):
        return _empty("Need transport_index and nature_index columns")
    df = hex_gdf_merged[["transport_index", "nature_index",
                          *(c for c in ["FAR", "CMI"] if c in hex_gdf_merged.columns)]].dropna()
    if df.empty: return _empty("Insufficient data for urban quality index")
    size_col  = df["FAR"].clip(lower=0.01) if "FAR" in df.columns else pd.Series(8, index=df.index)
    color_col = df["CMI"] if "CMI" in df.columns else df["transport_index"]
    med_t, med_n = df["transport_index"].median(), df["nature_index"].median()
    fig = go.Figure(go.Scatter(
        x=df["transport_index"], y=df["nature_index"], mode="markers",
        marker=dict(size=np.clip(size_col * 12, 4, 20), color=color_col, colorscale="Plasma",
                    showscale=True, colorbar=dict(title="CMI"), opacity=0.7,
                    line=dict(width=0.3, color="white")),
        hovertemplate="Transport: %{x:.3f}<br>Nature: %{y:.3f}<extra></extra>",
    ))
    try:
        m = np.polyfit(df["transport_index"], df["nature_index"], 1)
        x_line = np.linspace(df["transport_index"].min(), df["transport_index"].max(), 50)
        y_line = np.polyval(m, x_line)
        corr = float(np.corrcoef(df["transport_index"], df["nature_index"])[0, 1])
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color="rgba(100,100,100,0.5)", dash="dot", width=1.5),
            name=f"r={corr:.2f}", showlegend=True,
        ))
        fig.add_annotation(
            x=float(df["transport_index"].max()) * 0.95,
            y=float(df["nature_index"].max()) * 0.95,
            text=f"r = {corr:.2f}",
            showarrow=False, font=dict(size=11, color="#555555"),
        )
    except Exception:
        pass
    fig.add_hline(y=med_n, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=med_t, line_dash="dash", line_color="gray", opacity=0.5)
    for x, y, lbl, clr in [
        (med_t*1.05, med_n*1.05, "Green &<br>Connected", "#27AE60"),
        (med_t*0.3,  med_n*1.05, "Green &<br>Isolated",  "#F39C12"),
        (med_t*1.05, med_n*0.3,  "Dense &<br>Connected", "#3498DB"),
        (med_t*0.3,  med_n*0.3,  "Dense &<br>Isolated",  "#E74C3C"),
    ]:
        fig.add_annotation(x=x, y=y, text=lbl, showarrow=False,
                           font=dict(size=10, color=clr), opacity=0.8)
    fig.update_layout(
        title="Urban Quality Index: Transport vs Nature",
        xaxis_title="Transport Accessibility Index", yaxis_title="Nature Index",
        template="plotly_white", height=450,
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_density_gradient(buildings_gdf: gpd.GeoDataFrame,
                            city_center_lat: float, city_center_lon: float,
                            poi_df=None) -> go.Figure:
    if buildings_gdf is None or buildings_gdf.empty:
        return _empty("No building data for density gradient")
    if "height" not in buildings_gdf.columns:
        return _empty("No building height data for density gradient")
    b = buildings_gdf.copy().reset_index(drop=True)
    if b.crs is None: b = b.set_crs("EPSG:4326")
    b_m = b.to_crs("EPSG:3857")
    center_pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([city_center_lon], [city_center_lat]), crs="EPSG:4326"
    ).to_crs("EPSG:3857").geometry[0]
    b_m["dist_m"]    = b_m.geometry.centroid.distance(center_pt)
    b_m["height_num"] = pd.to_numeric(b_m["height"], errors="coerce")
    b_m = b_m.dropna(subset=["height_num", "dist_m"])
    b_m = b_m[b_m["height_num"].between(0, 200)].copy()
    if b_m.empty: return _empty("Insufficient height data for density gradient")
    b_m["dist_band"] = (b_m["dist_m"] // 200 * 200).astype(int)
    bh = b_m.groupby("dist_band")["height_num"].mean().reset_index()
    bh.columns = ["dist_m", "mean_height"]
    bh = bh.sort_values("dist_m")
    bh["smooth"] = bh["mean_height"].rolling(window=3, center=True, min_periods=1).mean()
    max_h = bh["smooth"].max() if not bh.empty else 1
    norm = bh["smooth"] / max(max_h, 1)
    clrs = [f"rgba({int(20+180*(1-v))},{int(20+60*(1-v))},{int(100+100*(1-v))},0.8)" for v in norm]
    has_poi = (poi_df is not None and not poi_df.empty
               and "lat" in poi_df.columns and "lon" in poi_df.columns)
    layout_kw = dict(yaxis2=dict(title="POI Density", overlaying="y", side="right",
                                 showgrid=False)) if has_poi else {}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bh["dist_m"] / 1000, y=bh["smooth"],
        fill="tozeroy", mode="lines+markers",
        line=dict(color="#2C3E50", width=2),
        fillcolor="rgba(52,152,219,0.15)",
        marker=dict(color=clrs, size=6), name="Mean Height",
    ))
    if has_poi:
        try:
            p = poi_df.copy()
            p_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(p["lon"], p["lat"]),
                                     crs="EPSG:4326").to_crs("EPSG:3857")
            p_dist = p_gdf.geometry.distance(center_pt)
            p_band = (p_dist // 200 * 200).astype(int)
            poi_counts = p_band.value_counts().sort_index().reset_index()
            poi_counts.columns = ["dist_m", "poi_count"]
            fig.add_trace(go.Scatter(
                x=poi_counts["dist_m"] / 1000, y=poi_counts["poi_count"],
                mode="lines", line=dict(color="#E74C3C", width=1.5, dash="dot"),
                name="POI Density", yaxis="y2",
            ))
        except Exception:
            pass
    fig.update_layout(
        title="Urban Density Gradient from City Center",
        xaxis_title="Distance from Center (km)", yaxis_title="Mean Building Height (m)",
        template="plotly_white", height=400,
        font=_FONT, title_font=_TITLE_FONT,
        **layout_kw,
    )
    return fig


def chart_landuse_crossref(crossref_df: pd.DataFrame) -> go.Figure:
    if crossref_df is None or crossref_df.empty:
        return _empty("No land use cross-reference data available")
    required = {"mean_transport", "mean_nature", "area_ha", "landuse_class"}
    if not required.issubset(crossref_df.columns):
        return _empty("Insufficient columns for land use cross-reference")
    df = crossref_df.dropna(subset=["mean_transport", "mean_nature"])
    if df.empty: return _empty("No valid land use cross-reference data")
    unique_cls = df["landuse_class"].unique()
    palette    = px.colors.qualitative.Set2
    color_map  = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_cls)}
    fig = go.Figure()
    for cls in unique_cls:
        row = df[df["landuse_class"] == cls]
        fig.add_trace(go.Scatter(
            x=row["mean_transport"], y=row["mean_nature"], mode="markers+text",
            marker=dict(size=np.clip(row["area_ha"]**0.4, 8, 40), color=color_map[cls],
                        opacity=0.8, line=dict(width=1, color="white")),
            text=row["landuse_class"].str.title(), textposition="top center",
            textfont=dict(size=10), name=str(cls).title(), showlegend=True,
        ))
    fig.update_layout(
        title="Land Use Performance: Transport vs Nature Access",
        xaxis_title="Mean Transport Index", yaxis_title="Mean Nature Index",
        template="plotly_white", height=450, legend=dict(title="Land Use"),
    )
    return fig


# ── Module 1: Urban Stress ────────────────────────────────────────────────────

def chart_urban_stress_map(stress_gdf) -> go.Figure:
    if stress_gdf is None or stress_gdf.empty or "urban_stress" not in stress_gdf.columns:
        return _empty("No urban stress data available")
    hover = [c for c in ["urban_stress", "stress_level", "dominant_driver", "density_stress",
                          "green_deficit", "transit_deficit", "flood_stress", "mono_stress"]
             if c in stress_gdf.columns]
    return _choropleth_hex(
        stress_gdf, "urban_stress",
        "Urban Stress Index — Composite of 5 Dimensions",
        colorscale="RdYlGn_r", hover_cols=hover,
    )


def chart_urban_stress_decomposition(stress_gdf) -> go.Figure:
    comp_cols = {
        "density_stress":  ("#E74C3C", "Density"),
        "green_deficit":   ("#27AE60", "Green Deficit"),
        "transit_deficit": ("#3498DB", "Transit Deficit"),
        "flood_stress":    ("#8E44AD", "Flood"),
        "mono_stress":     ("#F39C12", "Mono-Function"),
    }
    if stress_gdf is None or stress_gdf.empty or "stress_level" not in stress_gdf.columns:
        return _empty("No stress decomposition data available")
    if not any(c in stress_gdf.columns for c in comp_cols):
        return _empty("No stress component columns found")
    order = ["low", "moderate", "high", "critical"]
    df = stress_gdf.copy()
    df["stress_level"] = df["stress_level"].astype(str)
    grouped = df.groupby("stress_level")[[c for c in comp_cols if c in df.columns]].mean()
    grouped = grouped.reindex([o for o in order if o in grouped.index])
    fig = go.Figure()
    for col, (color, label) in comp_cols.items():
        if col in grouped.columns:
            fig.add_trace(go.Bar(name=label, x=grouped.index, y=grouped[col], marker_color=color))
    fig.update_layout(
        title="Stress Decomposition by Urban Zone Type",
        xaxis_title="Stress Level", yaxis_title="Mean Component Score",
        barmode="stack", template="plotly_white", height=450, legend=dict(title="Component"),
    )
    return fig


# ── Module 2: Opportunity Surface ─────────────────────────────────────────────

def chart_opportunity_surface(merged_hex_gdf) -> go.Figure:
    if merged_hex_gdf is None or merged_hex_gdf.empty:
        return _empty("No data for opportunity surface", 600)
    required = {"transport_index", "nature_index", "CMI"}
    missing = required - set(merged_hex_gdf.columns)
    if missing: return _empty(f"Missing columns: {missing}", 600)
    df = merged_hex_gdf[list(required) +
                        [c for c in ["urban_stress", "diversity_score"] if c in merged_hex_gdf.columns]].dropna()
    if len(df) < 3: return _empty("Insufficient data for opportunity surface", 600)

    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn) if mx != mn else pd.Series(0.5, index=s.index)

    x = _norm(df["transport_index"])
    y = _norm(df["nature_index"])
    z = 1.0 - _norm(df["CMI"])
    color_col = _norm(df["urban_stress"]) if "urban_stress" in df.columns else _norm(df.get("diversity_score", x))
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=[0.0] * len(x), mode="markers",
        marker=dict(size=2, color="rgba(100,100,100,0.2)", showscale=False),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=4, color=color_col, colorscale="RdYlGn_r",
                    showscale=True, colorbar=dict(title="Stress"), opacity=0.75),
        hovertemplate="Transport: %{x:.2f}<br>Nature: %{y:.2f}<br>Inv-Density: %{z:.2f}<extra></extra>",
    ))
    _annots = [
        dict(x=0.9, y=0.9, z=0.9, text="<b>Optimal Zone</b>",        ax=40, ay=-40, font=dict(color="#27AE60", size=11)),
        dict(x=0.9, y=0.1, z=0.5, text="Transit-Rich,<br>Nature-Poor", ax=40, ay=40,  font=dict(color="#F39C12", size=10)),
        dict(x=0.1, y=0.9, z=0.5, text="Green but<br>Isolated",        ax=-40, ay=-40, font=dict(color="#3498DB", size=10)),
        dict(x=0.1, y=0.1, z=0.1, text="<b>Urban Stress Core</b>",   ax=-40, ay=40,  font=dict(color="#E74C3C", size=11)),
    ]
    fig.update_layout(
        title="Opportunity Surface: Transport × Nature × Density",
        scene=dict(
            xaxis=dict(title="Transport Index", range=[0, 1]),
            yaxis=dict(title="Nature Index", range=[0, 1]),
            zaxis=dict(title="Density Inverse (1-CMI)", range=[0, 1]),
            bgcolor="rgba(15,15,25,0.95)",
            xaxis_gridcolor="rgba(255,255,255,0.1)", yaxis_gridcolor="rgba(255,255,255,0.1)",
            zaxis_gridcolor="rgba(255,255,255,0.1)",
            annotations=[go.layout.scene.Annotation(**a) for a in _annots],
        ),
        template="plotly_white", height=600, margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ── Module 3: Urban Fabric Typology ───────────────────────────────────────────

_FABRIC_LABELS_CHART = {
    "compact_vibrant": "Historic Mixed Core", "compact_mixed": "Dense Residential",
    "compact_low_mix": "Dense Mono",          "compact_mono": "Tower Block",
    "medium_vibrant":  "Active Mid-Rise",     "medium_mixed": "Typical Urban",
    "medium_low_mix":  "Mid-Rise Residential","medium_mono": "Office District",
    "low_vibrant":     "Active Low-Rise",     "low_mixed": "Mixed Suburb",
    "low_low_mix":     "Residential Suburb",  "low_mono": "Dormitory",
    "sprawl_vibrant":  "Commercial Strip",    "sprawl_mixed": "Peri-Urban Mixed",
    "sprawl_low_mix":  "Peri-Urban",          "sprawl_mono": "Fringe / Industrial",
}
_COMP_ORDER = ["sprawl", "low", "medium", "compact"]
_MIX_ORDER  = ["mono", "low_mix", "mixed", "vibrant"]


def chart_fabric_typology_matrix(fabric_gdf) -> go.Figure:
    if fabric_gdf is None or fabric_gdf.empty:
        return _empty("No fabric typology data available", 400)
    if "compactness" not in fabric_gdf.columns or "mix" not in fabric_gdf.columns:
        return _empty("Need compactness and mix columns", 400)
    df = fabric_gdf[["compactness", "mix"]].copy().astype(str)
    cross_tab = pd.crosstab(df["mix"], df["compactness"])
    cross_tab = cross_tab.reindex(index=_MIX_ORDER, columns=_COMP_ORDER, fill_value=0)
    text_matrix = []
    for mix_val in _MIX_ORDER:
        row = []
        for comp_val in _COMP_ORDER:
            key   = f"{comp_val}_{mix_val}"
            label = _FABRIC_LABELS_CHART.get(key, "Mixed")
            count = int(cross_tab.loc[mix_val, comp_val]) if (mix_val in cross_tab.index and comp_val in cross_tab.columns) else 0
            row.append(f"{label}<br>n={count}")
        text_matrix.append(row)
    fig = go.Figure(go.Heatmap(
        z=cross_tab.values.tolist(), x=_COMP_ORDER, y=_MIX_ORDER,
        colorscale="YlOrRd", text=text_matrix, texttemplate="%{text}",
        textfont=dict(size=9), showscale=True, colorbar=dict(title="Hex Count"), xgap=2, ygap=2,
    ))
    fig.update_layout(
        title="Urban Fabric Typology Matrix (16 Types)",
        xaxis_title="Compactness (FAR) →", yaxis_title="Land Use Mix →",
        template="plotly_white", height=400, margin=dict(l=80, r=20, t=60, b=60),
        annotations=[go.layout.Annotation(
            x=0.5, y=-0.12, xref="paper", yref="paper",
            text="Based on FAR × Land Use Mix", showarrow=False, font=dict(size=10, color="gray"),
        )],
    )
    return fig


def chart_fabric_typology_map(fabric_gdf) -> go.Figure:
    if fabric_gdf is None or fabric_gdf.empty or "planning_label" not in fabric_gdf.columns:
        return _empty("No fabric typology map data available")
    if "h3_cell" not in fabric_gdf.columns:
        return _empty("Need h3_cell column for fabric typology map")
    unique_labels = sorted(fabric_gdf["planning_label"].dropna().unique())
    palette   = px.colors.qualitative.Dark24
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique_labels)}
    hover = [c for c in ["planning_label", "fabric_type", "FAR", "diversity_score"] if c in fabric_gdf.columns]
    return _choropleth_hex(
        fabric_gdf, "planning_label", "Urban Fabric Types — Spatial Distribution",
        color_discrete_map=color_map, hover_cols=hover,
    )


# ── Module 4: Temporal Vulnerability ─────────────────────────────────────────

def chart_vulnerability_map(vuln_gdf) -> go.Figure:
    if vuln_gdf is None or vuln_gdf.empty or "vulnerability_score" not in vuln_gdf.columns:
        return _empty("No vulnerability data available")
    hover = [c for c in ["vulnerability_score", "vuln_class"] if c in vuln_gdf.columns]
    return _choropleth_hex(
        vuln_gdf, "vulnerability_score",
        "Temporal Vulnerability Index — Flood × Density × Isolation",
        colorscale="YlOrRd", mapbox_style="carto-darkmatter", hover_cols=hover,
    )


def chart_vulnerability_vs_stress(vuln_gdf) -> go.Figure:
    if vuln_gdf is None or vuln_gdf.empty or "vulnerability_score" not in vuln_gdf.columns:
        return _empty("No vulnerability data available")
    x_col = "urban_stress" if "urban_stress" in vuln_gdf.columns else None
    if x_col is None:
        return _empty("Need urban_stress column — run compute_urban_stress_index first")
    df = vuln_gdf[[x_col, "vulnerability_score"] +
                  [c for c in ["vuln_class", "FAR"] if c in vuln_gdf.columns]].dropna(
        subset=[x_col, "vulnerability_score"])
    if df.empty: return _empty("No overlapping stress + vulnerability data")
    _vuln_colors = {"resilient": "#27AE60", "moderate": "#F39C12", "high": "#E67E22", "extreme": "#E74C3C"}
    size_vals = np.clip(df["FAR"].fillna(0.3) * 12, 4, 20) if "FAR" in df.columns else 8
    fig = go.Figure()
    if "vuln_class" in df.columns:
        for cls, color in _vuln_colors.items():
            sub = df[df["vuln_class"].astype(str) == cls]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub["vulnerability_score"], mode="markers", name=cls.title(),
                marker=dict(color=color, size=size_vals[sub.index], opacity=0.75,
                            line=dict(width=0.5, color="white")),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df["vulnerability_score"], mode="markers",
            marker=dict(color=df["vulnerability_score"], colorscale="YlOrRd", size=8, opacity=0.75),
        ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    for x_pos, y_pos, label, color in [
        (0.75, 0.75, "Double Risk Zone",         "#E74C3C"),
        (0.25, 0.75, "Flood Risk,<br>Low Stress", "#8E44AD"),
        (0.75, 0.25, "Urban Stress,<br>Resilient","#3498DB"),
        (0.25, 0.25, "Balanced Zone",             "#27AE60"),
    ]:
        fig.add_annotation(x=x_pos, y=y_pos, text=label, showarrow=False,
                           font=dict(size=10, color=color), opacity=0.8)
    fig.update_layout(
        title="Urban Stress vs Climate Vulnerability",
        xaxis_title="Urban Stress Score", yaxis_title="Vulnerability Score",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        template="plotly_white", height=450,
    )
    return fig


# ── Module 5: Segregation Proxy ───────────────────────────────────────────────

def chart_segregation_map(seg_gdf) -> go.Figure:
    if seg_gdf is None or seg_gdf.empty or "segregation_score" not in seg_gdf.columns:
        return _empty("No segregation data available")
    hover = [c for c in ["segregation_score", "seg_class"] if c in seg_gdf.columns]
    return _choropleth_hex(
        seg_gdf, "segregation_score",
        "Urban Segregation Proxy — Mono-Function × Transit Isolation × Density",
        colorscale="RdPu", hover_cols=hover,
    )




# ── Module 6: Morphotype Radar Comparison ────────────────────────────────────

def chart_morphotype_radar_comparison(merged_hex_gdf) -> go.Figure:
    if merged_hex_gdf is None or merged_hex_gdf.empty or "cluster" not in merged_hex_gdf.columns:
        return _empty("Need cluster column — run compute_morphological_index first")
    _axes = [
        ("FAR", "FAR"), ("transport_index", "Transport"), ("nature_index", "Nature"),
        ("diversity_score", "Diversity"), ("road_hierarchy_mix", "Street Mix"),
        ("green_space_ratio", "Green Ratio"), ("height_norm", "Building Height"),
    ]
    df = merged_hex_gdf.copy()
    if "flood_stress" in df.columns:
        df["_flood_safety"] = 1.0 - pd.to_numeric(df["flood_stress"], errors="coerce").fillna(0.5)
    elif "flood_risk_tier" in df.columns:
        df["_flood_safety"] = 1.0 - df["flood_risk_tier"].map({"high": 1.0, "medium": 0.5, "low": 0.0}).fillna(0.5)
    else:
        df["_flood_safety"] = 0.5
    all_axes = _axes + [("_flood_safety", "Flood Safety")]
    axis_labels = [a[1] for a in all_axes]
    axis_cols   = [a[0] for a in all_axes]
    normed = {}
    for col, _ in all_axes:
        s = pd.to_numeric(df[col], errors="coerce").fillna(0.5) if col in df.columns else pd.Series(0.5, index=df.index)
        mn, mx = s.min(), s.max()
        normed[col] = (s - mn) / (mx - mn) if mx != mn else pd.Series(0.5, index=df.index)
    fig = go.Figure()
    cluster_counts = df["cluster"].value_counts()
    for cluster_name, color in MORPHOTYPE_COLORS.items():
        sub_idx = df["cluster"] == cluster_name
        if sub_idx.sum() == 0: continue
        vals = [float(normed[col][sub_idx].mean()) for col in axis_cols]
        count = int(cluster_counts.get(cluster_name, 0))
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=axis_labels + [axis_labels[0]],
            fill="toself",
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.18,)}",
            line=dict(color=color, width=2),
            name=f"{cluster_name.replace('_', ' ').title()} (n={count})",
        ))
    fig.update_layout(
        title="Morphotype DNA — 8-Dimension Urban Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9))),
        template="plotly_white", height=500,
        legend=dict(orientation="v", x=1.05, font=dict(size=10)),
    )
    return fig


# ── Terrain & Topography Charts ───────────────────────────────────────────────

def chart_terrain_elevation(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty or "elevation_m" not in hex_gdf.columns:
        return _empty("No elevation data available")
    hover = [c for c in ["elevation_m", "slope_deg", "twi", "terrain_flood_risk"]
             if c in hex_gdf.columns]
    return _choropleth_hex(
        hex_gdf, "elevation_m", "Terrain Elevation (metres)",
        colorscale="Spectral_r", hover_cols=hover, height=500,
    )


def chart_terrain_flood_risk(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty or "terrain_flood_risk" not in hex_gdf.columns:
        return _empty("No terrain flood risk data available")
    hover = [c for c in ["terrain_flood_risk", "elevation_m", "slope_deg", "twi"]
             if c in hex_gdf.columns]
    return _choropleth_hex(
        hex_gdf, "terrain_flood_risk",
        "Terrain-Based Flood Risk (Elevation + Slope + TWI)",
        colorscale=[[0, "#2ECC71"], [0.4, "#F1C40F"], [0.7, "#E67E22"], [1, "#C0392B"]],
        hover_cols=hover, height=500,
    )


def chart_terrain_cross_buildings(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty:
        return _empty("No terrain data available")
    required = {"elevation_m", "FAR"}
    if not required.issubset(hex_gdf.columns):
        return _empty("Need elevation_m and FAR columns")

    df = hex_gdf[list(required | {c for c in ["terrain_flood_risk", "BCR"]
                                   if c in hex_gdf.columns})].dropna()
    if df.empty:
        return _empty("No data for terrain vs buildings chart")

    color_vals = df["terrain_flood_risk"] if "terrain_flood_risk" in df.columns else df["FAR"]
    size_vals  = np.clip(df["BCR"] * 50, 4, 30) if "BCR" in df.columns else 8
    med_far    = float(df["FAR"].median())
    low_elev   = 10.0  # rough low-lying threshold

    fig = go.Figure(go.Scatter(
        x=df["elevation_m"], y=df["FAR"],
        mode="markers",
        marker=dict(
            size=size_vals, color=color_vals, colorscale="RdYlGn_r",
            showscale=True, colorbar=dict(title="Flood Risk"),
            opacity=0.75, line=dict(width=0.3, color="white"),
        ),
        hovertemplate="Elevation: %{x:.1f} m<br>FAR: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=med_far, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Median FAR {med_far:.2f}", annotation_position="top right")
    fig.add_vline(x=low_elev, line_dash="dot", line_color="#E74C3C", opacity=0.6,
                  annotation_text="10 m", annotation_position="top left")

    for (x_pos, y_pos, lbl, clr) in [
        (low_elev * 0.5, med_far * 1.5, "⚠️ High-Risk Dense Zone",  "#E74C3C"),
        (low_elev * 3.0, med_far * 1.5, "Dense Elevated Zone",      "#3498DB"),
        (low_elev * 0.5, med_far * 0.3, "Low-Risk Open Land",       "#27AE60"),
        (low_elev * 3.0, med_far * 0.3, "Elevated Sparse Zone",     "#95A5A6"),
    ]:
        fig.add_annotation(x=x_pos, y=y_pos, text=lbl, showarrow=False,
                           font=dict(size=10, color=clr), opacity=0.85)

    fig.update_layout(
        title="Building Density vs Terrain Elevation",
        xaxis_title="Elevation (m)", yaxis_title="Floor Area Ratio (FAR)",
        template="plotly_white", height=450,
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_twi_distribution(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty or "twi" not in hex_gdf.columns:
        return _empty("No TWI data available")

    twi = pd.to_numeric(hex_gdf["twi"], errors="coerce").dropna()
    if twi.empty:
        return _empty("No valid TWI values")

    # Colour-coded bars by risk zone
    fig = go.Figure()
    for lo, hi, color, label in [
        (0,  8,  "#27AE60", "Dry / Elevated (TWI < 8)"),
        (8,  12, "#F39C12", "Moderate (TWI 8–12)"),
        (12, 21, "#E74C3C", "Flood-Prone (TWI > 12)"),
    ]:
        subset = twi[(twi >= lo) & (twi < hi)]
        if not subset.empty:
            fig.add_trace(go.Histogram(
                x=subset, name=label, marker_color=color,
                xbins=dict(start=lo, end=hi, size=0.5),
                opacity=0.8,
            ))

    fig.add_vline(x=8,  line_dash="dot", line_color="#F39C12",
                  annotation_text="Moderate risk", annotation_position="top right")
    fig.add_vline(x=12, line_dash="dot", line_color="#E74C3C",
                  annotation_text="High risk", annotation_position="top right")

    fig.update_layout(
        title="Topographic Wetness Index Distribution",
        xaxis_title="TWI", yaxis_title="Hex Cell Count",
        barmode="stack", template="plotly_white", height=400,
        font=_FONT, title_font=_TITLE_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_slope_elevation_2d(hex_gdf) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty:
        return _empty("No terrain data available")
    required = {"elevation_m", "slope_deg"}
    if not required.issubset(hex_gdf.columns):
        return _empty("Need elevation_m and slope_deg columns")

    df = hex_gdf[list(required | {c for c in ["terrain_flood_risk"]
                                   if c in hex_gdf.columns})].dropna()
    if len(df) < 5:
        return _empty("Insufficient terrain data for 2D density chart")

    color_vals = df["terrain_flood_risk"] if "terrain_flood_risk" in df.columns else None

    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=df["elevation_m"], y=df["slope_deg"],
        colorscale="Blues", showscale=False, opacity=0.6,
        contours=dict(showlabels=False), line=dict(width=0.5),
        name="Density",
    ))
    scatter_kw = dict(
        x=df["elevation_m"], y=df["slope_deg"],
        mode="markers",
        name="Hex cells",
    )
    if color_vals is not None:
        scatter_kw["marker"] = dict(
            size=5, color=color_vals, colorscale="RdYlGn_r",
            showscale=True, colorbar=dict(title="Flood Risk"), opacity=0.6,
        )
    else:
        scatter_kw["marker"] = dict(size=5, color="#3498DB", opacity=0.5)
    fig.add_trace(go.Scatter(**scatter_kw))

    fig.update_layout(
        title="Terrain Profile: Elevation × Slope Distribution",
        xaxis_title="Elevation (m)", yaxis_title="Slope (degrees)",
        template="plotly_white", height=450,
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


# ── Admin boundary overlay ────────────────────────────────────────────────────

def extract_polygon_coords(geom) -> list:
    """Extract coordinate rings as [(lat, lon), …] from any geometry type."""
    rings = []
    try:
        if geom is None or not hasattr(geom, "geom_type"):
            return rings
        t = geom.geom_type
        if t == "Polygon":
            rings.append([(lat, lon) for lon, lat in geom.exterior.coords])
        elif t == "MultiPolygon":
            for poly in geom.geoms:
                rings.append([(lat, lon) for lon, lat in poly.exterior.coords])
        elif t == "LineString":
            rings.append([(lat, lon) for lon, lat in geom.coords])
        elif t == "MultiLineString":
            for line in geom.geoms:
                rings.append([(lat, lon) for lon, lat in line.coords])
        elif t == "GeometryCollection":
            for g in geom.geoms:
                rings.extend(extract_polygon_coords(g))
    except Exception as e:
        print(f"[extract_coords] {e}")
    return rings


def add_admin_boundaries_to_fig(fig, admin_boundaries: dict,
                                 show_city: bool = True,
                                 show_districts: bool = True):
    """Overlay admin boundary lines (shadow + white) onto any choropleth_mapbox figure."""
    if admin_boundaries is None:
        return fig

    # City boundary — shadow pass then solid white line on top
    if show_city and admin_boundaries.get("city") is not None:
        try:
            city_gdf = admin_boundaries["city"]
            for geom in city_gdf.geometry:
                for ring in extract_polygon_coords(geom):
                    if len(ring) < 3:
                        continue
                    lats = [c[0] for c in ring]
                    lons = [c[1] for c in ring]
                    # Close the ring explicitly
                    if lats[0] != lats[-1] or lons[0] != lons[-1]:
                        lats.append(lats[0])
                        lons.append(lons[0])
                    # Dark shadow
                    fig.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons, mode="lines",
                        line=dict(width=6, color="rgba(0,0,0,0.4)"),
                        showlegend=False, hoverinfo="skip",
                    ))
                    # White outline
                    fig.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons, mode="lines",
                        line=dict(width=3, color="rgba(255,255,255,1.0)"),
                        showlegend=False, hoverinfo="skip",
                    ))
        except Exception as e:
            print(f"[boundary] city draw error: {e}")

    # District boundaries
    for level in ["admin_9", "admin_10"]:
        if show_districts and admin_boundaries.get(level) is not None:
            try:
                gdf = admin_boundaries[level]
                for _, row in gdf.iterrows():
                    for ring in extract_polygon_coords(row.geometry):
                        if len(ring) < 3:
                            continue
                        lats = [c[0] for c in ring]
                        lons = [c[1] for c in ring]
                        if lats[0] != lats[-1] or lons[0] != lons[-1]:
                            lats.append(lats[0])
                            lons.append(lons[0])
                        fig.add_trace(go.Scattermapbox(
                            lat=lats, lon=lons, mode="lines",
                            line=dict(width=1.0, color="rgba(255,255,255,0.5)"),
                            showlegend=False, hoverinfo="skip",
                        ))
            except Exception as e:
                print(f"[boundary] district draw error: {e}")

    return fig


# ── Climate & Heat Island ─────────────────────────────────────────────────────

def chart_heat_island_map(hex_gdf, base_temp: float) -> go.Figure:
    if hex_gdf is None or hex_gdf.empty or 'uhi_delta' not in hex_gdf.columns:
        return _empty("No heat island data available")
    poly_gdf = hex_to_polygon_gdf(hex_gdf)
    if poly_gdf.empty:
        return _empty("No heat island polygon data")
    poly_gdf = poly_gdf.copy()
    poly_gdf['geojson_id'] = poly_gdf['h3_cell'].astype(str)
    bounds = poly_gdf.geometry.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds
    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0
    zoom = compute_zoom_level(poly_gdf)
    geojson = poly_gdf.__geo_interface__
    hover_cols = [c for c in ['uhi_delta', 'estimated_temp', 'BCR'] if c in poly_gdf.columns]
    keep_cols = ['geojson_id', 'uhi_delta'] + [c for c in hover_cols if c != 'uhi_delta']
    plot_df = poly_gdf[[c for c in keep_cols if c in poly_gdf.columns]].copy()
    try:
        fig = px.choropleth_mapbox(
            plot_df,
            geojson=geojson,
            locations='geojson_id',
            featureidkey='properties.geojson_id',
            color='uhi_delta',
            color_continuous_scale='RdYlBu_r',
            mapbox_style='carto-darkmatter',
            opacity=0.45,
            zoom=zoom,
            center={'lat': center_lat, 'lon': center_lon},
            hover_data={c: True for c in hover_cols if c in plot_df.columns},
        )
    except Exception:
        return _empty("Heat island map render failed")
    fig.update_layout(
        title=f"Urban Heat Island Proxy (Base: {base_temp:.1f}°C)",
        height=450,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar_title="ΔTemp (°C)",
        mapbox=dict(style='carto-darkmatter', zoom=zoom,
                    center={'lat': center_lat, 'lon': center_lon}),
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig




# ── District Scorecard ────────────────────────────────────────────────────────

def chart_district_scorecard(scores_df: pd.DataFrame) -> go.Figure:
    if scores_df is None or scores_df.empty:
        return _empty("No district scores available", 400)
    grade_cols = [c for c in scores_df.columns if c.endswith('_grade')]
    if not grade_cols:
        return _empty("No grade columns found", 400)
    display_cols = [c.replace('_grade', '').title() for c in grade_cols]
    _grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    if "overall_grade" in scores_df.columns:
        scores_df = scores_df.copy()
        scores_df["_sort_key"] = scores_df["overall_grade"].map(_grade_order).fillna(4)
        scores_df = scores_df.sort_values("_sort_key").drop(columns=["_sort_key"])
    z_text, z_colors = [], []
    for _, row in scores_df.iterrows():
        row_text, row_colors = [], []
        for col in grade_cols:
            grade = row.get(col, '?')
            row_text.append(str(grade) if grade else '?')
            row_colors.append({'A': 4, 'B': 3, 'C': 2, 'D': 1}.get(grade, 0))
        z_text.append(row_text)
        z_colors.append(row_colors)

    fig = go.Figure(data=go.Heatmap(
        z=z_colors,
        text=z_text,
        texttemplate='%{text}',
        textfont={'size': 16, 'color': 'white'},
        x=display_cols,
        y=scores_df['district_name'].tolist(),
        colorscale=[[0, '#95A5A6'], [0.25, '#E74C3C'], [0.5, '#F39C12'],
                    [0.75, '#82E0AA'], [1.0, '#27AE60']],
        showscale=False,
        hovertemplate='%{y}<br>%{x}: %{text}<extra></extra>',
    ))
    fig.update_layout(
        title='District Scorecard',
        title_subtitle_text='Sorted by overall performance',
        height=max(300, len(scores_df) * 40 + 100),
        template='plotly_white',
        xaxis=dict(side='top'),
        margin=dict(l=150, r=20, t=80, b=20),
        font=dict(size=12),
    )
    return fig


# ── New charts (Part 4) ───────────────────────────────────────────────────────

def chart_poi_density_contour(poi_df, city_polygon=None) -> go.Figure:
    if poi_df is None or poi_df.empty:
        return _empty("No POI data for density contour")
    if "lat" not in poi_df.columns or "lon" not in poi_df.columns:
        return _empty("POI data missing lat/lon columns")
    df = poi_df.dropna(subset=["lat", "lon"])
    if len(df) < 10:
        return _empty("Insufficient POI data for density contour")
    if city_polygon is not None:
        try:
            from shapely.geometry import Point
            mask = [city_polygon.contains(Point(lon, lat)) for lon, lat in zip(df["lon"], df["lat"])]
            df = df[mask].reset_index(drop=True)
        except Exception:
            pass
    if len(df) < 10:
        return _empty("Insufficient POIs within city boundary")
    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())
    try:
        from scipy.stats import gaussian_kde
        coords = np.vstack([df["lon"].values, df["lat"].values])
        kde = gaussian_kde(coords, bw_method=0.05)
        lon_range = np.linspace(df["lon"].min(), df["lon"].max(), 80)
        lat_range = np.linspace(df["lat"].min(), df["lat"].max(), 80)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        z = kde(np.vstack([lon_grid.ravel(), lat_grid.ravel()])).reshape(lon_grid.shape)
        fig = go.Figure(go.Densitymap(
            lat=lat_grid.ravel(), lon=lon_grid.ravel(), z=z.ravel(),
            radius=15, colorscale="YlOrRd", showscale=True,
            colorbar=dict(title="POI Density"),
            opacity=0.65,
        ))
    except Exception:
        fig = go.Figure(go.Densitymap(
            lat=df["lat"], lon=df["lon"], z=[1] * len(df),
            radius=20, colorscale="YlOrRd", showscale=False, opacity=0.65,
        ))
    zoom_val = compute_zoom_level(gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"))
    fig.update_layout(
        title="POI Density Distribution",
        map=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=zoom_val),
        template="plotly_white", height=500, margin=dict(l=0, r=0, t=50, b=0),
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_morphological_transition(merged_hex_gdf, city_center_lat: float,
                                    city_center_lon: float) -> go.Figure:
    if merged_hex_gdf is None or merged_hex_gdf.empty:
        return _empty("No hex data for morphological transition")
    if "lat" not in merged_hex_gdf.columns or "lon" not in merged_hex_gdf.columns:
        return _empty("Need lat/lon columns in hex grid")
    metrics_available = [c for c in ["FAR", "CMI", "transport_index", "green_space_ratio",
                                      "diversity_score"] if c in merged_hex_gdf.columns]
    if not metrics_available:
        return _empty("No morphological metrics available for transition chart")
    df = merged_hex_gdf.copy()
    try:
        center_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([city_center_lon], [city_center_lat]), crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        pts_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        df["dist_km"] = pts_gdf.geometry.distance(center_gdf.geometry[0]) / 1000
    except Exception:
        return _empty("Distance computation failed for morphological transition")
    df = df.dropna(subset=["dist_km"])
    df["dist_band"] = (df["dist_km"] // 0.5 * 0.5).round(1)
    _colors = {"FAR": "#E74C3C", "CMI": "#D4691E", "transport_index": "#3498DB",
               "green_space_ratio": "#27AE60", "diversity_score": "#9B59B6"}
    fig = go.Figure()
    for col in metrics_available:
        band_mean = df.groupby("dist_band")[col].mean().reset_index()
        band_mean = band_mean.sort_values("dist_band")
        s = band_mean[col]
        mn, mx = s.min(), s.max()
        normed = (s - mn) / (mx - mn) if mx != mn else pd.Series(0.5, index=s.index)
        fig.add_trace(go.Scatter(
            x=band_mean["dist_band"], y=normed,
            mode="lines+markers", name=col.replace("_", " ").title(),
            line=dict(color=_colors.get(col, "#888888"), width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        title="Morphological Transition from City Center",
        xaxis_title="Distance from Center (km)", yaxis_title="Normalised Metric Value",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


def chart_urban_efficiency_pareto(stress_gdf, admin_boundaries=None) -> go.Figure:
    if stress_gdf is None or stress_gdf.empty or "urban_stress" not in stress_gdf.columns:
        return _empty("No stress data for efficiency Pareto")
    comp_cols = {
        "density_stress": "Density",
        "green_deficit": "Green Deficit",
        "transit_deficit": "Transit Deficit",
        "flood_stress": "Flood",
        "mono_stress": "Mono-Function",
    }
    available = {k: v for k, v in comp_cols.items() if k in stress_gdf.columns}
    if not available:
        return _empty("No stress component columns for Pareto chart")
    means = {v: float(stress_gdf[k].mean()) for k, v in available.items()}
    labels = sorted(means, key=means.get, reverse=True)
    values = [means[l] for l in labels]
    cumulative = np.cumsum(values) / sum(values) * 100
    _bar_colors = ["#E74C3C" if v > 0.6 else "#F39C12" if v > 0.3 else "#27AE60" for v in values]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=_bar_colors,
        name="Mean Stress Component",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels, y=cumulative, mode="lines+markers",
        line=dict(color="#2C3E50", width=2), marker=dict(size=7),
        name="Cumulative %",
    ), secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color="gray", opacity=0.6, secondary_y=True,
                  annotation_text="80% threshold")
    fig.update_layout(
        title="Urban Stress Pareto Analysis",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=_FONT, title_font=_TITLE_FONT,
    )
    fig.update_yaxes(title_text="Mean Component Score", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 110])
    return fig


def chart_accessibility_spider(poi_df, city_center_lat: float,
                                city_center_lon: float) -> go.Figure:
    if poi_df is None or poi_df.empty:
        return _empty("No POI data for accessibility spider")
    if "lat" not in poi_df.columns or "lon" not in poi_df.columns:
        return _empty("POI data missing lat/lon columns")
    _services = {
        "Food & Drink":   ["food_and_beverage", "restaurant", "cafe", "bakery", "bar"],
        "Healthcare":     ["health_and_medical", "hospital", "pharmacy", "clinic"],
        "Education":      ["education", "school", "university", "kindergarten"],
        "Retail":         ["retail", "shop", "market", "supermarket"],
        "Recreation":     ["entertainment", "leisure", "sport", "gym", "park"],
        "Finance":        ["financial_service", "bank", "atm"],
    }
    try:
        center_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([city_center_lon], [city_center_lat]), crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        center_pt = center_gdf.geometry[0]
        poi_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(poi_df["lon"], poi_df["lat"]), crs="EPSG:4326"
        ).to_crs("EPSG:3857")
    except Exception:
        return _empty("Coordinate projection failed for accessibility spider")
    scores = {}
    for label, kws in _services.items():
        if "category" in poi_df.columns:
            mask = poi_df["category"].fillna("").str.lower().apply(lambda c: any(k in c for k in kws))
        else:
            mask = pd.Series(False, index=poi_df.index)
        sub_idx = poi_df[mask].index
        if len(sub_idx) == 0:
            scores[label] = 0.0
            continue
        dists = poi_gdf.geometry.iloc[sub_idx].distance(center_pt)
        med_dist = float(dists.median())
        scores[label] = max(0.0, 1.0 - med_dist / 5000)
    labels = list(scores.keys())
    values = [scores[l] for l in labels]
    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(52,152,219,0.25)",
        line=dict(color="#2980B9", width=2),
    ))
    fig.update_layout(
        title="Service Accessibility from City Center",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_white", height=420, showlegend=False,
        font=_FONT, title_font=_TITLE_FONT,
    )
    return fig


_CAT_GROUPS = {
    "Food & Hospitality": [
        "restaurant", "cafe", "bar", "bakery", "food", "drink",
        "french_restaurant", "pizza", "burger", "coffee", "tea",
        "pub", "brewery", "bistro", "brasserie", "caterer",
        "fast_food", "hotel", "hostel", "motel", "accommodation",
        "lodging", "inn", "resort", "catering",
    ],
    "Retail & Commerce": [
        "clothing_store", "shopping", "retail", "store", "shop",
        "supermarket", "grocery", "market", "mall", "boutique",
        "flowers_and_gifts", "bookstore", "electronics", "furniture",
        "hardware", "jewelry", "toy", "sporting_goods", "pet_store",
        "real_estate", "real_estate_agent", "professional_services",
        "office", "business", "lawyer", "accountant", "consultant",
        "insurance", "bank", "banks", "financial", "notary",
        "advertising", "marketing", "travel_agency", "automotive",
        "auto", "car", "gas_station",
    ],
    "Health & Education": [
        "hospital", "clinic", "doctor", "dentist", "pharmacy",
        "health", "medical", "optician", "veterinary", "spa",
        "wellness", "naturopathic", "holistic", "physiotherapy",
        "psychologist", "therapist", "naturopathic_holistic",
        "beauty_salon", "hair_salon", "nail_salon", "massage",
        "barber", "school", "university", "college", "kindergarten",
        "education", "college_university", "private_school",
        "tutoring", "library", "language_school", "driving_school",
    ],
    "Culture & Leisure": [
        "museum", "theater", "cinema", "gallery", "arts",
        "entertainment", "sports", "gym", "fitness", "park",
        "recreation", "club", "sports_club_and_league", "stadium",
        "bowling", "casino", "arcade", "concert", "nightclub",
        "dance", "yoga", "pilates", "swimming", "tennis", "golf",
    ],
    "Community & Services": [
        "community_services", "non_profit", "government", "post_office",
        "police", "fire_station", "church", "mosque", "temple",
        "synagogue", "religious", "social_service", "charity",
        "community_services_non_profits", "social_service_organizations",
        "transportation", "train_station", "bus_station", "parking",
    ],
}

_FUNC_COLORS = {
    "Food & Hospitality":   "#E74C3C",
    "Retail & Commerce":    "#F39C12",
    "Health & Education":   "#27AE60",
    "Culture & Leisure":    "#9B59B6",
    "Community & Services": "#3498DB",
    "Mixed":                "#95A5A6",
    "Other":                "#ECF0F1",
}

_VALID_FUNC_GROUPS = set(_CAT_GROUPS.keys()) | {"Mixed", "Other"}


def _categorize_poi(raw: str) -> str:
    cat_lower = str(raw).lower().strip()
    for group, keywords in _CAT_GROUPS.items():
        for kw in keywords:
            if kw in cat_lower or cat_lower in kw:
                return group
    return "Other"


def _get_dominant_or_mixed(group_series) -> str:
    counts = group_series.value_counts()
    if len(counts) == 0:
        return "Other"
    top_group = counts.index[0]
    top_count = counts.iloc[0]
    if top_group == "Other" and len(counts) > 1:
        top_group = counts.index[1]
        top_count = counts.iloc[1]
    total = counts.sum()
    if total > 0 and top_count / total < 0.40:
        return "Mixed"
    return top_group


def chart_poi_dominance_map(poi_df, merged_hex_gdf=None) -> go.Figure:
    if poi_df is None or poi_df.empty:
        return _empty("No POI data for dominance map")
    if "lat" not in poi_df.columns or "lon" not in poi_df.columns:
        return _empty("POI data missing lat/lon columns")
    if "category" not in poi_df.columns:
        return _empty("POI data missing category column")
    resolution = 8
    df = poi_df.dropna(subset=["lat", "lon", "category"]).copy()
    df["h3_cell"] = df.apply(
        lambda r: h3.latlng_to_cell(float(r["lat"]), float(r["lon"]), resolution), axis=1
    )
    df["functional_group"] = df["category"].apply(_categorize_poi)
    dominant = (
        df.groupby("h3_cell")["functional_group"]
        .apply(_get_dominant_or_mixed)
        .reset_index()
    )
    dominant.columns = ["h3_cell", "dominant_function"]
    dominant["dominant_function"] = dominant["dominant_function"].apply(
        lambda x: x if x in _VALID_FUNC_GROUPS else "Other"
    )
    group_counts = dominant["dominant_function"].value_counts()
    print(f"[poi_dominance] group distribution: {group_counts.to_dict()}")
    return _choropleth_hex(
        dominant, "dominant_function", "POI Functional Zone Dominance",
        color_discrete_map=_FUNC_COLORS, hover_cols=["dominant_function"], height=520,
    )


def chart_street_centrality_edges(graph, city_center_lat: float,
                                   city_center_lon: float) -> go.Figure:
    if graph is None:
        return _empty("No street graph for centrality edges")
    try:
        import networkx as nx
        nodes = list(graph.nodes)
        if len(nodes) < 100:
            return _empty("Insufficient street network nodes for centrality analysis")
        bc = nx.betweenness_centrality(graph, k=min(200, len(nodes)), normalized=True,
                                        weight="length", seed=42)
        edges = list(graph.edges(data=True))
        if len(edges) > 3000:
            edges = edges[:3000]
        center_lat, center_lon = city_center_lat, city_center_lon
        lats, lons, vals = [], [], []
        for u, v, data in edges:
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            u_bc = bc.get(u, 0)
            v_bc = bc.get(v, 0)
            edge_bc = (u_bc + v_bc) / 2
            lats += [u_data.get("y", 0), v_data.get("y", 0), None]
            lons += [u_data.get("x", 0), v_data.get("x", 0), None]
            vals.append(edge_bc)
        if not lats:
            return _empty("No edge coordinates found")
        val_arr = np.array(vals)
        mn, mx = val_arr.min(), val_arr.max()
        normed = (val_arr - mn) / (mx - mn) if mx != mn else val_arr
        colors = [f"rgba({int(255*(1-v))},{int(50+100*v)},{int(50+200*v)},0.8)"
                  for v in normed]
        edge_colors = []
        for i, clr in enumerate(colors):
            edge_colors += [clr, clr, "rgba(0,0,0,0)"]
        fig = go.Figure(go.Scattermap(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=1.5, color=edge_colors[0] if edge_colors else "#888888"),
            hoverinfo="skip",
        ))
        fig.update_layout(
            title="Street Network Betweenness Centrality",
            map=dict(style="carto-darkmatter",
                     center=dict(lat=center_lat, lon=center_lon), zoom=12),
            template="plotly_white", height=520, margin=dict(l=0, r=0, t=50, b=0),
            font=_FONT, title_font=_TITLE_FONT,
        )
        return fig
    except Exception as exc:
        return _empty(f"Street centrality failed: {exc}")

