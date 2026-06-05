import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px


def chart_poi_distribution(category_series: pd.Series) -> go.Figure:
    if category_series.empty:
        fig = go.Figure()
        fig.update_layout(title="No POI data available", template="plotly_white")
        return fig

    df = category_series.reset_index()
    df.columns = ["category", "count"]
    df = df.sort_values("count")

    n = len(df)
    colors = px.colors.sample_colorscale("Teal", [i / max(n - 1, 1) for i in range(n)])

    fig = go.Figure(go.Bar(
        x=df["count"],
        y=df["category"],
        orientation="h",
        marker_color=colors,
        text=df["count"],
        textposition="outside",
    ))

    fig.update_layout(
        title="Top POI Categories",
        xaxis_title="Count",
        yaxis_title="Category",
        template="plotly_white",
        height=450,
        margin=dict(l=10, r=40, t=50, b=40),
    )
    return fig


def chart_building_heights(buildings_gdf: gpd.GeoDataFrame) -> go.Figure:
    if buildings_gdf.empty or "height" not in buildings_gdf.columns:
        fig = go.Figure()
        fig.update_layout(title="No building height data available", template="plotly_white")
        return fig

    heights = pd.to_numeric(buildings_gdf["height"], errors="coerce").dropna()
    heights = heights[heights <= 200]

    if heights.empty:
        fig = go.Figure()
        fig.update_layout(title="No valid building heights found", template="plotly_white")
        return fig

    median_h = float(heights.median())
    p75_h = float(heights.quantile(0.75))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=heights,
        nbinsx=40,
        marker_color="#2196F3",
        opacity=0.8,
        name="Buildings",
    ))
    fig.add_vline(x=median_h, line_dash="dash", line_color="#E91E63",
                  annotation_text=f"Median: {median_h:.1f}m", annotation_position="top right")
    fig.add_vline(x=p75_h, line_dash="dot", line_color="#FF9800",
                  annotation_text=f"P75: {p75_h:.1f}m", annotation_position="top left")

    fig.update_layout(
        title="Building Height Distribution",
        xaxis_title="Height (m)",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def chart_diversity_heatmap(diversity_df: pd.DataFrame) -> go.Figure:
    if diversity_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No diversity data available", template="plotly_white")
        return fig

    center_lat = diversity_df["lat"].mean()
    center_lon = diversity_df["lon"].mean()

    fig = go.Figure(go.Scattermap(
        lat=diversity_df["lat"],
        lon=diversity_df["lon"],
        mode="markers",
        marker=dict(
            size=10,
            color=diversity_df["diversity_score"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Diversity"),
            opacity=0.8,
        ),
        text=diversity_df["diversity_score"].round(3).astype(str),
        hovertemplate="Diversity: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title="Land Use Diversity (H3 Hex Grid)",
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12,
        ),
        template="plotly_white",
        height=450,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def chart_street_network_radar(stats_dict: dict, city_name: str) -> go.Figure:
    if not stats_dict:
        fig = go.Figure()
        fig.update_layout(title="No street network data available", template="plotly_white")
        return fig

    avg_degree = stats_dict.get("avg_degree", 0)
    avg_street_length = stats_dict.get("avg_street_length", 200)
    intersection_count = stats_dict.get("intersection_count", 0)
    circuity_avg = stats_dict.get("circuity_avg", 1.0)

    # Normalize each metric to 0-1
    connectivity = min(avg_degree / 6.0, 1.0)
    # Shorter blocks = higher score (invert; typical urban block ~100-300m)
    block_size_score = max(0.0, 1.0 - (avg_street_length / 400.0))
    # intersection_count normalized against a large city reference (~5000)
    intersection_density = min(intersection_count / 5000.0, 1.0)
    # circuity 1.0 = straight, higher = more winding; score = proximity to 1
    network_efficiency = max(0.0, 1.0 - (circuity_avg - 1.0))

    categories = ["Connectivity", "Block Size Score", "Intersection Density", "Network Efficiency"]
    values = [connectivity, block_size_score, intersection_density, network_efficiency]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(33, 150, 243, 0.25)",
        line=dict(color="#2196F3", width=2),
        name=city_name,
    ))

    fig.update_layout(
        title=f"Street Network Profile — {city_name}",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
        ),
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig
