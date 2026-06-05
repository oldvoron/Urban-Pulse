import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from data import fetch_overture_pois, fetch_osm_data
from metrics import (
    poi_category_distribution,
    building_height_stats,
    land_use_diversity,
    street_network_stats,
)
from charts import (
    chart_poi_distribution,
    chart_building_heights,
    chart_diversity_heatmap,
    chart_street_network_radar,
)
load_dotenv()

def _ai_available() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(key) and key != "your_key_here"

st.set_page_config(
    page_title="UrbanPulse",
    page_icon="🏙️",
    layout="wide",
)

st.title("UrbanPulse — Urban Spatial Analytics")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    city_name = st.text_input("City", value="Tours, France")
    analyze = st.button("Analyze", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Data: Overture Maps + OpenStreetMap")
    st.caption("AI: Claude Sonnet")

# --- Cached data fetchers ---
@st.cache_data(ttl=3600)
def load_pois(city: str) -> pd.DataFrame:
    return fetch_overture_pois(city)

@st.cache_data(ttl=3600)
def load_osm(city: str):
    return fetch_osm_data(city)

# --- Main pipeline ---
if analyze:
    with st.spinner("Fetching Overture Maps POIs…"):
        poi_df = load_pois(city_name)

    with st.spinner("Fetching OSM street network & buildings…"):
        graph, buildings_gdf = load_osm(city_name)

    with st.spinner("Computing metrics…"):
        category_series = poi_category_distribution(poi_df)
        height_stats = building_height_stats(buildings_gdf)
        diversity_df = land_use_diversity(poi_df)
        net_stats = street_network_stats(graph)

    if _ai_available():
        from ai import get_urban_insights
        with st.spinner("Generating AI insights…"):
            insights = get_urban_insights(
                city_name,
                poi_stats=category_series.head(5).to_dict() if not category_series.empty else {},
                height_stats=height_stats,
                network_stats=net_stats,
            )
    else:
        insights = "AI analysis unavailable — add API key to .env"

    # --- Charts ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            chart_poi_distribution(category_series),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_building_heights(buildings_gdf),
            use_container_width=True,
        )

    with col_right:
        st.plotly_chart(
            chart_diversity_heatmap(diversity_df),
            use_container_width=True,
        )
        st.plotly_chart(
            chart_street_network_radar(net_stats, city_name),
            use_container_width=True,
        )

    # --- AI Insights ---
    st.markdown("---")
    st.subheader("AI Analysis")
    st.markdown(insights)

    # --- Download summary ---
    st.markdown("---")
    summary_lines = [
        f"UrbanPulse — Spatial Summary for {city_name}",
        "=" * 50,
        "",
        "POI Category Distribution (top 10):",
    ]
    for cat, cnt in category_series.head(10).items():
        summary_lines.append(f"  {cat}: {cnt}")

    summary_lines += ["", "Building Height Stats (metres):"]
    for k, v in height_stats.items():
        summary_lines.append(f"  {k}: {v:.2f}")

    summary_lines += ["", "Street Network Stats:"]
    for k, v in net_stats.items():
        summary_lines.append(f"  {k}: {v}")

    summary_lines += ["", "AI Insights:", insights]

    summary_text = "\n".join(summary_lines)

    st.download_button(
        label="Download Stats Summary",
        data=summary_text,
        file_name=f"urbanpulse_{city_name.replace(', ', '_').replace(' ', '_').lower()}.txt",
        mime="text/plain",
    )

else:
    st.info("Enter a city name in the sidebar and click **Analyze** to begin.")
