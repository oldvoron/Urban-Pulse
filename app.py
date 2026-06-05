import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.io as pio
from dotenv import load_dotenv

from data import (
    fetch_overture_pois, fetch_osm_data, fetch_osm_landuse,
    fetch_transport_data, fetch_nature_data, fetch_terrain_data,
    fetch_admin_boundaries, fetch_climate_data, fetch_population_grid,
    get_city_polygon,
)
from metrics import (
    poi_category_distribution,
    building_height_stats,
    land_use_diversity,
    street_network_stats,
    compute_morphological_index,
    landuse_composition,
    transport_accessibility_index,
    nature_accessibility_index,
    landuse_transport_crossref,
    compute_urban_stress_index,
    compute_fabric_typology,
    compute_temporal_vulnerability,
    compute_segregation_proxy,
    sample_terrain_for_hexes,
    compute_district_scores,
    compute_heat_island_proxy,
    create_hex_grid_from_bbox,
    create_hex_grid_from_pois,
)
from charts import (
    chart_poi_distribution,
    chart_building_heights,
    chart_diversity_heatmap,
    chart_street_network_radar,
    chart_morphotype_clusters,
    chart_far_heatmap,
    chart_landuse_composition,
    chart_street_orientation,
    chart_building_typology,
    chart_cmi_distribution,
    chart_transport_accessibility,
    chart_road_hierarchy,
    chart_transit_heatmap,
    chart_green_space_access,
    chart_flood_risk_zones,
    chart_nature_radar,
    chart_cross_heatmap_morph_transport,
    chart_cross_heatmap_nature_morph,
    chart_cross_heatmap_transport_nature,
    chart_15min_city_score,
    chart_urban_quality_index,
    chart_density_gradient,
    chart_landuse_crossref,
    chart_urban_stress_map,
    chart_urban_stress_decomposition,
    chart_opportunity_surface,
    chart_fabric_typology_matrix,
    chart_fabric_typology_map,
    chart_vulnerability_map,
    chart_vulnerability_vs_stress,
    chart_segregation_map,
    chart_segregation_profile,
    chart_morphotype_radar_comparison,
    chart_terrain_elevation,
    chart_terrain_flood_risk,
    chart_terrain_cross_buildings,
    chart_twi_distribution,
    chart_slope_elevation_2d,
    add_admin_boundaries_to_fig,
    chart_heat_island_map,
    chart_climate_summary,
    chart_district_scorecard,
    chart_population_density,
)

load_dotenv()

# Optional: streamlit-folium (for future drawing support)
try:
    import streamlit_folium  # noqa: F401
except ImportError:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ai_available() -> bool:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    return bool(key) and key != "your_key_here"


def _safe_chart(fn, *args, key: str, ss_key: str = None,
                apply_admin_bounds: bool = False, **kw):
    """Render a plotly chart; optionally overlay admin boundaries. Returns the figure."""
    try:
        fig = fn(*args, **kw)
        if apply_admin_bounds:
            ab = st.session_state.get('admin_boundaries')
            if ab:
                fig = add_admin_boundaries_to_fig(
                    fig, ab,
                    show_city=st.session_state.get('show_city_boundary', True),
                    show_districts=st.session_state.get('show_districts', True),
                )
        if ss_key is not None:
            st.session_state[f"fig_{ss_key}"] = fig
        st.plotly_chart(fig, use_container_width=True, key=key)
        return fig
    except Exception as e:
        st.info(f"Chart unavailable: {e}")
        return None


def _tab_export(charts_in_tab: list, tab_name: str):
    """Unified per-tab export expander."""
    with st.expander("⬇️ Export this tab's charts"):
        export_cols = st.columns(3)
        for i, (fig_key, label) in enumerate(charts_in_tab):
            fig = st.session_state.get(f'fig_{fig_key}')
            if fig:
                with export_cols[i % 3]:
                    try:
                        img = pio.to_image(fig, format='png', width=2400, height=1400, scale=3)
                        st.download_button(
                            f"⬇️ {label}",
                            data=img,
                            file_name=f"urbanpulse_{fig_key}.png",
                            mime="image/png",
                            key=f"exp_{fig_key}_{tab_name}",
                        )
                    except Exception:
                        pass


def chart_download_button(fig, filename: str, key: str) -> None:
    if fig is None:
        return
    try:
        img = pio.to_image(fig, format="png", width=2400, height=1400, scale=3)
        st.download_button(label="⬇️ PNG (high-res)", data=img,
                           file_name=filename, mime="image/png", key=key)
    except Exception:
        pass


def chart_download_svg(fig, filename: str, key: str) -> None:
    if fig is None:
        return
    try:
        svg = pio.to_image(fig, format="svg", width=1600, height=900)
        st.download_button(label="⬇️ SVG (vector)", data=svg,
                           file_name=filename, mime="image/svg+xml", key=key)
    except Exception:
        pass


def _has_cols(gdf, *cols, min_rows: int = 10) -> bool:
    if gdf is None or gdf.empty or len(gdf) < min_rows:
        return False
    return all(c in gdf.columns for c in cols)


def safe_merge(base, other, on: str = "h3_cell", how: str = "left"):
    if base is None or (hasattr(base, "empty") and base.empty):
        return base
    if other is None or (hasattr(other, "empty") and other.empty):
        return base
    if on not in base.columns or on not in other.columns:
        return base
    try:
        return base.merge(other, on=on, how=how, suffixes=("", "_dup"))
    except Exception:
        return base


def get_analysis_bbox(city_name, zone_mode, selected_district, admin_boundaries, custom_bbox):
    """Return (min_lon, min_lat, max_lon, max_lat) for the active zone selection."""
    import osmnx as ox
    if zone_mode == "Draw custom bbox" and all(v != 0.0 for v in custom_bbox):
        return custom_bbox
    if zone_mode == "Select district" and selected_district != "All":
        for level in ['admin_9', 'admin_10']:
            gdf = (admin_boundaries or {}).get(level)
            if gdf is not None and not gdf.empty:
                match = gdf[gdf['admin_name'] == selected_district]
                if not match.empty:
                    b = match.total_bounds
                    return (b[0], b[1], b[2], b[3])
    try:
        bounds = ox.geocode_to_gdf(city_name).total_bounds
        return (bounds[0], bounds[1], bounds[2], bounds[3])
    except Exception:
        return None


def clip_to_city(data, city_polygon):
    """
    Clip a GeoDataFrame (spatial clip) or plain DataFrame with lat/lon columns
    to the city boundary polygon. Returns data unchanged if city_polygon is None
    or clipping fails.
    """
    if city_polygon is None or data is None:
        return data
    try:
        if hasattr(data, "geometry"):
            # GeoDataFrame — use geopandas spatial clip
            gdf = data.copy()
            if gdf.empty:
                return gdf
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            city_gdf = gpd.GeoDataFrame(geometry=[city_polygon], crs="EPSG:4326")
            clipped = gpd.clip(gdf, city_gdf)
            print(f"[clip] {len(data)} → {len(clipped)} rows")
            return clipped
        elif "lat" in data.columns and "lon" in data.columns:
            # Plain DataFrame — vectorised point-in-polygon
            from shapely.geometry import Point
            pts = [Point(lon, lat) for lon, lat in zip(data["lon"], data["lat"])]
            mask = [city_polygon.contains(p) for p in pts]
            clipped = data[mask].reset_index(drop=True)
            print(f"[clip] {len(data)} → {len(clipped)} rows (lat/lon)")
            return clipped
    except Exception as e:
        print(f"[clip] failed: {e}")
    return data


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="UrbanPulse", page_icon="🏙️", layout="wide")
st.title("UrbanPulse — Urban Spatial Analytics")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    city_name = st.text_input("City", value="Tours, France")
    analyze = st.button("Analyze", type="primary", use_container_width=True)
    st.markdown("---")

    # Map Layers
    st.subheader("Map Layers")
    show_city_boundary = st.checkbox("Show city boundary", value=True, key="show_city_boundary")
    show_districts = st.checkbox("Show districts", value=True, key="show_districts")
    st.markdown("---")

    # Analysis Zone
    st.subheader("Analysis Zone")
    zone_mode = st.radio(
        "Zone selection",
        ["Entire city", "Select district", "Draw custom bbox"],
        key="zone_mode",
    )

    selected_district = "All"
    custom_bbox = (0.0, 0.0, 0.0, 0.0)

    if zone_mode == "Select district":
        if st.session_state.get('admin_boundaries', {}).get('has_districts'):
            district_names = st.session_state['admin_boundaries']['district_names']
            selected_district = st.selectbox(
                "Select district", ["All"] + sorted(district_names),
                key="selected_district_sb",
            )
        else:
            st.info("Run analysis first to load districts")

    elif zone_mode == "Draw custom bbox":
        st.markdown("**Custom bounding box**")
        col1, col2 = st.columns(2)
        with col1:
            custom_min_lat = st.number_input("Min lat", value=0.0, format="%.4f", key="cmin_lat")
            custom_min_lon = st.number_input("Min lon", value=0.0, format="%.4f", key="cmin_lon")
        with col2:
            custom_max_lat = st.number_input("Max lat", value=0.0, format="%.4f", key="cmax_lat")
            custom_max_lon = st.number_input("Max lon", value=0.0, format="%.4f", key="cmax_lon")
        st.caption("Tip: copy bbox coords from the city overview map")
        custom_bbox = (
            st.session_state.get("cmin_lon", 0.0),
            st.session_state.get("cmin_lat", 0.0),
            st.session_state.get("cmax_lon", 0.0),
            st.session_state.get("cmax_lat", 0.0),
        )

    # Detect zone change and clear PDF
    _curr_zone = (zone_mode, selected_district, custom_bbox)
    if st.session_state.get('_last_zone') and st.session_state['_last_zone'] != _curr_zone:
        st.session_state['pdf_bytes'] = None

    st.markdown("---")

    # Analysis Modules
    st.subheader("Analysis Modules")
    with st.expander("Toggle modules", expanded=False):
        run_morphology = st.checkbox("Morphology & Buildings", value=True, key="run_morphology")
        run_transport  = st.checkbox("Transport & Accessibility", value=True, key="run_transport")
        run_nature     = st.checkbox("Nature & Green Space", value=True, key="run_nature")
        run_terrain    = st.checkbox("Terrain & Flood Risk", value=True, key="run_terrain")
        run_climate    = st.checkbox("Climate & Heat Island", value=True, key="run_climate")
        run_stress     = st.checkbox("Urban Stress Index", value=True, key="run_stress")
        run_typology   = st.checkbox("Urban Fabric Typology", value=True, key="run_typology")
        run_cross      = st.checkbox("Cross-Analysis", value=True, key="run_cross")

    st.markdown("---")
    st.caption("Data: Overture Maps + OpenStreetMap")
    st.caption("AI: Claude Sonnet")


# ── Cache wrappers ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_pois(city: str, bbox=None) -> pd.DataFrame:
    return fetch_overture_pois(city, bbox=bbox)

@st.cache_data(ttl=3600)
def load_osm(city: str, bbox=None):
    return fetch_osm_data(city, bbox=bbox)

@st.cache_data(ttl=3600)
def load_landuse(city: str, bbox=None):
    return fetch_osm_landuse(city, bbox=bbox)

@st.cache_data(ttl=3600)
def load_transport(city: str, bbox=None) -> dict:
    return fetch_transport_data(city, bbox=bbox)

@st.cache_data(ttl=3600)
def load_nature(city: str, bbox=None) -> dict:
    return fetch_nature_data(city, bbox=bbox)

@st.cache_data(ttl=3600)
def load_admin_boundaries(city: str) -> dict:
    return fetch_admin_boundaries(city)

@st.cache_data(ttl=3600)
def load_climate(lat: float, lon: float) -> dict:
    return fetch_climate_data(lat, lon)


# ── Analysis pipeline ─────────────────────────────────────────────────────────

if analyze:
    st.session_state["pdf_bytes"] = None
    st.session_state["analysis_done"] = False
    st.session_state['_last_zone'] = _curr_zone

    # Step 1: Admin boundaries + city polygon
    with st.spinner("Fetching administrative boundaries…"):
        try:
            admin_boundaries = load_admin_boundaries(city_name)
            st.session_state['admin_boundaries'] = admin_boundaries
            if not admin_boundaries.get('has_districts', False):
                st.info("No sub-city districts found — showing city boundary only.")
        except Exception as e:
            st.warning(f"Admin boundaries failed: {e}")
            admin_boundaries = {}
            st.session_state['admin_boundaries'] = {}

    city_polygon = get_city_polygon(admin_boundaries)
    st.session_state['city_polygon'] = city_polygon

    # Compute analysis bbox
    analysis_bbox = get_analysis_bbox(
        city_name, zone_mode, selected_district, admin_boundaries, custom_bbox
    )

    # Step 2: Fetch data
    status = st.empty()

    status.text("Fetching Overture Maps POIs…")
    poi_df = load_pois(city_name, bbox=analysis_bbox)
    poi_df = clip_to_city(poi_df, city_polygon)
    status.text(f"✅ POI: {len(poi_df):,} places loaded")

    graph = None
    buildings_raw = None          # raw GeoDataFrame from OSM (all columns)
    buildings_gdf = gpd.GeoDataFrame(columns=["geometry", "height"])

    if run_morphology or run_transport:
        status.text("Fetching OSM street network & buildings…")
        osm_result = load_osm(city_name, bbox=analysis_bbox)
        graph = osm_result.get("graph")
        buildings_raw = osm_result.get("buildings")

        # Clip raw buildings, then normalise → ['geometry', 'height']
        if buildings_raw is not None and not buildings_raw.empty:
            try:
                b = buildings_raw.copy()
                if b.crs is None:
                    b = b.set_crs("EPSG:4326")
                b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
                b = clip_to_city(b, city_polygon)
                if not b.empty:
                    from metrics import _resolve_heights
                    b["height"] = _resolve_heights(b).values
                    buildings_gdf = b[["geometry", "height"]].copy()
            except Exception as e:
                st.warning(f"Building normalisation failed: {e}")

        n_nodes = len(graph.nodes) if graph else 0
        n_bld   = len(buildings_gdf)
        status.text(f"✅ OSM: {n_nodes:,} street nodes, {n_bld:,} buildings")

    status.text("Fetching OSM land use…")
    landuse_gdf = load_landuse(city_name, bbox=analysis_bbox)
    landuse_gdf = clip_to_city(landuse_gdf, city_polygon)
    status.text(f"✅ Land use: {len(landuse_gdf):,} polygons")

    if run_transport:
        status.text("Fetching transport data…")
        transport_data = load_transport(city_name, bbox=analysis_bbox)
        transport_data["roads"] = clip_to_city(
            transport_data.get("roads", gpd.GeoDataFrame()), city_polygon)
        transport_data["transit_stops"] = clip_to_city(
            transport_data.get("transit_stops", gpd.GeoDataFrame()), city_polygon)
        n_stops = len(transport_data.get("transit_stops", gpd.GeoDataFrame()))
        status.text(f"✅ Transport: {n_stops:,} transit stops")
    else:
        transport_data = {"roads": gpd.GeoDataFrame(), "transit_stops": gpd.GeoDataFrame(),
                          "cycling": gpd.GeoDataFrame(), "parking": gpd.GeoDataFrame()}

    if run_nature:
        status.text("Fetching nature & green space data…")
        nature_data = load_nature(city_name, bbox=analysis_bbox)
        nature_data["green_spaces"] = clip_to_city(
            nature_data.get("green_spaces", gpd.GeoDataFrame()), city_polygon)
        nature_data["water_bodies"] = clip_to_city(
            nature_data.get("water_bodies", gpd.GeoDataFrame()), city_polygon)
        n_green = len(nature_data.get("green_spaces", gpd.GeoDataFrame()))
        status.text(f"✅ Nature: {n_green:,} green space polygons")
    else:
        nature_data = {"green_spaces": gpd.GeoDataFrame(), "water_bodies": gpd.GeoDataFrame(),
                       "flood_risk_proxy": gpd.GeoDataFrame()}

    status.empty()

    # Step 3: Core metrics
    with st.spinner("Computing core metrics…"):
        try:
            category_series = poi_category_distribution(poi_df)
        except Exception as e:
            st.warning(f"POI distribution failed: {e}")
            category_series = pd.Series(dtype=int)

        try:
            height_stats = building_height_stats(buildings_gdf) if run_morphology else {}
        except Exception as e:
            st.warning(f"Building height stats failed: {e}")
            height_stats = {}

        try:
            diversity_df = land_use_diversity(poi_df)
        except Exception as e:
            st.warning(f"Land use diversity failed: {e}")
            diversity_df = pd.DataFrame(columns=["h3_cell", "lat", "lon", "diversity_score"])

        try:
            net_stats = street_network_stats(graph) if run_morphology else {}
        except Exception as e:
            st.warning(f"Street network stats failed: {e}")
            net_stats = {}

        try:
            morph_data = (
                compute_morphological_index(buildings_gdf, graph) if run_morphology
                else {"hex_metrics": gpd.GeoDataFrame(),
                      "typology_counts": pd.Series(dtype=int),
                      "typologies": pd.Series(dtype=str)}
            )
        except Exception as e:
            st.warning(f"Morphological index failed: {e}")
            morph_data = {"hex_metrics": gpd.GeoDataFrame(),
                          "typology_counts": pd.Series(dtype=int)}

        try:
            lu_metrics = landuse_composition(landuse_gdf)
        except Exception as e:
            st.warning(f"Land use composition failed: {e}")
            lu_metrics = {}

    # Step 4: Accessibility
    with st.spinner("Computing transport & nature accessibility…"):
        transit_stops_gdf = transport_data.get("transit_stops", gpd.GeoDataFrame())
        green_spaces_gdf  = nature_data.get("green_spaces", gpd.GeoDataFrame())

        try:
            transport_hex = (
                transport_accessibility_index(buildings_gdf, transit_stops_gdf, graph)
                if run_transport else gpd.GeoDataFrame()
            )
        except Exception as e:
            st.warning(f"Transport accessibility failed: {e}")
            transport_hex = gpd.GeoDataFrame()

        try:
            nature_hex = (
                nature_accessibility_index(
                    buildings_gdf, green_spaces_gdf,
                    nature_data.get("water_bodies", gpd.GeoDataFrame()),
                )
                if run_nature else gpd.GeoDataFrame()
            )
        except Exception as e:
            st.warning(f"Nature accessibility failed: {e}")
            nature_hex = gpd.GeoDataFrame()

    # Step 5: Build merged hex GDF — with POI/bbox fallback when buildings unavailable
    hex_metrics     = morph_data.get("hex_metrics", gpd.GeoDataFrame())
    typology_counts = morph_data.get("typology_counts", pd.Series(dtype=int))

    if hex_metrics.empty and run_morphology:
        st.warning("Building data unavailable — using POI hex grid as base layer")
        poi_hex = create_hex_grid_from_pois(poi_df, resolution=9)
        if poi_hex.empty and analysis_bbox:
            poi_hex = create_hex_grid_from_bbox(analysis_bbox, resolution=9)
        hex_metrics = poi_hex

    merged_hex = hex_metrics.copy()

    if not diversity_df.empty:
        _div = diversity_df[[c for c in ["h3_cell", "diversity_score"] if c in diversity_df.columns]]
        if "h3_cell" in _div.columns:
            merged_hex = safe_merge(merged_hex, _div.drop_duplicates("h3_cell"))

    if not transport_hex.empty:
        _t_cols = [c for c in ["h3_cell", "transport_index", "transit_score",
                                "cycling_coverage", "road_hierarchy_mix"]
                   if c in transport_hex.columns]
        if "h3_cell" in _t_cols:
            merged_hex = safe_merge(merged_hex, transport_hex[_t_cols].drop_duplicates("h3_cell"))

    if not nature_hex.empty:
        _n_cols = [c for c in ["h3_cell", "nature_index", "green_space_ratio",
                                "park_access_score", "water_proximity", "flood_risk_tier"]
                   if c in nature_hex.columns]
        if "h3_cell" in _n_cols:
            merged_hex = safe_merge(merged_hex, nature_hex[_n_cols].drop_duplicates("h3_cell"))

    # Clip merged hex to city boundary
    merged_hex = clip_to_city(merged_hex, city_polygon)

    # Step 6: Terrain
    terrain_data: dict = {}
    if run_terrain:
        with st.spinner("Fetching terrain elevation data…"):
            try:
                if analysis_bbox:
                    _tbbox = analysis_bbox
                elif not poi_df.empty:
                    _tbbox = (float(poi_df["lon"].min()), float(poi_df["lat"].min()),
                              float(poi_df["lon"].max()), float(poi_df["lat"].max()))
                elif not hex_metrics.empty and "lon" in hex_metrics.columns:
                    _tbbox = (float(hex_metrics["lon"].min()), float(hex_metrics["lat"].min()),
                              float(hex_metrics["lon"].max()), float(hex_metrics["lat"].max()))
                else:
                    _tbbox = None
                if _tbbox:
                    terrain_data = fetch_terrain_data(city_name, _tbbox)
            except Exception as e:
                st.warning(f"Terrain data fetch failed: {e}")

    # Step 7: Advanced analytics
    with st.spinner("Computing advanced analytics…"):
        analysis_hex = merged_hex.copy()

        if run_stress:
            try:
                analysis_hex = compute_urban_stress_index(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute urban stress: {e}")

        if run_typology:
            try:
                analysis_hex = compute_fabric_typology(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute fabric typology: {e}")

        for _fn, _label in [
            (compute_temporal_vulnerability, "temporal vulnerability"),
            (compute_segregation_proxy,      "segregation proxy"),
        ]:
            try:
                analysis_hex = _fn(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute {_label}: {e}")

    # Population density proxy (buildings × 3.5 persons per unit)
    if not analysis_hex.empty and 'BCR' in analysis_hex.columns:
        try:
            analysis_hex['pop_density_proxy'] = (
                analysis_hex['BCR'].fillna(0) * 105332.51 / 100 * 3.5
            )
        except Exception:
            pass

    # Step 8: Terrain sampling
    if run_terrain:
        with st.spinner("Sampling terrain elevation per hex cell…"):
            try:
                if not analysis_hex.empty and terrain_data.get("elevation_grid") is not None:
                    analysis_hex = sample_terrain_for_hexes(analysis_hex, terrain_data)
            except Exception as e:
                st.warning(f"Terrain sampling failed: {e}")

    # Step 9: Scalar summaries
    total_buildings = (
        len(buildings_raw) if buildings_raw is not None and not buildings_raw.empty
        else len(buildings_gdf) if buildings_gdf is not None and not buildings_gdf.empty
        else 0
    )
    total_pois      = len(poi_df) if not poi_df.empty else 0
    green_pct       = lu_metrics.get("green_space_ratio", 0.0) * 100
    dominant_morphotype = (
        hex_metrics["cluster"].mode()[0].replace("_", " ").title()
        if not hex_metrics.empty and "cluster" in hex_metrics.columns else "N/A"
    )
    orientation_entropy = net_stats.get("orientation_entropy", 0.0)
    dead_end_ratio      = net_stats.get("dead_end_ratio", 0.0)
    block_size_median   = net_stats.get("block_size_median", 0.0)
    orientation_hist    = net_stats.get("orientation_histogram", [])
    if not hex_metrics.empty and "lat" in hex_metrics.columns:
        city_center_lat = float(hex_metrics["lat"].mean())
        city_center_lon = float(hex_metrics["lon"].mean())
    elif not poi_df.empty:
        city_center_lat = float(poi_df["lat"].mean())
        city_center_lon = float(poi_df["lon"].mean())
    else:
        city_center_lat, city_center_lon = 0.0, 0.0

    roads_gdf = transport_data.get("roads", gpd.GeoDataFrame())
    road_type_counts = {}
    if not roads_gdf.empty and "highway" in roads_gdf.columns:
        def _flat_hw(h):
            if isinstance(h, list): return h[0] if h else "other"
            return str(h) if h else "other"
        road_type_counts = roads_gdf["highway"].apply(_flat_hw).value_counts().to_dict()

    transit_stops_count = len(transit_stops_gdf) if not transit_stops_gdf.empty else 0
    cycling_gdf = transport_data.get("cycling", gpd.GeoDataFrame())
    cycling_km = 0.0
    if not cycling_gdf.empty:
        try:
            cycling_km = cycling_gdf.to_crs("EPSG:3857").geometry.length.sum() / 1000
        except Exception:
            pass
    dominant_road      = max(road_type_counts, key=road_type_counts.get) if road_type_counts else "N/A"
    water_bodies_count = len(nature_data.get("water_bodies", gpd.GeoDataFrame()))
    high_flood_pct = 0.0
    if not nature_hex.empty and "flood_risk_tier" in nature_hex.columns:
        high_flood_pct = (nature_hex["flood_risk_tier"] == "high").mean() * 100

    nature_metrics_dict: dict = {}
    if not nature_hex.empty:
        nature_metrics_dict = {
            "green_ratio":     float(nature_hex["green_space_ratio"].mean()) if "green_space_ratio" in nature_hex.columns else 0.0,
            "park_access":     min(float(nature_hex["park_access_score"].mean()) / 10, 1.0) if "park_access_score" in nature_hex.columns else 0.0,
            "water_proximity": float(nature_hex["water_proximity"].mean()) if "water_proximity" in nature_hex.columns else 0.0,
            "flood_safety":    float((nature_hex["flood_risk_tier"] != "high").mean()) if "flood_risk_tier" in nature_hex.columns else 1.0,
        }

    # Step 10: Climate + UHI
    climate_data: dict = {}
    if run_climate and city_center_lat != 0.0:
        with st.spinner("Fetching climate data (Open-Meteo)…"):
            try:
                climate_data = load_climate(city_center_lat, city_center_lon)
            except Exception as e:
                st.warning(f"Climate data failed: {e}")

    if run_climate and climate_data and 'error' not in climate_data:
        try:
            analysis_hex = compute_heat_island_proxy(analysis_hex, climate_data)
        except Exception as e:
            st.warning(f"Heat island proxy failed: {e}")

    # Step 11: Land use cross-reference
    with st.spinner("Computing land use cross-reference…"):
        try:
            crossref_df = (
                landuse_transport_crossref(landuse_gdf, analysis_hex)
                if run_cross else pd.DataFrame()
            )
        except Exception as e:
            st.warning(f"Land use cross-reference failed: {e}")
            crossref_df = pd.DataFrame()

    # Step 12: District scores
    district_scores_df = pd.DataFrame()
    try:
        if not analysis_hex.empty and admin_boundaries.get('has_districts'):
            district_scores_df = compute_district_scores(analysis_hex, admin_boundaries)
    except Exception as e:
        st.warning(f"District scores failed: {e}")

    # Session state for PDF export
    st.session_state["total_buildings"]      = total_buildings
    st.session_state["total_pois"]           = total_pois
    st.session_state["median_height"]        = height_stats.get("median", 0)
    st.session_state["green_space_pct"]      = green_pct
    st.session_state["transport_index_mean"] = (
        float(transport_hex["transport_index"].mean())
        if not transport_hex.empty and "transport_index" in transport_hex.columns else 0.0
    )
    st.session_state["urban_stress_mean"] = (
        float(analysis_hex["urban_stress"].mean())
        if not analysis_hex.empty and "urban_stress" in analysis_hex.columns else 0.0
    )
    st.session_state["dominant_morphotype"] = dominant_morphotype

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "Overview", "Morphology", "Street Network",
        "Transport", "Nature & Risk", "Cross-Analysis",
        "Stress & Risk", "Typology", "District Scores",
    ])

    # ── Tab 1: Overview ───────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Buildings", f"{total_buildings:,}")
        c2.metric("Total POIs", f"{total_pois:,}")
        c3.metric("Green Space", f"{green_pct:.1f}%")
        c4.metric("Dominant Morphotype", dominant_morphotype)

        _safe_chart(chart_poi_distribution, category_series,
                    key="t1_poi", ss_key="poi_distribution")
        _safe_chart(chart_landuse_composition, lu_metrics,
                    key="t1_landuse", ss_key="landuse_composition")

        if not analysis_hex.empty and 'pop_density_proxy' in analysis_hex.columns:
            st.subheader("Population Density Proxy")
            _safe_chart(chart_population_density, analysis_hex,
                        key="t1_pop_density", ss_key="pop_density",
                        apply_admin_bounds=True)

        _tab_export([
            ("poi_distribution",    "POI Distribution"),
            ("landuse_composition", "Land Use Composition"),
            ("pop_density",         "Population Density"),
        ], "tab1")

    # ── Tab 2: Morphology ─────────────────────────────────────────────────────
    with tab2:
        if not run_morphology:
            st.info("Module disabled — enable in sidebar")
        else:
            _safe_chart(chart_far_heatmap, hex_metrics,
                        key="t2_far", ss_key="far_heatmap", apply_admin_bounds=True)

            col_l, col_r = st.columns(2)
            with col_l:
                _safe_chart(chart_building_typology, typology_counts,
                            key="t2_typology", ss_key="building_typology")
            with col_r:
                _safe_chart(chart_cmi_distribution, hex_metrics,
                            key="t2_cmi", ss_key="cmi_distribution")

            _safe_chart(chart_morphotype_clusters, hex_metrics,
                        key="t2_clusters", ss_key="morphotype_clusters", apply_admin_bounds=True)
            _safe_chart(chart_density_gradient, buildings_gdf, city_center_lat, city_center_lon,
                        key="t2_density_grad", ss_key="density_gradient")

            if _has_cols(analysis_hex, "transport_index", "nature_index"):
                _safe_chart(chart_urban_quality_index, analysis_hex,
                            key="t2_quality", ss_key="urban_quality")
            else:
                st.info("Urban quality index needs transport + nature data.")

        _tab_export([
            ("far_heatmap",         "FAR Heatmap"),
            ("morphotype_clusters", "Morphotype Clusters"),
            ("building_typology",   "Building Typology"),
            ("cmi_distribution",    "CMI Distribution"),
            ("density_gradient",    "Density Gradient"),
        ], "tab2")

    # ── Tab 3: Street Network ─────────────────────────────────────────────────
    with tab3:
        if not run_morphology:
            st.info("Module disabled — enable Morphology & Buildings in sidebar")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Orientation Entropy", f"{orientation_entropy:.3f}")
            c2.metric("Dead-End Ratio", f"{dead_end_ratio:.1%}")
            c3.metric("Block Size Median", f"{block_size_median:,.0f} m²")

            _safe_chart(chart_street_orientation, orientation_entropy, orientation_hist,
                        key="t3_orientation", ss_key="street_orientation")
            _safe_chart(chart_street_network_radar, net_stats, city_name,
                        key="t3_radar", ss_key="street_radar")

        _tab_export([
            ("street_orientation", "Street Orientation"),
            ("street_radar",       "Street Network Radar"),
        ], "tab3")

    # ── Tab 4: Transport ──────────────────────────────────────────────────────
    with tab4:
        if not run_transport:
            st.info("Module disabled — enable in sidebar")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Transit Stops", f"{transit_stops_count:,}")
            c2.metric("Cycling Infrastructure", f"{cycling_km:.1f} km")
            c3.metric("Dominant Road Type", dominant_road.title())

            col_l, col_r = st.columns(2)
            with col_l:
                _safe_chart(chart_road_hierarchy, road_type_counts,
                            key="t4_road_hier", ss_key="road_hierarchy")
            with col_r:
                _safe_chart(chart_transit_heatmap, transit_stops_gdf,
                            key="t4_transit_heat", ss_key="transit_heatmap")

            _safe_chart(chart_transport_accessibility, transport_hex,
                        key="t4_transport_acc", ss_key="transport_map", apply_admin_bounds=True)

        _tab_export([
            ("road_hierarchy",  "Road Hierarchy"),
            ("transit_heatmap", "Transit Heatmap"),
            ("transport_map",   "Transport Accessibility"),
        ], "tab4")

    # ── Tab 5: Nature & Risk ──────────────────────────────────────────────────
    with tab5:
        if not run_nature:
            st.info("Module disabled — enable in sidebar")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Green Space Coverage", f"{green_pct:.1f}%")
            c2.metric("High Flood Risk Zones", f"{high_flood_pct:.1f}%")
            c3.metric("Water Bodies", f"{water_bodies_count:,}")

            col_l, col_r = st.columns(2)
            with col_l:
                _safe_chart(chart_green_space_access, nature_hex,
                            key="t5_green_access", ss_key="nature_map", apply_admin_bounds=True)
            with col_r:
                _safe_chart(chart_flood_risk_zones, nature_hex,
                            key="t5_flood", ss_key="flood_risk_map", apply_admin_bounds=True)

            col_l2, col_r2 = st.columns(2)
            with col_l2:
                _safe_chart(chart_nature_radar, nature_metrics_dict,
                            key="t5_nature_radar", ss_key="nature_radar")
            with col_r2:
                _safe_chart(
                    chart_15min_city_score,
                    poi_df, transit_stops_gdf, green_spaces_gdf,
                    city_center_lat, city_center_lon,
                    key="t5_15min", ss_key="city_15min",
                )

        # Terrain (independent of run_nature)
        if run_terrain and _has_cols(analysis_hex, "elevation_m", "terrain_flood_risk", min_rows=3):
            st.subheader("Terrain Analysis")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                _safe_chart(chart_terrain_elevation, analysis_hex,
                            key="t5_terrain_elev", ss_key="terrain_elevation",
                            apply_admin_bounds=True)
            with col_t2:
                _safe_chart(chart_terrain_flood_risk, analysis_hex,
                            key="t5_terrain_flood", ss_key="terrain_flood_risk",
                            apply_admin_bounds=True)

            _safe_chart(chart_terrain_cross_buildings, analysis_hex,
                        key="t5_terrain_cross", ss_key="terrain_cross")

            col_t3, col_t4 = st.columns(2)
            with col_t3:
                _safe_chart(chart_twi_distribution, analysis_hex,
                            key="t5_twi", ss_key="twi_distribution")
            with col_t4:
                _safe_chart(chart_slope_elevation_2d, analysis_hex,
                            key="t5_slope_elev", ss_key="slope_elevation")
        elif run_terrain:
            st.info("Terrain elevation data not available for this location.")

        # Urban Heat Island
        if run_climate:
            st.subheader("Urban Heat Island")
            if climate_data and 'error' not in climate_data and 'uhi_delta' in analysis_hex.columns:
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    _safe_chart(chart_heat_island_map, analysis_hex,
                                climate_data.get('temp_max_avg', 20),
                                key="t5_uhi_map", ss_key="heat_island_map",
                                apply_admin_bounds=True)
                with col_u2:
                    _safe_chart(chart_climate_summary, climate_data,
                                key="t5_climate", ss_key="climate_summary")
            else:
                st.info("Climate data unavailable — check network connection or API status.")

        _tab_export([
            ("nature_map",        "Green Space Access"),
            ("flood_risk_map",    "Flood Risk Zones"),
            ("nature_radar",      "Nature Radar"),
            ("terrain_elevation", "Terrain Elevation"),
            ("terrain_flood_risk","Terrain Flood Risk"),
            ("heat_island_map",   "Heat Island Map"),
            ("climate_summary",   "Climate Summary"),
        ], "tab5")

    # ── Tab 6: Cross-Analysis ─────────────────────────────────────────────────
    with tab6:
        if not run_cross:
            st.info("Module disabled — enable in sidebar")
        else:
            _need = {"CMI", "transport_index", "nature_index"}
            if not _has_cols(analysis_hex, *_need, min_rows=10):
                st.info("Cross-analysis requires morphology + transport + nature layers with ≥10 hex cells.")
            else:
                _safe_chart(chart_opportunity_surface, analysis_hex,
                            key="t6_opp_surface", ss_key="opportunity_surface")

                col1, col2, col3 = st.columns(3)
                with col1:
                    _safe_chart(chart_cross_heatmap_morph_transport, analysis_hex,
                                key="t6_cross_mt", ss_key="cross_morph_transport")
                with col2:
                    _safe_chart(chart_cross_heatmap_nature_morph, analysis_hex,
                                key="t6_cross_nm", ss_key="cross_nature_density")
                with col3:
                    _safe_chart(chart_cross_heatmap_transport_nature, analysis_hex,
                                key="t6_cross_tn", ss_key="cross_transport_nature")

                _safe_chart(chart_landuse_crossref, crossref_df,
                            key="t6_lu_crossref", ss_key="landuse_crossref")

                if _has_cols(analysis_hex, "transport_index", "nature_index", min_rows=10):
                    _safe_chart(chart_urban_quality_index, analysis_hex,
                                key="t6_quality", ss_key="urban_quality_cross")

        _tab_export([
            ("opportunity_surface",    "Opportunity Surface"),
            ("cross_morph_transport",  "Morphology × Transport"),
            ("cross_nature_density",   "Nature × Density"),
            ("cross_transport_nature", "Transport × Nature"),
            ("landuse_crossref",       "Land Use Cross-Ref"),
        ], "tab6")

    # ── Tab 7: Stress & Risk ──────────────────────────────────────────────────
    with tab7:
        if not run_stress:
            st.info("Module disabled — enable in sidebar")
        else:
            if not _has_cols(analysis_hex, "urban_stress", min_rows=5):
                st.info("Urban stress requires morphology + transport + nature layers.")
            else:
                _safe_chart(chart_urban_stress_map, analysis_hex,
                            key="t7_stress_map", ss_key="stress_map", apply_admin_bounds=True)

                col_l, col_r = st.columns(2)
                with col_l:
                    _safe_chart(chart_urban_stress_decomposition, analysis_hex,
                                key="t7_stress_decomp", ss_key="stress_decomp")
                with col_r:
                    if _has_cols(analysis_hex, "vulnerability_score", min_rows=5):
                        _safe_chart(chart_vulnerability_map, analysis_hex,
                                    key="t7_vuln_map", ss_key="vulnerability_map",
                                    apply_admin_bounds=True)
                    else:
                        st.info("Vulnerability map needs nature/flood data.")

                if _has_cols(analysis_hex, "urban_stress", "vulnerability_score", min_rows=5):
                    _safe_chart(chart_vulnerability_vs_stress, analysis_hex,
                                key="t7_vuln_vs_stress", ss_key="vuln_vs_stress")
                else:
                    st.info("Vulnerability vs Stress scatter needs both computed layers.")

        _tab_export([
            ("stress_map",       "Stress Map"),
            ("stress_decomp",    "Stress Decomposition"),
            ("vulnerability_map","Vulnerability Map"),
            ("vuln_vs_stress",   "Vulnerability vs Stress"),
        ], "tab7")

    # ── Tab 8: Typology ───────────────────────────────────────────────────────
    with tab8:
        if not run_typology:
            st.info("Module disabled — enable in sidebar")
        else:
            if not _has_cols(analysis_hex, "fabric_type", "planning_label", min_rows=5):
                st.info("Typology analysis requires morphology + land-use diversity layers.")
            else:
                col_l, col_r = st.columns(2)
                with col_l:
                    _safe_chart(chart_fabric_typology_matrix, analysis_hex,
                                key="t8_fabric_matrix", ss_key="fabric_matrix")
                with col_r:
                    _safe_chart(chart_fabric_typology_map, analysis_hex,
                                key="t8_fabric_map", ss_key="fabric_map",
                                apply_admin_bounds=True)

                if _has_cols(analysis_hex, "cluster", min_rows=5):
                    _safe_chart(chart_morphotype_radar_comparison, analysis_hex,
                                key="t8_radar_comp", ss_key="morphotype_radar")
                else:
                    st.info("Morphotype radar needs cluster labels from morphological analysis.")

                col_l2, col_r2 = st.columns(2)
                with col_l2:
                    if _has_cols(analysis_hex, "segregation_score", min_rows=5):
                        _safe_chart(chart_segregation_map, analysis_hex,
                                    key="t8_seg_map", ss_key="segregation_map",
                                    apply_admin_bounds=True)
                    else:
                        st.info("Segregation map needs transport + diversity layers.")
                with col_r2:
                    if _has_cols(analysis_hex, "segregation_score", "compactness", min_rows=5):
                        _safe_chart(chart_segregation_profile, analysis_hex,
                                    key="t8_seg_profile", ss_key="segregation_profile")
                    else:
                        st.info("Segregation profile needs fabric typology + segregation layers.")

        _tab_export([
            ("fabric_matrix",     "Fabric Typology Matrix"),
            ("fabric_map",        "Fabric Typology Map"),
            ("morphotype_radar",  "Morphotype Radar"),
            ("segregation_map",   "Segregation Map"),
            ("segregation_profile","Segregation Profile"),
        ], "tab8")

    # ── Tab 9: District Scores ────────────────────────────────────────────────
    with tab9:
        st.subheader("District Performance Scorecard")
        if district_scores_df.empty:
            if not admin_boundaries.get('has_districts'):
                st.info(
                    "No district boundaries found for this city. "
                    "Try a larger city or verify admin boundaries exist in OpenStreetMap."
                )
            else:
                st.info(
                    "Insufficient hex coverage in districts to compute grades "
                    "(minimum 5 hex cells required per district)."
                )
        else:
            _safe_chart(chart_district_scorecard, district_scores_df,
                        key="t9_scorecard", ss_key="district_scorecard")

            with st.expander("Raw scores table"):
                st.dataframe(district_scores_df, use_container_width=True)

            csv_bytes = district_scores_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download scores CSV",
                data=csv_bytes,
                file_name=f"district_scores_{city_name.replace(', ', '_').replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="dl_district_scores_csv",
            )

        _tab_export([("district_scorecard", "District Scorecard")], "tab9")

    # ── AI Analysis ───────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("AI Analysis", expanded=False):
        if _ai_available():
            from ai import get_urban_insights
            with st.spinner("Generating AI insights…"):
                _morph_stats = {
                    "dominant_morphotype": dominant_morphotype,
                    "green_space_ratio":   lu_metrics.get("green_space_ratio", 0.0),
                    "far_mean": float(hex_metrics["FAR"].mean()) if not hex_metrics.empty and "FAR" in hex_metrics.columns else 0.0,
                    "dead_end_ratio":      dead_end_ratio,
                    "orientation_entropy": orientation_entropy,
                    "transit_stops":       transit_stops_count,
                    "cycling_km":          round(cycling_km, 1),
                    "high_flood_pct":      round(high_flood_pct, 1),
                }
                insights = get_urban_insights(
                    city_name,
                    poi_stats=category_series.head(5).to_dict() if not category_series.empty else {},
                    height_stats=height_stats,
                    network_stats={k: v for k, v in net_stats.items() if k != "orientation_histogram"},
                    morph_stats=_morph_stats,
                )
            st.session_state["ai_insights"] = insights
            st.markdown(insights)
            with st.expander("📋 Copy-friendly text"):
                st.code(insights, language=None)
        else:
            st.info("Add ANTHROPIC_API_KEY to .env to enable AI insights.")

    st.session_state["analysis_done"] = True
    st.session_state["city_name"] = city_name

else:
    st.info("Enter a city name in the sidebar and click **Analyze** to begin.")


# ── PDF Export ────────────────────────────────────────────────────────────────

if st.session_state.get("analysis_done", False):
    st.divider()
    st.subheader("Export Report")

    if st.session_state.get("pdf_bytes") is None:
        with st.spinner("Preparing PDF report… (30-60 s, requires kaleido)"):
            try:
                from report import generate_pdf_report
                _ms = {
                    "total_buildings":      st.session_state.get("total_buildings", 0),
                    "total_pois":           st.session_state.get("total_pois", 0),
                    "median_height":        st.session_state.get("median_height", 0),
                    "green_space_pct":      st.session_state.get("green_space_pct", 0),
                    "transport_index_mean": st.session_state.get("transport_index_mean", 0),
                    "urban_stress_mean":    st.session_state.get("urban_stress_mean", 0),
                    "dominant_morphotype":  st.session_state.get("dominant_morphotype", "Unknown"),
                }
                _figs = {k: st.session_state.get(f"fig_{k}") for k in [
                    "poi_distribution", "building_heights", "morphotype_clusters",
                    "far_heatmap", "transport_map", "nature_map", "stress_map",
                    "fabric_matrix", "morphotype_radar", "opportunity_surface",
                    "cross_morph_transport", "cross_nature_density",
                    "terrain_elevation", "terrain_flood_risk",
                ]}
                _city = st.session_state.get("city_name", "city")
                pdf_bytes = generate_pdf_report(
                    city_name=_city,
                    metrics_summary=_ms,
                    figures=_figs,
                    ai_insights=st.session_state.get("ai_insights", ""),
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["pdf_city"] = _city
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.session_state["pdf_bytes"] = b""

    _pdf = st.session_state.get("pdf_bytes")
    if _pdf:
        _city = st.session_state.get("pdf_city", "city")
        st.download_button(
            label="⬇️ Download PDF Report",
            data=_pdf,
            file_name=f"urbanpulse_{_city.replace(', ', '_').replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            key="download_pdf_final",
        )
        st.caption(f"Report ready — {len(_pdf) // 1024} KB")
