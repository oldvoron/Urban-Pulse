import gc
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import concurrent.futures
from math import radians, cos
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.io as pio
from dotenv import load_dotenv

from data import (
    fetch_overture_pois, fetch_osm_data, fetch_osm_landuse,
    fetch_transport_data, fetch_nature_data, fetch_terrain_data,
    fetch_admin_boundaries, fetch_climate_data,
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
    chart_street_network_radar,
    chart_morphotype_clusters,
    chart_far_heatmap,
    chart_landuse_composition,
    chart_street_orientation,
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
    chart_morphotype_radar_comparison,
    chart_terrain_elevation,
    chart_terrain_flood_risk,
    chart_terrain_cross_buildings,
    chart_twi_distribution,
    chart_slope_elevation_2d,
    add_admin_boundaries_to_fig,
    chart_heat_island_map,
    chart_district_scorecard,
    chart_poi_density_contour,
    chart_morphological_transition,
    chart_urban_efficiency_pareto,
    chart_nearest_services,
    chart_poi_dominance_map,
    chart_street_centrality_edges,
)

load_dotenv()


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
        st.plotly_chart(fig, width='stretch', key=key)
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


def has_data(df_or_gdf, min_rows=3, required_cols=None):
    """Check if dataframe has enough data to plot."""
    if df_or_gdf is None:
        return False
    if hasattr(df_or_gdf, 'empty') and df_or_gdf.empty:
        return False
    if len(df_or_gdf) < min_rows:
        return False
    if required_cols:
        for col in required_cols:
            if col not in df_or_gdf.columns:
                return False
            if df_or_gdf[col].isna().all():
                return False
    return True


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
    # Reuse the city boundary already fetched in fetch_admin_boundaries()
    # instead of issuing a second, unprotected geocode request.
    city_gdf = (admin_boundaries or {}).get("city")
    if city_gdf is not None and not city_gdf.empty:
        b = city_gdf.total_bounds
        return (b[0], b[1], b[2], b[3])
    return None


def bbox_area_km2(bbox):
    """Approximate area of a (min_lon, min_lat, max_lon, max_lat) bbox in km²."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_km = (max_lat - min_lat) * 111
    lon_km = (max_lon - min_lon) * 111 * cos(radians((min_lat + max_lat) / 2))
    return abs(lat_km * lon_km)


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

st.set_page_config(
    page_title="UrbanPulse",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'initialized' not in st.session_state:
    st.session_state.clear()
    st.session_state['initialized'] = True

st.title("UrbanPulse — Urban Spatial Analytics")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    city_size = st.selectbox(
        "City size",
        ["Small (< 100k)", "Medium (100k–500k)", "Large (500k+)"],
        index=1,
        key="city_size",
    )
    city_name = st.text_input(
        "City name",
        value="",
        placeholder="e.g. Blois, France",
        key="city_input",
    )

    SIZE_CONFIGS = {
        "Small (< 100k)": {
            "overture_limit": 5000,
            "building_limit": 15000,
            "network_dist": 3000,
            "hex_resolution": 9,
            "terrain_grid": 15,
            "road_tags": ['primary', 'secondary', 'tertiary'],
        },
        "Medium (100k–500k)": {
            "overture_limit": 15000,
            "building_limit": 25000,
            "network_dist": 5000,
            "hex_resolution": 9,
            "terrain_grid": 20,
            "road_tags": ['primary', 'secondary', 'tertiary'],
        },
        "Large (500k+)": {
            "overture_limit": 30000,
            "building_limit": 40000,
            "network_dist": 8000,
            "hex_resolution": 8,
            "terrain_grid": 20,
            "road_tags": ['primary', 'secondary'],
        },
    }
    config = SIZE_CONFIGS[city_size]
    st.session_state['config'] = config

    analyze = st.button("Analyze", type="primary", width='stretch')
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

if not city_name.strip():
    st.info("Enter a city name in the sidebar to begin analysis.")
    st.stop()

SKIP_MODULES = (
    ['segregation', 'vulnerability', 'fabric_typology']
    if city_size == "Small (< 100k)" else []
)

# ── Cache wrappers ────────────────────────────────────────────────────────────

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_pois(city: str, bbox=None, overture_limit: int = 15000) -> pd.DataFrame:
    return fetch_overture_pois(city, bbox=bbox, overture_limit=overture_limit)

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_osm(city: str, bbox=None, network_dist: int = 5000):
    return fetch_osm_data(city, bbox=bbox, network_dist=network_dist)

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_landuse(city: str, bbox=None):
    return fetch_osm_landuse(city, bbox=bbox)

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_transport(city: str, bbox=None, road_tags: tuple = ('primary', 'secondary', 'tertiary')) -> dict:
    return fetch_transport_data(city, bbox=bbox, road_tags=list(road_tags))

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_nature(city: str, bbox=None) -> dict:
    return fetch_nature_data(city, bbox=bbox)

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_admin_boundaries(city: str) -> dict:
    return fetch_admin_boundaries(city)

@st.cache_data(max_entries=1, ttl=1800, show_spinner=False)
def load_climate(lat: float, lon: float) -> dict:
    return fetch_climate_data(lat, lon)


def _run_parallel_fetches(city_name, analysis_bbox, config,
                           run_morph, run_transport, run_nature, run_terrain):
    """Fetch all data sources simultaneously using a thread pool."""
    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures_map['poi'] = executor.submit(
            load_pois, city_name, analysis_bbox, config['overture_limit'])
        futures_map['landuse'] = executor.submit(load_landuse, city_name, analysis_bbox)
        if run_morph:
            futures_map['osm'] = executor.submit(
                load_osm, city_name, analysis_bbox, config['network_dist'])
        if run_transport:
            futures_map['transport'] = executor.submit(
                load_transport, city_name, analysis_bbox, tuple(config['road_tags']))
        if run_nature:
            futures_map['nature'] = executor.submit(load_nature, city_name, analysis_bbox)
        if run_terrain and analysis_bbox:
            futures_map['terrain'] = executor.submit(
                fetch_terrain_data, city_name, analysis_bbox, config['terrain_grid'])

        results = {}
        timed_out_keys = []
        for key, future in futures_map.items():
            try:
                results[key] = future.result(timeout=240)
            except concurrent.futures.TimeoutError as e:
                print(f"[parallel] {key} timed out: {e}")
                results[key] = None
                timed_out_keys.append(key)
            except Exception as e:
                print(f"[parallel] {key} failed: {type(e).__name__}: {e}")
                results[key] = None
    return results, timed_out_keys


# ── Analysis pipeline ─────────────────────────────────────────────────────────

if analyze:
    st.session_state["pdf_bytes"] = None
    st.session_state["analysis_done"] = False
    st.session_state['_last_zone'] = _curr_zone

    # Step 1: Admin boundaries + city polygon (cached — skip refetch for the same city)
    with st.spinner("Fetching administrative boundaries…"):
        if (st.session_state.get('admin_boundaries') is not None
                and st.session_state.get('last_city') == city_name):
            admin_boundaries = st.session_state['admin_boundaries']
        else:
            try:
                admin_boundaries = load_admin_boundaries(city_name)
            except Exception as e:
                st.warning(f"Admin boundaries failed: {e}")
                admin_boundaries = {}
            st.session_state['admin_boundaries'] = admin_boundaries
            st.session_state['last_city'] = city_name

    city_polygon = get_city_polygon(admin_boundaries)
    st.session_state['city_polygon'] = city_polygon

    # Compute analysis bbox
    analysis_bbox = get_analysis_bbox(
        city_name, zone_mode, selected_district, admin_boundaries, custom_bbox
    )

    MAX_ENTIRE_CITY_AREA_KM2 = 800
    if zone_mode == "Entire city" and analysis_bbox is not None:
        area = bbox_area_km2(analysis_bbox)
        if area > MAX_ENTIRE_CITY_AREA_KM2:
            st.error(
                "❌ This city's area is too large for a full-city analysis via "
                "free Overpass servers. Please select a specific district "
                f"(detected area: {area:.0f} km², limit: {MAX_ENTIRE_CITY_AREA_KM2} km²)."
            )
            st.stop()

    # Step 2: Fetch all data sources in parallel
    status = st.empty()
    with st.spinner("Fetching all data sources…"):
        all_data, timed_out_layers = _run_parallel_fetches(
            city_name, analysis_bbox, config,
            run_morph=(run_morphology or run_transport),
            run_transport=run_transport,
            run_nature=run_nature,
            run_terrain=run_terrain,
        )
    gc.collect()

    poi_raw = all_data.get('poi')
    poi_df_raw = poi_raw if poi_raw is not None else pd.DataFrame(columns=["name", "category", "lat", "lon"])
    osm_result = all_data.get('osm') or {"graph": None, "buildings": None}
    landuse_raw = all_data.get('landuse')
    landuse_gdf = landuse_raw if landuse_raw is not None else gpd.GeoDataFrame()
    transport_data = all_data.get('transport') or {
        "roads": gpd.GeoDataFrame(), "transit_stops": gpd.GeoDataFrame(), "cycling": gpd.GeoDataFrame()}
    nature_data = all_data.get('nature') or {
        "green_spaces": gpd.GeoDataFrame(), "water_bodies": gpd.GeoDataFrame(),
        "flood_risk_proxy": gpd.GeoDataFrame()}
    terrain_data = all_data.get('terrain') or {}

    poi_df = clip_to_city(poi_df_raw, city_polygon)
    del poi_df_raw
    if poi_df is None or len(poi_df) == 0:
        st.error("❌ No data found for this city. Make sure to include the country, e.g. 'Blois, France'")
        st.stop()
    elif len(poi_df) < 15:
        st.warning("⚠️ Very few POIs found. Results may be incomplete.")
    status.text(f"✅ Data fetched: {len(poi_df):,} POIs")

    if timed_out_layers:
        st.warning(
            "Some data layers timed out for this large area. Try selecting "
            "a smaller zone (district) instead of the entire city."
        )

    graph = None
    buildings_raw = None
    buildings_gdf = gpd.GeoDataFrame(columns=["geometry", "height"])

    if run_morphology or run_transport:
        graph = osm_result.get("graph")
        buildings_raw = osm_result.get("buildings")

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

    landuse_gdf = clip_to_city(landuse_gdf, city_polygon)

    if run_transport:
        transport_data["roads"] = clip_to_city(
            transport_data.get("roads", gpd.GeoDataFrame()), city_polygon)
        transport_data["transit_stops"] = clip_to_city(
            transport_data.get("transit_stops", gpd.GeoDataFrame()), city_polygon)

    if run_nature:
        nature_data["green_spaces"] = clip_to_city(
            nature_data.get("green_spaces", gpd.GeoDataFrame()), city_polygon)
        nature_data["water_bodies"] = clip_to_city(
            nature_data.get("water_bodies", gpd.GeoDataFrame()), city_polygon)

    status.empty()

    # Sample buildings for metric computation — config-based limit
    _building_cap = config['building_limit']
    if buildings_gdf is not None and len(buildings_gdf) > _building_cap:
        buildings_sample = buildings_gdf.sample(_building_cap, random_state=42)
        print(f"[perf] Sampling {_building_cap} from {len(buildings_gdf)} buildings for metrics")
    else:
        buildings_sample = buildings_gdf
    gc.collect()

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
                compute_morphological_index(buildings_sample, graph, diversity_df=diversity_df) if run_morphology
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
    gc.collect()

    # Step 4: Accessibility
    with st.spinner("Computing transport & nature accessibility…"):
        transit_stops_gdf = transport_data.get("transit_stops", gpd.GeoDataFrame())
        green_spaces_gdf  = nature_data.get("green_spaces", gpd.GeoDataFrame())

        try:
            transport_hex = (
                transport_accessibility_index(buildings_sample, transit_stops_gdf, graph)
                if run_transport else gpd.GeoDataFrame()
            )
        except Exception as e:
            st.warning(f"Transport accessibility failed: {e}")
            transport_hex = gpd.GeoDataFrame()

        try:
            nature_hex = (
                nature_accessibility_index(
                    buildings_sample, green_spaces_gdf,
                    nature_data.get("water_bodies", gpd.GeoDataFrame()),
                )
                if run_nature else gpd.GeoDataFrame()
            )
        except Exception as e:
            st.warning(f"Nature accessibility failed: {e}")
            nature_hex = gpd.GeoDataFrame()
    gc.collect()

    # Step 5: Build merged hex GDF — with POI/bbox fallback when buildings unavailable
    hex_metrics     = morph_data.get("hex_metrics", gpd.GeoDataFrame())
    typology_counts = morph_data.get("typology_counts", pd.Series(dtype=int))

    if hex_metrics.empty and run_morphology:
        st.warning("Building data unavailable — using POI hex grid as base layer")
        poi_hex = create_hex_grid_from_pois(poi_df)
        if poi_hex.empty and analysis_bbox:
            poi_hex = create_hex_grid_from_bbox(analysis_bbox, resolution=config['hex_resolution'])
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

    # Drop unused columns to reduce memory footprint
    _keep_cols = ['h3_cell', 'lat', 'lon', 'geometry', 'FAR', 'BCR', 'CMI',
                  'transport_index', 'nature_index', 'urban_stress',
                  'diversity_score', 'cluster', 'terrain_flood_risk',
                  'elevation_m', 'green_space_ratio', 'flood_risk_tier',
                  'dominant_driver', 'vulnerability_score', 'segregation_score']
    merged_hex = merged_hex[[c for c in _keep_cols if c in merged_hex.columns]].copy()
    gc.collect()

    # Step 7: Advanced analytics
    with st.spinner("Computing advanced analytics…"):
        analysis_hex = merged_hex.copy()

        if run_stress:
            try:
                analysis_hex = compute_urban_stress_index(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute urban stress: {e}")

        if run_typology and 'fabric_typology' not in SKIP_MODULES:
            try:
                analysis_hex = compute_fabric_typology(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute fabric typology: {e}")

        if 'vulnerability' not in SKIP_MODULES:
            try:
                analysis_hex = compute_temporal_vulnerability(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute temporal vulnerability: {e}")

        if 'segregation' not in SKIP_MODULES:
            try:
                analysis_hex = compute_segregation_proxy(analysis_hex)
            except Exception as e:
                st.warning(f"Failed to compute segregation proxy: {e}")
    gc.collect()

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
        gc.collect()

    # Step 9: Scalar summaries
    total_buildings = (
        len(buildings_raw) if buildings_raw is not None and not buildings_raw.empty
        else len(buildings_gdf) if buildings_gdf is not None and not buildings_gdf.empty
        else 0
    )
    del buildings_raw
    gc.collect()
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
    st.session_state["city_center_lat"] = city_center_lat
    st.session_state["city_center_lon"] = city_center_lon

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

        if has_data(poi_df, min_rows=1):
            _safe_chart(chart_poi_dominance_map, poi_df, analysis_hex if not analysis_hex.empty else None,
                        key="t1_poi_dominance", ss_key="poi_dominance_map",
                        apply_admin_bounds=True)
            st.caption("Dominant POI functional zone per H3 hex cell (resolution 8, ~460m width). Each cell is coloured by its primary function — the category group with the highest POI count. Cells where no single group exceeds 40% of POIs are labelled 'Mixed', indicating genuinely multi-functional urban fabric. Source: Overture Maps Places. Groups: Food & Hospitality (restaurants, cafes, hotels), Retail & Commerce (shops, offices, banks, real estate), Health & Education (clinics, schools, salons), Culture & Leisure (museums, gyms, parks), Community & Services (government, religious, transport).")

        col_t1l, col_t1r = st.columns(2)
        with col_t1l:
            if has_data(category_series, min_rows=1):
                _safe_chart(chart_poi_distribution, category_series,
                            key="t1_poi", ss_key="poi_distribution")
                st.caption("Distribution of Points of Interest by functional category. Source: Overture Maps Places dataset (global POI database aggregating Meta, Microsoft, TomTom and OSM contributions). Each bar represents the count of POIs in that category within the city boundary. Look for dominant functions to understand the city's economic and service profile.")
        with col_t1r:
            if lu_metrics:
                _safe_chart(chart_landuse_composition, lu_metrics,
                            key="t1_landuse", ss_key="landuse_composition")
                st.caption("Donut chart of land use area composition across the city. Source: OSM landuse, leisure, and natural tags (polygonal features only). Percentages represent the share of mapped polygonal area — unmapped or unclassified areas are excluded. The green space indicator compares the city's green coverage against the EU urban average of 15%.")

        if has_data(poi_df, min_rows=10):
            _safe_chart(chart_poi_density_contour, poi_df,
                        st.session_state.get("city_polygon"),
                        key="t1_poi_contour", ss_key="poi_density_contour",
                        apply_admin_bounds=True)
            st.caption("POI activity density shown as an H3 hex heatmap (resolution 8, log-scaled colour). Source: Overture Maps Places. Each cell is coloured by the number of POIs it contains — warmer colours (orange to dark red) indicate higher concentrations of urban activity, revealing commercial cores and service clusters that stay readable at every zoom level.")

        if has_data(poi_df, min_rows=5) and city_center_lat != 0.0:
            _safe_chart(chart_nearest_services, poi_df, city_center_lat, city_center_lon,
                        key="t1_nearest_services", ss_key="nearest_services")
            st.caption("Distance from the city centre to the nearest facility of each service type, in kilometres. Green bars: within 5-minute walk (<0.5km); orange: within 15-minute walk (<1.5km); red: beyond walkable range. Source: Overture Maps Places. Computed using nearest-neighbour search from the city centroid.")

        _tab_export([
            ("poi_dominance_map",    "POI Dominance Map"),
            ("poi_distribution",     "POI Distribution"),
            ("landuse_composition",  "Land Use Composition"),
            ("poi_density_contour",  "POI Density Contour"),
            ("nearest_services",     "Nearest Services"),
        ], "tab1")

    # ── Tab 2: Morphology ─────────────────────────────────────────────────────
    with tab2:
        if not run_morphology:
            st.info("Module disabled — enable in sidebar")
        else:
            if has_data(hex_metrics, min_rows=1, required_cols=['FAR']):
                _safe_chart(chart_far_heatmap, hex_metrics,
                            key="t2_far", ss_key="far_heatmap", apply_admin_bounds=True)
                st.caption("Floor Area Ratio (FAR) per H3 hex cell: total built floor area divided by hex cell area. Source: OSM buildings (footprint area × estimated floors, where floors = height / 3.5m). FAR is a standard urban density metric — values above 2.0 indicate dense urban fabric; below 0.3 indicates suburban or low-rise areas.")

            if has_data(hex_metrics, min_rows=1, required_cols=['cluster']):
                _safe_chart(chart_morphotype_clusters, hex_metrics,
                            key="t2_clusters", ss_key="morphotype_clusters", apply_admin_bounds=True)
                st.caption("Urban morphotype classification derived from KMeans clustering (k=5) applied to six normalised features per H3 cell: floor-area ratio, building coverage ratio, median height, street density, POI density, and land-use diversity. Source: OSM buildings, OSMnx street network and Overture Maps POI. Clusters are labelled by their dominant profile: historic_core (high diversity, walkable mixed-use), dense_urban (highest FAR and height), residential (moderate FAR, low diversity), low_rise_commercial (high coverage, low height), suburban (low FAR and coverage, large plots).")

            if has_data(buildings_gdf, min_rows=5, required_cols=['height']):
                _safe_chart(chart_density_gradient, buildings_gdf,
                            poi_df=poi_df,
                            city_center_lat=city_center_lat, city_center_lon=city_center_lon,
                            key="t2_density_grad", ss_key="density_gradient")
                st.caption("Mean building height profiles along two perpendicular cross-sections through the city centre — north↔south (left, blue) and west↔east (right, purple) — in 0.3km bands, smoothed with a 3-band rolling average. Source: OSM buildings (height tag). The red dashed line marks the city centre. Compare the two profiles to spot directional asymmetries in the urban skyline — e.g. a taller core to the south or a denser corridor to the east.")

            if has_data(analysis_hex, min_rows=5) and city_center_lat != 0.0:
                _safe_chart(chart_morphological_transition, analysis_hex,
                            city_center_lat, city_center_lon,
                            key="t2_morph_transition", ss_key="morphological_transition")
                st.caption("Multi-line chart showing how four key urban metrics change with distance from the city centre in 500m bands: building density (FAR), green space ratio, transport accessibility, and POI diversity. All metrics normalised to 0–1 for comparability. Source: OSM buildings, OSM green space, OSM transit, and Overture Maps POI, aggregated to H3 hex cells and grouped by distance band. Vertical dotted lines mark 1km and 3km thresholds.")

            if _has_cols(analysis_hex, "transport_index", "nature_index"):
                _safe_chart(chart_urban_quality_index, analysis_hex,
                            key="t2_quality", ss_key="urban_quality")
                st.caption("Scatter plot of Transport Accessibility Index (x) versus Nature Index (y) for each H3 hex cell, sized by FAR and coloured by Composite Morphological Index. The dashed regression line and correlation coefficient (r) show how strongly transport and nature co-vary across the city. Source: OSM and Overture data. Quadrant labels classify cells by their combined performance profile.")

        _tab_export([
            ("far_heatmap",              "FAR Heatmap"),
            ("morphotype_clusters",      "Morphotype Clusters"),
            ("density_gradient",         "Density Gradient"),
            ("morphological_transition", "Morphological Transition"),
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

            if orientation_hist and any(h > 0 for h in orientation_hist):
                _safe_chart(chart_street_orientation, orientation_entropy, orientation_hist,
                            key="t3_orientation", ss_key="street_orientation")
                st.caption("Polar bar chart (wind rose) showing the distribution of street orientations in 36 bins of 5° each, mirrored for bidirectionality. Source: OSM street network via OSMnx, using bearing of each street segment. A uniform distribution indicates an isotropic organic network; strong peaks at 0°/90° indicate a cardinal grid. Shannon entropy value shown in title — lower entropy = more structured grid pattern.")
            if net_stats:
                _safe_chart(chart_street_network_radar, net_stats, city_name,
                            key="t3_radar", ss_key="street_radar")
                st.caption("Radar chart of seven street network metrics normalised to 0–1: connectivity (mean node degree), block size score (inverse of mean street length), intersection density, network efficiency (inverse circuity), dead-end ratio, orientation entropy (street direction diversity), and share of major roads. Source: OSM street network via OSMnx. A regular grid city scores high on connectivity and low on orientation entropy; an organic medieval network scores the opposite.")

            if graph is not None and city_center_lat != 0.0:
                _safe_chart(chart_street_centrality_edges, graph,
                            city_center_lat, city_center_lon,
                            key="t3_centrality", ss_key="street_centrality",
                            apply_admin_bounds=True)
                st.caption("Street network betweenness centrality mapped onto individual street segments, coloured from blue (low) to red (critical). Betweenness centrality measures how often each street lies on the shortest path between all pairs of nodes — high-centrality streets are structural movement corridors. Source: OSM street network via OSMnx and NetworkX. Computed using k=500 random sample pairs for performance. Dark background for visual contrast with the centrality gradient.")

        _tab_export([
            ("street_orientation", "Street Orientation"),
            ("street_radar",       "Street Network Radar"),
            ("street_centrality",  "Street Centrality"),
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
                if road_type_counts:
                    _safe_chart(chart_road_hierarchy, road_type_counts,
                                key="t4_road_hier", ss_key="road_hierarchy")
            with col_r:
                if has_data(transit_stops_gdf, min_rows=1):
                    _safe_chart(chart_transit_heatmap, transit_stops_gdf,
                                key="t4_transit_heat", ss_key="transit_heatmap",
                                apply_admin_bounds=True)

            if has_data(transport_hex, min_rows=1, required_cols=['transport_index']):
                _safe_chart(chart_transport_accessibility, transport_hex,
                            key="t4_transport_acc", ss_key="transport_map", apply_admin_bounds=True)
                st.caption("Composite transport accessibility index per H3 cell, combining: transit stop density within 400m walk (50% weight), cycling infrastructure coverage in km/km² (30%), and road type entropy (20%). Source: OSM public_transport, highway, and cycleway tags via OSMnx. Higher values (green) indicate well-served areas; red cells lack both transit and cycling infrastructure.")

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
                if has_data(nature_hex, min_rows=1, required_cols=['nature_index']):
                    _safe_chart(chart_green_space_access, nature_hex,
                                key="t5_green_access", ss_key="nature_map", apply_admin_bounds=True)
                    st.caption("Green space coverage ratio per H3 hex cell: proportion of cell area occupied by parks, forests, gardens, and natural vegetation. Source: OSM leisure=park/garden, landuse=forest/meadow/grass, natural=wood/scrub tags. Cells are coloured from red (no green cover) to dark green (high coverage). Use alongside the flood risk layer to identify green buffers near waterways.")
            with col_r:
                if has_data(nature_hex, min_rows=1, required_cols=['flood_risk_tier']):
                    _safe_chart(chart_flood_risk_zones, nature_hex,
                                key="t5_flood", ss_key="flood_risk_map", apply_admin_bounds=True)
                    st.caption("Flood risk classification based on proximity to waterways and terrain elevation. Source: OSM natural=water and waterway tags for water body locations; Open-Meteo Elevation API (SRTM 90m resolution) for terrain height. Risk tiers: high = within 50m of waterway or low elevation; medium = 50–200m; low = beyond 200m and elevated terrain. This is a proxy indicator — not an official flood hazard map.")

            col_l2, col_r2 = st.columns(2)
            with col_l2:
                if nature_metrics_dict:
                    _safe_chart(chart_nature_radar, nature_metrics_dict,
                                key="t5_nature_radar", ss_key="nature_radar")
            with col_r2:
                if has_data(poi_df, min_rows=1):
                    _safe_chart(
                        chart_15min_city_score,
                        poi_df, transit_stops_gdf, green_spaces_gdf,
                        city_center_lat, city_center_lon,
                        key="t5_15min", ss_key="city_15min",
                    )
                    st.caption("15-minute city accessibility score by service type: percentage of city area where each service category is reachable within a 1.25km walk (15 minutes at 5km/h). Source: Overture Maps POI for food, healthcare, education; OSM transit stops for public transit; OSM leisure polygons for green space. Scored using a 10×10 sample grid and nearest-neighbour distance. Green ≥ 70%, orange 40–70%, red < 40%.")

        # Terrain (independent of run_nature)
        if run_terrain and _has_cols(analysis_hex, "elevation_m", "terrain_flood_risk", min_rows=3):
            st.subheader("Terrain Analysis")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                _safe_chart(chart_terrain_elevation, analysis_hex,
                            key="t5_terrain_elev", ss_key="terrain_elevation",
                            apply_admin_bounds=True)
                st.caption("Terrain elevation in metres above sea level, sampled at H3 hex cell centroids from a regular grid. Source: Open-Meteo Elevation API using SRTM global digital elevation model at 90m resolution, queried via a 30×30 point grid over the city bounding box and interpolated to hex centroids via nearest-neighbour assignment.")
            with col_t2:
                _safe_chart(chart_terrain_flood_risk, analysis_hex,
                            key="t5_terrain_flood", ss_key="terrain_flood_risk",
                            apply_admin_bounds=True)
                st.caption("Terrain-based flood risk composite index per H3 cell, combining three components: elevation risk (40% weight, lower = higher risk), Topographic Wetness Index proxy (35%, computed as ln(cell_area / tan(slope))), and slope risk (25%, flatter terrain retains water). Source: Open-Meteo Elevation API / SRTM. Higher values indicate greater terrain-based flood susceptibility.")

            _safe_chart(chart_terrain_cross_buildings, analysis_hex,
                        key="t5_terrain_cross", ss_key="terrain_cross")
            st.caption("Scatter plot comparing building density (FAR) against terrain elevation for each H3 hex cell, with cell size proportional to building coverage ratio (BCR). Source: OSM buildings for FAR/BCR; Open-Meteo Elevation API for terrain height. The vertical line marks 10m elevation (approximate low-lying threshold); the horizontal line shows the city median FAR. Top-left quadrant (dense + low elevation) indicates highest combined risk.")

            col_t3, col_t4 = st.columns(2)
            with col_t3:
                _safe_chart(chart_twi_distribution, analysis_hex,
                            key="t5_twi", ss_key="twi_distribution")
                st.caption("Distribution of Topographic Wetness Index (TWI) values across the city's H3 hex cells. TWI = ln(contributing_area / tan(slope)), a standard hydrological index indicating areas prone to water accumulation. Source: computed from SRTM elevation data via Open-Meteo API using numpy gradient. Thresholds: TWI < 8 = dry/elevated; 8–12 = moderate; > 12 = flood-prone hollows.")
            with col_t4:
                _safe_chart(chart_slope_elevation_2d, analysis_hex,
                            key="t5_slope_elev", ss_key="slope_elevation")

        # Urban Heat Island
        if run_climate:
            st.subheader("Urban Heat Island")
            if climate_data and 'error' not in climate_data and 'uhi_delta' in analysis_hex.columns:
                _safe_chart(chart_heat_island_map, analysis_hex,
                            climate_data.get('temp_max_avg', 20),
                            key="t5_uhi_map", ss_key="heat_island_map",
                            apply_admin_bounds=True)
                st.caption("Urban Heat Island proxy per H3 hex cell, estimated as ΔTemperature from the city baseline. Formula: ΔT = BCR × 3.0 − green_ratio × 2.0, where BCR is building coverage ratio and green_ratio is green space fraction. Source: OSM buildings for BCR; OSM green space for green ratio; Open-Meteo 7-day forecast for baseline temperature. This is a structural proxy — not measured air temperature data.")

        _tab_export([
            ("nature_map",        "Green Space Access"),
            ("flood_risk_map",    "Flood Risk Zones"),
            ("nature_radar",      "Nature Radar"),
            ("terrain_elevation", "Terrain Elevation"),
            ("terrain_flood_risk","Terrain Flood Risk"),
            ("heat_island_map",   "Heat Island Map"),
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
                st.caption("3D scatter plot positioning each H3 hex cell in a three-dimensional space: transport accessibility (x), nature index (y), and inverse density (z, so low-density = high z). Colour indicates urban stress level. Source: OSM, Overture Maps, elevation data. The 'Optimal Zone' corner (high transport + high nature + low density) represents the most livable urban configuration — cells far from that corner are candidates for targeted investment.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    _safe_chart(chart_cross_heatmap_morph_transport, analysis_hex,
                                key="t6_cross_mt", ss_key="cross_morph_transport")
                    st.caption("Cross-tabulation matrix of Morphological Index quintiles (x) against Transport Accessibility quintiles (y), showing the count of H3 cells in each combination. Source: OSM buildings for morphology; OSM transit and cycling for transport. A concentration of cells in the top-right (dense + well-connected) indicates a compact transit-oriented city; concentration in bottom-left signals disconnected sprawl.")
                with col2:
                    _safe_chart(chart_cross_heatmap_nature_morph, analysis_hex,
                                key="t6_cross_nm", ss_key="cross_nature_density")
                    st.caption("Cross-tabulation of Green Space Ratio quintiles (x) against Building Density / FAR quintiles (y), coloured by median Composite Morphological Index. Source: OSM landuse and leisure for green space; OSM buildings for FAR. Green cells in the top-left (high nature + high density) are rare urban jewels — areas that manage to combine density with greenery. Red cells in the bottom-right indicate dense, nature-deprived zones.")
                with col3:
                    _safe_chart(chart_cross_heatmap_transport_nature, analysis_hex,
                                key="t6_cross_tn", ss_key="cross_transport_nature")
                    st.caption("Cross-tabulation of Transit Score quintiles (x) against Nature Index quintiles (y), showing concentration of H3 cells. Source: OSM transit stops for transit score; OSM green spaces and water proximity for nature index. The 'Transit-Rich Green Zones' annotation marks the ideal quadrant (top-right); 'Urban Stress Zones' marks areas with both poor transit and low nature access — typically peripheral mono-functional districts.")

                if has_data(crossref_df, min_rows=1):
                    _safe_chart(chart_landuse_crossref, crossref_df,
                                key="t6_lu_crossref", ss_key="landuse_crossref")
                    st.caption("Bubble chart comparing mean transport accessibility (x) and mean nature index (y) for each land use type, with bubble size proportional to total area in hectares. Source: OSM landuse polygons spatially joined to the H3 transport and nature hex grids. Reveals which land use types are systematically under-served by transport or lacking green access — useful for targeting policy interventions.")

                if _has_cols(analysis_hex, "transport_index", "nature_index", min_rows=10):
                    _safe_chart(chart_urban_quality_index, analysis_hex,
                                key="t6_quality", ss_key="urban_quality_cross")
                    st.caption("Scatter plot of Transport Accessibility Index (x) versus Nature Index (y) for each H3 hex cell, sized by FAR and coloured by Composite Morphological Index. The dashed regression line and correlation coefficient (r) show how strongly transport and nature co-vary across the city. Source: OSM and Overture data. Quadrant labels classify cells by their combined performance profile.")

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
                st.caption("Urban Stress Index per H3 cell — a composite of five equally-weighted components: building density stress (FAR-based), green space deficit (inverse nature index), transit deficit (inverse transport index), flood stress (terrain-based), and mono-functional stress (inverse POI diversity). Sources: OSM buildings, Overture POI, OSM transit, Open-Meteo elevation. Hover to see which driver dominates in each cell.")

                col_l, col_r = st.columns(2)
                with col_l:
                    _safe_chart(chart_urban_stress_decomposition, analysis_hex,
                                key="t7_stress_decomp", ss_key="stress_decomp")
                    st.caption("Decomposition of Urban Stress Index by stress class (low / moderate / high / critical), showing mean contribution of each of five components per class. Source: derived from OSM and Overture Maps data. Use this to understand what drives stress in different parts of the city — whether transit, density, green deficit, or flood risk is the primary factor.")
                with col_r:
                    if _has_cols(analysis_hex, "vulnerability_score", min_rows=5):
                        _safe_chart(chart_vulnerability_map, analysis_hex,
                                    key="t7_vuln_map", ss_key="vulnerability_map",
                                    apply_admin_bounds=True)
                        st.caption("Temporal Vulnerability Index per H3 cell: composite of flood risk (35%), building density (30%), transport isolation (20%), and proximity to water (15%). Source: OSM waterways, OSM buildings, transport accessibility index, Open-Meteo elevation. Darker red cells are both physically exposed to flood risk and poorly connected — indicating limited evacuation or emergency response capacity.")

                if _has_cols(analysis_hex, "urban_stress", min_rows=5):
                    _safe_chart(chart_urban_efficiency_pareto, analysis_hex,
                                key="t7_pareto", ss_key="stress_pareto")
                    st.caption("Pareto analysis of urban stress distribution: hex cells sorted from highest to lowest stress (left to right), with the red line showing cumulative stress accumulation. Source: Urban Stress Index computed from OSM and Overture data. The annotation shows what percentage of the city's area concentrates 80% of total urban stress — a standard spatial inequality metric used in urban policy analysis.")

                if _has_cols(analysis_hex, "urban_stress", "vulnerability_score", min_rows=5):
                    _safe_chart(chart_vulnerability_vs_stress, analysis_hex,
                                key="t7_vuln_vs_stress", ss_key="vuln_vs_stress")
                    st.caption("Scatter plot comparing Urban Stress Index (x-axis) against Temporal Vulnerability Index (y-axis) for each H3 cell, sized by FAR and coloured by vulnerability class. Source: both indices derived from OSM, Overture Maps, and SRTM elevation. Quadrant lines at 0.5/0.5 divide the city into four risk profiles — 'Double Risk Zone' (top-right) requires priority planning attention.")

        _tab_export([
            ("stress_map",       "Stress Map"),
            ("stress_decomp",    "Stress Decomposition"),
            ("stress_pareto",    "Stress Pareto"),
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
                    st.caption("16-cell Urban Fabric Typology Matrix classifying H3 hex cells by two dimensions: compactness (FAR, x-axis: sprawl → compact) and land use mix (POI Shannon entropy, y-axis: mono → vibrant). Source: OSM buildings for FAR; Overture Maps POI for diversity. Cell counts show how many hex cells fall in each fabric type — the dominant type reveals the city's overall urban character.")
                with col_r:
                    _safe_chart(chart_fabric_typology_map, analysis_hex,
                                key="t8_fabric_map", ss_key="fabric_map",
                                apply_admin_bounds=True)
                    st.caption("Spatial distribution of 16 urban fabric types across the city, based on the compactness × mix classification. Source: OSM buildings and Overture Maps POI, aggregated to H3 resolution 9. Each colour represents a distinct fabric archetype from 'Historic Mixed Core' (compact + vibrant) to 'Fringe / Industrial' (sprawl + mono). Use to identify which areas share similar planning challenges or opportunities.")

                if _has_cols(analysis_hex, "cluster", min_rows=5):
                    _safe_chart(chart_morphotype_radar_comparison, analysis_hex,
                                key="t8_radar_comp", ss_key="morphotype_radar")
                    st.caption("Radar (spider) chart comparing 5 urban morphotype clusters across 8 analytical dimensions: FAR, transport index, nature index, POI diversity, street connectivity, green ratio, flood safety, and normalised building height. Source: all OSM, Overture, and elevation sources combined. Each polygon represents the average profile of one morphotype cluster — the shape reveals what makes each zone type analytically distinct.")

                if _has_cols(analysis_hex, "segregation_score", min_rows=5):
                    _safe_chart(chart_segregation_map, analysis_hex,
                                key="t8_seg_map", ss_key="segregation_map",
                                apply_admin_bounds=True)
                    st.caption("Urban Segregation Proxy per H3 cell: composite of mono-functionality (40%), transit isolation (35%), and density isolation (25%, high FAR in mono-functional zone). Source: Overture Maps POI diversity, OSM transit stops, OSM buildings. This is a spatial proxy for functional segregation — not a socioeconomic segregation measure. High values indicate areas that are simultaneously dense, poorly served, and mono-functional.")

        _tab_export([
            ("fabric_matrix",   "Fabric Typology Matrix"),
            ("fabric_map",      "Fabric Typology Map"),
            ("morphotype_radar","Morphotype Radar"),
            ("segregation_map", "Segregation Map"),
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
            st.caption("Letter-grade scorecard for each administrative district across five dimensions: transport accessibility, nature access, morphological quality, urban stress, and flood risk. Grades: A (top 25%), B (25–45%), C (45–65%), D (bottom 35%). Source: H3 hex cell metrics spatially joined to OSM administrative boundaries (admin_level 9–10). Minimum 2 hex cells required per district to compute a grade. Districts sorted by overall performance.")

            with st.expander("Raw scores table"):
                st.dataframe(district_scores_df, width='stretch')

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

    # Free large data structures after figures are stored in session_state
    del buildings_gdf, buildings_sample, poi_df, landuse_gdf
    del transport_data, nature_data
    gc.collect()

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
