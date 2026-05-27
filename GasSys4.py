import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import requests, zipfile, os
from shapely.geometry import shape, box
import plotly.express as px
import json
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(layout="wide", page_title="US Biomass Gasification Explorer")

# ---------------------------------------------
# 🗺️ MAP LAYER BUILDER
# ---------------------------------------------
def build_choropleth_layer(merged_gdf, states_gdf, counties_gdf, layer_key):
    """
    Build a Plotly choropleth figure for the requested layer.
    layer_key: "plants" | "lcoh" | "lca" | "transport"
    Everything except the data trace is identical to the original map code.
    """
    fig = go.Figure()

    county_trace = px.choropleth_mapbox(
        counties_gdf,
        geojson=counties_gdf.__geo_interface__,
        locations=counties_gdf.index,
        color_discrete_sequence=["rgba(0,0,0,0)"],
    ).data[0]
    county_trace.marker.line.width = 0.2
    county_trace.marker.line.color = "gray"
    county_trace.showlegend = False
    fig.add_trace(county_trace)

    state_lines = states_gdf.boundary
    for geom in state_lines:
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                lon, lat = line.xy
                fig.add_trace(go.Scattermapbox(
                    lon=list(lon), lat=list(lat), mode="lines",
                    line=dict(color="black", width=1.5), hoverinfo="skip", showlegend=False,
                ))
        elif geom.geom_type == "LineString":
            lon, lat = geom.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(lon), lat=list(lat), mode="lines",
                line=dict(color="black", width=1.5), hoverinfo="skip", showlegend=False,
            ))

    if layer_key == "plants":
        category_colors = {
            "<1": "#FFFFFF", "≥1 and <2": "#F47C20", "≥2 and <3": "#FBCF86",
            "≥3 and <5": "#D1E7A0", "≥5 and <10": "#9ACD68", "≥10 and <20": "#62A652",
            "≥20 and <50": "#337C75", "≥50 and <100": "#235A82", "100+": "#1B3D8C",
        }
        data_fig = px.choropleth_mapbox(
            merged_gdf,
            geojson=merged_gdf.__geo_interface__,
            locations=merged_gdf.index,
            color="Plant Category",
            category_orders={"Plant Category": list(category_colors.keys())},
            color_discrete_map=category_colors,
            mapbox_style="carto-positron",
            opacity=0.8,
            hover_name="NAME",
            hover_data={"State": True, "Gasification Plants": True,
                        "Total Biomass (dry tons)": ":,.0f", "Plant Category": True},
            center={"lat": 37.8, "lon": -96},
            zoom=3.2,
        )
        for t in data_fig.data:
            t.marker.line.width = 0.3
            t.marker.line.color = "black"
            fig.add_trace(t)
        fig.update_layout(
            legend_title_text="Gasification Plants",
            legend=dict(
                x=0.98, y=0.5, xanchor="right", yanchor="middle",
                bgcolor="rgba(255,255,255,1)",
                font=dict(color="black"),
                title_font=dict(color="black", size=12),
            )
        )

    else:
        col_map = {
            "lcoh":      ("LCOH ($/kg H2)",                    "RdYlGn_r", "LCOH<br>($/kg H2)"),
            "lca":       ("Total LCA (kg CO2e/kg H2)",         "RdYlGn_r", "Total LCA<br>(kg CO2e/kg H2)"),
            "transport": ("Total Transportation Cost ($/dt)",  "YlOrRd",   "Transport Cost<br>($/dry ton)"),
        }
        col_name, colorscale, cb_title = col_map[layer_key]
        plot_df = merged_gdf.copy()
        plot_df["_val"] = pd.to_numeric(plot_df.get(col_name), errors="coerce")

        # Mask out counties with fewer than 1 gasification plant → display as white
        insufficient_mask = pd.to_numeric(
            plot_df.get("Gasification Plants", pd.Series(0, index=plot_df.index)),
            errors="coerce"
        ).fillna(0) < 1
        plot_df.loc[insufficient_mask, "_val"] = np.nan

        valid = plot_df["_val"].dropna()
        vmin = float(valid.quantile(0.05)) if len(valid) else 0
        vmax = float(valid.quantile(0.95)) if len(valid) else 1
        data_fig = px.choropleth_mapbox(
            plot_df,
            geojson=plot_df.__geo_interface__,
            locations=plot_df.index,
            color="_val",
            color_continuous_scale=colorscale,
            range_color=[vmin, vmax],
            mapbox_style="carto-positron",
            opacity=0.8,
            hover_name="NAME",
            hover_data={"State": True, col_name: ":.2f", "_val": False},
            center={"lat": 37.8, "lon": -96},
            zoom=3.2,
            labels={"_val": col_name},
        )
        data_fig.update_coloraxes(colorbar_title=cb_title)
        for t in data_fig.data:
            t.marker.line.width = 0.3
            t.marker.line.color = "black"
            fig.add_trace(t)
        fig.update_layout(coloraxis=data_fig.layout.coloraxis)

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center={"lat": 37.8, "lon": -96}, zoom=3.5, style="carto-positron"),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,1)",
        plot_bgcolor="rgba(0,0,0,1)",
        height=650,
    )
    return fig


# ---------------------------------------------
# ⚡ CACHED LOADERS
# ---------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(data_path):
    df = pd.read_csv(data_path)
    df["FIPSCODE"] = df["FIPSCODE"].astype(str).str.zfill(5)
    return df

@st.cache_data(show_spinner=True)
def load_boundaries():
    base_dir = "CountyData"
    os.makedirs(base_dir, exist_ok=True)

    # Correct 2021 URLs because connecticut changed counties after 2021 and BTRS uses 2021 boundaries
    urls = {
        "states": "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip",
        "counties": "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip",
    }

    paths = {}
    for name, url in urls.items():
        zip_name = os.path.basename(url)
        zip_path = os.path.join(base_dir, zip_name)
        extract_dir = os.path.join(base_dir, name)
        os.makedirs(extract_dir, exist_ok=True)

        # Force re-download if missing or corrupted
        if not os.path.exists(zip_path):
            print(f"📥 Downloading {name} shapefile...")
            r = requests.get(url)
            if r.status_code != 200:
                raise Exception(f"❌ Download failed for {url} (HTTP {r.status_code})")

            with open(zip_path, "wb") as f:
                f.write(r.content)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        shp_files = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
        match = next((f for f in shp_files), None)

        if match:
            paths[name] = os.path.join(extract_dir, match)
        else:
            raise Exception(f"❌ No .shp file found inside {extract_dir}")

    # Load shapefiles
    states = gpd.read_file(paths["states"]).to_crs("EPSG:4326")
    counties = gpd.read_file(paths["counties"]).to_crs("EPSG:4326")

    counties.rename(columns={"GEOID": "FIPSCODE"}, inplace=True)

    states["geometry"] = states["geometry"].simplify(0.05, preserve_topology=True)
    counties["geometry"] = counties["geometry"].simplify(0.03, preserve_topology=True)

    return states, counties

@st.cache_data(show_spinner=True)
def load_counties():
    states, counties = load_boundaries()
    return counties

@st.cache_data(show_spinner=True)
def merge_data(_counties_df, biomass_df):
    merged = _counties_df.merge(biomass_df, on="FIPSCODE", how="left")
    return merged

@st.cache_data(show_spinner=True)
def load_formations(formations_path):
    df = pd.read_csv(formations_path)
    df["formation_id"] = df["formation_id"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()
    df["uid"] = df["uid"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=True)
def load_eor_data(co2ecom_path, source_site_coords_path):
    """
    Build EOR formation lookup from co2_e_com.csv + source_site_coords.csv.
    Returns a DataFrame with columns:
        uid, co2_price_per_tonne, state, province,
        candidacy, capacity_flag, shovel_ready, lat, lon
    The co2_price_per_tonne is the storage cost from the CO2 source perspective
    (negative = operator pays for CO2, i.e. source earns revenue;
     positive = source pays operator to take CO2).
    Only fields with candidacy==YES-YES and capacity_flag==YES are valid storage sites.
    """
    ecom = pd.read_csv(co2ecom_path, skiprows=2, low_memory=False)

    # Province centroid coordinates
    coords = pd.read_csv(source_site_coords_path, skiprows=3, low_memory=False)
    prov_df = coords.iloc[:, [9, 10, 11]].copy()
    prov_df.columns = ["province", "lat", "lon"]
    prov_df = prov_df.dropna(subset=["province"])
    prov_df = prov_df[~prov_df["province"].isin(["Region (i.e., Province Name)", "EIA Basin"])]
    prov_df["lat"] = pd.to_numeric(prov_df["lat"], errors="coerce")
    prov_df["lon"] = pd.to_numeric(prov_df["lon"], errors="coerce")
    prov_df = prov_df.dropna()
    prov_df["province"] = prov_df["province"].str.strip()
    province_map = dict(zip(prov_df["province"], zip(prov_df["lat"], prov_df["lon"])))

    # Manual fallbacks for provinces not in coords table
    province_map.update({
        "Cambridge Arch-Central Kansas": (38.5, -98.0),
        "East Texas Basin/LA-MS Salt Basins": (32.0, -94.5),
        "Eastern Oregon-Wash.": (45.5, -119.0),
        "Bend Arch-Ft. Worth Basin": (32.5, -98.7),
        "Big Horn Basin": (44.0, -108.0),
        "Wyoming Thrust Belt": (42.5, -110.5),
    })

    rows = []
    for _, row in ecom.iterrows():
        uid = str(row.iloc[2]).strip()
        co2_price = pd.to_numeric(row.iloc[3], errors="coerce")
        state = str(row.iloc[7]).strip()
        province = str(row.iloc[6]).strip()
        candidacy = str(row.iloc[13]).strip()
        capacity = str(row.iloc[12]).strip()
        shovel_ready = str(row.iloc[14]).strip()
        lat, lon = province_map.get(province, (None, None))
        rows.append({
            "uid": uid,
            "formation_id": uid,   # align with SALINE schema
            "storage_type": "EOR",
            "co2_price_per_tonne": -co2_price if pd.notna(co2_price) else co2_price,  # flip sign: co2_e_com stores operator perspective, we need source perspective
            "state": state,
            "province": province,
            "candidacy": candidacy,
            "capacity_flag": capacity,
            "shovel_ready": shovel_ready,
            "lat": lat,
            "lon": lon,
        })

    df = pd.DataFrame(rows)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])

@st.cache_data(show_spinner=True)
def load_transport_costs(co2tcom_path):
    """
    CO2_T_COM CSV.
    Row 0: description text (skipped)
    Row 1: column headers — first col blank, second col is the vlookup key
            e.g. "0.1;100" / "10-0.1-100"
    The key we need is col B (index 1): the "Case number" style key
    Cost col: "First year break-even CO2 price in first year of project" (2018$)
    which is the column containing the per-tonne 2018$ breakeven price.
    """
    # Try skiprows=1 first (one description row), fall back to 0
    for skip in [1, 2, 0]:
        try:
            df = pd.read_csv(co2tcom_path, skiprows=skip)
            df.columns = df.columns.str.strip()
            cols = list(df.columns)

            # Find the key column: col B is index 1 after skipping
            # It contains values like "10-0.1-100", "20-0.1-100" etc.
            # Find it by looking for a column whose values match that pattern
            key_col = None
            for c in cols:
                sample = df[c].dropna().astype(str).head(5).tolist()
                if any("-" in v and v.count("-") >= 2 for v in sample):
                    key_col = c
                    break
            if key_col is None:
                # Fall back to second column
                key_col = cols[1]

            # Find the 2018$ cost column — it contains the breakeven price
            # From the data it is the column with header containing "first year of project"
            # or we fall back to the column at index position 18 (0-indexed)
            cost_col = None
            for c in cols:
                if "first year of project" in c.lower() or ("break" in c.lower() and "2018" in c.lower()):
                    cost_col = c
                    break
            if cost_col is None:
                # From the pasted T_COM data, the 2018$/tonne price is column index 18
                cost_col = cols[18]

            df["key"] = df[key_col].astype(str).str.strip()
            df["_cost"] = pd.to_numeric(df[cost_col], errors="coerce")
            result = dict(zip(df["key"], df["_cost"]))
            # Sanity check: should have hundreds of entries
            if len(result) > 50:
                return result
        except Exception:
            continue
    raise ValueError(
        f"Could not parse CO2_T_COM CSV at {co2tcom_path}. "
        "Please check that it is exported directly from the CO2_T_COM Results sheet "
        "with the first row being the description and the second row being column headers."
    )

@st.cache_data(show_spinner=True)
def load_storage_data(co2scom_path):
    """
    CO2_S_COM CSV.
    Key col (B): "{pisc}-{state}-{flow_rate}-{formation_id}-{capacity_factor}"
    Col C: Capacity flag YES/NO
    Col AK (index 37 from col A, 0-indexed): 2018$ cost
    Col AL (index 38): 2023$ cost

    Parses valid pisc, flow_rate, and capacity_factor values directly from the
    keys in the file so sidebar selectors always reflect what is actually available.
    """
    from collections import Counter

    df = pd.read_csv(co2scom_path, skiprows=4, low_memory=False)
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        df.columns[1]: "lookup_key",
        df.columns[2]: "capacity_flag",
        df.columns[37]: "cost_2018",
        df.columns[38]: "cost_2023",
    })

    df["lookup_key"] = df["lookup_key"].astype(str).str.strip()
    df["capacity_flag"] = df["capacity_flag"].astype(str).str.strip().str.upper()
    df["cost_2018"] = pd.to_numeric(df["cost_2018"], errors="coerce")
    df["cost_2023"] = pd.to_numeric(df["cost_2023"], errors="coerce")

    result = df[["lookup_key", "capacity_flag", "cost_2018", "cost_2023"]].copy()

    # Parse unique valid option values directly from keys in the file
    flow_counts = Counter()
    pisc_set, cap_set = set(), set()
    for key in result["lookup_key"].dropna():
        parts = key.split("-")
        if len(parts) >= 5:
            pisc_set.add(parts[0])
            flow_counts[parts[2]] += 1
            cap_set.add(parts[-1])

    # Flow rates appearing 300+ times are real user-selectable options (not site-specific computed values)
    valid_piscs = sorted(pisc_set, key=lambda x: int(x))
    valid_flows = sorted([f for f, c in flow_counts.items() if c >= 300], key=lambda x: float(x))
    # Only the three clean round capacity factors
    valid_caps = sorted([c for c in cap_set if float(c) in {80, 85, 100}], key=lambda x: float(x))

    return result, valid_piscs, valid_flows, valid_caps

# ---------------------------------------------
# 🧮 CO2 T&S ENGINE
# ---------------------------------------------
def mround(value, multiple):
    if multiple == 0:
        return value
    return round(value / multiple) * multiple

def compute_ts_costs(
    county_centroids_df,
    formations_df,
    transport_cost_table,
    storage_df,
    pipeline_type,
    tortuosity,
    pisc_years,
    ts_flow_rate,
    ts_capacity_factor,
    cost_year,
    max_distance_miles=1000,
):
    cost_col = "cost_2018" if cost_year == "2018" else "cost_2023"

    # Build fast storage lookup dict
    storage_lookup = {}
    for _, row in storage_df.iterrows():
        storage_lookup[row["lookup_key"]] = {
            "capacity": row["capacity_flag"],
            "cost": row[cost_col],
        }

    parallel_mult = {
        "Dedicated Pipeline": 1,
        "Trunkline": 1,
        "2 Parallel Trunklines": 2,
        "3 Parallel Trunklines": 3,
    }.get(pipeline_type, 1)

    # Pre-convert formations to numpy arrays for vectorized haversine
    fm_lats = np.radians(formations_df["lat"].values)
    fm_lons = np.radians(formations_df["lon"].values)
    fm_ids = formations_df["formation_id"].values
    fm_states = formations_df["state"].values

    def haversine_vectorized(clat_deg, clon_deg):
        clat = math.radians(clat_deg)
        clon = math.radians(clon_deg)
        cos_angle = (
            np.sin(fm_lats) * math.sin(clat) +
            np.cos(fm_lats) * math.cos(clat) * np.cos(fm_lons - clon)
        )
        np.clip(cos_angle, -1.0, 1.0, out=cos_angle)
        return 3443.8985 * np.arccos(cos_angle) * 1.15078

    def adjusted_dist(raw_miles):
        d = mround((1 + tortuosity) * raw_miles, 10)
        return max(d, 50)

    def feeder_dist(adj_miles):
        return min(mround(0.1 * adj_miles, 10), 50)

    def transport_key(dist_miles):
        return f"{int(dist_miles)}-{ts_flow_rate}-{ts_capacity_factor}"

    def storage_key(formation_id, state):
        return f"{pisc_years}-{state}-{ts_flow_rate}-{formation_id}-{ts_capacity_factor}"

    results = []

    for _, county in county_centroids_df.iterrows():
        fips = county["FIPSCODE"]
        clat = county["centroid_lat"]
        clon = county["centroid_lon"]

        raw_dists = haversine_vectorized(clat, clon)

        valid_mask = raw_dists <= max_distance_miles
        if not valid_mask.any():
            results.append({"FIPSCODE": fips, "T&S Cost ($/t)": None})
            continue

        best_ts = None
        best_row = None

        for i in np.where(valid_mask)[0]:
            raw_dist = raw_dists[i]
            adj = adjusted_dist(raw_dist)
            feed = feeder_dist(adj)

            main_key = transport_key(adj)
            feed_key = transport_key(feed)

            main_cost = transport_cost_table.get(main_key)
            if main_cost is None:
                continue

            if pipeline_type == "Dedicated Pipeline":
                transport_cost = main_cost
            else:
                feed_cost = transport_cost_table.get(feed_key)
                if feed_cost is None:
                    continue
                transport_cost = (main_cost * parallel_mult) + feed_cost + feed_cost

            skey = storage_key(fm_ids[i], fm_states[i])
            sdata = storage_lookup.get(skey)

            if (sdata is None
                    or sdata["capacity"] != "YES"
                    or sdata["cost"] is None
                    or (isinstance(sdata["cost"], float) and np.isnan(sdata["cost"]))):
                continue

            ts_cost = transport_cost + sdata["cost"]

            if best_ts is None or ts_cost < best_ts:
                best_ts = ts_cost
                best_row = {
                    "FIPSCODE": fips,
                    "Best Formation": fm_ids[i],
                    "Best Formation State": fm_states[i],
                    "Raw Distance (mi)": round(float(raw_dist), 1),
                    "Adjusted Distance (mi)": adj,
                    "Transport Cost ($/t)": round(float(transport_cost), 2),
                    "Storage Cost ($/t)": round(float(sdata["cost"]), 2),
                    "T&S Cost ($/t)": round(float(ts_cost), 2),
                }

        results.append(best_row if best_row else {"FIPSCODE": fips, "T&S Cost ($/t)": None})

    return pd.DataFrame(results)

# ---------------------------------------------
# 🏭 EOR T&S ENGINE
# ---------------------------------------------
def compute_ts_costs_eor(
    county_centroids_df,
    eor_df,
    transport_cost_table,
    pipeline_type,
    tortuosity,
    ts_flow_rate,
    ts_capacity_factor,
    max_distance_miles=1000,
):
    """
    EOR-specific T&S engine.
    - Transport cost: same pipeline haversine/lookup logic as SALINE.
    - Storage cost: co2_price_per_tonne from co2_e_com (the break-even CO2 price
      that the EOR operator charges/pays for CO2).
      Negative = operator pays source (revenue for source, reduces T&S cost).
      Positive = source pays operator (adds to T&S cost).
    - Only fields with candidacy==YES-YES and capacity_flag==YES are eligible.
    """
    parallel_mult = {
        "Dedicated Pipeline": 1,
        "Trunkline": 1,
        "2 Parallel Trunklines": 2,
        "3 Parallel Trunklines": 3,
    }.get(pipeline_type, 1)

    # Filter to eligible EOR fields only
    eligible = eor_df[
        (eor_df["candidacy"] == "YES-YES") &
        (eor_df["capacity_flag"] == "YES") &
        eor_df["co2_price_per_tonne"].notna() &
        eor_df["lat"].notna() &
        eor_df["lon"].notna()
    ].reset_index(drop=True)

    fm_lats = np.radians(eligible["lat"].values)
    fm_lons = np.radians(eligible["lon"].values)
    fm_ids  = eligible["formation_id"].values
    fm_states = eligible["state"].values
    fm_costs = eligible["co2_price_per_tonne"].values

    def haversine_vectorized(clat_deg, clon_deg):
        clat = math.radians(clat_deg)
        clon = math.radians(clon_deg)
        cos_angle = (
            np.sin(fm_lats) * math.sin(clat) +
            np.cos(fm_lats) * math.cos(clat) * np.cos(fm_lons - clon)
        )
        np.clip(cos_angle, -1.0, 1.0, out=cos_angle)
        return 3443.8985 * np.arccos(cos_angle) * 1.15078

    def adjusted_dist(raw_miles):
        return max(mround((1 + tortuosity) * raw_miles, 10), 50)

    def feeder_dist(adj_miles):
        return min(mround(0.1 * adj_miles, 10), 50)

    def transport_key(dist_miles):
        return f"{int(dist_miles)}-{ts_flow_rate}-{ts_capacity_factor}"

    results = []

    for _, county in county_centroids_df.iterrows():
        fips = county["FIPSCODE"]
        clat = county["centroid_lat"]
        clon = county["centroid_lon"]

        raw_dists = haversine_vectorized(clat, clon)
        valid_mask = raw_dists <= max_distance_miles

        if not valid_mask.any():
            results.append({"FIPSCODE": fips, "T&S Cost ($/t)": None})
            continue

        best_ts = None
        best_row = None

        for i in np.where(valid_mask)[0]:
            raw_dist = raw_dists[i]
            adj = adjusted_dist(raw_dist)
            feed = feeder_dist(adj)

            main_cost = transport_cost_table.get(transport_key(adj))
            if main_cost is None:
                continue

            if pipeline_type == "Dedicated Pipeline":
                transport_cost = main_cost
            else:
                feed_cost = transport_cost_table.get(transport_key(feed))
                if feed_cost is None:
                    continue
                transport_cost = (main_cost * parallel_mult) + feed_cost + feed_cost

            # EOR storage cost: CO2 price from operator perspective
            # Negative means operator pays for CO2 (free/revenue storage for source)
            storage_cost = float(fm_costs[i])
            ts_cost = transport_cost + storage_cost

            if best_ts is None or ts_cost < best_ts:
                best_ts = ts_cost
                best_row = {
                    "FIPSCODE": fips,
                    "Best Formation": fm_ids[i],
                    "Best Formation State": fm_states[i],
                    "Raw Distance (mi)": round(float(raw_dist), 1),
                    "Adjusted Distance (mi)": adj,
                    "Transport Cost ($/t)": round(float(transport_cost), 2),
                    "Storage Cost ($/t)": round(float(storage_cost), 2),
                    "T&S Cost ($/t)": round(float(ts_cost), 2),
                }

        results.append(best_row if best_row else {"FIPSCODE": fips, "T&S Cost ($/t)": None})

    return pd.DataFrame(results)

# ---------------------------------------------
# 📂 LOAD DATA
# ---------------------------------------------
GITHUB_RAW = "https://raw.githubusercontent.com/TyMcC123/Gas-Sys-Test/main"
data_path = f"{GITHUB_RAW}/AllGasSysData3.csv"
counties = load_counties()
biomass_df = load_data(data_path)
co2scom_path = f"{GITHUB_RAW}/co2scom.csv"
co2tcom_path = f"{GITHUB_RAW}/co2tcom.csv"
formations_path = f"{GITHUB_RAW}/formations.csv"
co2ecom_path = f"{GITHUB_RAW}/co2_e_com.csv"
source_site_coords_path = f"{GITHUB_RAW}/source_site_coords.csv"

_load_errors = []

try:
    transport_cost_table = load_transport_costs(co2tcom_path)
except Exception as e:
    transport_cost_table = {}
    _load_errors.append(f"CO2_T_COM failed to load ({co2tcom_path}): {e}")

try:
    storage_df, scom_piscs, scom_flows, scom_caps = load_storage_data(co2scom_path)
except Exception as e:
    storage_df = pd.DataFrame(columns=["lookup_key", "capacity_flag", "cost_2018", "cost_2023"])
    scom_piscs, scom_flows, scom_caps = ["10", "15", "50"], ["2.4"], ["100"]
    _load_errors.append(f"CO2_S_COM failed to load ({co2scom_path}): {e}")

try:
    formations_df = load_formations(formations_path)
except Exception as e:
    formations_df = pd.DataFrame(columns=["storage_type", "uid", "basin", "formation_id", "state", "lon", "lat"])
    _load_errors.append(f"formations.csv failed to load ({formations_path}): {e}")

try:
    eor_df = load_eor_data(co2ecom_path, source_site_coords_path)
except Exception as e:
    eor_df = pd.DataFrame(columns=["uid", "formation_id", "storage_type", "co2_price_per_tonne",
                                    "state", "province", "candidacy", "capacity_flag",
                                    "shovel_ready", "lat", "lon"])
    _load_errors.append(f"co2_e_com failed to load ({co2ecom_path}): {e}")

# ---------------------------------------------
# 🎛️ USER FILTERS (with Apply button)
# ---------------------------------------------
# Show any data load errors prominently at the top of the page
if _load_errors:
    for err in _load_errors:
        st.error(err)
    # Diagnostic: show actual column names of the T_COM CSV to help fix key mismatch
    with st.expander("🔍 Diagnostic: show raw CSV column names"):
        try:
            for skip in [1, 2, 0]:
                try:
                    _diag = pd.read_csv(co2tcom_path, skiprows=skip, nrows=3)
                    st.write(f"**co2tcom.csv** (skiprows={skip}) columns:")
                    st.write(list(_diag.columns))
                    st.dataframe(_diag.head(3))
                    break
                except Exception as _e:
                    st.write(f"skiprows={skip} failed: {_e}")
        except Exception as _e2:
            st.write(f"Diagnostic failed: {_e2}")
        try:
            for skip in [4, 3, 2, 0]:
                try:
                    _diag2 = pd.read_csv(co2scom_path, skiprows=skip, nrows=3)
                    st.write(f"**co2scom.csv** (skiprows={skip}) columns (first 10):")
                    st.write(list(_diag2.columns[:10]))
                    break
                except Exception:
                    continue
        except Exception as _e3:
            st.write(f"S_COM diagnostic failed: {_e3}")
biomass_types = ["Herbaceous", "Woody", "Forest", "Paper", "Plastic"]
scenarios = ["Near Term", "Emerging", "Mature Market Low", "Mature Market Medium", "Mature Market High"]
radii = ["20", "40", "60", "80"]

st.sidebar.header("Gasification Systems Tool")
st.sidebar.subheader("Press Apply Selections Button at Bottom for Selections to Update")
st.sidebar.header("Biomass Selections")
selected_biomass = st.sidebar.selectbox("Biomass Type", biomass_types)
selected_scenario = st.sidebar.selectbox("Economic Scenario", scenarios)
selected_radius = st.sidebar.selectbox("Radius (miles)", radii)

st.sidebar.header("Pretreatment Selections")

pretreatment_options = {
    "Torrefaction": 0.082,  # 8.2% moisture
    "Pelleting": 0.10       # 10% moisture
}

selected_pretreatment = st.sidebar.selectbox(
    "Pretreatment Type",
    list(pretreatment_options.keys())
)

# Moisture content for chosen treatment
moisture_value = pretreatment_options[selected_pretreatment]

# Display moisture content as a read-only box
st.sidebar.text_input(
    "Pretreated Moisture Content (%)",
    value=f"{moisture_value*100:.1f}%",
    disabled=True
)

technology_a = {
    "Pretreated Biomass Needed": 292000,
    "Raw Biomass Needed": 292000 / (1 - moisture_value)
}


st.sidebar.header("CO2 Transport & Storage Selections")

storage_formation_type = st.sidebar.selectbox(
    "Storage Formation Type",
    ["SALINE", "EOR"],
    index=0,
    help="SALINE: saline aquifer formations (314 sites). EOR: conventional oilfields for CO2-EOR (2,094 sites)."
)

pipeline_type = st.sidebar.selectbox(
    "Pipeline Type",
    ["Trunkline", "Dedicated Pipeline", "2 Parallel Trunklines", "3 Parallel Trunklines"],
    index=0
)

tortuosity = st.sidebar.slider(
    "Tortuosity Factor", min_value=0.0, max_value=0.5, value=0.15, step=0.05,
    help="Accounts for route deviation vs. straight line. 0.15 = 15% longer (default)."
)

ts_flow_rate = st.sidebar.selectbox(
    "CO2 Mass Flow Rate (Mtpa)",
    scom_flows,
    index=scom_flows.index("2.4") if "2.4" in scom_flows else 0,
    help="Options reflect values available in the loaded S_COM file."
)

ts_capacity_factor = st.sidebar.selectbox(
    "Pipeline Capacity Factor (%)",
    scom_caps,
    index=scom_caps.index("100") if "100" in scom_caps else 0,
    format_func=lambda x: f"{x}%",
)

pisc_years = st.sidebar.selectbox(
    "Post-Injection Site Care (years)",
    scom_piscs,
    index=scom_piscs.index("50") if "50" in scom_piscs else len(scom_piscs) - 1,
    format_func=lambda x: f"{x} years",
)

cost_year = st.sidebar.selectbox(
    "Cost Basis Year", ["2018", "2023"], index=0
)


# ✅ Add Apply button
apply_filters = st.sidebar.button("Apply Selections")

# ---------------------------------------------------------
# 🧮 Transportation Costs Dictionaries
# ---------------------------------------------------------
operational_data = {
    "operational_days_per_year": 292,
    "truck_labor_per_hr": 26.14,
    "diesel_costs_per_gallon": 3.18,
    "truck_depreciation_period_yrs": 5,
    "avg_speed_mph": 35,
    "loading_unloading_time_mins": 45,
}

annual_ownership_costs = {
    "total_annual_cost": 72454.56,
    "truck_payments": 35063.20,
    "tires": 8917.43,
    "maintenance_repair": 7760.52,
    "insurance": 7710.37,
    "shop": 3581.78,
    "support_personnel": 3428.95,
    "licenses_tags": 1873.27,
    "employment_screening": 241.17,
    "other": 3877.87
}

biomass_specs = {
    "Woody": {
        "max_weight_fed_reg": 80000,
        "semi_truck_cab_wt": 10000,
        "trailer_wt": 10625,
        "total_equip_wt": 20625,
        "truck_payload_tons": 29,
        "capacity_factor": 0.95
    },
    "Herbaceous": {
        "max_weight_fed_reg": 80000,
        "semi_truck_cab_wt": 10000,
        "trailer_wt": 10895,
        "total_equip_wt": 20895,
        "truck_payload_tons": 23,
        "capacity_factor": 0.95
    },
    "Paper": {
        "max_weight_fed_reg": 80000,
        "semi_truck_cab_wt": 10000,
        "trailer_wt": 12000,
        "total_equip_wt": 22000,
        "truck_payload_tons": 29,
        "capacity_factor": 0.95
    },
    "Forest": {
        "max_weight_fed_reg": 80000,
        "semi_truck_cab_wt": 10000,
        "trailer_wt": 14000,
        "total_equip_wt": 24000,
        "truck_payload_tons": 28,
        "capacity_factor": 0.95
    },
    "Plastic": {
        "max_weight_fed_reg": 80000,
        "semi_truck_cab_wt": 10000,
        "trailer_wt": 12000,
        "total_equip_wt": 22000,
        "truck_payload_tons": 29,
        "capacity_factor": 0.95
    }
}

logistics_requirements = {
    "avg_dist_to_plant_miles": selected_radius,
    "winding_factor": 1.2
}

dist = float(logistics_requirements["avg_dist_to_plant_miles"])
factor = float(logistics_requirements["winding_factor"])

avg_round_trip_miles =  2 * dist * factor

trip_time_hrs = (avg_round_trip_miles / operational_data["avg_speed_mph"]) + ((2 * operational_data["loading_unloading_time_mins"]) / 60)
trips_per_day = 1 if trip_time_hrs >= 8 else math.floor(8 / trip_time_hrs)

print(f"Trips per day: {trips_per_day}")
print(f"time per trip (hrs): {trip_time_hrs}")
print(f"Avg round trip miles: {avg_round_trip_miles}")
print(f"Avg one-way distance: {dist}")

# Only process and render when the button is pressed
if apply_filters:
    code_value = f"{selected_scenario} {selected_biomass} {selected_radius}"

    # ---------------------------------------------
    # 🧮 FILTER & CLEAN DATA
    # ---------------------------------------------
    filtered = biomass_df[biomass_df["Code"].str.strip().eq(code_value)].copy()

    # Convert biomass column to numeric BEFORE using it
    filtered["Total Biomass (dry tons)"] = (
        filtered["Total Biomass (dry tons)"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    filtered["Total Biomass (dry tons)"] = pd.to_numeric(
        filtered["Total Biomass (dry tons)"], errors="coerce"
    ).fillna(0)

    # Calculate plants AFTER conversion
    filtered["Gasification Plants"] = np.floor(filtered["Total Biomass (dry tons)"] / technology_a["Pretreated Biomass Needed"]).astype(int)

    # ---------------------------------------------------------
    # 🔗 FEEDSTOCK MC / LCA / COST LOOKUP TABLE (including Paper & Plastic)
    # ---------------------------------------------------------
    feedstock_table = pd.DataFrame({
        "Feedstock": [
            "Fire reduction thinnings", "Forest processing waste", "Logging residues", "Other forest waste",
            "Small diameter trees", "Barley straw", "Biomass sorghum", "Corn stover",
            "Cotton field residues", "Cotton gin trash", "Energy cane", "Miscanthus",
            "Oats straw", "Sorghum stubble", "Switchgrass", "Wheat straw",
            "Eucalyptus", "Willow", "Pine", "Poplar",
            "Pruning residues, citrus", "Pruning residues, noncitrus", "Pruning residues, tree nuts",
            "Paper", "Plastic"
        ],
        "MC": [
            0.3, 0.25, 0.2, 0.15, 0.15, 0.3, 0.25, 0.2,
            0.15, 0.15, 0.4, 0.15, 0.18, 0.4, 0.15, 0.16,
            0.543, 0.6, 0.57, 0.45,
            0.45, 0.45, 0.45,
            0.055, 0.02              
        ],
        "LCA": [
            30.21445636, 32.36756084, -1.428796954, -1.522533863, -1.522533863,
            30.21445636, 32.36756084, -1.428796954, -1.522533863, -1.522533863,
            12.41233907, 36.67376981, 35.38190712, 25.90824739, 44.66766991,
            36.24314891, -461.116656, -488.9808, -550.68552, -709.7013,
            -709.7013, -709.7013, -709.7013,
            4767.5628, 1901.592      
        ],
        "Cost": [
            70, 54, 40, 50, 58.54, 70, 70, 70, 60, 48, 70, 70, 70, 70, 70,
            70, 70, 70, 70, 70, 40, 40, 40,
            None, None               # ← Paper and plastic cost comes from dataframe, not table
        ]
    })

    feedstock_table.set_index("Feedstock", inplace=True)

    # ---------------------------------------------------------
    # 🧮 Calculate raw biomass, MC, LCA, cost including paper & plastic
    # ---------------------------------------------------------

    feedstock_cols = list(feedstock_table.index)

    paper_cost_col = "Paper Avg Price"
    plastic_cost_col = "Plastic Avg Cost"

    # Ensure columns exist
    for col in feedstock_cols + [paper_cost_col, plastic_cost_col]:
        if col not in filtered.columns:
            filtered[col] = 0

    # Convert numeric fields
    filtered[feedstock_cols] = filtered[feedstock_cols].apply(
        lambda x: pd.to_numeric(x, errors="coerce").fillna(0)
    )
    filtered[[paper_cost_col, plastic_cost_col]] = filtered[
        [paper_cost_col, plastic_cost_col]
    ].apply(lambda x: pd.to_numeric(x, errors="coerce").fillna(0))

    # ---------------------------------------------------------
    # 🌡️ Correct raw biomass calculation (dry + moisture)
    # raw_i = dry_i * (1 + MC_i)
    # ---------------------------------------------------------

    for col in feedstock_cols:
        mc_val = feedstock_table.loc[col, "MC"]
        filtered[f"raw_{col}"] = filtered[col] / (1 - mc_val)

    # Total raw biomass across all feedstocks
    raw_cols = [f"raw_{c}" for c in feedstock_cols]
    filtered["Raw Biomass (tons)"] = filtered[raw_cols].sum(axis=1)

    # ---------------------------------------------------------
    # 📊 Weighted MC (using raw biomass weighting, not dry)
    # ---------------------------------------------------------

    filtered["Weighted Average MC"] = (
        sum(filtered[f"raw_{col}"] * feedstock_table.loc[col, "MC"] for col in feedstock_cols)
        / filtered["Raw Biomass (tons)"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # ✨ New Pretreated Biomass Calculation
    # ---------------------------------------------------------
    filtered["Pretreated Biomass"] = filtered["Raw Biomass (tons)"] * (1 - (filtered["Weighted Average MC"] - moisture_value))

    # ---------------------------------------------------------
    # 🌎 Weighted LCA (kg CO2e/dt), using raw biomass
    # ---------------------------------------------------------

    filtered["Weighted Average LCA (kg CO2e/dt)"] = (
        sum(filtered[f"raw_{col}"] * feedstock_table.loc[col, "LCA"] for col in feedstock_cols)
        / filtered["Raw Biomass (tons)"].replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Number of trucks calculation
    # ---------------------------------------------------------

    filtered["Dry Biomass needed for one plant"] = technology_a["Raw Biomass Needed"] * (1 - filtered["Weighted Average MC"])

    # 1. Get the constants from your dictionaries
    op_days = operational_data["operational_days_per_year"]
    specs = biomass_specs.get(selected_biomass)

    # 2. Extract payload and capacity factor for the specific type
    payload = specs["truck_payload_tons"]
    cap_factor = specs["capacity_factor"]

    # 3. Calculate "Trucks per Day" for the whole column
   
    filtered["Loads Per Day"] = math.ceil(technology_a["Raw Biomass Needed"] / (payload * cap_factor) / op_days)

    filtered["Trucks Per Day"] = filtered["Loads Per Day"] / trips_per_day 
    
    filtered["Total Trucks per Day"] = filtered["Trucks Per Day"] + (0.1 * filtered["Trucks Per Day"])  # Adding 10% buffer

    filtered["Annual Cost of Trucks ($/yr)"] = filtered["Total Trucks per Day"] *  annual_ownership_costs["total_annual_cost"]

    filtered["Annual Fuel Cost ($/yr)"] = (filtered["Loads Per Day"] * operational_data["operational_days_per_year"]) * avg_round_trip_miles * operational_data["diesel_costs_per_gallon"] / 6.73

    filtered["Annual Labor Cost ($/yr)"] = (filtered["Loads Per Day"] * operational_data["operational_days_per_year"]) * (avg_round_trip_miles / operational_data["avg_speed_mph"] + 2 * operational_data["loading_unloading_time_mins"] / 60) * operational_data["truck_labor_per_hr"]

    filtered["Total Transportation Cost ($/dt)"] = (filtered["Annual Cost of Trucks ($/yr)"] + filtered["Annual Fuel Cost ($/yr)"] + filtered["Annual Labor Cost ($/yr)"])/filtered["Dry Biomass needed for one plant"]

    # ---------------------------------------------------------
    # 💰 COST CALCULATION
    # ---------------------------------------------------------

    # Feedstock cost (dry tons × cost)
    feedstock_cost_total = sum(
        filtered[col] * feedstock_table.loc[col, "Cost"]
        for col in feedstock_cols
        if pd.notna(feedstock_table.loc[col, "Cost"])
    )

    # Paper & plastic LCA/MC cost included in feedstock_table
    # Paper cost = dry tons × price
    paper_cost_total = filtered["Paper"] * filtered[paper_cost_col]

    # Plastic cost = dry tons × price
    plastic_cost_total = filtered["Plastic"] * filtered[plastic_cost_col]

    # Total cost from everything
    filtered["Total Cost ($)"] = (
        feedstock_cost_total
        + paper_cost_total
        + plastic_cost_total
    )

    # Weighted cost per dry ton
    filtered["Weighted Average Cost ($/dt)"] = (
        filtered["Total Cost ($)"]
        / filtered["Raw Biomass (tons)"].replace(0, np.nan)
    )

    # ---------------------------------------------
    # 🗺️ MERGE DATA
    # ---------------------------------------------
    merged = merge_data(counties, filtered)
    merged = merged[merged.geometry.notnull()]

    # Recalculate after merge for consistency
    merged["Total Biomass (dry tons)"] = pd.to_numeric(
        merged["Total Biomass (dry tons)"], errors="coerce"
    ).fillna(0)
    merged["Gasification Plants"] = np.floor(
        merged["Pretreated Biomass"] / technology_a["Pretreated Biomass Needed"]
    ).fillna(0).astype(int)

    # ---------------------------------------------
    # 🏭 CO2 T&S COST COMPUTATION
    # ---------------------------------------------
    merged["centroid_lat"] = merged.geometry.centroid.y
    merged["centroid_lon"] = merged.geometry.centroid.x
    county_centroids = merged[["FIPSCODE", "centroid_lat", "centroid_lon"]].drop_duplicates()

    if _load_errors:
        st.warning("T&S cost computation skipped — fix the data load errors shown above first.")
        ts_results = pd.DataFrame(columns=["FIPSCODE"])
    elif storage_formation_type == "EOR":
        with st.spinner("Computing lowest T&S cost per county against EOR formations..."):
            ts_results = compute_ts_costs_eor(
                county_centroids_df=county_centroids,
                eor_df=eor_df,
                transport_cost_table=transport_cost_table,
                pipeline_type=pipeline_type,
                tortuosity=tortuosity,
                ts_flow_rate=ts_flow_rate,
                ts_capacity_factor=ts_capacity_factor,
            )
    else:
        active_formations = formations_df[formations_df["storage_type"] == "SALINE"].copy()
        with st.spinner("Computing lowest T&S cost per county against SALINE formations..."):
            ts_results = compute_ts_costs(
                county_centroids_df=county_centroids,
                formations_df=active_formations,
                transport_cost_table=transport_cost_table,
                storage_df=storage_df,
                pipeline_type=pipeline_type,
                tortuosity=tortuosity,
                pisc_years=pisc_years,
                ts_flow_rate=ts_flow_rate,
                ts_capacity_factor=ts_capacity_factor,
                cost_year=cost_year,
            )

    merged = merged.merge(ts_results, on="FIPSCODE", how="left")

    # ---------------------------------------------
    # 🧮 TOTAL LCA CALCULATION (col S from LCA sheet)
    # S = (O + P + Q + R) / C28  [kg CO2e / kg H2]
    # O = land prep LCA:    M * N
    # P = transport LCA:    C2 * (1-L) * 907.2/0.621371 * N * one_way_dist
    # Q = gasification LCA: gasif_gwp * C28
    # R = T&S LCA:          E2 * 1000 * C44
    # ---------------------------------------------
    _LCA_C28  = 23190135.3          # kg H2/yr per plant (H2 output)
    _LCA_C44  = 447573.8            # kg CO2/yr captured per plant
    _LCA_C2   = 0.000117762475522534 # kg CO2e/(kg-km), transport GWP factor
    _LCA_E2   = 0.01094             # kg CO2e/kg CO2, T&S GWP factor
    _LCA_GASIF_GWP = {              # gasification GWP by biomass type
        "Woody":      20.21285714,
        "Herbaceous": 20.22454545,
        "Forest":     20.22,
        "Paper":      3.21,
        "Plastic":    3.21,
    }
    _gasif_gwp = _LCA_GASIF_GWP.get(selected_biomass, 20.22)
    _one_way_dist = dist * factor / 2   # avg one-way miles to plant

    # (LCA values computed on plants_df below after filtering)
    _pass = None  # placeholder

    # ---------------------------------------------
    # 💲 LCOH CONSTANTS (from LCOH.xlsx)
    # W = C49 + C50 + C51 + C52  — all fixed plant-model constants in 2022$
    #   C49 = Capital         = 2.5548 $/kg H2
    #   C50 = Fixed O&M       = 1.3472 $/kg H2
    #   C51 = Variable O&M    = 0.9264 $/kg H2
    #   C52 = Fuel (2022$)    = 2.2630 $/kg H2  (reference woody biomass price)
    # Y = T&S component = (ts_2018$/t * CPI2022/CPI2018 * C44) / C28
    # Z = W + Y  (LCOH $/kg H2 in 2022$)
    # ---------------------------------------------
    _LCOH_W      = 7.091399314857891   # C53: fixed sum C49+C50+C51+C52 (2022$)
    _LCOH_C28    = 23190135.315836746  # C28: annual H2 at actual CF (kg H2/yr)
    _LCOH_C44    = 447573.79526630434  # C44: CO2 captured at actual CF (kg CO2/yr)
    _LCOH_CPI_2018 = 251.1
    _LCOH_CPI_2022 = 292.7

    # ---------------------------------------------
    # 🔍 T&S DIAGNOSTIC
    # ---------------------------------------------
    with st.expander(f"🔍 T&S Diagnostic — {storage_formation_type}"):
        _matched = ts_results["T&S Cost ($/t)"].notna().sum() if "T&S Cost ($/t)" in ts_results.columns else 0
        _total = len(ts_results)
        st.write(f"**Counties with a valid T&S match: {_matched} / {_total}**")

        if storage_formation_type == "EOR":
            st.write(f"**Eligible EOR fields (YES-YES candidacy + YES capacity):** {len(eor_df[(eor_df['candidacy']=='YES-YES') & (eor_df['capacity_flag']=='YES')])}")
            st.write("**Sample EOR fields used:**")
            st.dataframe(eor_df[(eor_df["candidacy"]=="YES-YES") & (eor_df["capacity_flag"]=="YES")][
                ["formation_id","state","province","co2_price_per_tonne","lat","lon"]
            ].head(5))
        else:
            _sample_fm = formations_df[formations_df["storage_type"] == "SALINE"].head(3)
            _sample_keys = [
                f"{pisc_years}-{row['state']}-{ts_flow_rate}-{row['formation_id']}-{ts_capacity_factor}"
                for _, row in _sample_fm.iterrows()
            ]
            st.write("**Sample SALINE storage keys being generated:**")
            for k in _sample_keys:
                st.code(k)
            _scom_sample = storage_df.head(5)
            st.write("**Sample S_COM lookup keys:**")
            for k in _scom_sample["lookup_key"].tolist():
                st.code(k)

    # ---------------------------------------------
    # 🧮 BINNING INTO "Plant Category"
    # ---------------------------------------------
    def categorize_plants(x):
        if x < 1:
            return "<1"
        elif 1 <= x < 2:
            return "≥1 and <2"
        elif 2 <= x < 3:
            return "≥2 and <3"
        elif 3 <= x < 5:
            return "≥3 and <5"
        elif 5 <= x < 10:
            return "≥5 and <10"
        elif 10 <= x < 20:
            return "≥10 and <20"
        elif 20 <= x < 50:
            return "≥20 and <50"
        elif 50 <= x < 100:
            return "≥50 and <100"
        else:
            return "100+"

    merged["Plant Category"] = merged["Gasification Plants"].apply(categorize_plants)

    # Add state abbreviations
    state_fips_map = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
        "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
        "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
        "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
        "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
        "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
        "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
        "54": "WV", "55": "WI", "56": "WY",
    }
    merged["STATEFP"] = merged["STATEFP"].astype(str).str.zfill(2)
    merged["State"] = merged["STATEFP"].map(state_fips_map)

    # ---------------------------------------------
    # 🗺️ Compute LCA & LCOH on full merged df for map layers
    # ---------------------------------------------
    _mL = pd.to_numeric(merged["Weighted Average MC"], errors="coerce").fillna(0)
    _mM = pd.to_numeric(merged["Weighted Average LCA (kg CO2e/dt)"], errors="coerce").fillna(0)
    _mPlants_map = merged["Gasification Plants"].replace(0, np.nan).fillna(1)
    _mN = pd.to_numeric(merged["Raw Biomass (tons)"], errors="coerce").fillna(0) / _mPlants_map
    _mO = _mM * _mN
    _mP = _LCA_C2 * (1 - _mL) * 907.2 / 0.621371 * _mN * _one_way_dist
    _mQ = _gasif_gwp * _LCA_C28
    _mR = _LCA_E2 * 1000 * _LCA_C44
    merged["Total LCA (kg CO2e/kg H2)"] = (_mO + _mP + _mQ + _mR) / _LCA_C28
    _mTS = pd.to_numeric(merged.get("T&S Cost ($/t)", pd.Series(np.nan, index=merged.index)), errors="coerce")
    _mY = (_mTS * (_LCOH_CPI_2022 / _LCOH_CPI_2018) * _LCOH_C44) / _LCOH_C28
    merged["LCOH ($/kg H2)"] = _LCOH_W + _mY
    # Blank out counties with no raw biomass
    _no_biomass_mask = pd.to_numeric(merged["Raw Biomass (tons)"], errors="coerce").fillna(0) == 0
    merged.loc[_no_biomass_mask, "Total LCA (kg CO2e/kg H2)"] = np.nan
    merged.loc[_no_biomass_mask, "LCOH ($/kg H2)"] = np.nan

    category_colors = {
        "<1": "#FFFFFF",
        "≥1 and <2": "#F47C20",
        "≥2 and <3": "#FBCF86",
        "≥3 and <5": "#D1E7A0",
        "≥5 and <10": "#9ACD68",
        "≥10 and <20": "#62A652",
        "≥20 and <50": "#337C75",
        "≥50 and <100": "#235A82",
        "100+": "#1B3D8C",
    }

    states_gdf, counties_gdf = load_boundaries()
    counties_gdf["geometry"] = counties_gdf["geometry"].simplify(0.05, preserve_topology=True)

    fig = go.Figure()

    county_trace = px.choropleth_mapbox(
        counties_gdf,
        geojson=counties_gdf.__geo_interface__,
        locations=counties_gdf.index,
        color_discrete_sequence=["rgba(0,0,0,0)"],
    ).data[0]
    county_trace.marker.line.width = 0.2
    county_trace.marker.line.color = "gray"
    county_trace.showlegend = False
    fig.add_trace(county_trace)

    state_lines = states_gdf.boundary
    for geom in state_lines:
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                lon, lat = line.xy
                fig.add_trace(go.Scattermapbox(
                    lon=list(lon),
                    lat=list(lat),
                    mode="lines",
                    line=dict(color="black", width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                ))
        elif geom.geom_type == "LineString":
            lon, lat = geom.xy
            fig.add_trace(go.Scattermapbox(
                lon=list(lon),
                lat=list(lat),
                mode="lines",
                line=dict(color="black", width=1.5),
                hoverinfo="skip",
                showlegend=False,
            ))

    biomass_fig = px.choropleth_mapbox(
        merged,
        geojson=merged.__geo_interface__,
        locations=merged.index,
        color="Plant Category",
        category_orders={"Plant Category": list(category_colors.keys())},
        color_discrete_map=category_colors,
        mapbox_style="carto-positron",
        opacity=0.8,
        hover_name="NAME",
        hover_data={
            "State": True,
            "Gasification Plants": True,
            "Total Biomass (dry tons)": ":,.0f",
            "Plant Category": True,
        },
        center={"lat": 37.8, "lon": -96},
        zoom=3.2,
    )

    for t in biomass_fig.data:
        t.marker.line.width = 0.3
        t.marker.line.color = "black"
        fig.add_trace(t)

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center={"lat": 37.8, "lon": -96}, zoom=3.5, style="carto-positron"),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,1)",
        plot_bgcolor="rgba(0,0,0,1)",
        height=650,
        legend_title_text="Gasification Plants",
        legend=dict(
            x=0.98,
            y=0.5,
            xanchor="right",
            yanchor="middle",
            bgcolor="rgba(255,255,255,1)",
            font=dict(color="black"),
            title_font=dict(color="black", size=12)
        )
    )

    # Save computed data to session_state so the map layer selector
    # can redraw without re-running the full apply_filters block.
    st.session_state["_map_fig"] = fig
    st.session_state["_map_merged"] = merged
    st.session_state["_map_states_gdf"] = states_gdf
    st.session_state["_map_counties_gdf"] = counties_gdf

    merged["State"] = merged["STATEFP"].map(state_fips_map)
    merged["County"] = merged["NAME"]

    _ts_cols = ["Best Formation", "Best Formation State", "Best Formation Province",
                "Raw Distance (mi)", "Transport Cost ($/t)", "Storage Cost ($/t)", "T&S Cost ($/t)"]
    _base_cols_all = ["County", "State", "Total Biomass (dry tons)", "Gasification Plants",
                      "Weighted Average Cost ($/dt)", "Weighted Average MC",
                      "Weighted Average LCA (kg CO2e/dt)",
                      "Raw Biomass (tons)", "Pretreated Biomass",
                      "Trucks Per Day", "Total Transportation Cost ($/dt)"]
    _available_ts = [c for c in _ts_cols if c in merged.columns]
    _base_cols = [c for c in _base_cols_all if c in merged.columns]
    plants_df = merged.loc[
        merged["Gasification Plants"] >= 1,
        _base_cols + _available_ts
    ].copy()
    plants_df = plants_df.sort_values(by="Gasification Plants", ascending=False)

    # Ensure Raw Biomass is numeric before using it
    plants_df["Raw Biomass (tons)"] = (
        plants_df["Raw Biomass (tons)"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    plants_df["Raw Biomass (tons)"] = pd.to_numeric(
        plants_df["Raw Biomass (tons)"],
        errors="coerce"
    )

    # ---------------------------------------------
    # 🧮 TOTAL LCA (computed on plants_df to avoid GeoDataFrame cache issues)
    # ---------------------------------------------
    _pL = pd.to_numeric(plants_df["Weighted Average MC"], errors="coerce").fillna(0)
    _pM = pd.to_numeric(plants_df["Weighted Average LCA (kg CO2e/dt)"], errors="coerce").fillna(0)
    _pPlants = plants_df["Gasification Plants"].replace(0, np.nan).fillna(1)
    _pN = plants_df["Raw Biomass (tons)"].fillna(0) / _pPlants
    _pO = _pM * _pN
    _pP = _LCA_C2 * (1 - _pL) * 907.2 / 0.621371 * _pN * _one_way_dist
    _pQ = _gasif_gwp * _LCA_C28
    _pR = _LCA_E2 * 1000 * _LCA_C44
    plants_df["Total LCA (kg CO2e/kg H2)"] = (_pO + _pP + _pQ + _pR) / _LCA_C28

    # ---------------------------------------------
    # 💲 LCOH CALCULATION (column Z from LCOH.xlsx)
    # W = fixed plant-model cost (Capital + Fixed O&M + Variable O&M + Fuel), all 2022$
    #     = C49 + C50 + C51 + C52 = 7.0914 $/kg H2  (constant for all counties)
    # Y = per-county T&S contribution in $/kg H2
    #     = (ts_cost_2018$/t * CPI2022/CPI2018 * C44_kg_CO2/yr) / C28_kg_H2/yr
    # Z = W + Y
    # ---------------------------------------------
    _pTS_lcoh = (
        pd.to_numeric(plants_df["T&S Cost ($/t)"], errors="coerce")
        if "T&S Cost ($/t)" in plants_df.columns
        else pd.Series(np.nan, index=plants_df.index)
    )
    _pY_lcoh = (_pTS_lcoh * (_LCOH_CPI_2022 / _LCOH_CPI_2018) * _LCOH_C44) / _LCOH_C28
    plants_df["LCOH ($/kg H2)"] = _LCOH_W + _pY_lcoh

    # Ensure plants_df is a clean standalone DataFrame before reorder and display
    plants_df = plants_df.copy()

    # Reorder columns so Total LCA and LCOH appear after Gasification Plants
    _col_order = ["County", "State", "Total Biomass (dry tons)", "Gasification Plants","Raw Biomass (tons)", "Pretreated Biomass",
                  "Weighted Average Cost ($/dt)", "Weighted Average MC",
                  "Trucks Per Day", "Total Transportation Cost ($/dt)",
                  "Best Formation", "Best Formation State", "Best Formation Province",
                  "Raw Distance (mi)", "Transport Cost ($/t)", "Storage Cost ($/t)",
                  "Weighted Average LCA (kg CO2e/dt)", "T&S Cost ($/t)","Total LCA (kg CO2e/kg H2)", "LCOH ($/kg H2)"]
    _col_order_present = [c for c in _col_order if c in plants_df.columns]
    _extra_cols = [c for c in plants_df.columns if c not in _col_order_present]
    plants_df = plants_df[_col_order_present + _extra_cols]

    # Save plants_df to session_state so the table persists across layer changes
    st.session_state["_map_plants_df"] = plants_df


# ---------------------------------------------
# 🗺️ MAP LAYER SELECTOR (outside apply_filters so layer changes don't reset)
# ---------------------------------------------
_layer_labels = {
    "plants":    "🏭 Gasification Plants",
    "lcoh":      "💲 LCOH ($/kg H₂)",
    "lca":       "🌿 Total LCA (kg CO₂e/kg H₂)",
    "transport": "🚛 Biomass Transportation Cost ($/dt)",
}
_selected_layer = st.radio(
    label="Select map layer:",
    options=list(_layer_labels.keys()),
    format_func=lambda k: _layer_labels[k],
    horizontal=True,
    label_visibility="collapsed",
)

if "_map_fig" in st.session_state:
    if _selected_layer == "plants":
        st.plotly_chart(st.session_state["_map_fig"], use_container_width=True)
    else:
        _layer_fig = build_choropleth_layer(
            st.session_state["_map_merged"],
            st.session_state["_map_states_gdf"],
            st.session_state["_map_counties_gdf"],
            _selected_layer,
        )
        st.plotly_chart(_layer_fig, use_container_width=True)
else:
    st.info("👆 Select your filters in the sidebar and click **Apply Filters** to update the map and table.")

# ---------------------------------------------
# 📋 DATA TABLE (outside apply_filters so it persists across layer changes)
# ---------------------------------------------
if "_map_plants_df" in st.session_state:
    _display_df = st.session_state["_map_plants_df"]
    st.markdown("### Counties Supporting One or More Gasification Plants")
    if len(_display_df) > 0:
        _numeric_cols = [
            "Total Biomass (dry tons)",
            "Gasification Plants",
            "Weighted Average Cost ($/dt)",
            "Weighted Average MC",
            "Weighted Average LCA (kg CO2e/dt)",
            "Raw Biomass (tons)",
            "Pretreated Biomass",
            "Trucks Per Day",
            "Total Transportation Cost ($/dt)",
            "Total LCA (kg CO2e/kg H2)",
            "LCOH ($/kg H2)",
            "Raw Distance (mi)",
            "Transport Cost ($/t)",
            "Storage Cost ($/t)",
            "T&S Cost ($/t)",
        ]
        for col in _numeric_cols:
            if col in _display_df.columns:
                _display_df[col] = pd.to_numeric(
                    _display_df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(),
                    errors="coerce"
                )
        _all_formats = {
            "Total Biomass (dry tons)": "{:,.0f}",
            "Gasification Plants": "{:.0f}",
            "Weighted Average Cost ($/dt)": "${:,.2f}",
            "Weighted Average MC": "{:.1%}",
            "Weighted Average LCA (kg CO2e/dt)": "{:.1f}",
            "Raw Biomass (tons)": "{:,.0f}",
            "Pretreated Biomass": "{:,.0f}",
            "Trucks Per Day": "{:.0f}",
            "Total Transportation Cost ($/dt)": "${:,.2f}",
            "Total LCA (kg CO2e/kg H2)": "{:,.2f}",
            "LCOH ($/kg H2)": "${:,.2f}",
            "Raw Distance (mi)": "{:,.1f}",
            "Transport Cost ($/t)": "${:,.2f}",
            "Storage Cost ($/t)": "${:,.2f}",
            "T&S Cost ($/t)": "${:,.2f}",
        }
        _fmt = {k: v for k, v in _all_formats.items() if k in _display_df.columns}
        def highlight_ts_cols(styler):
            ts_highlight_cols = [
                c for c in [
                    "Best Formation", "Best Formation State", "Best Formation Province",
                    "Raw Distance (mi)", "Transport Cost ($/t)", "Storage Cost ($/t)", "T&S Cost ($/t)"
                ]
                if c in styler.data.columns
            ]
            transport_highlight_cols = [
                c for c in [
                    "Trucks Per Day", "Total Feedstock Transportation Cost ($/dt)"
                ]
                if c in styler.data.columns
            ]
            lca_highlight_cols = [
                c for c in [
                    "Feedstock Weighted Average LCA (kg CO2e/dt)",
                    "Total LCA (kg CO2e/kg H2)",
                    "LCOH ($/kg H2)",
                ]
                if c in styler.data.columns
            ]
            first_8_cols = list(styler.data.columns[:8])
            return styler.set_properties(
                subset=first_8_cols,
                **{"background-color": "#d4edda", "color": "black"}
            ).set_properties(
                subset=ts_highlight_cols,
                **{"background-color": "#d0eaf8", "color": "black"}
            ).set_properties(
                subset=transport_highlight_cols,
                **{"background-color": "#fff9c4", "color": "black"}
            ).set_properties(
                subset=lca_highlight_cols,
                **{"background-color": "#ffe0b2", "color": "black"}
            ).set_properties(
                **{"border": "1px solid black"}
            )



        
        _col_rename = {
            "Weighted Average LCA (kg CO2e/dt)": "Feedstock Weighted Average LCA (kg CO2e/dt)",
            "Weighted Average Cost ($/dt)": "Feedstock Weighted Average Cost ($/dt)",
            "Weighted Average MC": "Feedstock Weighted Average MC",
            "Pretreated Biomass": "Pretreated Biomass (tons)",
            "Total Transportation Cost ($/dt)": "Total Feedstock Transportation Cost ($/dt)",
        }

        _display_col_order = [
            "County", "State", "Total Biomass (dry tons)", "Gasification Plants",
            "Raw Biomass (tons)", "Pretreated Biomass (tons)",
            "Feedstock Weighted Average Cost ($/dt)", "Feedstock Weighted Average MC",
            "Trucks Per Day", "Total Feedstock Transportation Cost ($/dt)",
            "Best Formation", "Best Formation State", "Best Formation Province",
            "Raw Distance (mi)", "Transport Cost ($/t)", "Storage Cost ($/t)", "T&S Cost ($/t)",
            "Feedstock Weighted Average LCA (kg CO2e/dt)", 
            "Total LCA (kg CO2e/kg H2)", "LCOH ($/kg H2)",
        ]

        _renamed_df = _display_df.rename(columns=_col_rename)
        _display_col_order_present = [c for c in _display_col_order if c in _renamed_df.columns]

        st.dataframe(
            _renamed_df[_display_col_order_present]
            .style.format({_col_rename.get(k, k): v for k, v in _fmt.items()})
            .pipe(highlight_ts_cols),
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No counties in the current selection can support a full gasification plant.")
        _layer_labels