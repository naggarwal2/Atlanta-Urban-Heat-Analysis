import os
import streamlit as st
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests

st.set_page_config(page_title="Atlanta UHI Cooling Simulator", layout="wide")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8080")
BOUNDARY_FILE = "Official Atlanta Neighborhood Boundaries.geojson"
ARTIFACT_FILE = "sar_model_artifact.joblib"

SQ_M_PER_ACRE = 4046.86
CANOPY_SQ_M_PER_TREE = 50.0
TREE_PIT_IMPERVIOUS_OFFSET_SQ_M = 9.0


def c_to_f(temp_c: float) -> float:
    return (temp_c * 9.0 / 5.0) + 32.0


@st.cache_resource
def load_local_artifact():
    return joblib.load(ARTIFACT_FILE)


def run_local_sar_prediction(
    neighborhood_name: str,
    new_dev_acres: float,
    trees_added: int,
    parking_removed_acres: float,
    building_removed_acres: float,
    park_added_acres: float,
):
    """Standalone SAR equilibrium inference used when FastAPI backend is not reachable (e.g., Streamlit Cloud)."""
    artifact = load_local_artifact()
    df_ref = artifact["df_ref"]
    best_features = artifact["best_features"]
    scaler = artifact["scaler"]
    betas = artifact["betas"]
    W_ref = artifact["W_ref"]
    spatial_multiplier = artifact["spatial_multiplier"]

    nbhd_idx = int(df_ref.index[df_ref["NAME"] == neighborhood_name][0])
    df_sim = df_ref.copy()
    area = float(df_sim.loc[nbhd_idx, "area_sq_m"])

    # 1. Plant mature urban trees
    delta_canopy_trees = (trees_added * CANOPY_SQ_M_PER_TREE) / area
    delta_pit_imp = (trees_added * TREE_PIT_IMPERVIOUS_OFFSET_SQ_M) / area
    df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] + delta_canopy_trees, 0.0, 1.0
    )
    df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] = np.clip(
        df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] + delta_canopy_trees * 100.0, 0.0, 100.0
    )
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] - delta_pit_imp, 0.0, 1.0
    )

    # 2. Remove asphalt parking lot
    delta_imp_lot = (parking_removed_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] - delta_imp_lot, 0.0, 1.0
    )
    shift_pct_lot = min(float(df_sim.loc[nbhd_idx, "pct_nlcd_dev_high_24"]), delta_imp_lot * 100.0)
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_high_24"] -= shift_pct_lot
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_open_21"] += shift_pct_lot

    # 3. Remove building / structure footprint
    delta_imp_bldg = (building_removed_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] - delta_imp_bldg, 0.0, 1.0
    )
    target_pct_bldg = delta_imp_bldg * 100.0
    shift_med = min(float(df_sim.loc[nbhd_idx, "pct_nlcd_dev_med_23"]), target_pct_bldg)
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_med_23"] -= shift_med
    rem_pct_bldg = target_pct_bldg - shift_med
    shift_high = min(float(df_sim.loc[nbhd_idx, "pct_nlcd_dev_high_24"]), rem_pct_bldg)
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_high_24"] -= shift_high
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_open_21"] += shift_med + shift_high

    # 4. Add urban park / green space
    delta_park_share = (park_added_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] = np.clip(
        df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] + delta_park_share * 100.0, 0.0, 100.0
    )
    df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] + delta_park_share, 0.0, 1.0
    )

    # 5. New construction
    delta_dev_share = (new_dev_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] + delta_dev_share, 0.0, 1.0
    )
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_med_23"] = np.clip(
        df_sim.loc[nbhd_idx, "pct_nlcd_dev_med_23"] + delta_dev_share * 100.0, 0.0, 100.0
    )

    X_scaled = scaler.transform(df_sim[best_features].values)
    X_const = np.column_stack([np.ones(len(df_sim)), X_scaled])
    sim_temps_c = np.exp(spatial_multiplier @ (X_const @ betas))

    baseline_c = float(df_ref.loc[nbhd_idx, "baseline_pred_c"])
    simulated_c = float(sim_temps_c[nbhd_idx])
    delta_c = simulated_c - baseline_c

    neighbor_mask = W_ref[nbhd_idx] > 0
    avg_neighbor_delta_c = float(
        np.mean(sim_temps_c[neighbor_mask] - df_ref.loc[neighbor_mask, "baseline_pred_c"].values)
    )

    all_predictions = [
        {
            "NAME": str(df_ref.loc[i, "NAME"]),
            "baseline_pred_c": float(df_ref.loc[i, "baseline_pred_c"]),
            "simulated_pred_c": float(sim_temps_c[i]),
            "delta_c": float(sim_temps_c[i] - df_ref.loc[i, "baseline_pred_c"]),
        }
        for i in range(len(df_ref))
    ]

    return {
        "target_neighborhood": neighborhood_name,
        "baseline_temp_c": round(baseline_c, 4),
        "simulated_temp_c": round(simulated_c, 4),
        "temp_change_c": round(delta_c, 4),
        "avg_neighbor_spillover_c": round(avg_neighbor_delta_c, 5),
        "all_neighborhoods": all_predictions,
    }


@st.cache_data
def load_boundary_and_baseline():
    gdf = gpd.read_file(BOUNDARY_FILE)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    try:
        resp = requests.get(f"{API_URL}/baseline", timeout=3)
        resp.raise_for_status()
        df_base = pd.DataFrame(resp.json()["neighborhoods"])
        use_api = True
    except Exception:
        artifact = load_local_artifact()
        df_base = artifact["df_ref"].copy()
        use_api = False

    return gdf, df_base, use_api


def classify_heat_risk(temp_c, q33, q66):
    if temp_c >= q66:
        return "High Heat Risk", "#ff6b6b"
    elif temp_c >= q33:
        return "Moderate Heat Risk", "#ffd166"
    return "Low / Cool Zone", "#4ecdc4"


# --- Load Map Boundaries & Baseline Model State ---
try:
    gdf, df_base, use_api = load_boundary_and_baseline()
except Exception as e:
    st.error(f"Could not load boundary file '{BOUNDARY_FILE}' or model artifact '{ARTIFACT_FILE}'.\n\nDetails: {e}")
    st.stop()

q33, q66 = np.percentile(df_base["baseline_pred_c"], [33, 66])
neighborhood_names = sorted(df_base["NAME"].unique())

# Keep selected neighborhood synced between map clicks and dropdown
if "selected_nbhd" not in st.session_state:
    st.session_state.selected_nbhd = "Midtown" if "Midtown" in neighborhood_names else neighborhood_names[0]

st.title("Atlanta Urban Heat Island (UHI) & Cooling Simulator")
st.caption(
    "Interactive urban planning tool powered by a Spatial Autoregressive (SAR, k=12) model "
    "capturing direct surface impacts and neighbor spillover effects."
)

col_map, col_controls = st.columns([1.35, 1.0], gap="large")

with col_controls:
    st.subheader("1. Neighborhood Profile & Heat Drivers")
    selected_nbhd = st.selectbox(
        "Select Atlanta Neighborhood (or click directly on the map)",
        neighborhood_names,
        index=neighborhood_names.index(st.session_state.selected_nbhd),
    )
    st.session_state.selected_nbhd = selected_nbhd

    row = df_base.loc[df_base["NAME"] == selected_nbhd].iloc[0]
    base_temp_c = float(row["baseline_pred_c"])
    base_temp_f = c_to_f(base_temp_c)
    risk_tier, risk_color = classify_heat_risk(base_temp_c, q33, q66)

    m1, m2, m3 = st.columns([1.35, 1.15, 0.65])
    with m1:
        st.caption("Baseline Summer LST")
        st.markdown(
            f"<div style='font-size: 1.55rem; font-weight: 600; line-height: 1.25;'>"
            f"{base_temp_c:.2f} °C<br>"
            f"<span style='font-size: 1.15rem; font-weight: 400; opacity: 0.9;'>({base_temp_f:.2f} °F)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.caption("Heat Risk Tier")
        st.markdown(
            f"<div style='font-size: 1.35rem; font-weight: 600; color: {risk_color}; "
            f"white-space: normal; word-wrap: break-word; line-height: 1.25;'>"
            f"{risk_tier}</div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.caption("NPU")
        st.markdown(
            f"<div style='font-size: 1.55rem; font-weight: 600; line-height: 1.25;'>{row['NPU']}</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("Key Environmental Drivers in Neighborhood", expanded=True):
        area_acres = float(row["area_sq_m"]) / 4046.86
        st.write(f"- **Neighborhood Area:** {area_acres:,.1f} acres")
        st.write(f"- **Impervious Surface Intensity:** {row['impervious_surfaces_per_sq_meter_unit'] * 100:.1f}%")
        st.write(f"- **Tree Canopy Coverage:** {row['tree_canopy_per_sq_meter_unit'] * 100:.1f}%")
        st.write(
            f"- **Developed Land (Med + High Intensity):** "
            f"{row['pct_nlcd_dev_med_23'] + row['pct_nlcd_dev_high_24']:.1f}%"
        )
        st.write(f"- **Total Forest Cover:** {row['pct_nlcd_forest_total']:.1f}%")

    st.divider()
    st.subheader("2. 'What-If' Development & Mitigation Simulator")

    new_dev_acres = st.slider("New Construction / Impervious Footprint (acres)", 0.0, 20.0, 0.0, 0.5)
    trees_added = st.slider("Plant New Mature Trees (count)", 0, 1000, 0, 25)
    parking_removed = st.slider("Remove Asphalt Parking Lot (acres)", 0.0, 15.0, 0.0, 0.5)
    building_removed = st.slider("Remove Building / Structure Footprint (acres)", 0.0, 15.0, 0.0, 0.5)
    park_added = st.slider("Add New Urban Park / Green Space (acres)", 0.0, 20.0, 0.0, 0.5)

    if use_api:
        payload = {
            "neighborhood_name": selected_nbhd,
            "new_dev_acres": new_dev_acres,
            "trees_added": trees_added,
            "parking_removed_acres": parking_removed,
            "building_removed_acres": building_removed,
            "park_added_acres": park_added,
        }
        pred_resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        pred_resp.raise_for_status()
        sim_data = pred_resp.json()
    else:
        sim_data = run_local_sar_prediction(
            neighborhood_name=selected_nbhd,
            new_dev_acres=new_dev_acres,
            trees_added=trees_added,
            parking_removed_acres=parking_removed,
            building_removed_acres=building_removed,
            park_added_acres=park_added,
        )

    sim_temp_c = sim_data["simulated_temp_c"]
    sim_temp_f = c_to_f(sim_temp_c)
    delta_temp_c = sim_data["temp_change_c"]
    delta_temp_f = delta_temp_c * 9.0 / 5.0
    spillover_temp_c = sim_data["avg_neighbor_spillover_c"]

    r1, r2 = st.columns([1.25, 1.0])
    with r1:
        st.metric(
            "Simulated Neighborhood LST",
            f"{sim_temp_c:.2f} °C",
            delta=f"{delta_temp_c:+.3f} °C ({delta_temp_f:+.3f} °F)",
            delta_color="inverse",
        )
        st.markdown(
            f"<div style='font-size: 1.2rem; font-weight: 500; opacity: 0.9; margin-top: -6px;'>"
            f"({sim_temp_f:.2f} °F)</div>",
            unsafe_allow_html=True,
        )
    with r2:
        st.metric(
            "Avg. Neighbor Spillover (k=12)",
            f"{spillover_temp_c:+.4f} °C",
            help="Average temperature ripple across the 12 nearest neighboring areas via (I - ρW)⁻¹.",
        )

with col_map:
    st.subheader("Interactive Atlanta Neighborhood Map")
    map_layer = st.radio(
        "Map Color Metric",
        ["Simulated LST (°C)", "Temperature Change (°C)"],
        horizontal=True,
    )

    df_preds = pd.DataFrame(sim_data["all_neighborhoods"])
    df_preds = df_preds.rename(
        columns={
            "simulated_pred_c": "Simulated LST (°C)",
            "delta_c": "Temperature Change (°C)",
        }
    )

    gdf_plot = gdf.merge(df_preds, on="NAME", how="inner")
    color_scale = "RdYlBu_r" if map_layer == "Simulated LST (°C)" else "RdBu_r"
    color_midpoint = None if map_layer == "Simulated LST (°C)" else 0.0

    fig = px.choropleth_map(
        gdf_plot,
        geojson=gdf_plot.geometry.__geo_interface__,
        locations=gdf_plot.index,
        color=map_layer,
        hover_name="NAME",
        hover_data={
            "baseline_pred_c": ":.2f",
            "Simulated LST (°C)": ":.2f",
            "Temperature Change (°C)": ":+.4f",
        },
        color_continuous_scale=color_scale,
        color_continuous_midpoint=color_midpoint,
        map_style="carto-positron",
        zoom=10.3,
        center={"lat": 33.755, "lon": -84.390},
        opacity=0.75,
        height=720,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points")

    # Update selected neighborhood if user clicked a polygon on the map
    if selection and selection.get("selection") and selection["selection"].get("points"):
        clicked_idx = selection["selection"]["points"][0].get("point_index")
        if clicked_idx is not None and 0 <= clicked_idx < len(gdf_plot):
            clicked_name = str(gdf_plot.iloc[clicked_idx]["NAME"])
            if clicked_name != st.session_state.selected_nbhd:
                st.session_state.selected_nbhd = clicked_name
                st.rerun()