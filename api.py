import time
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# -------------------------------------------------------------
# Step 7: Configure Production Monitoring & Inference Logging
# -------------------------------------------------------------
LOG_FILE = "inference.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("atlanta_uhi_api")

app = FastAPI(
    title="Atlanta UHI Cooling Simulator API",
    description="Spatial Autoregressive (SAR, k=12) inference service for urban heat island mitigation.",
    version="1.2.1",
)

# Load serialized SAR model artifact once at startup
artifact = joblib.load("sar_model_artifact.joblib")
df_ref = artifact["df_ref"]
best_features = artifact["best_features"]
scaler = artifact["scaler"]
betas = artifact["betas"]
W_ref = artifact["W_ref"]
spatial_multiplier = artifact["spatial_multiplier"]

SQ_M_PER_ACRE = 4046.86
CANOPY_SQ_M_PER_TREE = 50.0
TREE_PIT_IMPERVIOUS_OFFSET_SQ_M = 9.0  # ~3m x 3m permeable tree cutout replacing pavement


class SimulationRequest(BaseModel):
    neighborhood_name: str = Field(..., example="Midtown")
    new_dev_acres: float = Field(0.0, ge=0.0, description="New impervious construction footprint (acres)")
    trees_added: int = Field(0, ge=0, description="Number of mature urban trees planted")
    parking_removed_acres: float = Field(0.0, ge=0.0, description="Asphalt parking lot removed (acres)")
    building_removed_acres: float = Field(0.0, ge=0.0, description="Building / structure footprint removed (acres)")
    park_added_acres: float = Field(0.0, ge=0.0, description="New urban park / green space added (acres)")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_metadata": artifact["metadata"],
        "neighborhood_count": len(df_ref),
    }


@app.get("/baseline")
def get_baseline():
    """Returns baseline predicted temperatures and environmental drivers for all neighborhoods."""
    records = df_ref[
        [
            "NAME",
            "NPU",
            "area_sq_m",
            "baseline_pred_c",
            "impervious_surfaces_per_sq_meter_unit",
            "tree_canopy_per_sq_meter_unit",
            "pct_nlcd_dev_open_21",
            "pct_nlcd_dev_low_22",
            "pct_nlcd_dev_med_23",
            "pct_nlcd_dev_high_24",
            "pct_nlcd_forest_total",
        ]
    ].to_dict(orient="records")
    return {"neighborhoods": records}


@app.get("/logs")
def get_recent_logs(limit: int = 20):
    """Returns the most recent inference log entries for live monitoring."""
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        return {"entries": []}
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return {"total_logged_events": len(lines), "recent_entries": lines[-limit:]}


@app.post("/predict")
def predict(req: SimulationRequest):
    start_time = time.time()

    matches = df_ref.index[df_ref["NAME"] == req.neighborhood_name].tolist()
    if not matches:
        logger.error(f"NeighborhoodNotFound | input='{req.neighborhood_name}'")
        raise HTTPException(status_code=404, detail=f"Neighborhood '{req.neighborhood_name}' not found.")

    nbhd_idx = int(matches[0])
    df_sim = df_ref.copy()
    area = float(df_sim.loc[nbhd_idx, "area_sq_m"])
    area_acres = area / SQ_M_PER_ACRE

    # Basic input drift / physical boundary check
    total_modified_acres = (
        req.new_dev_acres + req.parking_removed_acres + req.building_removed_acres + req.park_added_acres
    )
    if total_modified_acres > area_acres:
        logger.warning(
            f"InputDriftWarning | Neighborhood='{req.neighborhood_name}' | "
            f"ModifiedAcres={total_modified_acres:.2f} exceeds NeighborhoodAreaAcres={area_acres:.2f}"
        )

    # 1. Plant mature urban trees -> increases canopy + vegetative cover & offsets tree-pit impervious surface
    delta_canopy_trees = (req.trees_added * CANOPY_SQ_M_PER_TREE) / area
    delta_pit_imp = (req.trees_added * TREE_PIT_IMPERVIOUS_OFFSET_SQ_M) / area
    df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] + delta_canopy_trees, 0.0, 1.0
    )
    df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] = np.clip(
        df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] + delta_canopy_trees * 100.0, 0.0, 100.0
    )
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] - delta_pit_imp, 0.0, 1.0
    )

    # 2. Remove asphalt parking lot -> reduces impervious intensity & shifts high-intensity dev to open space
    delta_imp_lot = (req.parking_removed_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] - delta_imp_lot, 0.0, 1.0
    )
    shift_pct_lot = min(float(df_sim.loc[nbhd_idx, "pct_nlcd_dev_high_24"]), delta_imp_lot * 100.0)
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_high_24"] -= shift_pct_lot
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_open_21"] += shift_pct_lot

    # 3. Remove building / structure footprint -> reduces impervious intensity & shifts med/high dev to open space
    delta_imp_bldg = (req.building_removed_acres * SQ_M_PER_ACRE) / area
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

    # 4. Add urban park -> increases forest % and tree canopy share
    delta_park_share = (req.park_added_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] = np.clip(
        df_sim.loc[nbhd_idx, "pct_nlcd_forest_total"] + delta_park_share * 100.0, 0.0, 100.0
    )
    df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "tree_canopy_per_sq_meter_unit"] + delta_park_share, 0.0, 1.0
    )

    # 5. New construction -> increases impervious intensity & medium-intensity dev %
    delta_dev_share = (req.new_dev_acres * SQ_M_PER_ACRE) / area
    df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] = np.clip(
        df_sim.loc[nbhd_idx, "impervious_surfaces_per_sq_meter_unit"] + delta_dev_share, 0.0, 1.0
    )
    df_sim.loc[nbhd_idx, "pct_nlcd_dev_med_23"] = np.clip(
        df_sim.loc[nbhd_idx, "pct_nlcd_dev_med_23"] + delta_dev_share * 100.0, 0.0, 100.0
    )

    # Solve spatial equilibrium: y_hat = (I - rho * W)^(-1) * (X * beta)
    X_scaled = scaler.transform(df_sim[best_features].values)
    X_const = np.column_stack([np.ones(len(df_sim)), X_scaled])
    sim_temps_c = np.exp(spatial_multiplier @ (X_const @ betas))

    baseline_c = float(df_ref.loc[nbhd_idx, "baseline_pred_c"])
    simulated_c = float(sim_temps_c[nbhd_idx])
    delta_c = simulated_c - baseline_c

    # Average spillover across the k=12 spatial neighbors
    neighbor_mask = W_ref[nbhd_idx] > 0
    avg_neighbor_delta_c = float(
        np.mean(sim_temps_c[neighbor_mask] - df_ref.loc[neighbor_mask, "baseline_pred_c"].values)
    )

    latency_ms = (time.time() - start_time) * 1000.0
    logger.info(
        f"Neighborhood='{req.neighborhood_name}' | Dev={req.new_dev_acres}ac | Trees={req.trees_added} | "
        f"LotRem={req.parking_removed_acres}ac | BldgRem={req.building_removed_acres}ac | Park={req.park_added_acres}ac | "
        f"PredLST={simulated_c:.2f}C | Delta={delta_c:+.4f}C | Spillover={avg_neighbor_delta_c:+.5f}C | "
        f"Latency={latency_ms:.2f}ms"
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
        "target_neighborhood": req.neighborhood_name,
        "baseline_temp_c": round(baseline_c, 4),
        "simulated_temp_c": round(simulated_c, 4),
        "temp_change_c": round(delta_c, 4),
        "avg_neighbor_spillover_c": round(avg_neighbor_delta_c, 5),
        "latency_ms": round(latency_ms, 2),
        "all_neighborhoods": all_predictions,
    }