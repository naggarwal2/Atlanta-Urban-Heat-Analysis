import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# -------------------------------------------------------------
# 0. Load Exported Train/Val & Test Datasets
# -------------------------------------------------------------
df_train_val = pd.read_csv("train_val_data.csv").reset_index(drop=True)
df_test = pd.read_csv("test_data.csv").reset_index(drop=True)

# Shift df_test index so df_cleaned has unique contiguous indices
df_test.index = df_test.index + len(df_train_val)
df_cleaned = pd.concat([df_train_val, df_test], axis=0)


# spatial weights matrix (najja)

import libpysal

# Extract the X and Y centroid coordinates into a NumPy array
coords = np.column_stack((df_cleaned['centroid_x'], df_cleaned['centroid_y']))

# Construct the K-Nearest Neighbors weights matrix
# (k=5 is a standard starting point, meaning each neighborhood connects to its 5 closest neighbors)
w_knn = libpysal.weights.KNN.from_array(coords, k=5)

# Row-standardize the weights matrix (highly recommended for spatial modeling)
w_knn.transform = 'r'

# Calculate the spatial lag (the average LST of the 5 nearest neighboring areas)
df_cleaned['lst_mean_c_lag'] = libpysal.weights.lag_spatial(w_knn, df_cleaned['lst_mean_c'])


# Define features, target, and GroupKFold for SAR modeling & CV
date_cols = sorted([col for col in df_cleaned.columns if col.startswith("scene_date_")])
features = [
    "area_sq_m",
    "pct_nlcd_dev_open_21",
    "pct_nlcd_dev_low_22",
    "pct_nlcd_dev_med_23",
    "pct_nlcd_dev_high_24",
    "pct_nlcd_forest_total",
    "tree_canopy_per_sq_meter_unit",
    "impervious_surfaces_per_sq_meter_unit",
    *date_cols,
    "centroid_x",
    "centroid_y",
]
target = "log_lst_mean_c"

X_tv = df_train_val[features]
y_tv = df_train_val[target]
groups_tv = df_train_val["NPU"]
gkf = GroupKFold(n_splits=5)


# spatial auto regressive model (sydnee)

# helper functions & training
from functools import reduce
from spreg import ML_Lag
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------
# 1. SAR Helper Functions (Scene-Blocked Weights + BLUP Prediction)
# -------------------------------------------------------------
def build_scene_blocked_weights(df_subset, scene_col="scene_date", k=5):
    """Connects each neighborhood ONLY to its k nearest spatial neighbors on the same scene_date."""
    df_sub = df_subset.copy()

    # Reconstruct scene_date if only one-hot dummy columns are present
    if scene_col not in df_sub.columns:
        dummy_cols = sorted([c for c in df_sub.columns if c.startswith(f"{scene_col}_")])
        conditions = [df_sub[c] == 1 for c in dummy_cols]
        choices = [c.replace(f"{scene_col}_", "") for c in dummy_cols]
        df_sub[scene_col] = np.select(conditions, choices, default="baseline_date")

    df_sorted = df_sub.sort_values(scene_col).reset_index(drop=True)
    w_blocks = []

    for _, grp in df_sorted.groupby(scene_col, sort=False):
        coords = np.column_stack((grp["centroid_x"], grp["centroid_y"]))
        k_eff = min(k, len(grp) - 1)
        w_scene = libpysal.weights.KNN.from_array(coords, k=k_eff, ids=grp.index.tolist())
        w_blocks.append(w_scene)

    w_panel = reduce(libpysal.weights.w_union, w_blocks) if len(w_blocks) > 1 else w_blocks[0]
    w_panel.transform = "r"
    return df_sorted, w_panel


def predict_sar_blup(sar_model, scaler, df_map, eval_index, feature_cols, target_col, k=5):
    """
    Predicts out-of-sample neighborhoods using Best Linear Unbiased Prediction (BLUP),
    conditioning on known training temperatures of adjacent neighbors on the same scene_date:
    y_u = (I - rho * W_uu)^(-1) * (X_u * beta + rho * W_uk * y_known)
    """
    df_all = df_map.copy()
    df_all["is_eval"] = df_all.index.isin(eval_index)
    df_sorted, w_all = build_scene_blocked_weights(df_all, scene_col="scene_date", k=k)

    X_scaled = scaler.transform(df_sorted[feature_cols].values)
    X_const = np.column_stack([np.ones(len(df_sorted)), X_scaled])

    betas = sar_model.betas[:-1].flatten()
    rho = float(sar_model.rho)
    W_dense = w_all.full()[0]

    u_mask = df_sorted["is_eval"].values
    k_mask = ~u_mask

    W_uu = W_dense[np.ix_(u_mask, u_mask)]
    W_uk = W_dense[np.ix_(u_mask, k_mask)]

    xb_u = X_const[u_mask] @ betas
    y_k = df_sorted.loc[k_mask, target_col].values

    rhs = xb_u + rho * (W_uk @ y_k)
    y_hat_u = np.linalg.solve(np.eye(u_mask.sum()) - rho * W_uu, rhs)

    return df_sorted.loc[u_mask].copy(), y_hat_u


def evaluate_summer_sar_celsius(df_eval_subset, y_true_log, y_pred_log, title="SAR MODEL"):
    """Converts log predictions back to Celsius and prints overall pooled summer R², MAE, and RMSE."""
    df_res = df_eval_subset.copy()
    df_res["y_true_c"] = np.exp(y_true_log)
    df_res["y_pred_c"] = np.exp(y_pred_log)

    summer_r2 = float(r2_score(df_res["y_true_c"], df_res["y_pred_c"]))
    summer_mae = float(mean_absolute_error(df_res["y_true_c"], df_res["y_pred_c"]))
    summer_rmse = float(np.sqrt(mean_squared_error(df_res["y_true_c"], df_res["y_pred_c"])))

    print(f"\n{'='*55}\n=== {title} ===\n{'='*55}")
    print(f"Summer R²:   {summer_r2:.4f}")
    print(f"Summer MAE:  {summer_mae:.2f} °C")
    print(f"Summer RMSE: {summer_rmse:.2f} °C")
    print("=" * 55)

    return {
        "summer_r2": summer_r2,
        "summer_mae_c": summer_mae,
        "summer_rmse_c": summer_rmse,
        "count": int(len(df_res))
    }


# -------------------------------------------------------------
# 2. Untuned SAR Baseline (k=5, No Leakage / No Coords)
# -------------------------------------------------------------
sar_base_features = [c for c in features if c not in ["lst_mean_c_lag", "centroid_x", "centroid_y"]]

df_tv_s, w_tv_s = build_scene_blocked_weights(df_train_val, scene_col="scene_date", k=5)
scaler_untuned = StandardScaler()
X_tv_s = scaler_untuned.fit_transform(df_tv_s[sar_base_features].values)
y_tv_s = df_tv_s[target].values.reshape(-1, 1)

untuned_sar = ML_Lag(y_tv_s, X_tv_s, w=w_tv_s, name_y=target, name_x=sar_base_features)
print(f"Untuned SAR Rho (ρ): {float(untuned_sar.rho):.4f} (p = {untuned_sar.z_stat[-1][1]:.4e}) | AIC: {float(untuned_sar.aic):.2f}")

df_test_untuned, untuned_preds = predict_sar_blup(
    untuned_sar, scaler_untuned, df_cleaned, df_test.index, sar_base_features, target, k=5
)
untuned_metrics = evaluate_summer_sar_celsius(
    df_test_untuned, df_test_untuned[target].values, untuned_preds, title="UNTUNED SAR (k=5)"
)


# hyperparamter tuning
# -------------------------------------------------------------
# 3. Spatial CV Tuning (k Neighbors & Feature Specification)
# -------------------------------------------------------------
date_dummy_cols = [c for c in sar_base_features if c.startswith("scene_date_")]
land_only_cols = [c for c in sar_base_features if c not in date_dummy_cols + ["area_sq_m"]]

candidate_specs = {
    "land_cover_only": land_only_cols + date_dummy_cols,
    "land_cover_plus_area": sar_base_features
}
k_grid = [3, 5, 8, 12]
tuning_records = []

print("\n=== TUNING SAR (5-FOLD SPATIAL GROUPKFOLD CV) ===")
for k in k_grid:
    for spec_name, spec_cols in candidate_specs.items():
        fold_maes, fold_rmses, fold_rhos = [], [], []

        for train_idx, val_idx in gkf.split(X_tv, y_tv, groups_tv):
            df_tr = df_train_val.iloc[train_idx]
            df_va = df_train_val.iloc[val_idx]

            df_tr_s, w_tr = build_scene_blocked_weights(df_tr, scene_col="scene_date", k=k)
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(df_tr_s[spec_cols].values)
            y_tr_arr = df_tr_s[target].values.reshape(-1, 1)

            sar_cv = ML_Lag(y_tr_arr, X_tr_sc, w=w_tr, name_y=target, name_x=spec_cols)

            # Predict validation fold conditioned ONLY on training fold neighbors (zero test leakage)
            df_va_eval, va_preds_log = predict_sar_blup(
                sar_cv, scaler, df_train_val, df_va.index, spec_cols, target, k=k
            )
            y_va_c = np.exp(df_va_eval[target].values)
            y_va_pred_c = np.exp(va_preds_log)

            fold_maes.append(mean_absolute_error(y_va_c, y_va_pred_c))
            fold_rmses.append(np.sqrt(mean_squared_error(y_va_c, y_va_pred_c)))
            fold_rhos.append(float(sar_cv.rho))

        mean_mae = float(np.mean(fold_maes))
        mean_rmse = float(np.mean(fold_rmses))
        mean_rho = float(np.mean(fold_rhos))
        tuning_records.append({
            "k": k, "spec_name": spec_name, "features": spec_cols,
            "cv_mae_c": mean_mae, "cv_rmse_c": mean_rmse, "mean_rho": mean_rho
        })
        print(f"k={k:<2} | {spec_name:<21} | CV MAE: {mean_mae:.3f} °C | CV RMSE: {mean_rmse:.3f} °C | ρ: {mean_rho:.4f}")

best_cfg = min(tuning_records, key=lambda x: x["cv_mae_c"])
best_k = int(best_cfg["k"])
best_spec = best_cfg["spec_name"]
best_features = best_cfg["features"]
print(f"\n>>> Optimal SAR Config Selected: k={best_k}, {best_spec} (CV MAE: {best_cfg['cv_mae_c']:.3f} °C)")


# final tuned SAR fit & testing
# -------------------------------------------------------------
# 4. Final Tuned SAR Fit & Hold-Out Test Evaluation
# -------------------------------------------------------------
df_tv_final, w_tv_final = build_scene_blocked_weights(df_train_val, scene_col="scene_date", k=best_k)
scaler_final = StandardScaler()
X_tv_final = scaler_final.fit_transform(df_tv_final[best_features].values)
y_tv_final = df_tv_final[target].values.reshape(-1, 1)

final_sar = ML_Lag(
    y_tv_final,
    X_tv_final,
    w=w_tv_final,
    name_y=target,
    name_x=best_features
)

# Print Parameter Summary Table
param_names = ["CONSTANT"] + list(best_features) + [f"W_{target} (rho)"]
df_sar_params = pd.DataFrame({
    "Parameter": param_names,
    "Estimate": final_sar.betas.flatten(),
    "Std_Error": final_sar.std_err,
    "z_stat": [s[0] for s in final_sar.z_stat],
    "p_value": [s[1] for s in final_sar.z_stat]
})

print("\n=== FINAL TUNED SAR PARAMETER ESTIMATES ===")
print(df_sar_params.to_string(index=False))
print(f"\nLog-Likelihood: {float(final_sar.logll):.2f} | AIC: {float(final_sar.aic):.2f}")

# Evaluate Final Tuned SAR on Hold-Out Test Set (Using Known-Neighbor BLUP)
df_test_final, final_preds_log = predict_sar_blup(
    final_sar, scaler_final, df_cleaned, df_test.index, best_features, target, k=best_k
)

final_metrics = evaluate_summer_sar_celsius(
    df_test_final,
    df_test_final[target].values,
    final_preds_log,
    title=f"FINAL TUNED SAR (k={best_k}, {best_spec})"
)




# -------------------------------------------------------------
# 5. Prepare & Serialize Model Artifact for Deployment
# -------------------------------------------------------------
import joblib
from datetime import datetime

# Build a single-date 248-neighborhood reference map (baseline scene_date where dummies == 0)
date_dummy_cols = [c for c in best_features if c.startswith("scene_date_")]
baseline_mask = (df_cleaned[date_dummy_cols].sum(axis=1) == 0)
df_ref = df_cleaned.loc[baseline_mask].copy().reset_index(drop=True)

# Construct the 248 x 248 spatial weights matrix (k=12) for the single reference scene
coords_ref = np.column_stack((df_ref["centroid_x"], df_ref["centroid_y"]))
w_ref = libpysal.weights.KNN.from_array(coords_ref, k=best_k)
w_ref.transform = "r"
W_ref_dense = w_ref.full()[0]

# Precompute the spatial multiplier M = (I - rho * W)^(-1) for instant equilibrium inference
rho_final = float(final_sar.rho)
betas_final = final_sar.betas[:-1].flatten()  # [CONSTANT, beta_1, ..., beta_p]
n_nbhd = len(df_ref)
spatial_multiplier = np.linalg.inv(np.eye(n_nbhd) - rho_final * W_ref_dense)

# Store baseline predicted Celsius temperatures across all 248 neighborhoods
X_ref_scaled = scaler_final.transform(df_ref[best_features].values)
X_ref_const = np.column_stack([np.ones(n_nbhd), X_ref_scaled])
df_ref["baseline_pred_c"] = np.exp(spatial_multiplier @ (X_ref_const @ betas_final))

# Serialize model artifact + metadata for reproducibility and traceability
sar_artifact = {
    "metadata": {
        "model_name": "Atlanta_UHI_SAR_Model",
        "model_version": "1.0.0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "k_neighbors": best_k,
        "feature_spec": best_spec,
        "target": target,
        "holdout_metrics": final_metrics,
    },
    "best_features": list(best_features),
    "scaler": scaler_final,
    "betas": betas_final,
    "rho": rho_final,
    "W_ref": W_ref_dense,
    "spatial_multiplier": spatial_multiplier,
    "df_ref": df_ref,
}

artifact_path = "sar_model_artifact.joblib"
joblib.dump(sar_artifact, artifact_path)
print(f"\nModel saved as {artifact_path}")