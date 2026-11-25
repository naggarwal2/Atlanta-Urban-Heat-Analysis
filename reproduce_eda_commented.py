"""Atlanta Urban Heat EDA.

Process Landsat and NLCD data by neighborhood and export EDA results.
Update BASE_DIR and OUTPUT_DIR before running.
"""


# 1. IMPORT LIBRARIES AND SET FILE LOCATIONS

"""Load libraries and set paths for the input data and output files."""

import glob
import os

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio

from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds, transform as window_transform
from scipy.stats import pearsonr

matplotlib.use("Agg")  # Save figures without opening a display window.
import matplotlib.pyplot as plt


# Change these paths if running on a different computer.
BASE_DIR = "/mnt/data/urban_heat_work/data"
OUTPUT_DIR = "/mnt/data/Atlanta_Heat_EDA"

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the output folder if needed.


# 2. LOAD ATLANTA NEIGHBORHOOD BOUNDARIES

"""Load Atlanta neighborhood boundaries for geographic analysis."""

neighborhoods = gpd.read_file(
    os.path.join(BASE_DIR, "boundaries", "atlanta_neighborhoods.geojson")
)

neighborhood_count = len(neighborhoods)


# 3. MATCH RASTER PIXELS TO NEIGHBORHOODS

"""Match raster pixels to neighborhood boundaries using location IDs."""


def get_neighborhood_pixels(dataset):
    # Align the neighborhood coordinates with this raster.
    aligned_neighborhoods = neighborhoods.to_crs(dataset.crs)
    bounds = aligned_neighborhoods.total_bounds

    # Read only the raster window covering the neighborhood boundaries.
    window = from_bounds(*bounds, transform=dataset.transform)
    window = window.round_offsets().round_lengths()
    window = window.intersection(
        Window(0, 0, dataset.width, dataset.height)
    )

    pixel_values = dataset.read(1, window=window)
    pixel_transform = window_transform(window, dataset.transform)

    # Give pixels inside each neighborhood a numeric ID.
    neighborhood_labels = rasterize(
        [
            (geometry, index + 1)
            for index, geometry in enumerate(aligned_neighborhoods.geometry)
        ],
        out_shape=pixel_values.shape,
        transform=pixel_transform,
        fill=0,
        dtype="int16",
    )

    return pixel_values, neighborhood_labels, window


# 4. PROCESS LANDSAT LAND SURFACE TEMPERATURE (LST)

"""Filter Landsat pixels, convert LST to °F, and average by neighborhood."""

lst_sums = np.zeros(neighborhood_count)
lst_counts = np.zeros(neighborhood_count, dtype=int)

landsat_pattern = os.path.join(BASE_DIR, "landsat", "*20240626*ST_B10.TIF")

for temperature_file in sorted(glob.glob(landsat_pattern)):
    quality_file = temperature_file.replace("ST_B10", "QA_PIXEL")

    with rasterio.open(temperature_file) as temperature_dataset, \
            rasterio.open(quality_file) as quality_dataset:

        pixels, labels, window = get_neighborhood_pixels(temperature_dataset)
        quality = quality_dataset.read(1, window=window)

        # Keep labeled, nonzero pixels that pass the Landsat quality-bit test.
        valid = (
            (labels > 0)
            & (pixels > 0)
            & ((quality & 0b10011110) == 0)
        )

        # Landsat scale/offset -> Kelvin -> Celsius -> Fahrenheit.
        temperature_f = (
            (pixels.astype("float64") * 0.00341802 + 149 - 273.15)
            * 1.8
            + 32
        )

        # Exclude temperatures outside the expected surface-temperature range.
        valid &= (temperature_f > 40) & (temperature_f < 180)

        # Add valid temperatures and pixel counts for each neighborhood.
        lst_sums += np.bincount(
            labels[valid] - 1,
            weights=temperature_f[valid],
            minlength=neighborhood_count,
        )
        lst_counts += np.bincount(
            labels[valid] - 1,
            minlength=neighborhood_count,
        )

        print(
            "LST", os.path.basename(temperature_file), np.sum(valid),
            flush=True,
        )


# Create the initial neighborhood-level table.
eda_df = pd.DataFrame(
    {
        "neighborhood": neighborhoods.NAME,
        "npu": neighborhoods.NPU,
        "lst_f": np.divide(
            lst_sums,
            lst_counts,
            out=np.full(neighborhood_count, np.nan),
            where=lst_counts > 0,
        ),
        "lst_valid_pixels": lst_counts,
    }
)


# 5. PROCESS NLCD ENVIRONMENTAL VARIABLES

"""Calculate neighborhood percentages for canopy, developed land, and imperviousness."""

nlcd_files = [
    ("canopy", "tree_canopy_2024.tiff"),
    ("land", "annual_nlcd_landcover_2024.tiff"),
    ("impervious", "annual_nlcd_impervious_2024.tiff"),
]

for variable_key, filename in nlcd_files:
    nlcd_path = os.path.join(BASE_DIR, "nlcd", filename)

    with rasterio.open(nlcd_path) as dataset:
        pixels, labels, window = get_neighborhood_pixels(dataset)

        # Only consider pixels inside neighborhoods and outside NoData.
        valid = (labels > 0) & (pixels != dataset.nodata)

        if variable_key == "land":
            valid &= pixels != 11  # Exclude open water.
            values = np.isin(pixels, [22, 23, 24]).astype(float) * 100
            output_column = "developed_pct"
        else:
            valid &= pixels <= 100  # Exclude invalid percentages.
            values = pixels.astype(float)
            output_column = variable_key + "_pct"

        # Find pixel counts and summed values for each neighborhood.
        pixel_counts = np.bincount(
            labels[valid] - 1,
            minlength=neighborhood_count,
        )
        value_sums = np.bincount(
            labels[valid] - 1,
            weights=values[valid],
            minlength=neighborhood_count,
        )

        # Convert pixel totals into neighborhood-level percentages.
        eda_df[output_column] = np.divide(
            value_sums,
            pixel_counts,
            out=np.full(neighborhood_count, np.nan),
            where=pixel_counts > 0,
        )
        eda_df[variable_key + "_pixels"] = pixel_counts

        print(variable_key, np.sum(valid), flush=True)


# Save the structured dataset for later analysis and modeling.
eda_df.to_csv(os.path.join(OUTPUT_DIR, "neighborhood_eda.csv"), index=False)


# 6. SET UP CONSISTENT GRAPH FORMATTING

"""Set chart formatting and define a helper for saving figures."""

plt.rcParams.update({"font.size": 11, "figure.dpi": 140})


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches="tight")
    plt.close()


# 7. CREATE THE LST HISTOGRAM

"""Plot the distribution of average neighborhood surface temperatures."""

plt.figure(figsize=(8, 4.6))
plt.hist(eda_df.lst_f.dropna(), bins=20, edgecolor="white")
plt.xlabel("Mean neighborhood LST (°F)")
plt.ylabel("Neighborhoods")
plt.title("Atlanta land surface temperature · June 26, 2024")
save_figure("01_lst_histogram.png")


# 8. CREATE BOXPLOTS FOR ALL FOUR VARIABLES

"""Compare the spread and outliers of the four analysis variables."""

figure, axes = plt.subplots(1, 4, figsize=(12, 4))

analysis_columns = ["lst_f", "canopy_pct", "developed_pct", "impervious_pct"]
boxplot_names = ["LST (°F)", "Canopy (%)", "Developed (%)", "Impervious (%)"]

for axis, column, name in zip(axes, analysis_columns, boxplot_names):
    axis.boxplot(eda_df[column].dropna())
    axis.set_title(name)
    axis.set_xticks([])

figure.suptitle("Neighborhood distributions")
save_figure("02_boxplots.png")


# 9. CREATE SCATTER PLOTS AND LINEAR TREND LINES

"""Plot LST relationships, Pearson correlations, and fitted trend lines."""

scatterplots = [
    ("03_lst_vs_canopy.png", "canopy_pct", "Tree canopy"),
    ("04_lst_vs_developed.png", "developed_pct", "Developed land"),
    ("05_lst_vs_impervious.png", "impervious_pct", "Impervious surface"),
]

for filename, column, label in scatterplots:
    # Exclude neighborhoods missing either plotted value.
    paired_data = eda_df[[column, "lst_f"]].dropna()

    # Calculate Pearson correlation and its p-value.
    correlation, p_value = pearsonr(
        paired_data[column], paired_data.lst_f
    )

    plt.figure(figsize=(7.5, 5))
    plt.scatter(paired_data[column], paired_data.lst_f, alpha=0.65, s=24)

    # Fit a straight line to show the overall relationship.
    slope, intercept = np.polyfit(paired_data[column], paired_data.lst_f, 1)
    x_values = np.linspace(paired_data[column].min(), paired_data[column].max())
    plt.plot(x_values, slope * x_values + intercept, "--")

    plt.xlabel(label + " (%)")
    plt.ylabel("Mean LST (°F)")
    plt.title(f"LST vs {label.lower()} · r = {correlation:.2f}")
    save_figure(filename)


# 10. EXPORT THE DATA QUALITY SUMMARY

"""Export missing-value counts and descriptive statistics for each variable."""

quality_rows = []

for column in analysis_columns:
    values = eda_df[column]

    quality_rows.append(
        {
            "variable": column,
            "count": int(values.count()),
            "missing": int(values.isna().sum()),
            "mean": values.mean(),
            "median": values.median(),
            "minimum": values.min(),
            "maximum": values.max(),
            "std": values.std(),
        }
    )

pd.DataFrame(quality_rows).to_csv(
    os.path.join(OUTPUT_DIR, "data_quality_summary.csv"),
    index=False,
)


# 11. WRITE THE TEXT RESULTS AND CORRELATION SUMMARY

"""Save a text summary of neighborhood statistics and correlations."""

results_path = os.path.join(OUTPUT_DIR, "results.txt")

with open(results_path, "w") as results_file:
    # Report the number of neighborhoods and descriptive statistics.
    results_file.write(
        f"Neighborhoods: {neighborhood_count}\n"
        + eda_df[analysis_columns].describe().to_string()
        + "\n"
    )

    # Report each environmental variable's correlation with LST.
    for column in analysis_columns[1:]:
        paired_data = eda_df[[column, "lst_f"]].dropna()
        correlation, p_value = pearsonr(
            paired_data[column], paired_data.lst_f
        )
        results_file.write(
            f"{column}: r={correlation:.4f}, "
            f"p={p_value:.4g}, n={len(paired_data)}\n"
        )

    # Report missing values in the processed neighborhood-level dataset.
    results_file.write(
        "Missing:\n" + eda_df[analysis_columns].isna().sum().to_string()
    )


# Display the summary after all outputs have been saved.
with open(results_path, "r") as results_file:
    print(results_file.read(), flush=True)
