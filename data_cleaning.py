# Cleaning and Validating ATL neighboorhood data
import pandas as pd

# Load the aggregated dataset
df = pd.read_csv("aggregated_data.csv")

# Preview the data
print(df.head())

# Dataset Size
print("Dataset shape:", df.shape)

# Column Names
print("\nColumns:")
print(df.columns.tolist())

# Data Types
print("\nData types:")
print(df.dtypes)

# Missing Values

# Count missing values per column
missing = df.isnull().sum()

print("\nMissing values:")
print(missing)

# Remove OBJECTID only if every value is missing
if "OBJECTID" in df.columns:
    if df["OBJECTID"].isnull().all():
        df = df.drop(columns=["OBJECTID"])
        print("Removed empty OBJECTID column")

# Duplicate Rows
print("\nDuplicate rows:", df.duplicated().sum())

# Repeated Neighborhood Names
print("Duplicate names:", df["NAME"].duplicated().sum())

# Display Neighborhoods w/ repeated names
repeated = df[df["NAME"].duplicated(keep=False)]
print(repeated)

# Inconsistent Temperature Measurements
invalid_temp = df[
    (df["lst_min_c"] > df["lst_mean_c"]) |
    (df["lst_mean_c"] > df["lst_max_c"]) |
    (df["lst_std_c"] < 0)
]
print("\nInvalid temperature records:")
print(invalid_temp)

# Percentage Ranges
percentage_cols = [
    "tree_canopy_mean",
    "impervious_surfaces_mean",
    "pct_nlcd_water_11",
    "pct_nlcd_dev_open_21",
    "pct_nlcd_dev_low_22",
    "pct_nlcd_dev_med_23",
    "pct_nlcd_dev_high_24",
    "pct_nlcd_forest_total"
]

for col in percentage_cols:
    invalid = df[~df[col].between(0, 100)]
    print(f"{col}: {len(invalid)} invalid values")

# Check standard deviation values
std_cols = [
    "tree_canopy_std",
    "impervious_surfaces_std"
]

for col in std_cols:
    invalid = (df[col] < 0).sum()
    print(f"{col}: {invalid} negative values")

# Invalid neighborhood areas
invalid_area = df[df["area_sq_m"] <= 0]

print("\nInvalid neighborhood areas:")
print(invalid_area)

# Identify unusual average temperatures
Q1 = df["lst_mean_c"].quantile(0.25)
Q3 = df["lst_mean_c"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["lst_mean_c"] < lower) |
    (df["lst_mean_c"] > upper)
]

print("\nPotential temperature outliers:")
print(outliers[["NAME", "lst_mean_c"]])

# Confirm measurements passed validation
invalid_percentages = sum(
    (~df[col].between(0, 100)).sum()
    for col in percentage_cols
)

invalid_std = sum(
    (df[col] < 0).sum()
    for col in std_cols
)

if (
    len(invalid_temp) > 0
    or len(invalid_area) > 0
    or invalid_percentages > 0
    or invalid_std > 0
):
    raise ValueError(
        "Invalid measurements found. Review before exporting."
    )

# Save cleaned data without changing the original
df.to_csv("cleaned_data.csv", index=False)

print("\nCleaning complete!")
print("Final dataset shape:", df.shape)

# Summarize the final dataset
print("\nDATA CLEANING SUMMARY")

print("Total neighborhoods:", len(df))
print("Total columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Temperature issues:", len(invalid_temp))
print("Invalid areas:", len(invalid_area))
print("Potential temperature outliers:", len(outliers))