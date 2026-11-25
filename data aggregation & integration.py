import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterstats import zonal_stats

# loading raw data
atlanta_neighborhods = "/Users/sydneelear/Desktop/capstone data/Official Atlanta Neighborhood Boundaries.geojson"
lst_rasters = {
    "2023-06-03": "/Users/sydneelear/Desktop/capstone data/LST/LST 06_03_2023 17_17_40.tif",
    "2023-06-07": "/Users/sydneelear/Desktop/capstone data/LST/LST 06_07_2023 15_39_58.tif",
    "2023-06-27": "/Users/sydneelear/Desktop/capstone data/LST/LST 06_27_2023 15_35_08.tif",
    "2023-08-14": "/Users/sydneelear/Desktop/capstone data/LST/LST 08_14_2023 12_55_45.tif",
    "2023-08-23": "/Users/sydneelear/Desktop/capstone data/LST/LST 08_23_2023 17_07_13.tif",
}

tree_canopy_raster = "/Users/sydneelear/Desktop/capstone data/NLCD Land Cover Data 2023/nlcd_tcc_conus_wgs84_v2025-6_20230101_20231231_76c83eeb-36c7-4a4f-88ce-4d385a180cce.tiff"
impervious_surfaces_raster = "/Users/sydneelear/Desktop/capstone data/NLCD Land Cover Data 2023/Annual_NLCD_FctImp_2023_CU_C1V2_76c83eeb-36c7-4a4f-88ce-4d385a180cce.tiff"
land_cover_raster = "/Users/sydneelear/Desktop/capstone data/NLCD Land Cover Data 2023/Annual_NLCD_LndCov_2023_CU_C1V2_76c83eeb-36c7-4a4f-88ce-4d385a180cce.tiff"

# output path configuration
output_dir = "/Users/sydneelear/Desktop/capstone data"
output_csv = os.path.join(output_dir, "aggregated_data.csv")
os.makedirs(output_dir, exist_ok = True)

# base geo dataframe to align neighborhood boundaries with rasters
base_gdf = gpd.read_file(atlanta_neighborhods)

# area and location of each neighborhood in sq meters
base_gdf['area_sq_m'] = base_gdf.to_crs(epsg=32616).geometry.area
base_gdf['centroid_x'] = base_gdf.to_crs(epsg=32616).geometry.centroid.x
base_gdf['centroid_y'] = base_gdf.to_crs(epsg=32616).geometry.centroid.y



# tree canopy percentage
# aligning neighborhood boundaries
with rasterio.open(tree_canopy_raster) as src:
    gdf_canopy = base_gdf.to_crs(src.crs)

# extracting tree canopy pct
tree_canopy_stats = zonal_stats(gdf_canopy, tree_canopy_raster, stats = ['mean', 'std'], nodata = 255)
base_gdf['tree_canopy_mean'] = [s['mean'] for s in tree_canopy_stats]
base_gdf['tree_canopy_std'] = [s['std'] for s in tree_canopy_stats]



# impervious surface percentage
# aligning neighborhood boundaries
with rasterio.open(impervious_surfaces_raster) as src:
    gdf_imp = base_gdf.to_crs(src.crs)

# extracting impervious surfaces pct
impervious_surfaces_stats = zonal_stats(gdf_imp, impervious_surfaces_raster, stats = ['mean', 'std'], nodata = 255)
base_gdf['impervious_surfaces_mean'] = [s['mean'] for s in impervious_surfaces_stats]
base_gdf ['impervious_surfaces_std'] = [s['std'] for s in impervious_surfaces_stats]



# nlcd classes (categorical)
# aligning neighborhood boundaries
with rasterio.open(land_cover_raster) as src:
    gdf_lc = base_gdf.to_crs(src.crs)

# extracting nlcd classes
land_cover_stats = zonal_stats(gdf_lc, land_cover_raster, categorical = True)
land_cover_df = pd.DataFrame(land_cover_stats).fillna(0)
total_pixels = land_cover_df.sum(axis = 1)

# class 11: open water
base_gdf['pct_nlcd_water_11'] = ((land_cover_df.get(11,0)/total_pixels)*100.0 if 11 in land_cover_df else 0.0)
# class 21: developed, open space
base_gdf['pct_nlcd_dev_open_21'] = ((land_cover_df.get(21,0)/total_pixels)*100.0 if 21 in land_cover_df else 0.0)
# class 22: developed, low intensity
base_gdf['pct_nlcd_dev_low_22'] = ((land_cover_df.get(22,0)/total_pixels)*100.0 if 22 in land_cover_df else 0.0)
# class 23: developed, medium intensity
base_gdf['pct_nlcd_dev_med_23'] = ((land_cover_df.get(23,0)/total_pixels)*100.0 if 23 in land_cover_df else 0.0)
# class 24: developed, high intensity
base_gdf['pct_nlcd_dev_high_24'] = ((land_cover_df.get(24,0)/total_pixels)*100.0 if 24 in land_cover_df else 0.0)
# combining forest classes (41: deciduous, 42: evergreen, 43: mixed, 90: woody wetlands)
forest_pixels = (land_cover_df.get(41,0) + land_cover_df.get(42,0) + land_cover_df.get(43,0) + land_cover_df.get (90,0))
base_gdf['pct_nlcd_forest_total'] = (forest_pixels/total_pixels) * 100.0


# lst (target) across all five summer dates
lst_gdfs = []

for scene_date, raster_path in lst_rasters.items():
    date_gdf = base_gdf.copy()
    with rasterio.open(raster_path) as src:
        date_gdf_lst = date_gdf.to_crs(src.crs)

    lst_stats = zonal_stats(date_gdf_lst, raster_path, stats=["mean", "min", "max", "std"], nodata=np.nan)

    # converting from kelvins to celsius
    date_gdf['lst_mean_c'] = [(s["mean"] - 273.15) if (s["mean"] is not None and s["mean"] > 100) else np.nan for s in lst_stats]
    date_gdf['lst_min_c'] = [(s["min"] - 273.15) if (s["min"] is not None and s["min"] > 100) else np.nan for s in lst_stats]
    date_gdf['lst_max_c'] = [(s["max"] - 273.15) if (s["max"] is not None and s["max"] > 100) else np.nan for s in lst_stats]
    date_gdf['lst_std_c'] = [s['std'] for s in lst_stats]
    date_gdf['scene_date'] = scene_date

    lst_gdfs.append(date_gdf)

# aggregating all dates
full_gdf = pd.concat(lst_gdfs, ignore_index=True)


# export csv
id_cols = [col for col in ['OBJECTID', 'NAME', 'NEIGHBORHO', 'NPU', 'GlobalID', 'STATENAME'] if col in full_gdf.columns]

data_cols = [
    'area_sq_m',
    'centroid_x',
    'centroid_y',
    'scene_date',
    'lst_mean_c',
    'lst_min_c',
    'lst_max_c',
    'lst_std_c',
    'tree_canopy_mean',
    'tree_canopy_std',
    'impervious_surfaces_mean',
    'impervious_surfaces_std',
    'pct_nlcd_water_11',
    'pct_nlcd_dev_open_21',
    'pct_nlcd_dev_low_22',
    'pct_nlcd_dev_med_23',
    'pct_nlcd_dev_high_24',
    'pct_nlcd_forest_total'
]

export_df = pd.DataFrame(full_gdf[id_cols + data_cols])
export_df.to_csv(output_csv, index = False)