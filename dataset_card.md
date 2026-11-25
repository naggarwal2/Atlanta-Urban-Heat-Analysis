# Dataset Card: Atlanta Urban Heat Island Predictor

## I. Dataset Overview & Identification
- **Dataset Name:** Atlanta Neighborhood Land Surface Temperature and Land Cover
- **Version:** v1.0.0 (October 2026)
- **Authors & Owners:**
  - Sydnee Lear
  - Najja McGee
  - Nitika Aggarwal
  - Kem Ahanonu
- **Summary:** An integrated spatial dataset linking satellite sensing observations (land surface temperature (LST), tree canopy cover, impervious surface percentages, and land cover classifications) with 248 official Atlanta municipal neighborhood boundaries to quantify and predict heat intensities and evaluate targeted cooling interventions.

---

## II. Motivations & Intended Use
- **Context & Motivation:** Metro Atlanta is surging in development as the 6th largest metropolitan area in the United States, adding nearly 400,000 new residents since 2020. However, this rapid development often replaces trees and green spaces with pavement and buildings, raising urban temperatures and intensifying the Urban Heat Island effect. While tree canopies and vegetation cool neighborhoods, municipal planners and developers currently lack data-driven tools to find the trade-off between development and vegetation.
- **Intended Use:** To determine whether environmental characteristics can accurately predict land surface temperatures at the neighborhood level, establish clear coefficients (the cooling effect of tree canopy vs the heating effect of concrete), and power an interactive "what if" simulator. This enables planners and developers to forecast post-development temperatures and quantify how much added vegetation is required to offset heat from new infrastructure.

---

## III. Data Provenance & Pipeline

### Data Sources
- **Official Atlanta Boundaries Polygon:**
  - Sourced from City of Atlanta Department of City Planning GIS Open Data Hub
  - **Format:** GeoJSON/Vector polygons; base geographic coordinate system in WGS84
  - Establishes the official spatial units of analysis (248 neighborhoods). Used to clip and aggregate raster pixel data into consistent tabular boundaries.
- **LST Raster:**
  - Sourced from NASA Earth Data Search ECOSTRESS
  - Peak summer pass on August 23, 2023 at 17:07 UTC (1:07 PM EDT) recorded in Kelvins
  - **Format:** GeoTIFF raster
  - Serves as the project's primary dependent/target variable ($y$) for measuring urban heat island intensity.
- **Tree Canopy Raster:**
  - Sourced from USGS National Land Cover Database (NLCD)
  - **Format:** GeoTIFF raster; continuous pixel values ranging from 0% to 100% canopy density, with a standard 30-meter spatial resolution.
  - Measures how much of the neighborhood is covered by trees, showing where leaves block direct sunlight with shade and naturally cool the air by releasing moisture.
- **Impervious Surfaces Raster:**
  - Sourced from USGS National Land Cover Database (NLCD)
  - **Format:** GeoTIFF raster; continuous pixel values ranging from 0% to 100% impervious cover, with a standard 30-meter spatial resolution.
  - Directly measures the concentration of paved roads, parking lots, roofs, and concrete structures that absorb solar radiation and trap heat during daylight hours.
- **Land Cover Raster:**
  - Sourced from USGS National Land Cover Database (NLCD)
  - **Format:** Categorical GeoTIFF raster containing discrete integer land-use classifications (e.g., water, open space, low/medium/high built density, forest) at 30-meter spatial resolution.
  - Provides the physical context of each neighborhood by classifying surface types; whether concrete is a high-rise downtown building vs. a suburban home with a yard, or whether greenery is a park lawn vs. a dense forest.

### Integration Pipeline
1. Loaded Atlanta neighborhood boundaries and converted them from GPS coordinates into standard ground meters so we could calculate the physical size of each neighborhood in meters.
2. Aligned the map coordinate systems between the neighborhood boundaries with each satellite raster file to ensure neighborhood outlines lined up over the exact same locations on the map.
3. Pulled satellite data that falls within each neighborhood border:
   - a. Averaged LST
   - b. Averaged percentage of tree canopy & impervious surface temperature
   - c. Counted pixels for each land cover type to find the portion of neighborhood that is open space, housing, dense development, or forest.
4. Converted satellite temperature data from Kelvin to Celsius and converted land cover counts into percentages of the total neighborhood area.
5. Combined neighborhood names, planning units, physical area, and all extracted metrics into our final dataset.

---

## IV. Feature Dictionary & Summary Statistics

| Feature | Description | Data Type | Mean | Min – Max |
| :--- | :--- | :--- | :--- | :--- |
| **Land Surface Temperature** (`lst_mean_c`) | Average ground surface temperatures in Celsius, converted from satellite Kelvin measurements. Target variable that model is trying to predict to measure heat across Atlanta neighborhoods. | Float | $39.56^\circ\text{C}$ | $32.60\text{ – }47.86^\circ\text{C}$ |
| **Neighborhood** (`NAME`) | Official name of each Atlanta neighborhood | Object | — | — |
| **Neighborhood Planning Unit** (`NPU`) | Letter (A–Z) that groups adjacent neighborhoods. Used for spatial cross-validation to keep neighboring areas together and prevent data leakage where model cheats by memorizing nearby climates. | Object | — | — |
| **Area** (`area_sq_m`) | Total physical size of the neighborhood in square meters. Allows feature engineering to turn percentages into real measurements so users can test adding specific amounts of tree coverage/impervious surfaces. | Float | $134,520.3\text{ m}^2$ | $6,396.1\text{ – }778,897.7\text{ m}^2$ |
| **Tree Canopy** (`tree_canopy_per_sq_meter_unit`) *(in $\text{m}^2 \rightarrow$ `tree_canopy_sq_m`)* | Average percentage of the neighborhood covered by tree leaves (0–100%). Main green predictor to see how much shade and vegetative cover cool the neighborhood down. | Float | 41.99% | 1.04% – 77.55% |
| **Impervious Surfaces** (`impervious_surfaces_per_sq_meter_unit`) *(in $\text{m}^2 \rightarrow$ `impervious_surfaces_sq_m`)* | Average percentage of paved ground like roads, parking lots, roofs, etc. (0–100%). Main built-environment predictor to see how much paved surfaces trap and retain heat. | Float | 32.67% | 4.23% – 84.10% |
| **NLCD Class 11 - Water** (`pct_nlcd_water_11`) | Percentage of the neighborhood covered by open water (lakes, ponds, rivers). **Dropped;** the majority of neighborhoods have zero open water; features with almost no variation add noise rather than useful signal. | Float | 0.27% | 0.00% – 14.97% |
| **NLCD Class 21 - Open Development** (`pct_nlcd_dev_open_21`) | Percentage of land with open land like lawns, parks, and golf courses (less than 20% paved). Shows whether open grass and yards cool the area differently than dense trees. | Float | 30.44% | 0.00% – 82.67% |
| **NLCD Class 22 - Low Density Development** (`pct_nlcd_dev_low_22`) | Percentage of land with low-density development like single-family suburban homes (20%–49% paved). | Float | 31.01% | 0.65% – 80.95% |
| **NLCD Class 23 - Medium Density Development** (`pct_nlcd_dev_med_23`) | Percentage of land with medium-density development like townhomes and commercial strip malls (50%–79% paved). | Float | 17.49% | 0.00% – 81.71% |
| **NLCD Class 24 - High Density Development** (`pct_nlcd_dev_high_24`) | Percentage of land with high-density development like downtown buildings and industrial centers (80%–100% paved). Most important class for urban areas; identifies the areas that trap significantly more heat than flatter pavements. | Float | 9.66% | 0.00% – 83.04% |
| **NLCD Class 41–43, 90 - Forests** (`pct_nlcd_forest_total`) | Combined percentage of deciduous, evergreen, mixed forest, and woody wetlands. Measures large, dense forested patches compared to scattered street trees. | Float | 10.87% | 0.00% – 80.83% |

---

## V. Considerations, Limitations, & Ethics
- **Spatial Autocorrelation:** Adjacent neighborhoods share similar climates and canopy coverage. To prevent data leakage and artificial score inflation, model evaluation must use grouped spatial cross-validation rather than random train/test splits.
- **Feature Exclusion:** `pct_nlcd_water_11` is dropped from all modeling pipelines because the majority of neighborhoods contain 0% water, offering no variation and introducing noise.
- **Temporal Snapshot:** Satellite measurements capture a single summer afternoon snapshot (August 23, 2023 at 1:07 PM EDT) and do not reflect cooling at night or seasonal temperature changes.
- **Data Privacy:** The dataset carries no personal privacy risk because it contains no personally identifiable information, individual household data, or private property records. All inputs are derived from aggregated public government boundaries and 30-meter satellite observations, which ensures individual residents are anonymous across all 248 neighborhoods.
