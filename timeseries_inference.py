# based on the Canadian notebook
import sys
import os
from os import path, makedirs
import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import geopandas as gpd
from geocube.api.core import make_geocube
import rasterio
import rasterio.features
from rasterio.features import geometry_mask
import xarray as xr
from shapely.geometry import Point
from datetime import datetime
import inquirer
from rasterio.warp import reproject, Resampling, transform as warp_transform
from scipy.spatial import KDTree


class WaterMask:
    """
    Computes a water mask for a given port directory.
    FIRST checks if a coastline/water shapefile (*.shp) exists in masks_dir.
    SECOND (Fallback) computes an NDWI-based land mask from Sentinel-2 JP2 bands.
    """

    def __init__(self, masks_dir):
        self.masks_dir = masks_dir
        self.land_mask = None
        self.mask_transform = None
        self.mask_crs = None
        self.water_coordinates_m = None  # Storage for high-speed fallback lookup
        self._load_and_compute()

    def _find_band(self, band_id):
        """Return the path of the JP2 file whose name contains `_{band_id}_`."""
        if not path.isdir(self.masks_dir):
            return None
        for fname in os.listdir(self.masks_dir):
            if fname.lower().endswith('.jp2') and f'_{band_id}_' in fname:
                return path.join(self.masks_dir, fname)
        return None

    def _load_and_compute(self):
        b03_path = self._find_band('B03')
        b08_path = self._find_band('B08')
        
        if b03_path is None or b08_path is None:
            print(f' Δ Warning: B03/B08 JP2 files not found in {self.masks_dir}. Masking framework disabled.')
            return

        with rasterio.open(b03_path) as src:
            self.mask_transform = src.transform
            self.mask_crs = src.crs
            mask_shape = src.shape

        shp_files = glob.glob(path.join(self.masks_dir, '*.shp'))
        
        if shp_files:
            shp_path = shp_files[0]
            print(f' ! Shapefile Found in water_mask_dir: {path.basename(shp_path)}')
            
            water_gdf = gpd.read_file(shp_path)
            if water_gdf.crs != self.mask_crs:
                water_gdf = water_gdf.to_crs(self.mask_crs)
            
            inside_geom_mask = geometry_mask(water_gdf.geometry, out_shape=mask_shape, 
                                            transform=self.mask_transform, invert=False)
            
            fname_lower = path.basename(shp_path).lower()
            if 'land' in fname_lower or 'coastline' in fname_lower:
                self.land_mask = inside_geom_mask.astype(np.uint8)
            else:
                self.land_mask = (~inside_geom_mask).astype(np.uint8)
            
            print(f' ✓ Successfully rasterized shapefile boundary.')
            
            # For shapefiles, we can comfortably extract water boundary pixel coordinates
            water_pixels = np.argwhere(self.land_mask == 0)
            # Convert row/col to projection coordinates (meters)
            xs, ys = rasterio.transform.xy(self.mask_transform, water_pixels[:, 0], water_pixels[:, 1])
            self.water_coordinates_m = np.column_stack((xs, ys))
            
        else:
            print(f' ! No shapefile found in {self.masks_dir}. Processing heavy NDWI matrices dynamically...')
            if path.exists(b03_path) and path.exists(b08_path):
                print(' ✓ Band 03 and 08 are available')
            else:
                print(' Δ Band 03 and 08 are NOT available')
                return
            with rasterio.open(b03_path) as src:
                b03 = src.read(1).astype(np.float32)
            with rasterio.open(b08_path) as src:
                b08 = src.read(1).astype(np.float32)
                
            # denom = b03 + b08
            # denom[denom == 0] = np.nan
            # ndwi = (b03 - b08) / denom
            ndwi = (b03 - b08) / (b03 + b08)
            print(f"Min NDWI: {np.nanmin(ndwi)}")
            print(f"Max NDWI: {np.nanmax(ndwi)}")
            print(f"Mean NDWI: {np.nanmean(ndwi)}")
            
            del b03, b08
            
            self.land_mask = (ndwi < 0.0).astype(np.uint8)
            
            # Find coordinates where water exists (ndwi >= 0)
            print(" ✓ Extracting water coordinate spatial indexes...")
            water_pixels = np.argwhere(self.land_mask == 0)
            
            # Sample every 3rd pixel if memory is tight, or keep all for precise coastal mapping
            xs, ys = rasterio.transform.xy(self.mask_transform, water_pixels[:, 0], water_pixels[:, 1])
            self.water_coordinates_m = np.column_stack((xs, ys))
            
            del ndwi

    def compute_point_distances(self, target_xs, target_ys):
        """
        High-performance point-to-point distance algorithm using KDTree.
        Avoids dense grid space distance allocations.
        """
        if self.water_coordinates_m is None or len(self.water_coordinates_m) == 0:
            return np.full(len(target_xs), 9999.0)
            
        target_pts = np.column_stack((target_xs, target_ys))
        tree = KDTree(self.water_coordinates_m)
        distances, _ = tree.query(target_pts, workers=-1) # Uses all CPU cores
        return distances

    def apply_to_raster(self, raster_path):
        if self.land_mask is None:
            return
        with rasterio.open(raster_path) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            dst_transform = src.transform
            dst_crs = src.crs
            dst_shape = src.shape
        land_resampled = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=self.land_mask.astype(np.float32),
            destination=land_resampled,
            src_transform=self.mask_transform,
            src_crs=self.mask_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        land_pixels = land_resampled > 0.5
        print(f"--- MASKING DEBUG FOR {raster_path} ---")
        print(f"Total grid pixels: {data.size}")
        print(f"Pixels identified as LAND by the mask: {land_pixels.sum()}")
        print(f"Pixels identified as WATER by the mask: {(~land_pixels).sum()}")
        data[~land_pixels] = -9999.0
        profile.update(dtype=rasterio.float32, nodata=-9999.0)
        with rasterio.open(raster_path, 'w', **profile) as dst:
            dst.write(data, 1)
        print(f' ✓ Water mask applied: {int(land_pixels.sum()):,} land pixels retained.')


curr_time = datetime.now()
dt_str = curr_time.strftime('%Y%m%d_%H%M%S')
current_dir = path.dirname(path.realpath(__file__))
model = path.join(current_dir, 'flood_depth_train_results', 'rf_new_dd_model.pkl')
time = ['2031-01-01_12-00-00', '2036-01-01_12-00-00', '2041-01-01_12-00-00', '2046-01-01_12-00-00',
        '2051-01-01_12-00-00', '2056-01-01_12-00-00', '2061-01-01_12-00-00', '2066-01-01_12-00-00', 
        '2071-01-01_12-00-00', '2076-01-01_12-00-00', '2081-01-01_12-00-00', '2086-01-01_12-00-00', 
        '2091-01-01_12-00-00']

# ------------- criteria ---------------------
CMIP5_GCM = 'mohc_hadgem2_es'
RCM = ['smhi_rca4', 'dmi_hirham5', 'knmi_racmo22e']

port_answer = inquirer.prompt([inquirer.List('port', message='Which is the port?', choices=['Heraklion', 'Malaga', 'Azores', 'Raahe', 'Rauma', 'Bretagne'])])['port']

if port_answer == 'Azores':
    azores_answer = inquirer.prompt([inquirer.List('island', message='Which is the island?', choices=['Horta', 'PontaDelgada', 'PraiaDaVitoria'])])['island']
if port_answer == 'Bretagne':
    bretagne_answer = inquirer.prompt([inquirer.List('island', message='Which is the island?', choices=['Douarnenez', 'Audierne', 'Saint_Guenole', 'Guilvinec', 'Lesconil', 'Loctudy', 'Concarneau', 'Pont_l_Abbe'])])['island']

model_answer = inquirer.prompt([inquirer.List('model', message='For which RCP model is it?', choices=['rcp85', 'rcp45'])])['model']
if model_answer == 'rcp45':
    RCM = ['smhi_rca4', 'dmi_hirham5']

if port_answer == 'Azores':
    folder = f'{port_answer}_{azores_answer}'
    water_mask_dir = path.join(path.dirname(path.realpath(__file__)), 'water_masks', folder)
    print(f' > Searvhing in {water_mask_dir} for folder: {folder}')
elif port_answer == 'Bretagne':
    folder = f'{port_answer}_{bretagne_answer}'
    water_mask_dir = path.join(path.dirname(path.realpath(__file__)), 'water_masks', folder)
    print(f' > Searvhing in {water_mask_dir} for folder: {folder}')
else:
    water_mask_dir = path.join(path.dirname(path.realpath(__file__)), 'water_masks', port_answer)
    print(f' > Searvhing in {water_mask_dir} for folder: {port_answer}')
water_masker = WaterMask(water_mask_dir)

for t in time:
    for rcm in RCM:
        # input_basename = path.splitext(path.basename(input_csv))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # if port_answer == 'Azores':
        #     input_csv = path.join(current_dir, 'dataset4inference', f'{port_answer}_points_{model_answer}_{CMIP5_GCM}', f'Azores_{azores_answer}_random_points_{t}_{rcm}_merged.csv')
        # else:
        #     input_csv = path.join(current_dir, 'dataset4inference', f'{port_answer}_points_{model_answer}_{CMIP5_GCM}', f'{port_answer}_random_points_{t}_{rcm}_merged.csv')
        
        # if not path.exists(input_csv):
        #     continue
            
        # input_basename = path.splitext(path.basename(input_csv))[0]
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # output_csv = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{input_basename}_flood_predictions_{timestamp}.csv')
        # raster_filename = f'{input_basename}_flood_depth_{timestamp}.tif'

        if port_answer == 'Malaga':
            input_csv = path.join(current_dir, 'dataset4inference', f'{port_answer}_points_{model_answer}_{CMIP5_GCM}', f'{port_answer}_random_points_{t}_{rcm}_merged.csv')
            input_basename = path.splitext(path.basename(input_csv))[0]
            output_csv = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{input_basename}_flood_predictions_{timestamp}.csv')
            shp_path = path.join(current_dir, '..', 'CTI', 'geoms', 'port malaga_restricted.shp')
        elif port_answer == 'Azores':
            input_csv = path.join(current_dir, 'dataset4inference', f'{port_answer}_{azores_answer}_points_{model_answer}_{CMIP5_GCM}', f'Azores_{azores_answer}_random_points_{t}_{rcm}_merged.csv')
            input_basename = path.splitext(path.basename(input_csv))[0]
            output_csv = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{azores_answer}', f'{input_basename}_flood_predictions_{timestamp}.csv')
            shp_path = path.join(current_dir, '..','CTI', 'geoms', f'polygon_azores_{azores_answer}.shp')
        elif port_answer == 'Bretagne':
            input_csv = path.join(current_dir, 'dataset4inference', f'{port_answer}_{bretagne_answer}_points_{model_answer}_{CMIP5_GCM}', f'Bretagne_{bretagne_answer}_random_points_{t}_{rcm}_merged.csv')
            input_basename = path.splitext(path.basename(input_csv))[0]
            output_csv = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{bretagne_answer}', f'{input_basename}_flood_predictions_{timestamp}.csv')
            shp_path = path.join(current_dir, '..','CTI', 'geoms', f'polygon_bretagne_{bretagne_answer}.shp')
        else:
            input_csv = path.join(current_dir, 'dataset4inference', f'{port_answer}_points_{model_answer}_{CMIP5_GCM}', f'{port_answer}_random_points_{t}_{rcm}_merged.csv')
            input_basename = path.splitext(path.basename(input_csv))[0]
            output_csv = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{input_basename}_flood_predictions_{timestamp}.csv')
            shp_path = path.join(current_dir, '..', 'CTI', 'geoms', f'polygon_{port_answer.lower()}.shp')
        if path.exists(input_csv) and path.exists(shp_path):
            print(f' > Found input CSV for {port_answer}: {input_csv}\n shapefile: {shp_path}\nand will output to: {output_csv}')
        else:
            print(f' Δ Missing input CSV "{input_csv}" or shapefile "{shp_path}" for {port_answer}. Skipping this scenario.')
        raster_filename = f'{input_basename}_flood_depth_{timestamp}.tif'

        if not path.exists(input_csv):
            continue
        if path.exists(shp_path):
            herakleion_shp = gpd.read_file(shp_path)
            boundary = herakleion_shp.geometry.iloc[0]
        else:
            boundary = None

        predictive_points = pd.read_csv(input_csv, sep=',')
        essential_cols = [col for col in predictive_points.columns if col not in ['drainage_density_km_per_km2', 'nearest_river_id', 'basin_id']]
        valid_mask = predictive_points[essential_cols].notna().all(axis=1)
        
        if valid_mask.sum() == 0:
            print(f' Δ Warning: {path.basename(input_csv)} contains no valid row data. Skipping scenario')
            continue
        predictive_data = predictive_points[valid_mask].copy()
        if 'drainage_density_km_per_km2' in predictive_data.columns:
            predictive_data['drainage_density_km_per_km2'] = predictive_data['drainage_density_km_per_km2'].fillna(0)

        # ─── 1. OPTIMIZED SHORELINE BUFFER DETECTION ───
        # Reproject target points directly into the Water Mask metric CRS project space
        xs, ys = warp_transform('EPSG:4326', water_masker.mask_crs, predictive_data['longitude'].values, predictive_data['latitude'].values)
        
        # Calculate distance to water for the specific target points directly via KDTree 
        dist_to_sea = water_masker.compute_point_distances(xs, ys)

        coastal_buffer = dist_to_sea <= 15.0
        if 'elevation_m' in predictive_data.columns:
            predictive_data.loc[coastal_buffer, 'elevation_m'] = 0.0
        if 'slope_deg' in predictive_data.columns:
            predictive_data.loc[coastal_buffer, 'slope_deg'] = 0.0
        print(f' ✓ Processed distances. Overrode elevation & slope to 0 for {coastal_buffer.sum():,} points within 15m of shoreline.')

        # ─── 2. PIPELINE PREPARATION ───
        columns_to_drop = ['segment_id', 'latitude', 'longitude', 'dem_source', 'pr_source', 'country', 'flooded']
        predictive_data_features = predictive_data.drop(columns=columns_to_drop, axis=1, errors='ignore')

        with open(model, 'rb') as f:
            loaded_data = pickle.load(f)
            loaded_model = loaded_data['model']
            land_use_encoder = loaded_data['land_use_encoder']
            climate_zone_encoder = loaded_data.get('climate_zone_encoder', None)

        if land_use_encoder is not None and 'land_use' in predictive_data_features.columns:
            predictive_data_features['land_use'] = land_use_encoder.transform(predictive_data_features['land_use'])
        if climate_zone_encoder is not None and 'climate_zone' in predictive_data_features.columns:
            predictive_data_features['climate_zone'] = climate_zone_encoder.transform(predictive_data_features['climate_zone'])

        expected_features = list(loaded_model.feature_names_in_)
        predictive_data_features = predictive_data_features[expected_features]

        # ─── 3. MACHINE LEARNING INFERENCE ───
        flood_depth_pred = loaded_model.predict(predictive_data_features)
        predictive_data['flood_depth_cm'] = flood_depth_pred

        # ─── 4. INLAND WATER TAPER ───
        min_dist = 15.0
        max_dist = 100.0
        taper_scale = np.clip((dist_to_sea - min_dist) / (max_dist - min_dist), 0.0, 1.0)
        predictive_data['flood_depth_cm'] *= taper_scale

        # ─── 5. EXPORT AND GEOCUBE RASTERIZATION ───
        geometry = [Point(xy) for xy in zip(predictive_data['longitude'], predictive_data['latitude'])]
        predictive_points_gdf = gpd.GeoDataFrame(predictive_data, geometry=geometry, crs='EPSG:4326')

        if boundary is not None:
            if herakleion_shp.crs != predictive_points_gdf.crs:
                herakleion_shp_reproj = herakleion_shp.to_crs(predictive_points_gdf.crs)
                boundary = herakleion_shp_reproj.geometry.iloc[0]
            points_bounds = predictive_points_gdf.total_bounds
            boundary_bounds = gpd.GeoSeries([boundary]).bounds.iloc[0]
            
            overlap = not (boundary_bounds.iloc[2] < points_bounds[0] or boundary_bounds.iloc[0] > points_bounds[2] or 
                        boundary_bounds.iloc[3] < points_bounds[1] or boundary_bounds.iloc[1] > points_bounds[3])  
            if not overlap:
                boundary = None

        if not path.exists(path.dirname(output_csv)):
            makedirs(path.dirname(output_csv))
        predictive_data.to_csv(output_csv, index=False)

        grid_data = make_geocube(
            vector_data=predictive_points_gdf, measurements=['flood_depth_cm'], 
            resolution=0.0001, interpolate_na_method='linear'
        )
        
        if port_answer == 'Azores':
            raster_output_path = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{azores_answer}', raster_filename)
        elif port_answer == 'Bretagne':
            raster_output_path = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', f'{bretagne_answer}', raster_filename)
        else:
            raster_output_path = path.join(current_dir, f'rf_inference_results_{model_answer}', f'depth_LT50cm_{model_answer}_{CMIP5_GCM}', f'{port_answer}', raster_filename)
        grid_data['flood_depth_cm'] = grid_data['flood_depth_cm'].rio.write_crs('EPSG:4326')
        grid_data['flood_depth_cm'].rio.to_raster(raster_output_path)
        print(f' ✓ saved flood depth raster to: {raster_output_path}')

        if path.exists(raster_output_path):
            water_masker.apply_to_raster(raster_output_path)
        print('-'*60)