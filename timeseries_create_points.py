import warnings
# Silence warnings including the Google API core lifecycle alerts
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning, module='google.api_core')

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os import path, makedirs
from shapely.geometry import Point, box
import ee
import reverse_geocoder as rg
import pycountry
import inquirer

class PointGenerator:
    def __init__(self, shapefile_path, num_points, crs_out=4326):
        self.shapefile_path = shapefile_path
        self.num_points = num_points
        self.crs_out = crs_out
        
    def generate(self):
        print(f"Loading shapefile: {self.shapefile_path}")
        shp = gpd.read_file(self.shapefile_path).to_crs(epsg=self.crs_out)
        boundary = shp.geometry.iloc[0]
        
        print(f"Generating {self.num_points} random points within polygon boundary...")
        
        minx, miny, maxx, maxy = boundary.bounds
        points = []
        
        while len(points) < self.num_points:
            chunk_size = (self.num_points - len(points)) * 2
            lons = np.random.uniform(minx, maxx, chunk_size)
            lats = np.random.uniform(miny, maxy, chunk_size)
            
            for lon, lat in zip(lons, lats):
                pt = Point(lon, lat)
                if pt.within(boundary):
                    points.append(pt)
                if len(points) >= self.num_points:
                    break
                    
        gdf = gpd.GeoDataFrame(geometry=points, crs=f"EPSG:{self.crs_out}")
        print(f"✓ Generated {len(gdf)} points successfully.\n")
        return gdf, boundary

    @staticmethod
    def visualize(gdf, boundary):
        fig, ax = plt.subplots(figsize=(10, 12))
        gpd.GeoSeries([boundary], crs="EPSG:4326").plot(ax=ax, color="white", edgecolor="0.8", linewidth=0.5)
        gdf.plot(ax=ax, markersize=5, color="green", aspect=1)
        plt.title("Random Points in Polygon")
        plt.show()
        return fig

class GEEDataEnricher:
    """Enriches spatial coordinates using fast GEE batch FeatureCollection reductions."""
    def __init__(self, project_id):
        self.project_id = project_id
        self.initialized = self._initialize_gee()
        
    def _initialize_gee(self):
        try:
            ee.Initialize(project=self.project_id)
            print(f"✓ Connected to GEE (Project: {self.project_id})")
            return True
        except Exception:
            print("Δ Attempting GEE Authentication...")
            try:
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
                print(f"✓ GEE Authenticated successfully.")
                return True
            except Exception as auth_error:
                print(f"✕ GEE Initialization failed: {auth_error}")
                return False

    def enrich_points(self, df, dem_dataset="COPERNICUS", lulc_dataset="DYNAMICWORLD"):
        if not self.initialized:
            print("✕ GEE not initialized. Skipping remote lookups.")
            return df
            
        print(f"Batch processing {len(df)} features directly on Earth Engine servers...")
        
        features = []
        for idx, row in df.iterrows():
            geom = ee.Geometry.Point([row['longitude'], row['latitude']])
            feat = ee.Feature(geom, {'orig_idx': int(idx)})
            features.append(feat)
        fc = ee.FeatureCollection(features)
        
        wbm_img = ee.ImageCollection('COPERNICUS/DEM/GLO30').mosaic().select('WBM')
        
        dem_collection_name = 'COPERNICUS/DEM/GLO30' if dem_dataset == 'COPERNICUS' else 'MERIT/Hydro/v1_0_1'
        dem_band = 'DEM' if dem_dataset == 'COPERNICUS' else 'elv'
        dem_scale = 30 if dem_dataset == 'COPERNICUS' else 90
        
        dem_img = ee.ImageCollection(dem_collection_name).select(dem_band).mosaic()
        base_proj = ee.Image(ee.ImageCollection(dem_collection_name).first()).projection()
        dem_img_projected = dem_img.setDefaultProjection(base_proj)
        slope_img = ee.Terrain.slope(dem_img_projected)
        
        lulc_img = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').select('label').mode()
        
        combined_stack = ee.Image.cat([
            wbm_img.rename('wbm_val'),
            dem_img.rename('elevation_m'),
            slope_img.rename('slope_deg'),
            lulc_img.rename('land_use_idx')
        ])
        
        sampled_fc = combined_stack.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            scale=dem_scale
        )
        
        sampled_info = sampled_fc.getInfo()
        
        wbm_map, elev_map, slope_map, lulc_map = {}, {}, {}, {}
        lulc_classes = {0:'water', 1:'trees', 2:'grass', 3:'flooded_vegetation', 
                        4:'crops', 5:'shrub_and_scrub', 6:'built', 7:'bare', 8:'snow_and_ice'}
        
        for feat in sampled_info['features']:
            props = feat['properties']
            idx = props['orig_idx']
            wbm_map[idx] = props.get('wbm_val')
            elev_map[idx] = props.get('elevation_m')
            slope_map[idx] = props.get('slope_deg')
            
            l_idx = props.get('land_use_idx')
            lulc_map[idx] = lulc_classes.get(int(l_idx), 'unknown') if l_idx is not None else 'unknown'
            
        df['wbm_value'] = df.index.map(wbm_map)
        df['elevation_m'] = df.index.map(elev_map)
        df['dem_source'] = dem_dataset
        df['slope_deg'] = df.index.map(slope_map)
        df['land_use'] = df.index.map(lulc_map)
        
        initial_count = len(df)
        df = df[df['wbm_value'] != 1].copy()
        print(f"✓ Land filter passed: retained {len(df)}/{initial_count} dry points.")
        
        return df

class DrainageDensityCalculator:
    def __init__(self, basin_paths, rivers_path, buffer_distance_m=1000):
        self.basin_paths = basin_paths
        self.rivers_path = rivers_path
        self.buffer_distance_m = buffer_distance_m

    def compute(self, gdf_points):
        print("Calculating drainage density metrics...")
        bounds = gdf_points.total_bounds
        
        # Buffer the bounding box slightly (0.05 degrees) to catch nearby rivers/basins
        bbox = box(*bounds).buffer(0.05)
        
        basin_list = []
        for p in self.basin_paths:
            if path.exists(p):
                shp = gpd.read_file(p)
                # CRITICAL FIX: Reproject to match points CRS *BEFORE* clipping
                if shp.crs != gdf_points.crs:
                    shp = shp.to_crs(gdf_points.crs)
                shp = shp.clip(bbox) 
                basin_list.append(shp)
        
        if not basin_list:
            print("✕ Spatial Basins missing. Skipping drainage logic.")
            gdf_points['drainage_density_km_per_km2'] = None
            return gdf_points
            
        basins = pd.concat(basin_list, ignore_index=True)
        if basins.crs is None: 
            basins.set_crs(epsg=4326, inplace=True)
        
        try:
            rivers_raw = gpd.read_file(self.rivers_path, layer='HydroRIVERS_v10')
            # CRITICAL FIX: Reproject rivers to match points CRS *BEFORE* clipping
            if rivers_raw.crs != gdf_points.crs:
                rivers_raw = rivers_raw.to_crs(gdf_points.crs)
            rivers = rivers_raw.clip(bbox)
        except Exception as e:
            print(f"✕ Failed to pull layer HydroRIVERS: {e}")
            gdf_points['drainage_density_km_per_km2'] = None
            return gdf_points

        # Spatial Join: Link points directly to containing basin properties
        joined = gpd.sjoin(gdf_points, basins[['HYBAS_ID', 'SUB_AREA', 'geometry']], how='left', predicate='within')
        
        # Check matching performance
        matched_count = joined["HYBAS_ID"].notna().sum()
        print(f" ✓ Mapped {matched_count}/{len(gdf_points)} points into local basin geometries.")

        # Project to metric coordinate reference system for true buffer calculations
        joined_m = joined.to_crs(epsg=3035)
        rivers_m = rivers.to_crs(epsg=3035)
        
        densities = []
        for idx, row in joined_m.iterrows():
            pt_basin = row['HYBAS_ID']
            if pd.isna(pt_basin):
                densities.append(0.0)
                continue
                
            # Grab rivers within the specified metric search radius buffer
            local_rivers = rivers_m[rivers_m.intersects(row.geometry.buffer(self.buffer_distance_m))]
            if len(local_rivers) > 0:
                # Isolate the segment matching the primary sub-basin system structure ID
                basin_specific = local_rivers[local_rivers['HYBAS_L12'] == pt_basin]
                river_row = basin_specific.iloc[0] if len(basin_specific) > 0 else local_rivers.iloc[0]
                
                dd = river_row['LENGTH_KM'] / row['SUB_AREA'] if row['SUB_AREA'] > 0 else 0.0
                densities.append(dd)
            else:
                densities.append(0.0)
                
        joined['drainage_density_km_per_km2'] = densities
        
        if 'index_right' in joined.columns:
            joined = joined.drop(columns=['index_right'])
            
        return joined

class ClimateDataProcessor:
    def __init__(self, port_answer, azores_answer, bretagne_answer, pr_path, gcm_name, rcm_list, climate_zones_path):
        self.port_answer = port_answer
        self.azores_answer = azores_answer
        self.bretagne_answer = bretagne_answer
        self.pr_path = pr_path
        self.gcm = gcm_name
        self.rcm_list = rcm_list
        self.climate_zones_path = climate_zones_path

    def process_meteorology(self, base_df, target_time, output_csv_base):
        # read daily precipitation csv files like "mohc_hadgem2_es-smhi_rca4_daily_grid2x2_Malaga.csv"
        date_part, time_part = target_time.split('_')
        time_format_csv = f"{date_part} {time_part.replace('-', ':')}"

        pt_lats = base_df['latitude'].values
        pt_lons = base_df['longitude'].values

        pr_values = {}
        for rcm in self.rcm_list:
            if self.port_answer == 'Azores':
                pr_csv_file = f'{self.gcm}-{rcm}_daily_grid2x2_{self.port_answer}_{self.azores_answer}.csv'
            # elif self.port_answer == 'Bretagne':
            #     pr_csv_file = f'{self.gcm}-{rcm}_daily_grid2x2_{self.port_answer}_{self.bretagne_answer}.csv'
            else:
                pr_csv_file = f'{self.gcm}-{rcm}_daily_grid2x2_{self.port_answer}.csv'
            pr_csv_path = path.join(self.pr_path, pr_csv_file)
            if not path.exists(pr_csv_path):
                print(f' X Precipitation CSV file not found: {pr_csv_path}')
                pr_values[rcm] = pd.Series(0.0, index=base_df.index)
                continue
            else:
                print(f' > Found precipitation file: {pr_csv_path} for RCM: {rcm}')
            df_pr = pd.read_csv(pr_csv_path)
            grid_at_time = df_pr[df_pr['time'] == time_format_csv]
            if grid_at_time.empty:
                print(f' ✕ Time {time_format_csv} not found in {pr_csv_file}')
                pr_values[rcm] = pd.Series(0.0, index=base_df.index)
                continue

            # Nearest-grid-cell lookup: for each point find the closest lat/lon in the grid
            grid_lats = grid_at_time['lat'].values
            grid_lons = grid_at_time['lon'].values
            pr_grid = grid_at_time['pr'].values

            dist2 = (pt_lats[:, np.newaxis] - grid_lats) ** 2 + (pt_lons[:, np.newaxis] - grid_lons) ** 2
            nearest_idx = dist2.argmin(axis=1)
            pr_series = pd.Series(pr_grid[nearest_idx], index=base_df.index)
            pr_values[rcm] = pr_series
            print(f' > {pr_csv_file} | {time_format_csv} | pr range [{pr_series.min():.6e}, {pr_series.max():.6e}]')

        for rcm in self.rcm_list:
            rcm_csv_path = f"{path.splitext(output_csv_base)[0]}_{rcm}_merged.csv"
            print(f"Building scenario dataset: {path.basename(rcm_csv_path)}")

            df_slice = base_df.copy()
            df_slice['rainfall_intensity_mm_hr'] = pr_values.get(rcm, pd.Series(0.0, index=base_df.index))
            df_slice['pr_source'] = f"{self.gcm}-{rcm}"
            
            df_slice = self._add_climate_zones(df_slice)
            
            desired_cols = [
                'segment_id', 'latitude', 'longitude', 'elevation_m', 'dem_source',
                'slope_deg', 'land_use', 'rainfall_intensity_mm_hr', 'pr_source',
                'drainage_density_km_per_km2', 'country', 'climate_zone'
            ]
            df_slice = df_slice[[c for c in desired_cols if c in df_slice.columns]]
            df_slice.to_csv(rcm_csv_path, index=False)

    def _add_climate_zones(self, df):
        if 'latitude' not in df.columns: return df
        coords = list(zip(df['latitude'], df['longitude']))
        
        results = rg.search(coords, verbose=False)
        
        countries = []
        for r in results:
            try:
                c_obj = pycountry.countries.get(alpha_2=r['cc'])
                countries.append(c_obj.name if c_obj else 'unknown')
            except Exception:
                countries.append('unknown')
        df['country'] = countries
        
        if path.exists(self.climate_zones_path):
            cz_csv = pd.read_csv(self.climate_zones_path, sep=';')
            zone_map = {str(row['Countries / Regions']).strip(): row['Climate Zone'] for _, row in cz_csv.iterrows()}
            df['climate_zone'] = df['country'].apply(
                lambda c: next((v for k, v in zone_map.items() if c in k), 'unknown')
            )
        else:
            df['climate_zone'] = 'unknown'
            
        return df

def main():
    PROJECT_ID = 'ee-project-485013'
    NUM_POINTS = 5000
    CMIP5_GCM = 'mohc_hadgem2_es'
    RCM = ['smhi_rca4', 'dmi_hirham5', 'knmi_racmo22e']
    TIME_STEPS = ['2031-01-01_12-00-00', '2036-01-01_12-00-00', '2041-01-01_12-00-00', '2046-01-01_12-00-00',
                '2051-01-01_12-00-00', '2056-01-01_12-00-00', '2061-01-01_12-00-00', '2066-01-01_12-00-00', 
                '2071-01-01_12-00-00', '2076-01-01_12-00-00', '2081-01-01_12-00-00', '2086-01-01_12-00-00', 
                '2091-01-01_12-00-00']
    ID_PATTERN = 'GSH-{:05d}'

    port_question = [
        inquirer.List('port', 
                      message='Which is the port?',
                      choices=['Heraklion', 'Malaga', 'Azores', 'Raahe', 'Rauma', 'Bretagne'])
    ]
    port_answer = inquirer.prompt(port_question)['port']
    
    azores_answer = None
    bretagne_answer = None
    if port_answer == 'Azores':
        azores_question = [
            inquirer.List('azores_island', 
                          message='Which island?',
                          choices=['Horta', 'PontaDelgada', 'PraiaDaVitoria'])
        ]
        azores_answer = inquirer.prompt(azores_question)['azores_island']

    if port_answer == 'Bretagne':
        bretagne_question = [
            inquirer.List('bretagne_port', 
                            message='Which port?',
                            choices=['Douarnenez', 'Audierne', 'Saint_Guenole', 'Guilvinec', 
                                'Lesconil', 'Loctudy', 'Concarneau', 'Pont_l_Abbe'])
        ]
        bretagne_answer = inquirer.prompt(bretagne_question)['bretagne_port']

    model_question = [
        inquirer.List('model',
                      message='For which RCP model is it?',
                      choices=['rcp85', 'rcp45'])
    ]
    model_answer = inquirer.prompt(model_question)['model']
    

    current_dir = path.dirname(path.realpath(__file__))
    if port_answer == 'Malaga':
        shp_path = path.join(current_dir, '..', 'CTI', 'geoms', 'port malaga_restricted.shp')
        pr_path = path.join(current_dir, '..', 'ICCS', 'clim_dc', f'{model_answer}', f'{port_answer}_grid2x2')
        out_dir = path.join(current_dir, 'dataset4inference', f'{port_answer}_points_{model_answer}_{CMIP5_GCM}')
        makedirs(out_dir, exist_ok=True)
    elif port_answer == 'Azores':
        shp_path = path.join(current_dir, '..','CTI', 'geoms', f'polygon_azores_{azores_answer}.shp')
        pr_path = path.join(current_dir, '..', 'ICCS', 'clim_dc', f'{model_answer}', f'Azores_{azores_answer}_grid2x2')
        out_dir = path.join(current_dir, 'dataset4inference', f'{port_answer}_{azores_answer}_points_{model_answer}_{CMIP5_GCM}')
        makedirs(out_dir, exist_ok=True)
    elif port_answer == 'Bretagne':
        shp_path = path.join(current_dir, '..','CTI', 'geoms', f'polygon_bretagne_{bretagne_answer}.shp')
        pr_path = path.join(current_dir, '..', 'ICCS', 'clim_dc', f'{model_answer}', f'Bretagne_grid2x2')
        out_dir = path.join(current_dir, 'dataset4inference', f'{port_answer}_{bretagne_answer}_points_{model_answer}_{CMIP5_GCM}')
        makedirs(out_dir, exist_ok=True)
    else:
        shp_path = path.join(current_dir, '..', 'CTI', 'geoms', f'polygon_{port_answer.lower()}.shp')
        pr_path = path.join(current_dir, '..', 'ICCS', 'clim_dc', f'{model_answer}', f'{port_answer}_grid2x2')
        out_dir = path.join(current_dir, 'dataset4inference', f'{port_answer}_points_{model_answer}_{CMIP5_GCM}')
        makedirs(out_dir, exist_ok=True)

    if model_answer == 'rcp45':
        RCM = ['smhi_rca4', 'dmi_hirham5']

    if not path.exists(shp_path):
        print(f" X Shapefile not found at {shp_path}. Please check the path and try again")
        return
    if not path.exists(pr_path):
        print(f" X Climate data directory not found at {pr_path}. Please check the path and try again")
        return
    
    # shp_path = path.join(current_dir, '..', '..', 'CTI', 'geoms', 'polygon_heraklion.shp')
    # pr_path = path.join(current_dir, '..', '..', 'ICCS', 'clim_dc', f'{model_answer}', 'Heraklion_grid2x2')
    climate_zones_path = path.join(current_dir, '..', 'climate_zones', 'eu_climate_zones.csv')
    
    hydrosheds_dir = 'HydroSHEDS'
    basin_paths = [path.join(current_dir, '..', hydrosheds_dir, 'hybas_eu_lev01-12_v1c', 'hybas_eu_lev12_v1c.shp')] # nono keep the []
    rivers_path = path.join(current_dir, '..', hydrosheds_dir, 'HydroRIVERS_v10', 'HydroRIVERS_v10.gdb')
    
    generator = PointGenerator(shp_path, NUM_POINTS)
    gdf_points, boundary = generator.generate()
    
    base_df = pd.DataFrame({
        'segment_id': [ID_PATTERN.format(i+1) for i in range(len(gdf_points))],
        'latitude': gdf_points.geometry.y,
        'longitude': gdf_points.geometry.x
    })
    
    enricher = GEEDataEnricher(PROJECT_ID)
    base_df = enricher.enrich_points(base_df, dem_dataset='COPERNICUS')
    
    gdf_filtered = gpd.GeoDataFrame(
        base_df, geometry=gpd.points_from_xy(base_df.longitude, base_df.latitude), crs="EPSG:4326"
    )
    
    calc = DrainageDensityCalculator(basin_paths, rivers_path)
    enriched_spatial_df = calc.compute(gdf_filtered)
    
    climate_processor = ClimateDataProcessor(port_answer, azores_answer, bretagne_answer, pr_path, CMIP5_GCM, RCM, climate_zones_path)
    
    for target_time in TIME_STEPS:
        print(f"\n{'='*60}\nProcessing Scenario Time Window: {target_time}\n{'='*60}")
        if port_answer == 'Azores':
            output_csv_base = path.join(out_dir, f'{port_answer}_{azores_answer}_random_points_{target_time}.csv')
        elif port_answer == 'Bretagne':
            output_csv_base = path.join(out_dir, f'{port_answer}_{bretagne_answer}_random_points_{target_time}.csv')
        else:
            output_csv_base = path.join(out_dir, f'{port_answer}_random_points_{target_time}.csv')
        
        climate_processor.process_meteorology(enriched_spatial_df, target_time, output_csv_base)
        
    print("\n✓ Processing workflow pipeline operations finished successfully.")

if __name__ == '__main__':
    main()