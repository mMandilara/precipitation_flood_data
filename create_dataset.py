import os
from os import path, makedirs, listdir
import numpy as np
import sys
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Point, box
import ee
from pathlib import Path
import json
import warnings
import random
import pycountry
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Set PROJ_LIB safely
# os.environ['PROJ_LIB'] = r'C:\Users\USERNAME\miniconda3\envs\PROJECT_NAME\Library\share\proj'
GEE_CONFIG_PATH = Path.home() / '.gee_config.json'

def get_gee_project():
    saved_proj = None
    if GEE_CONFIG_PATH.exists():
        try:
            with open(GEE_CONFIG_PATH, 'r') as f:
                saved_proj = json.load(f).get('project_id')
        except (json.JSONDecodeError, KeyError):
            print('Warning: Could not read GEE project from config. Proceeding without it.')
            saved_proj = None

    if saved_proj:
        use_save = input(f'Use saved project "{saved_proj}"? (y/n): ').strip().lower()
        if use_save == 'y' or use_save == '':
            if _validate_gee_project(saved_proj):
                return saved_proj
            print(f' X Saved project "{saved_proj}" failed to initialize. Please check and try again')

    while True:
        project_id = input('Enter your GEE project ID (or leave blank to skip): ').strip()
        if not project_id:
            print('X Project ID cannot be empty...')
        if _validate_gee_project(project_id):
            break
        print(f' X Project ID "{project_id}" failed to initialize. Please check and try again.')
    
    save = input('Save this project ID for future runs? (y/n): ').strip().lower()
    if save == 'y':
        try:
            with open(GEE_CONFIG_PATH, 'w') as f:
                json.dump({'project_id': project_id}, f)
            print(f' ✓ Saved to {GEE_CONFIG_PATH}')
        except OSError as e:
            print(f' X Could not save config: {e}')

    return project_id


def _validate_gee_project(project_id):
    try:
        ee.Initialize(project=project_id)
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            return True
        except Exception as e:
            print(f' X GEE validation error: {e}')
            return False


class downloadFloodTIFs:
    def __init__(self, years, current_dir, efas_dir, base_url):
        self.years = years
        self.current_dir = current_dir
        self.efas_dir = efas_dir
        self.base_url = base_url
        self.download_path = path.join(self.efas_dir, self.years[0] if self.years else "YEARS_NOT_SPECIFIED")

    def fetch_from_ftp(self):
        for y in self.years:
            url = self.base_url + y + '/'
            download_path = path.join(self.efas_dir, y)
            if not path.exists(download_path):
                makedirs(download_path, exist_ok=True)
            self.download_path = download_path
            
            makedirs(download_path, exist_ok=True)
            
            try:
                print(f'\n >> Accessing URL: {url}')
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                print('  Success connection to JRC Open Data Repository!')

                tif_links = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.tif')]
                print(f'Found {len(tif_links)} TIF files on server')
                
                downloaded_months = set()
                for existing_file in listdir(download_path):
                    if existing_file.endswith('.tif'):
                        match = re.search(r'WD_MERGE_(\d{4})-(\d{2})-\d{2}', existing_file)
                        if match:
                            downloaded_months.add(f"{match.group(1)}-{match.group(2)}")
                
                files_downloaded = 0
                for filename in tif_links:
                    match = re.search(r'WD_MERGE_(\d{4})-(\d{2})-\d{2}', filename)
                    if match:
                        year_month = f'{match.group(1)}-{match.group(2)}'
                        if year_month not in downloaded_months:
                            file_path = path.join(download_path, filename)
                            print(f'  Downloading variant window for {year_month}: {filename}')

                            try:
                                with requests.get(url + filename, stream=True, timeout=60) as r:
                                    r.raise_for_status()
                                    with open(file_path, 'wb') as f:
                                        for chunk in r.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                files_downloaded += 1
                                downloaded_months.add(year_month)
                
                                print(f'Downloaded {files_downloaded} new raw rasters for year target context {y}')
                            except Exception as e:
                                print(f'  Error downloading {filename}: {e}')
            except Exception as e:
                print(f'Error accessing platform repository metadata mapping framework {url}: {e}')
        return self.download_path

class GEEBulkEnricher:
    """Manages robust, high-volume batch server-side feature extractions securely on a single thread."""
    def __init__(self, project_id):
        self.project_id = project_id
        self.initialized = self._init_gee()

    def _init_gee(self):
        try:
            ee.Initialize(project=self.project_id)
            return True
        except Exception:
            try:
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
                return True
            except Exception as e:
                print(f"✕ Massive GEE Instance failure: {e}")
                return False

    def _sample_batch(self, batch_df, static_stack, start_date, end_date, lulc_start, lulc_end, hours):
        """One GEE round trip for a small group of points. Returns {orig_idx: properties}."""
        features = [ee.Feature(ee.Geometry.Point([lon, lat]), {'orig_idx': int(idx)})
                    for idx, lon, lat in zip(batch_df.index, batch_df['longitude'], batch_df['latitude'])]
        fc = ee.FeatureCollection(features)

        # Restrict heavy collections to just the area covered by this batch
        region = ee.Geometry.Rectangle(
            [float(batch_df['longitude'].min()), float(batch_df['latitude'].min()),
             float(batch_df['longitude'].max()), float(batch_df['latitude'].max())],
            proj=None, geodesic=False)

        lulc = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                .filterDate(lulc_start, lulc_end).filterBounds(region)
                .select('label').mode().rename('lulc'))

        pr_collection = (ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                         .filterDate(start_date, end_date).filterBounds(region)
                         .select('total_precipitation'))
        avg_pr_rate = pr_collection.sum().multiply(1000).divide(hours).rename('pr_rate')  # m -> mm, per hour

        # Static layers (30 m) + land use, sampled at 30 m
        sampled_a = static_stack.addBands(lulc).reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=30).getInfo()['features']
        # ERA5-Land is ~11 km native, so sample it at its own resolution (much cheaper)
        sampled_b = avg_pr_rate.reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=11132).getInfo()['features']

        out = {}
        for feat in sampled_a:
            out[feat['properties']['orig_idx']] = dict(feat['properties'])
        for feat in sampled_b:
            out.setdefault(feat['properties']['orig_idx'], {}).update(feat['properties'])
        return out

    def _sample_with_retry(self, batch_df, min_batch=10, **kw):
        """If GEE complains about memory/timeouts, split the batch in half and retry."""
        try:
            return self._sample_batch(batch_df, **kw)
        except Exception as e:
            msg = str(e).lower()
            retryable = any(s in msg for s in ('memory', 'timed out', 'timeout', 'too many', 'computation'))
            if retryable and len(batch_df) > min_batch:
                mid = len(batch_df) // 2
                print(f"  ↻ Batch of {len(batch_df)} failed ({e}); splitting into {mid} + {len(batch_df) - mid}")
                out = self._sample_with_retry(batch_df.iloc[:mid], min_batch, **kw)
                out.update(self._sample_with_retry(batch_df.iloc[mid:], min_batch, **kw))
                return out
            print(f"✕ Reduction lookup timeout anomaly fallback (batch of {len(batch_df)} skipped): {e}")
            return {}

    def process_dataframe(self, df, start_date, end_date, dem_ds="COPERNICUS", pr_ds="ERA5",
                          batch_size=250, lulc_pad_days=365):
        if not self.initialized or df.empty:
            df['wbm_value'] = 0
            df['elevation_m'] = np.nan
            df['slope_deg'] = np.nan
            df['land_use'] = 'unknown'
            df['pr_avg_mm_hr'] = 0.0
            df['pr_source'] = 'failed'
            return df

        # Defaults, so that a failed batch never leaves missing columns.
        # wbm_value stays NaN on failure -> those rows are dropped later by the land mask (== 0).
        df['wbm_value'] = np.nan
        df['elevation_m'] = np.nan
        df['slope_deg'] = np.nan
        df['land_use'] = 'unknown'
        df['pr_avg_mm_hr'] = 0.0

        # Static datasets (lazy, built once)
        wbm = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1').mosaic().select('WBM')
        dem = ee.ImageCollection('COPERNICUS/DEM/GLO30_2024_1').select('DEM').mosaic()
        slope = ee.Terrain.slope(dem)
        static_stack = ee.Image.cat([wbm.rename('wbm'), dem.rename('elev'), slope.rename('slope')])

        hours = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).total_seconds() / 3600, 1)
        # Dynamic World: mode over a window around the event instead of the whole archive
        lulc_start = (pd.to_datetime(start_date) - pd.Timedelta(days=lulc_pad_days)).strftime('%Y-%m-%d')
        lulc_end = (pd.to_datetime(end_date) + pd.Timedelta(days=lulc_pad_days)).strftime('%Y-%m-%d')

        lulc_map = {0:'water', 1:'trees', 2:'grass', 3:'flooded_vegetation', 4:'crops', 5:'shrub_and_scrub', 6:'built', 7:'bare', 8:'snow_and_ice'}

        # Sort spatially so each batch covers a small area
        ordered = df.sort_values(['latitude', 'longitude'])
        for start in range(0, len(ordered), batch_size):
            batch = ordered.iloc[start:start + batch_size]
            print(f"  ▶ submitting batch {start // batch_size + 1}/{-(-len(ordered) // batch_size)} ({len(batch)} pts)...")
            results = self._sample_with_retry(
                batch, static_stack=static_stack, start_date=start_date, end_date=end_date,
                lulc_start=lulc_start, lulc_end=lulc_end, hours=hours)


            for idx, p in results.items():
                df.at[idx, 'wbm_value'] = p.get('wbm', 0)
                df.at[idx, 'elevation_m'] = p.get('elev', np.nan)
                df.at[idx, 'slope_deg'] = p.get('slope', np.nan)
                df.at[idx, 'pr_avg_mm_hr'] = p.get('pr_rate', 0.0)
                l_idx = p.get('lulc')
                df.at[idx, 'land_use'] = lulc_map.get(int(l_idx), 'unknown') if l_idx is not None else 'unknown'
            print(f"  GEE batch {start // batch_size + 1}/{-(-len(ordered) // batch_size)} done ({len(results)}/{len(batch)} points)")

        df['dem_source'] = dem_ds
        df['pr_source'] = np.where(df['wbm_value'].isna(), 'failed', pr_ds)
        return df


class createNonFloodPoints:
    def __init__(self, id_pattern, years, current_dir, efas_dir, download_path, num_points_nonflood=200, output_format='csv_wgs84', gee_project=None, precipitation_dataset=None, elevation_dataset=None, max_workers=1):
        self.id_pattern = id_pattern
        self.years = years
        self.current_dir = current_dir
        self.efas_dir = efas_dir
        self.download_path = download_path
        self.num_points_nonflood = num_points_nonflood
        self.gee_project = gee_project
        self.max_workers = max_workers
        self.lulc_dataset = 'DYNAMICWORLD'

    def extract_dates_from_raster(self, filename):
        match = re.search(r'WD_MERGE_(\d{4}-\d{2}-\d{2})---(\d{4}-\d{2}-\d{2})', filename)
        return (match.group(1), match.group(2)) if match else (None, None)

    def preload_hydro_data(self):
        print(' > Preloading Global HydroSHEDS references systematically...')
        hydrosheds_dir = path.join(self.current_dir, 'HydroSHEDS')
        continents = ['eu', 'af', 'as', 'na', 'sa', 'si', 'au', 'gr']
        basin_files = []

        for c in continents:
            p = path.join(hydrosheds_dir, f'hybas_{c}_lev01-12_v1c', f'hybas_{c}_lev12_v1c.shp')
            if path.exists(p):
                try:
                    gdf = gpd.read_file(p)
                    if gdf.crs is None:
                        gdf.set_crs(epsg=4326, inplace=True)
                    basin_files.append(gdf)
                    print(f"  Loaded basin file: {p}")
                except Exception as e:
                    print(f"  Warning: Could not read basin file {p}: {e}")

        # Create empty GeoDataFrame with proper geometry column if no files found
        if basin_files:
            print('')
            basins_gdf = pd.concat(basin_files, ignore_index=True)
        else:
            # Create an empty GeoDataFrame with the correct schema
            from shapely.geometry import Polygon
            empty_geom = Polygon()
            basins_gdf = gpd.GeoDataFrame(
                columns=['HYBAS_ID', 'NEXT_DOWN', 'NEXT_SINK', 'MAIN_BAS', 'DIST_SINK',
                        'DIST_MAIN', 'SUB_AREA', 'UP_AREA', 'PFAF_ID', 'ENDO',
                        'COAST', 'ORDER', 'SORT', 'geometry'],
                geometry=[empty_geom],
                crs="EPSG:4326"
            ).iloc[0:0]  # Create empty but with correct structure

        # For rivers
        rivers_path = path.join(hydrosheds_dir, 'HydroRIVERS_v10', 'HydroRIVERS_v10.gdb')
        if path.exists(rivers_path):
            try:
                rivers_gdf = gpd.read_file(rivers_path, layer='HydroRIVERS_v10')
                if rivers_gdf.crs is None:
                    rivers_gdf.set_crs(epsg=4326, inplace=True)
                print(f"  Loaded rivers file: {rivers_path}")
            except Exception as e:
                print(f"  Warning: Could not read rivers file: {e}")
                # Create empty rivers GeoDataFrame
                from shapely.geometry import LineString
                empty_geom = LineString()
                rivers_gdf = gpd.GeoDataFrame(
                    columns=['LENGTH_KM', 'geometry'],
                    geometry=[empty_geom],
                    crs="EPSG:4326"
                ).iloc[0:0]
        else:
            # Create empty rivers GeoDataFrame
            from shapely.geometry import LineString
            empty_geom = LineString()
            rivers_gdf = gpd.GeoDataFrame(
                columns=['LENGTH_KM', 'geometry'],
                geometry=[empty_geom],
                crs="EPSG:4326"
            ).iloc[0:0]

        return basins_gdf, rivers_gdf

    def _process_nonflood_tif(self, tif_path, boundary, basins_gdf, rivers_gdf):
        image_name = path.basename(tif_path)
        output_csv = path.join(path.dirname(tif_path), f'{image_name[:-4]}_nonflood_points.csv')
        if path.exists(output_csv):
            print(f'  ↷ skip (CSV exists): {image_name}')
            return
        print(f'  ... start: {image_name}')

        start_date, end_date = self.extract_dates_from_raster(image_name)
        if not start_date: return

        with rasterio.open(tif_path) as src:
            tif_crs = src.crs
            nodata = src.nodata

        geom = boundary.geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        candidates = []
        
        while len(candidates) < self.num_points_nonflood:
            lons = np.random.uniform(minx, maxx, self.num_points_nonflood * 2)
            lats = np.random.uniform(miny, maxy, self.num_points_nonflood * 2)
            for lon, lat in zip(lons, lats):
                if Point(lon, lat).within(geom):
                    candidates.append((lon, lat))
                if len(candidates) >= self.num_points_nonflood: break

        df = pd.DataFrame(candidates, columns=['longitude', 'latitude'])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
        
        gdf_raster = gdf.to_crs(tif_crs)
        with rasterio.open(tif_path) as src:
            df['tif_value'] = [v[0] for v in src.sample([(g.x, g.y) for g in gdf_raster.geometry])]

        enricher = GEEBulkEnricher(self.gee_project)
        df = enricher.process_dataframe(df, start_date, end_date)

        tif_mask = df['tif_value'].isna() | (df['tif_value'] == 0) | (df['tif_value'] == nodata)
        land_mask = df['wbm_value'] == 0
        df = df[tif_mask & land_mask].copy()

        if df.empty: return

        df['segment_id'] = [self.id_pattern.format(i+1) for i in range(len(df))]
        df['event_start_date'] = start_date
        df['event_end_date'] = end_date

        dd_adder = addDrainageDensity(output_csv, basins_gdf, rivers_gdf, 1000)
        df = dd_adder.calculate_dd_direct(df)

        df = addClimateZone(self.current_dir, output_csv, "").enrich_direct(df)
        df['flooded'] = 0
        
        df.to_csv(output_csv, index=False)
        print(f" ✓ Successfully completed extraction logic for: {image_name[:-4]}")

    def convert_boundary_to_tif_points(self):
        basins_gdf, rivers_gdf = self.preload_hydro_data()
        
        years_folders = self.efas_dir
        boundaries = []
        for folder in listdir(years_folders):
            folder_path = path.join(years_folders, folder)
            if path.isdir(folder_path) and folder in self.years:
                for tif_file in [f for f in listdir(folder_path) if f.endswith('.tif')]:
                    t_path = path.join(folder_path, tif_file)
                    with rasterio.open(t_path) as src:
                        b = src.bounds
                        poly = box(b.left, b.bottom, b.right, b.top)
                        rcrs = src.crs
                    gdf_b = gpd.GeoDataFrame({'geometry': [poly]}, crs=rcrs).to_crs(epsg=4326)
                    boundaries.append((t_path, gdf_b))
        print(f' > {len(boundaries)} TIF boundaries queued for non-flood points')

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_nonflood_tif, tp, bd, basins_gdf, rivers_gdf) for tp, bd in boundaries]
            for f in as_completed(futures): f.result()


class createFloodPoints:
    def __init__(self, id_pattern, years, current_dir, efas_dir, download_path, point_spacing=300, output_format='csv_wgs84', gee_project=None, precipitation_dataset=None, elevation_dataset=None, max_workers=1):
        self.id_pattern = id_pattern
        self.years = years
        self.current_dir = current_dir
        self.efas_dir = efas_dir
        self.download_path = download_path
        self.point_spacing = point_spacing
        self.gee_project = gee_project
        self.max_workers = max_workers

    def extract_dates_from_raster(self, filename):
        match = re.search(r'WD_MERGE_(\d{4}-\d{2}-\d{2})---(\d{4}-\d{2}-\d{2})', filename)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _process_flood_tif(self, tif_path, output_csv, basins_gdf, rivers_gdf):
        if path.exists(output_csv): return
        image_name = path.basename(tif_path)
        start_date, end_date = self.extract_dates_from_raster(image_name)
        if path.exists(output_csv):
            print(f'  ↷ skip flood (CSV exists): {image_name}')
            return
        print(f'  ⏳ start flood: {image_name}')

        with rasterio.open(tif_path) as src:
            transform = src.transform
            nodata = src.nodata
            crs_wkt = src.crs.to_wkt()
            data = src.read(1) 

        valid_mask = (data >= 50) if nodata is None else (data != nodata) & (data >= 50)
        rows, cols = np.where(valid_mask)
        if len(rows) == 0: return

        xs, ys = rasterio.transform.xy(transform, rows, cols)
        df = pd.DataFrame({'x': xs, 'y': ys, 'flood_depth_m': data[valid_mask] / 100.0}) # convert cm to meters
        
        if self.point_spacing > 1:
            df = df.iloc[::self.point_spacing].reset_index(drop=True)
        print(f'    {len(df)} flood points to enrich for {image_name}')

        # Reproject
        tx = Transformer.from_crs(CRS.from_wkt(crs_wkt), CRS.from_epsg(4326), always_xy=True)
        lon, lat = tx.transform(df['x'].values, df['y'].values)
        df['longitude'], df['latitude'] = lon, lat

        enricher = GEEBulkEnricher(self.gee_project)
        df = enricher.process_dataframe(df, start_date, end_date)
        
        # keep land points only
        df = df[df['wbm_value'] == 0].copy()
        if df.empty: return

        df['segment_id'] = [self.id_pattern.format(i+1) for i in range(len(df))]
        df['event_start_date'] = start_date
        df['event_end_date'] = end_date
        df['flooded'] = df['flood_depth_m']

        df = addDrainageDensity(output_csv, basins_gdf, rivers_gdf, 1000).calculate_dd_direct(df)
        df = addClimateZone(self.current_dir, output_csv, "").enrich_direct(df)

        df.to_csv(output_csv, index=False)
        print(f'  ✓ flood done: {image_name[:-4]}')

    def convert_tif_to_points_csv(self):
        nonf_inst = createNonFloodPoints(self.id_pattern, self.years, self.current_dir, self.efas_dir, self.download_path, gee_project=self.gee_project)
        basins_gdf, rivers_gdf = nonf_inst.preload_hydro_data()

        tasks = []
        print(f' > {len(tasks)} flood TIFs queued for year(s) {self.years}')
        for y in self.years:
            folder = path.join(self.efas_dir, y)
            if path.exists(folder):
                for img in [f for f in listdir(folder) if f.endswith('.tif')]:
                    tasks.append((path.join(folder, img), path.join(folder, f'{img[:-4]}_points.csv')))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_flood_tif, tp, oc, basins_gdf, rivers_gdf) for tp, oc in tasks]
            for f in as_completed(futures): f.result()


class addDrainageDensity:
    def __init__(self, csv_path, basins_gdf, rivers_gdf, buffer_distance_m=1000):
        self.csv_path = csv_path
        self.basins_gdf = basins_gdf
        self.rivers_gdf = rivers_gdf
        self.buffer_distance_m = buffer_distance_m

    def calculate_dd_direct(self, df):
        if df.empty: return df
        gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs='EPSG:4326')
        
        pt_basins = gpd.sjoin(gdf_pts, self.basins_gdf[['HYBAS_ID', 'geometry']], how='left', predicate='within')
        
        gdf_pts_m = gdf_pts.to_crs(epsg=3035)
        rivers_m = self.rivers_gdf.to_crs(epsg=3035)
        
        nearest = gpd.sjoin_nearest(gdf_pts_m, rivers_m, how='left', max_distance=self.buffer_distance_m)
        nearest = nearest[~nearest.index.duplicated(keep='first')]
        
        try:
            df['drainage_density_km_per_km2'] = (nearest['LENGTH_KM'] / nearest['SUB_AREA']).fillna(0.0).values
        except Exception as e:
            print(f'  Δ Warning: could not compute drainage density: {e}')
        return df


class addClimateZone:
    def __init__(self, current_dir, csv_path, climate_zones_path):
        self.current_dir = current_dir
        self.csv_path = csv_path
        self.climate_zones_path = path.join(self.current_dir, 'climate_zones', 'eu_climate_zones.csv')

    def enrich_direct(self, df):
        if df.empty: return df
        import reverse_geocoder as rg
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
            df['climate_zone'] = df['country'].apply(lambda c: next((v for k, v in zone_map.items() if c in k), 'unknown'))
        else:
            df['climate_zone'] = 'unknown'
        return df


class mergeFloodNonFlood:
    def __init__(self, current_dir, years, merged_csv):
        self.current_dir = current_dir
        self.years = years
        self.merged_csv = merged_csv

    def merge(self):
        modified_headers = ['segment_id', 'latitude', 'longitude', 'elevation_m', 'dem_source', 'slope_deg', 'land_use', 'rainfall_intensity_mm_hr', 'pr_source', 'drainage_density_km_per_km2', 'country', 'climate_zone', 'flooded']
        header_remap = {'pr_avg_mm_hr': 'rainfall_intensity_mm_hr'}
        
        all_dfs = []
        for y in self.years:
            folder = path.join(self.current_dir, 'floods_datasets', 'ftp_CEMS-EFAS', y)
            if not path.exists(folder):
                print(f'  Δ merge: folder not found: {folder}')
                continue
            
            for file in [f for f in listdir(folder) if f.endswith('.csv')]:
                df = pd.read_csv(path.join(folder, file)).rename(columns=header_remap)
                df = df[[c for c in modified_headers if c in df.columns]]
                all_dfs.append(df)
                
        if not all_dfs:
            print(' Δ merge: no per-TIF CSVs found — gt_flood_depth.csv NOT written')
            return
        merged = pd.concat(all_dfs, ignore_index=True).reindex(columns=modified_headers)
        merged.to_csv(self.merged_csv, index=False)
        print(f"✓ Output Dataset consolidated cleanly to target template format layout path: {self.merged_csv}")


# Empty formatting wrappers to preserve layout configurations mapping interface specifications
class addWater: pass
class addElevationSlope: pass
class floodedColumn: pass
class nonFloodedColumn: pass


def main():
    base_url = 'https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/European_Satellite-Derived_Flood_Depth_Maps/maps/'
    YEARS = ['2028', '2019', '2022', '2024'] 
    FLOOD_YEARS = ['2018', '2024'] # there are too many non-flooded data so...
    GEE_PROJECT = get_gee_project()
    NUM_POINTS_NONFLOOD = 200
    POINT_SPACING = 300
    ID_PATTERN = 'NEW-{:05d}'
    
    current_dir = path.dirname(path.realpath(__file__))
    efas_dir = path.join(current_dir,  'floods_datasets', 'ftp_CEMS-EFAS')
    merged_csv = path.join(current_dir, 'gt_flood_depth.csv')

    if not path.exists(efas_dir):
        makedirs(efas_dir, exist_ok=True)
        print(f"✓ Created directory for EFAS datasets: {efas_dir}")

    downloader = downloadFloodTIFs(YEARS, current_dir, efas_dir, base_url)
    downloader.fetch_from_ftp()

    nonfpoint_creator = createNonFloodPoints(ID_PATTERN, YEARS, current_dir, efas_dir, downloader.download_path, NUM_POINTS_NONFLOOD, gee_project=GEE_PROJECT)
    nonfpoint_creator.convert_boundary_to_tif_points()
    
    fpoint_creator = createFloodPoints(ID_PATTERN, FLOOD_YEARS, current_dir, efas_dir, downloader.download_path, POINT_SPACING, gee_project=GEE_PROJECT)
    fpoint_creator.convert_tif_to_points_csv()

    merger = mergeFloodNonFlood(current_dir, YEARS + FLOOD_YEARS, merged_csv)
    merger.merge()


if __name__ == '__main__':
    main()