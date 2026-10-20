import os
from os import path, makedirs, listdir, getcwd
import sys
import pandas as pd
import numpy as np
import xarray as xr
import glob
import zipfile
import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform
import cartopy.crs as ccrs
import inquirer
import shutil
import numpy as np

def inputs():
    gcm_question = [
        inquirer.List('GCM', 
                        message='Which Global Climate Model (GCM)?',
                        choices=['cccm_canesm2', 'ichec_ec_earth', 'ipsl_cm5a_mr', 'mohc_hadgem2_es', 'ncc_noresm1_m', 'cnrm_cerfacs_cm5', 'miroc_miroc5', 'mpi_m_mpi_esm_lr'])
    ]
    gcm_answer = inquirer.prompt(gcm_question)['GCM']


    # the following choices are the ones shown in CDS website for the corresponding GCM model. They have been written as shows in the website [September 2026]
    if gcm_answer == 'cccm_canesm2':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['gerics_remo2015','clmcom_clm_cclm4_8_17'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'ichec_ec_earth':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['clmcom_eth_cosmo_crclim', 'gerics_remo2015', 'ictp_regcm4_6', 'ipsl_wrf381p', 'smhi_rca4', 'uhoh_wrf361h', 'clmcom_clm_cclm4_8_17', 'dmi_hirham5', 'knmi_racmo22e', 'mohc_hadrem3_ga7_05'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'ipsl_cm5a_mr':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['gerics_remo2015', 'ictp_regcm4_6', 'ipsl_wrf381p', 'smhi_rca4', 'dmi_hirham5', 'knmi_racmo22e'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'mohc_hadgem2_es':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['clmcom_eth_cosmo_crclim', 'cnrm_aladin63', 'gerics_remo2015', 'ictp_regcm4_6', 'ipsl_wrf381p', 'smhi_rca4', 'uhoh_wrf361h', 'clmcom_clm_cclm4_8_17', 'dmi_hirham5', 'knmi_racmo22e', 'mohc_hadrem3_ga7_05'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'ncc_noresm1_m':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['clmcom_eth_cosmo_crclim', 'cnrm_aladin63', 'gerics_remo2015', 'ictp_regcm4_6', 'ipsl_wrf381p', 'smhi_rca4', 'dmi_hirham5', 'knmi_racmo22e', 'mohc_hadrem3_ga7_05'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'cnrm_cerfacs_cm5':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['clmcom_eth_cosmo_crclim', 'cnrm_aladin63', 'gerics_remo2015', 'ictp_regcm4_6', 'ipsl_wrf381p', 'smhi_rca4', 'uhoh_wrf361h', 'clmcom_clm_cclm4_8_17', 'dmi_hirham5', 'knmi_racmo22e', 'mohc_hadrem3_ga7_05'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'miroc_miroc5':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['gerics_remo2015', 'uhoh_wrf361h', 'clmcom_clm_cclm4_8_17'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']
    elif gcm_answer == 'mpi_m_mpi_esm_lr':
        rcm_question = [
            inquirer.List('RCM', 
                            message='Which Regional Climate Model (RCM)?',
                            choices=['clmcom_eth_cosmo_crclim', 'cnrm_aladin63', 'gerics_remo2015', 'ictp_regcm4_6', 'ipsl_wrf381p', 'smhi_rca4', 'uhoh_wrf361h', 'clmcom_clm_cclm4_8_17', 'dmi_hirham5', 'knmi_racmo22e', 'mohc_hadrem3_ga7_05', 'mpi_csc_remo2009'])
        ]
        rcm_answer = inquirer.prompt(rcm_question)['RCM']

    model_question = [
        inquirer.List('model',
                        message='For which RCP model is it?',
                        choices=['rcp85', 'rcp45'])
    ]
    model_answer = inquirer.prompt(model_question)['model']
    if model_answer == 'rcp85':
        model_answer = 'rcp_8_5'
    elif model_answer == 'rcp45':
        model_answer = 'rcp_4_5'

    in_start_year = input('Enter the start year for the data request (e.g., 2031): ')
    range_year_question = input('Is there a range of years? (y/n): ')
    if range_year_question.lower() == 'y':
        in_end_year = input('Enter the end year for the data request (e.g., 2095): ')
        years_list = list(range(int(in_start_year), int(in_end_year)))
        # print(f'Years range: {years_list}\n\n')
    else:
        years_list = [int(in_start_year)]
        # print(f'Year: {years_list}\n\n')
    
    return gcm_answer, rcm_answer, model_answer, years_list


def densify_box(lon_min, lat_min, lon_max, lat_max, n=100):
    lons = np.linspace(lon_min, lon_max, n)
    lats = np.linspace(lat_min, lat_max, n)
    top    = [(lon, lat_max) for lon in lons]
    right  = [(lon_max, lat) for lat in lats]
    bottom = [(lon, lat_min) for lon in lons[::-1]]
    left   = [(lon_min, lat) for lat in lats[::-1]]
    return Polygon(top + right + bottom + left)


def main(lon_min, lat_min, lon_max, lat_max):
    gcm_answer, rcm_answer, model_answer, years_list = inputs()
    densified_box = densify_box(lon_min, lat_min, lon_max, lat_max)
    wgs84_box = densify_box(lon_min, lat_min, lon_max, lat_max)

    # EURO-CORDEX coordinate system parameters
    CORDEX_CRS = pyproj.CRS.from_cf({
        'grid_mapping_name': 'rotated_latitude_longitude',
        'grid_north_pole_latitude': 39.25,
        'grid_north_pole_longitude': -162,
    })
    transformer = pyproj.Transformer.from_crs("epsg:4326", CORDEX_CRS, always_xy=True)

    cordex_poly = transform(transformer.transform, wgs84_box)
    cord_lon_min, cord_lat_min, cord_lon_max, cord_lat_max = cordex_poly.bounds

    # print(f'Looking at AOI: {cord_lon_min:.2f} to {cord_lon_max:.2f} longitude, {cord_lat_min:.2f} to {cord_lat_max:.2f} latitude')

    if min(years_list) == max(years_list):
        year_str = f'{min(years_list)}'
    else:
        year_str = f'{min(years_list)}-{max(years_list)}'

    zip_pattern = f'{gcm_answer}-{rcm_answer}_{min(years_list)}-{max(years_list)}_*{model_answer}_europe.zip'
    zip_files = glob.glob(zip_pattern)

    if not zip_files:
        print(f' X No zip files matched pattern: {zip_pattern}')
        print(f' > Files in current directory: {glob.glob("*.zip")}')

    for z_file in zip_files:
        print(f' > Extracting ZIP: {z_file}')
        extract_dir = z_file.replace('.zip', '_extracted')
        if path.exists(extract_dir) and any(f.endswith('.nc') for f in listdir(extract_dir)):
            print(f' > NetCDF files already exist in {extract_dir}, skipping extraction...')
        else:
            print(f' > Extracting {z_file}...')
            makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(z_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

        print(f' > Processing NetCDF files in {extract_dir}...')
        
        nc_files = glob.glob(path.join(extract_dir, '*.nc'))
            
        for nc_f in nc_files:
            print(f' > Processing NetCDF file: {nc_f}')
            
            with xr.open_mfdataset(nc_f) as dsc:
                if 'bnds' in dsc.dims:
                    dsc = dsc.drop_dims('bnds')
                try:
                    dsc_cropped = dsc.sel(
                        rlat=slice(cord_lat_min, cord_lat_max),
                        rlon=slice(cord_lon_min, cord_lon_max)
                    )
                    if dsc_cropped.sizes['rlat'] == 0 or dsc_cropped.sizes['rlon'] == 0:
                        print(f' Δ Warning: No data in cropped area for {path.basename(nc_f)}, skipping...')
                        continue
                except Exception as e:
                    print(f' X Error cropping {path.basename(nc_f)}: {e}, skipping...')
                    continue
                
                print(dsc['rotated_pole'].attrs)  # confirms real pole params from the file
                print(dsc.rlon.min().item(), dsc.rlon.max().item())
                print(dsc.rlat.min().item(), dsc.rlat.max().item())

                df = dsc_cropped.to_dataframe().reset_index()
                available_cols = ['time', 'lat', 'lon', 'pr']
                df_cleaned = df[[col for col in available_cols if col in df.columns]]

            # save
            base_nc_name = path.basename(nc_f).replace('.nc', '_europe.csv')
            out_csv = path.join(extract_dir, base_nc_name)
            df_cleaned.to_csv(out_csv, index=False)
            print(f' ✓ Saved cropped CSV to: {out_csv}')

        # merge CSV files
        europe_csv = f'{gcm_answer}-{rcm_answer}_daily_grid2x2_europe.csv'
        europe_csv_path = 'cds_csvdata_exported'
        if not path.exists(path.join(getcwd(), europe_csv_path)):
            makedirs(path.join(getcwd(), europe_csv_path), exist_ok=True)

        csv_list = [path.join(extract_dir, f) for f in listdir(extract_dir) if f.endswith('_europe.csv')]
        
        if csv_list:
            dfs = []
            for csv_path in csv_list:
                print(f' > Merging CSV file: {path.basename(csv_path)}')
                temp_df = pd.read_csv(csv_path)
                dfs.append(temp_df)
                
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_csv(path.join(europe_csv_path, europe_csv), index=False)
            print(f' ✓ Final Merged CSV saved successfully! Location: {path.join(europe_csv_path, europe_csv)}')

            del final_df
            del dfs
        
        print(' > Cleaning up temporary folder...')
        try:
            shutil.rmtree(extract_dir)
            print(' ✓ Temporary folder deleted successfully.')
        except Exception as e:
            print(f' X Could not remove directory {extract_dir}: {e}')

if __name__ == '__main__':
    from select_map_bbox import lon_min, lat_min, lon_max, lat_max # draw
    try:
        main(lon_min, lat_min, lon_max, lat_max)
    except Exception as e:
        print(f' X An error occurred: {e}')