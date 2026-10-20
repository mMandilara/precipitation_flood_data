# CDS API
import cdsapi

from dotenv import load_dotenv
import numpy as np
import xarray as xr
import os
import inquirer

import matplotlib.path as mpath
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

from glob import glob
from pathlib import Path
from os.path import basename
import zipfile
import urllib3 
urllib3.disable_warnings() # Disable "InsecureRequestWarning" 
                           # for data download via API
                           
load_dotenv()

URL = os.getenv('url')
KEY = os.getenv('key')

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


#########################################################
years_list = [str(year) for year in years_list]
print(f'You have selected the following:\n  -GCM: {gcm_answer}\n  -RCM: {rcm_answer}\n  -RCP model: {model_answer}\n  -Years: {years_list}\n\n')
check = input('Do you want to proceed with the data request? ([y]/n): ').strip().lower()

if check == 'n':
    print('Data request cancelled.')
    exit()
else:
    print('Proceeding with the data request...')
    

client = cdsapi.Client(url=URL, key=KEY)
dataset = "projections-cordex-domains-single-levels"

client.retrieve(
    'projections-cordex-domains-single-levels',
    {
        'download_format': 'zip',
        'data_format': 'netcdf_legacy',
        'domain': 'europe',
        'experiment': model_answer,
        'horizontal_resolution': '0_11_degree_x_0_11_degree',
        'temporal_resolution': 'daily_mean',    #other choises: 'monthly_mean', 'seasonal_mean', '3_hours', '6_hours', 'fixed'
        'variable': ['mean_precipitation_flux'],
        'gcm_model': gcm_answer,
        'rcm_model': rcm_answer,
        'ensemble_member': 'r1i1p1',
        'year': years_list,
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12'
        ],
        'area': [71.87, -31.65, 34.1, 31.66] # for the whole Euopre. If not set the default is global data
    },
    f'{gcm_answer}-{rcm_answer}_{min(years_list)}-{max(years_list)}_cordex_{model_answer}_europe.zip'
)