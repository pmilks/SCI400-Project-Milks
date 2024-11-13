import netCDF4 as nc
import pandas as pd
import numpy as np
url = '19820101135743-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA07_G_1982001_day-v02.0-fv01.0 (1).nc'
ds = nc.Dataset(url)
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
sst = ds.variables['sea_surface_temperature'][:]
sst_dev = ds.variables['dt_analysis'][:]
wind_spd = ds.variables['wind_speed'][:]
aero_thick = ds.variables['aerosol_dynamic_indicator'][:]
l2p_flag = ds.variables['l2p_flags'][:]


lat_grid, lon_grid = np.meshgrid(lat,lon,indexing='ij')

df = pd.DataFrame({
    'Latitude':lat_grid.ravel(),
    'Longitude':lon_grid.ravel(),
    'SST':sst.ravel(),
    'SST Deviation':sst_dev.ravel(),
    'Wind Speed':wind_spd.ravel(),
    'Aerosol Indicator':aero_thick.ravel(),
    'L2P Flags':l2p_flag.ravel()
})
print(df.head())
