import netCDF4 as nc
import pandas as pd
import numpy as np
url = 'D:\\SCI400 - Project\\SCI400-Project-Milks\\sst_1981_2023\\2023\\20230805135422-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_2023217_day-v02.0-fv01.0.nc'

with nc.Dataset(url,'r') as ds:
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    sst = ds.variables['sea_surface_temperature'][:]
    sst_dev = ds.variables['dt_analysis'][:]
    wind_spd = ds.variables['wind_speed'][:]
    l2p_flag = ds.variables['l2p_flags'][:]

    target_lat = np.arange(7,47.3,0.1)
    target_lon = np.arange(-97.4,-16.9,0.1)
    nearest_lat_index = [np.abs(lat-t).argmin() for t in target_lat]
    nearest_lon_index = [np.abs(lon-t).argmin() for t in target_lon]

    nearest_lat = lat[nearest_lat_index]
    nearest_lon = lon[nearest_lon_index]
    filtered_sst = sst[:,nearest_lat_index,:][:,:,nearest_lon_index]
    filtered_sst_dev = sst_dev[:,nearest_lat_index,:][:,:,nearest_lon_index]
    filtered_wind_spd = wind_spd[:,nearest_lat_index,:][:,:,nearest_lon_index]
    filtered_l2p_flag = l2p_flag[:,nearest_lat_index,:][:,:,nearest_lon_index]

lat_grid, lon_grid = np.meshgrid(nearest_lat,nearest_lon,indexing='ij')

df = pd.DataFrame({
    'Latitude':lat_grid.ravel(),
    'Longitude':lon_grid.ravel(),
    'SST':filtered_sst.ravel(),
    'SST Deviation':filtered_sst_dev.ravel(),
    'Wind Speed':filtered_wind_spd.ravel(),
    'L2P Flags':filtered_l2p_flag.ravel()
})

df = df.loc[df['L2P Flags'] == 0].drop(columns='L2P Flags').dropna() #.dropna()?
print(df.head)