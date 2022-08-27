import numpy as np
import pandas as pd
import netCDF4 as nc
from scipy import stats
import os
# os.environ['PROJ_LIB'] = r"C:\Users\96349\anaconda3\Lib\site-packages\mpl_toolkits\basemap"
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
import sys
# sys.path.append(r'C:\Users\96349\OneDrive - University of Southern California\Desktop\Downscaling_Project\AERONET')
import xarray as xr
import rioxarray as rxr
import geopandas as gpd




def resolution_downward(image, M_lats, M_lons, G_lats, G_lons):
    '''

    :param image:
    :param M_lats:
    :param M_lons:
    :param G_lats:
    :param G_lons:
    :return:
    '''
    if image.shape != (len(M_lats), len(M_lons)):
        print('Please check your input image')
        raise ValueError
    lat_gap = abs((M_lats[1] - M_lats[0]) / 2)
    lon_gap = abs((M_lons[1] - M_lons[0]) / 2)
    M_high_image = np.zeros((499, 788))
    for i in range(len(M_lats)):
        for j in range(len(M_lons)):
            aod = image[i, j]
            min_lat = np.argmin(np.abs(G_lats - M_lats[i] + lat_gap)) if i != 0 else 0
            max_lat = np.argmin(np.abs(G_lats - M_lats[i] - lat_gap)) if i != len(M_lats) - 1 else len(G_lats)
            min_lon = np.argmin(np.abs(G_lons - M_lons[j] + lon_gap)) if j != 0 else 0
            max_lon = np.argmin(np.abs(G_lons - M_lons[j] - lon_gap)) if j != len(M_lons) - 1 else len(G_lons)
            # print('lat:', min_lat, max_lat, '  lon:', min_lon, max_lon)
            # print(M_high_image[min_lat:max_lat, min_lon:max_lon].shape)
            M_high_image[min_lat:max_lat, min_lon:max_lon] = aod
    return M_high_image


def image_to_table(image, lats, lons, day, elev=None, rm_nan=False):
    '''

    :param image: AOD data
    :param elev: elevation data
    :param lats: latitude
    :param lons: longitude
    :param day: day
    :return:
    (np.prod(image.shape), 5)
    5 columns: latitude, longitude, day, elevation, AOD
    '''
    if image.shape != (len(lats), len(lons)):
        print('please check data consistency!')
        raise ValueError
    if elev is not None:
        out_array = np.zeros((len(lats) * len(lons), 5))
    else:
        out_array = np.zeros((len(lats) * len(lons), 4))
    out_array[:, -1] = image.reshape(len(lats) * len(lons))
    lat_in = []
    for lat in lats:
        lat_in += [lat] * len(lons)
    out_array[:, 0] = lat_in
    out_array[:, 1] = list(lons) * len(lats)
    out_array[:, 2] = float(day)
    if elev is not None:
        out_array[:, 3] = elev.reshape(len(lats) * len(lons))
    if rm_nan:
        out_array = out_array[~np.isnan(out_array[:, 0])]
    return out_array

def country_filter(sample_image, lats, lons, country_shape):
    """
    crop the image with a country shape
    :param sample_image: image need to be clopped
    :param lats: latitude of sample image
    :param lons: longitude of sample image
    :param country_shape: the shape of country, read by geopandas
    eg: crop_extent = gpd.read_file(os.path.join(path))
    :return: cropped image, latitude, longitude
    """
    data = xr.DataArray(sample_image, dims=('y', 'x'), coords={'y': lats, 'x': lons})
    lidar_clipped = data.rio.set_crs(country_shape.crs).rio.clip(country_shape.geometry)
    return lidar_clipped.values, lidar_clipped['y'].values, lidar_clipped['x'].values


def nc_subset_avg_series_cal(lat, lon, G5NR_data, target_var, window_width = 0.05):
    '''
    Take lat lon, window width, target variable and G5NR data, return daily time series of
    target variable averaged on the window at lat lon

    Parameters
    ----------
    lat : float
    lon : float
    G5NR_data : netCDF data
    target_var : string
    window_width : float, optional
        width of the window at lat lon. The default is 0.05.

    Returns
    -------
    daily time series of target variable averaged on the window at lat lon

    '''
    latbounds = [ lat-window_width/2 , lat+window_width/2 ]
    lonbounds = [ lon-window_width/2 , lon + window_width/2 ] # degrees east ?
    lats = G5NR_data.variables['lat'][:]
    lons = G5NR_data.variables['lon'][:]

    # latitude lower and upper index
    latli = np.argmin( np.abs( lats - latbounds[0] ) )
    latui = np.argmin( np.abs( lats - latbounds[1] ) )

    # longitude lower and upper index
    lonli = np.argmin( np.abs( lons - lonbounds[0] ) )
    lonui = np.argmin( np.abs( lons - lonbounds[1] ) )
    AOD_time_series = G5NR_data.variables[target_var][ : , latli:latui , lonli:lonui ]
    G5NR_mean_time_series = []
    for i in range(AOD_time_series.shape[0]):
        G5NR_mean_time_series.append(AOD_time_series[i,:,:].mean())
    return(pd.Series(G5NR_mean_time_series))

def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return((r_value**2, p_value))

def plot_series(title, G5NR, AERONET, R2, p, year):
    fig = plt.figure(figsize=(12, 6))
    plt.title('%d R2: ' % year + str(round(R2, 3)) + title)
    plt.plot(pd.Series(AERONET.index), AERONET,  color='red', label='AERONET')
    plt.plot(pd.Series(AERONET.index), G5NR, color='blue', label='G5NR')
    plt.legend()
    plt.xlabel('day')
    plt.ylabel('AOD 550nm')
    plt.savefig(str(year) + title[:-6] + '.jpg')


if __name__ == '__main__':
    # define some universal  variables
    # file path of G5NR 06-07, and 05-06
    file_path_g_06 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = r'C:\Users\96349\Documents\Downscale_data\MERRA2\G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    # file path of MERRA-2 data
    file_path_m = r'C:\Users\96349\Documents\Downscale_data\MERRA2\MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_ele = r'C:\Users\96349\Documents\Downscale_data\elevation\elevation_data.npy'
    # target variable
    target_var = 'TOTEXTTAU'
    # take a sample image from G5NR and MERRA-2 respectively
    # 2005/05/16
    g05_data = nc.Dataset(file_path_g_05)
    sample_G_image = g05_data.variables[target_var][0]
    m_data = nc.Dataset(file_path_m)
    sample_M_image = m_data.variables[target_var][1825]
    elev_data = np.load(file_path_ele)
    M_lons = m_data.variables['lon'][:]
    M_lats = m_data.variables['lat'][:]

    G_lons = g05_data.variables['lon'][:]
    G_lats = g05_data.variables['lat'][:]

    d_image = resolution_downward(sample_M_image, M_lats, M_lons, G_lats, G_lons)
    table_test = image_to_table(sample_G_image, G_lats, G_lons, 0)
    print('the shape of table is:', table_test.shape)
    print(table_test[499 * 788 - 10:])
