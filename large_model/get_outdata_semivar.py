import numpy as np
import sys
import os
import time

if '/scratch1/menglinw/S2S_Downscaling' not in sys.path:
    sys.path.append('/scratch1/menglinw/S2S_Downscaling')
import util_tools
from util_tools import downscale
from util_tools.data_loader import data_processer
import pandas as pd
from scipy import stats
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import netCDF4 as nc
from skgstat import Variogram
from matplotlib import pyplot as plt




def country_cut(image_list, country, lats, lons, day_list, data_path, picture_name):
    data_processor = data_processer()
    out_list = []
    for i in day_list:
        image = image_list[i]
        clipped_obj, image_out, out_lats, out_lons = data_processor.country_filter(image, lats, lons, country,
                                                                                   return_obj=True)
        clipped_obj.rio.to_raster(os.path.join(data_path, str(i)+picture_name))
        out_list.append(image_out)
    return out_list, out_lats, out_lons


def read_shape(file_path_country):
    country_shape = gpd.read_file(file_path_country[0])
    if len(file_path_country) > 1:
        for country_path in file_path_country[1:]:
            country_shape = pd.concat([country_shape, gpd.read_file(country_path)])
    return country_shape


def get_semivariogram(data, lats, lons, title):
    flat_data = pd.DataFrame({'z': data.reshape(np.prod(data.shape)),
                              'x': np.repeat(lats, len(lons)),
                              'y': list(lons) * len(lats)})
    flat_data.dropna(inplace=True)
    V1 = Variogram(flat_data[['x', 'y']].values, flat_data.z.values, normalize=True)
    V1.plot()
    plt.savefig(title)


# TODO: load true data
target_var = 'DUEXTTAU'
file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
file_path_ele = '/project/mereditf_284/menglin/Downscale_data/ELEV/elevation_data.npy'

file_path_country_afg = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp']
afg_shape = read_shape(file_path_country_afg)

file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                     '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']
g_shape = read_shape(file_path_country)

# load data
data_processor = data_processer()
g_data, _, [G_lats, G_lons, M_lats, M_lons], _ = data_processor.load_data(target_var,
                                                                          file_path_g_05,
                                                                          file_path_g_06,
                                                                          file_path_m,
                                                                          file_path_ele,
                                                                          file_path_country,
                                                                          normalize=False)

afg_data, _, [G_lats2, G_lons2, M_lats2, M_lons2], _ = data_processor.load_data(target_var,
                                                                                file_path_g_05,
                                                                                file_path_g_06,
                                                                                file_path_m,
                                                                                file_path_ele,
                                                                                file_path_country_afg,
                                                                                normalize=False)

# TODO: load downscaled data
data_path = '/scratch1/menglinw/Results/1_8_1'
d_data = np.load(os.path.join(data_path, 'out_downscaled_g_data.npy'))
d_afg_data = np.load(os.path.join(data_path, 'out_downscaled_AFG_data.npy'))

# TODO: get semivariogram at target date
target_days = [0, 100, 200, 300, 364]
target_days_g = [365, 465, 565, 665, 729]
g_data, g_cut_lats, g_cut_lons = country_cut(g_data, g_shape, G_lats, G_lons, target_days_g,
                                             data_path, '_true_g_data.tif')
print('G datashape:', len(g_data), g_data[0].shape)

g_data_AFG, AFG_cut_lats, AFG_cut_lons = country_cut(afg_data, afg_shape, G_lats2, G_lons2, target_days_g,
                                                     data_path, '_true_afg_data.tif')
print('AFG datashape:', len(g_data_AFG), g_data_AFG[0].shape)

season_downscaled_mean, _, _ = country_cut(d_data, g_shape, G_lats, G_lons, target_days,
                                           data_path, '_down_g_data.tif')
print('Downscaled G datashape:', len(season_downscaled_mean), season_downscaled_mean[0].shape)

season_downscaled_mean_AFG, _, _ = country_cut(d_afg_data, afg_shape, G_lats2, G_lons2, target_days,
                                               data_path, '_down_afg_data.tif')
print('Downscaled AFG datashape:', len(season_downscaled_mean_AFG), season_downscaled_mean_AFG[0].shape)

for i in range(len(g_data)):
    get_semivariogram(g_data[i], g_cut_lats, g_cut_lons,
                      os.path.join(data_path, str(target_days[i]) + '_true_g_semivariogram.jpg'))
    get_semivariogram(g_data_AFG[i], AFG_cut_lats, AFG_cut_lons,
                      os.path.join(data_path, str(target_days[i]) + '_true_afg_semivariogram.jpg'))
    get_semivariogram(season_downscaled_mean[i], g_cut_lats, g_cut_lons,
                      os.path.join(data_path, str(target_days[i]) + '_down_g_semivariogram.jpg'))
    get_semivariogram(season_downscaled_mean_AFG[i], AFG_cut_lats, AFG_cut_lons,
                      os.path.join(data_path, str(target_days[i]) + '_down_afg_semivariogram.jpg'))
