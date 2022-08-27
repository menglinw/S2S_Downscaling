import os
import sys
import math
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from math import exp, sqrt, log
import time
import geopandas as gpd
import pandas as pd


class data_processer():
    def __init__(self, file_path_g_05, file_path_g_06, file_path_m, file_path_ele, file_path_country,
                 test_proportion, n_lag, target_variable):
        self.test_proportion = test_proportion
        self.n_lag = int(n_lag)
        self.target_var = target_variable
        self.file_path_g = [file_path_g_05, file_path_g_06]
        self.file_path_m = file_path_m
        self.file_path_ele = file_path_ele
        self.load_data()

    def load_data(self):
        # load country shape file
        country_shape = gpd.read_file(file_path_country[0])
        for country_path in file_path_country[1:]:
            country_shape = pd.concat([country_shape, gpd.read_file(country_path)])

        # get outer bound of country shape
        latmin, lonmin, latmax, lonmax = country_shape.total_bounds

        # load G5NR, log, normalize
        g05_data = nc.Dataset(self.file_path_g[0])
        g06_data = nc.Dataset(self.file_path_g[1])
        g_data = np.concatenate((self._data_g5nr_to_array(g05_data), self._data_g5nr_to_array(g06_data)), axis=0)
        # log and normalize
        g_data = np.log(g_data)
        g_data = (g_data - g_data.min())/(np.quantile(g_data, 0.95) - g_data.min())
        # cut G5NR data
        G_lats = g05_data.variables['lat'][:]
        G_lons = g05_data.variables['lon'][:]
        latmin_ind = np.argmin(np.abs(G_lats - latmin))
        latmax_ind = np.argmin(np.abs(G_lats - latmax))
        lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
        self.g_data = g_data[:, latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]
        # load lat&lon of G5NR
        self.G_lats = g05_data.variables['lat'][latmin_ind - 1:latmax_ind + 1]
        self.G_lats = self._normalize(self.G_lats)
        self.G_lons = g05_data.variables['lon'][lonmin_ind:lonmax_ind + 2]
        self.G_lons = self._normlize(self.G_lons)

        # load MERRA2, log, normalize
        m_data = nc.Dataset(self.file_path_m)
        self.m_data = m_data.variables[self.target_var][:, :, :]
        # log and normalize
        self.m_data = np.log(self.m_data)
        self.m_data = (self.m_data - self.m_data.min())/(np.quantile(self.m_data, 0.95) - self.m_data.min())
        # cut MERRA2 data
        M_lats = m_data.variables['lat'][:]
        M_lons = m_data.variables['lon'][:]
        latmin_ind = np.argmin(np.abs(M_lats - latmin))
        latmax_ind = np.argmin(np.abs(M_lats - latmax))
        lonmin_ind = np.argmin(np.abs(M_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(M_lons - lonmax))
        self.m_data = self.m_data[:, latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]
        # load lat&lon of MERRA2
        self.M_lats = m_data.variables['lat'][latmin_ind - 1:latmax_ind + 1]
        self.M_lats = self._normlize(self.M_lats)
        self.M_lons = m_data.variables['lon'][lonmin_ind:lonmax_ind + 2]
        self.M_lons = self._normlize(self.M_lons)

        # load Elevation
        self.ele_data = np.load(self.file_path_ele)

    def _normlize(self, data):
        return (data-data.min())/(data.max() - data.min())

    def _data_g5nr_to_array(self, nc_data, time_start=0, time_length=365):
        time_interval = range(time_start, time_start + time_length)
        out = nc_data.variables[self.target_var][:][time_interval, :, :]
        return out


if __name__=="__main__":
    start = time.time()
    target_var = 'DUSMASS'
    file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_ele = '/project/mereditf_284/menglin/Downscale_data/ELEV/elevation_data.npy'
    file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']