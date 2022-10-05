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
    def __init__(self):
        pass

    def load_data(self, target_variable, file_path_g_05, file_path_g_06, file_path_m, file_path_ele, file_path_country):
        # load country shape file
        country_shape = gpd.read_file(file_path_country[0])
        for country_path in file_path_country[1:]:
            country_shape = pd.concat([country_shape, gpd.read_file(country_path)])

        # get outer bound of country shape
        latmin, lonmin, latmax, lonmax = country_shape.total_bounds

        # load G5NR, log, normalize
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        g_data = np.concatenate((self.data_g5nr_to_array(g05_data, target_var=target_variable),
                                 self.data_g5nr_to_array(g06_data, target_var=target_variable)), axis=0)
        # log and normalize
        g_data = np.log10(g_data)
        g_data = (g_data - g_data.min()) / (g_data.max() - g_data.min())
        # cut G5NR data
        G_lats = g05_data.variables['lat'][:]
        G_lons = g05_data.variables['lon'][:]
        latmin_ind = np.argmin(np.abs(G_lats - latmin))
        latmax_ind = np.argmin(np.abs(G_lats - latmax))
        lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
        g_data = g_data[:, latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]
        # load lat&lon of G5NR
        G_lats = g05_data.variables['lat'][latmin_ind - 1:latmax_ind + 1]
        G_lats = self.normalize(G_lats)
        G_lons = g05_data.variables['lon'][lonmin_ind:lonmax_ind + 2]
        G_lons = self.normalize(G_lons)

        # load Elevation
        ele_data = np.load(file_path_ele)
        ele_data = ele_data[latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]
        ele_data = self.normalize(ele_data)

        # load MERRA2, log, normalize
        m_ncdata = nc.Dataset(file_path_m)
        m_data = m_ncdata.variables[target_variable][:, :, :]
        # log and normalize
        m_data = np.log10(m_data)
        m_data = (m_data - m_data.min()) / (m_data.max() - m_data.min())
        # cut MERRA2 data
        M_lats = m_ncdata.variables['lat'][:]
        M_lons = m_ncdata.variables['lon'][:]
        latmin_ind = np.argmin(np.abs(M_lats - latmin))
        latmax_ind = np.argmin(np.abs(M_lats - latmax))
        lonmin_ind = np.argmin(np.abs(M_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(M_lons - lonmax))
        m_data = m_data[:, latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]
        # load lat&lon of MERRA2
        M_lats = m_ncdata.variables['lat'][latmin_ind - 1:latmax_ind + 1]
        M_lats = self.normalize(M_lats)
        M_lons = m_ncdata.variables['lon'][lonmin_ind:lonmax_ind + 2]
        M_lons = self.normalize(M_lons)

        return g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data

    def normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def data_g5nr_to_array(self, nc_data, target_var, time_start=0, time_length=365):
        time_interval = range(time_start, time_start + time_length)
        out = nc_data.variables[target_var][:][time_interval, :, :]
        return out

    def unify_m_data(self, g_data, m_data, G_lats, G_lons, M_lats, M_lons):
        lat_dim, lon_dim = g_data.shape[1:]
        unif_m_data = np.zeros((m_data.shape[0], lat_dim, lon_dim))
        for i in range(lat_dim):
            for j in range(lon_dim):
                lat = G_lats[i]
                lon = G_lons[j]
                m_lat_idx = np.argmin(np.abs(M_lats - lat))
                m_lon_idx = np.argmin(np.abs(M_lons - lon))
                unif_m_data[:, i, j] = m_data[:, m_lat_idx, m_lon_idx]
        return unif_m_data

    def flatten(self, h_data, l_data, ele_data, lat_lon, days, n_lag, n_pred, task_dim, is_perm=True, return_Y=True):
        # h_data and l_data should be in the same time range
        task_lat_dim, task_lon_dim = task_dim
        G_lats, G_lons = lat_lon
        n_instance = (h_data.shape[0]-n_lag-n_pred+1)*(h_data[1]-task_lat_dim+1)*(h_data[2]-task_lon_dim+1)
        X_high = []
        X_low = []
        X_ele = []
        # lat, lon, day
        X_other = []

        Y = []
        end_point = h_data.shape[0]-n_pred if return_Y else h_data.shape[0]
        for t in range(n_lag-1, end_point):
            for lat in range(task_lat_dim, h_data.shape[1]+1):
                for lon in range(task_lon_dim, h_data.shape[2]+1):
                    if h_data[(t-n_lag+1):t+1, (lat-task_lat_dim):lat, (lon-task_lon_dim):lon].shape != (n_lag, task_lat_dim, task_lon_dim):
                        print('t:', (t-n_lag+1), t+1)
                        print('lat: ', (lat-task_lat_dim), lat)
                        print('lon: ', (lon-task_lon_dim), lon)
                    else:
                        X_high.append(h_data[(t - n_lag + 1):t + 1, (lat - task_lat_dim):lat, (lon - task_lon_dim):lon])
                        if return_Y:
                            Y.append(h_data[t+1:(t+n_pred+1), (lat-task_lat_dim):lat, (lon-task_lon_dim):lon])
                        X_low.append(l_data[(t-n_lag+1):t+1, (lat-task_lat_dim):lat, (lon-task_lon_dim):lon])
                        X_ele.append(ele_data[(lat-task_lat_dim):lat, (lon-task_lon_dim):lon])
                        X_other.append([G_lats[lat-task_lat_dim], G_lons[lon-task_lon_dim], (days[t]%365)/365])
        if is_perm:
            perm = np.random.permutation(len(X_high))
            if return_Y:
                return np.expand_dims(np.array(X_high), -1)[perm], np.expand_dims(np.array(X_low), -1)[perm], \
                       np.expand_dims(np.array(X_ele), -1)[perm], np.array(X_other)[perm], np.array(Y)[perm]
            else:
                return np.expand_dims(np.array(X_high), -1)[perm], np.expand_dims(np.array(X_low), -1)[perm], \
                       np.expand_dims(np.array(X_ele), -1)[perm], np.array(X_other)[perm]
        if return_Y:
            return np.expand_dims(np.array(X_high), -1), np.expand_dims(np.array(X_low), -1), \
                   np.expand_dims(np.array(X_ele), -1), np.array(X_other), np.array(Y)
        else:
            return np.expand_dims(np.array(X_high), -1), np.expand_dims(np.array(X_low), -1), \
                   np.expand_dims(np.array(X_ele), -1), np.array(X_other)

    def data_process(self):
        # TODO: unify dimension
        self.unify_m_data()
        # TODO: split tran/test
        # TODO: write flatten function
        # TODO: cache data
        pass

if __name__=="__main__":
    start = time.time()