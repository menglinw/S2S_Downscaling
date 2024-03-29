import os
import sys
import math
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from math import exp, sqrt, log
import time
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd


class data_processer():
    def __init__(self):
        pass

    def country_filter(self, sample_image, lats, lons, country_shape, return_obj=False):
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
        if return_obj:
            return lidar_clipped, lidar_clipped.values, lidar_clipped['y'].values, lidar_clipped['x'].values
        else:
            return lidar_clipped.values, lidar_clipped['y'].values, lidar_clipped['x'].values

    def load_data(self, target_variable, file_path_g_05, file_path_g_06, file_path_m, file_path_ele, file_path_country,
                  normalize=True):
        '''
        load G5NR, MERRA2, elevation in the outer bound of given shape file
        :param target_variable:
        :param file_path_g_05:
        :param file_path_g_06:
        :param file_path_m:
        :param file_path_ele:
        :param file_path_country:
        :param normalize:
        :return:
        '''
        # load country shape file
        country_shape = gpd.read_file(file_path_country[0])
        if len(file_path_country) > 1:
            for country_path in file_path_country[1:]:
                country_shape = pd.concat([country_shape, gpd.read_file(country_path)])
        self.countryshape = country_shape

        # get outer bound of country shape
        lonmin, latmin, lonmax, latmax = country_shape.total_bounds

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
        if normalize:
            G_lats = self.normalize(G_lats)
        G_lons = g05_data.variables['lon'][lonmin_ind:lonmax_ind + 2]
        if normalize:
            G_lons = self.normalize(G_lons)

        # load Elevation
        ele_data = np.load(file_path_ele)
        ele_data = ele_data[latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]
        if normalize:
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
        if normalize:
            M_lats = self.normalize(M_lats)
        M_lons = m_ncdata.variables['lon'][lonmin_ind:lonmax_ind + 2]
        if normalize:
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

    def cut_country(self, g_data, m_data, lat_lon, ele_data):
        G_lats, G_lons, M_lats, M_lons = lat_lon
        if g_data.shape != m_data.shape:
            raise ValueError('G data and M data are not in a consistent shape!')
        croped_g = []
        croped_m = []
        for i in range(g_data.shape[0]):
            c_g, _, _ = self.country_filter(g_data[i], G_lats, G_lons, self.countryshape)
            c_m, _, _ = self.country_filter(m_data[i], G_lats, G_lons, self.countryshape)
            croped_g.append(c_g)
            croped_m.append(c_m)
        croped_ele, lats, lons = self.country_filter(ele_data, G_lats, G_lons, self.countryshape)
        return np.array(croped_g), np.array(croped_m), croped_ele, lats, lons

    def flatten(self, h_data, l_data, ele_data, lat_lon, days, n_lag, n_pred, task_dim, is_perm=True, return_Y=True,
                stride=1, return_nan=False):
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
            for lat in range(task_lat_dim, h_data.shape[1]+1, stride):
                for lon in range(task_lon_dim, h_data.shape[2]+1, stride):
                    if h_data[(t-n_lag+1):t+1, (lat-task_lat_dim):lat, (lon-task_lon_dim):lon].shape != (n_lag, task_lat_dim, task_lon_dim):
                        print('t:', (t-n_lag+1), t+1)
                        print('lat: ', (lat-task_lat_dim), lat)
                        print('lon: ', (lon-task_lon_dim), lon)
                    else:
                        if return_nan:
                            X_high.append(h_data[(t - n_lag + 1):t + 1, (lat - task_lat_dim):lat, (lon - task_lon_dim):lon])
                            if return_Y:
                                Y.append(h_data[t+1:(t+n_pred+1), (lat-task_lat_dim):lat, (lon-task_lon_dim):lon])
                            X_low.append(l_data[(t-n_lag+1):t+1, (lat-task_lat_dim):lat, (lon-task_lon_dim):lon])
                            X_ele.append(ele_data[(lat-task_lat_dim):lat, (lon-task_lon_dim):lon])
                            X_other.append([G_lats[lat-task_lat_dim], G_lons[lon-task_lon_dim], (days[t]%365)/365])
                        else:
                            if not np.isnan(h_data[t, lat-1, lon-1]):
                                X_high.append(
                                    h_data[(t - n_lag + 1):t + 1, (lat - task_lat_dim):lat, (lon - task_lon_dim):lon])
                                if return_Y:
                                    Y.append(h_data[t + 1:(t + n_pred + 1), (lat - task_lat_dim):lat,
                                             (lon - task_lon_dim):lon])
                                X_low.append(
                                    l_data[(t - n_lag + 1):t + 1, (lat - task_lat_dim):lat, (lon - task_lon_dim):lon])
                                X_ele.append(ele_data[(lat - task_lat_dim):lat, (lon - task_lon_dim):lon])

                                X_other.append(
                                    [G_lats[lat - task_lat_dim], G_lons[lon - task_lon_dim], (days[t] % 365) / 365])

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


if __name__=="__main__":
    start = time.time()