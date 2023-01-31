import numpy as np
import sys
import os
import tensorflow as tf
import time

if '/scratch1/menglinw/S2S_Downscaling' not in sys.path:
    sys.path.append('/scratch1/menglinw/S2S_Downscaling')
import util_tools
from util_tools import downscale
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
from util_tools.data_loader import data_processer
import pandas as pd
from scipy import stats
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import geopandas as gpd
import netCDF4 as nc
from skgstat import Variogram
from matplotlib import pyplot as plt


# define helper function
def mapping_tanh(x):
    x02 = tf.keras.backend.tanh(x) + 1  # x in range(0,2)
    scale = 1 / 2.
    return x02 * scale


def mapping_abs(x):
    x02 = x / (1 + tf.keras.backend.abs(x)) + 1
    scale = 1 / 2.
    return x02 * scale


def nnelu(x):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(x))


def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
    return random_sample


def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) -
                                              tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) -
                                              tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss


def get_generator(n_lag, n_pred, task_dim, latent_space_dim):
    high_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))

    x1 = tf.keras.layers.ConvLSTM2D(32, kernel_size=(3, 3), return_sequences=False,
                                    activation=tf.keras.layers.LeakyReLU())(high_input)
    x1 = tf.keras.layers.Flatten()(x1)

    low_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
    x2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3, 3), return_sequences=False,
                                    activation=tf.keras.layers.LeakyReLU())(low_input)
    x2 = tf.keras.layers.Flatten()(x2)

    ele_input = tf.keras.Input(shape=(task_dim[0], task_dim[1], 1))
    x3 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU())(ele_input)
    x3 = tf.keras.layers.Flatten()(x3)

    other_input = tf.keras.Input(shape=(3))
    x4 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU())(other_input)

    e1 = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3, x4])
    e2 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(e1)
    e3 = tf.keras.layers.BatchNormalization()(e2)
    e3 = tf.keras.layers.Dropout(0.5)(e3)
    e4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(e3)
    e5 = tf.keras.layers.BatchNormalization()(e4)
    e5 = tf.keras.layers.Dropout(0.5)(e5)
    e6 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(e5)
    e7 = tf.keras.layers.BatchNormalization()(e6)
    e7 = tf.keras.layers.Dropout(0.5)(e7)
    est_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(e7)
    est_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(e7)
    est_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([est_mu, est_log_variance])

    d1 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(est_output)
    d2 = tf.keras.layers.BatchNormalization()(d1)
    d3 = tf.keras.layers.Concatenate(axis=1)([d2, e7])
    d3 = tf.keras.layers.Dropout(0.5)(d3)
    d4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(d3)
    d5 = tf.keras.layers.BatchNormalization()(d4)
    d6 = tf.keras.layers.Concatenate(axis=1)([d5, e5])
    d6 = tf.keras.layers.Dropout(0.5)(d6)
    d7 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(d6)
    d8 = tf.keras.layers.BatchNormalization()(d7)
    d9 = tf.keras.layers.Concatenate(axis=1)([d8, e3])
    d9 = tf.keras.layers.Dropout(0.5)(d9)
    d10 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                activation=tf.keras.layers.LeakyReLU())(d9)
    d11 = tf.keras.layers.BatchNormalization()(d10)
    d12 = tf.keras.layers.Dense(n_pred * np.prod(task_dim), activation=mapping_abs)(d11)
    d13 = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(d12)
    generator = tf.keras.Model([high_input, low_input, ele_input, other_input], d13)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    generator.compile(optimizer=opt, loss=loss_func(est_mu, est_log_variance))
    return generator


def id_to_boundary(id):
    if id == 1:
        return 0, 120
    elif id == 2:
        return 120, 240
    else:
        return 240, 360


def get_area_data(down_g_data, down_AFG_data, season, area, n_lag):
    target_var = 'DUEXTTAU'
    area = int(area)
    AFG_only = False if area != 0 else True
    season = int(season)

    file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_ele = '/project/mereditf_284/menglin/Downscale_data/ELEV/elevation_data.npy'
    if AFG_only:
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp']
    else:
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']

    # load input data
    data_processor = data_processer()
    _, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor.load_data(target_var,
                                                                                     file_path_g_05,
                                                                                     file_path_g_06,
                                                                                     file_path_m,
                                                                                     file_path_ele,
                                                                                     file_path_country)
    g_data = down_AFG_data[-n_lag:] if AFG_only else down_g_data[-n_lag:]
    # unify the spatial dimension of low resolution data to high resolution data
    match_m_data_all = data_processor.unify_m_data(g_data, m_data, G_lats, G_lons, M_lats, M_lons)
    # only keep the range that is the same as G5NR
    season_days = 91 if season != 4 else 92
    match_m_data = match_m_data_all[
                   (1826 + 730 + (season - 1) * 91 - n_lag):(1826 + 730 + season_days + (season - 1) * 91), :, :]
    print('m_data shape:', match_m_data.shape)
    print('g_data shape: ', g_data.shape)
    days = list(range(366 - n_lag, 366)) + list(range(1, 366))
    days = days[(season - 1) * 91:season_days + n_lag + (season - 1) * 91]

    # subset the data
    '''
    # area subset
    if not AFG_only:
        lat_id = area // 3 + 1 if area % 3 != 0 else area // 3
        lon_id = area % 3 if area % 3 != 0 else 3
        print('area: ', area, 'lat:', lat_id, 'lon:', lon_id)
        lat_low, lat_high = id_to_boundary(lat_id)
        lon_low, lon_high = id_to_boundary(lon_id)
        print('lat:', lat_low, lat_high)
        print('lon:', lon_low, lon_high)
        g_data = g_data[:, lat_low:lat_high, lon_low:lon_high]
        match_m_data = match_m_data[:, lat_low:lat_high, lon_low:lon_high]
        ele_data = ele_data[lat_low:lat_high, lon_low:lon_high]
        G_lats = G_lats[lat_low:lat_high]
        G_lons = G_lons[lon_low:lon_high]
    '''
    return g_data, match_m_data, ele_data, [G_lats, G_lons], days


def combine_lon_data(a1, a2, a3):
    return np.concatenate([a1, a2, a3], axis=2)


def combine_lat_data(a1, a2, a3):
    return np.concatenate([a1, a2, a3], axis=1)


def reconstruct_season_data(area_data_list):
    a123_d = combine_lon_data(area_data_list[1], area_data_list[2], area_data_list[3])

    a456_d = combine_lon_data(area_data_list[4], area_data_list[5], area_data_list[6])

    a789_d = combine_lon_data(area_data_list[7], area_data_list[8], area_data_list[9])
    d_data = combine_lat_data(a123_d, a456_d, a789_d)

    AFG_d = area_data_list[0]
    return d_data, AFG_d


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2, p_value


def get_true_data(test_set):
    target_var = 'DUEXTTAU'

    file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
    file_path_ele = '/project/mereditf_284/menglin/Downscale_data/ELEV/elevation_data.npy'

    file_path_country_AFG = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp']

    file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']

    # load input data
    data_processor = data_processer()
    g_data, _, [_, _, _, _], _ = data_processor.load_data(target_var,
                                                          file_path_g_05,
                                                          file_path_g_06,
                                                          file_path_m,
                                                          file_path_ele,
                                                          file_path_country)
    g_data_AFG, _, [_, _, _, _], _ = data_processor.load_data(target_var,
                                                              file_path_g_05,
                                                              file_path_g_06,
                                                              file_path_m,
                                                              file_path_ele,
                                                              file_path_country_AFG)
    return g_data[test_set, :, :], g_data_AFG[test_set, :, :]


def save_downscaled_data(d_data, path, n_lag):
    np.save(path, d_data[n_lag:].filled(0))


def country_cut(image_list, country, lats, lons):
    data_processor = data_processer()
    out_list = []
    for image in image_list:
        image_out, out_lats, out_lons = data_processor.country_filter(image, lats, lons, country)
        out_list.append(image_out)
    return out_list, out_lats, out_lons


def read_shape(file_path_country):
    country_shape = gpd.read_file(file_path_country[0])
    if len(file_path_country) > 1:
        for country_path in file_path_country[1:]:
            country_shape = pd.concat([country_shape, gpd.read_file(country_path)])
    return country_shape


def get_lat_lon(country_shape, g05_data):
    lonmin, latmin, lonmax, latmax = country_shape.total_bounds
    G_lats = g05_data.variables['lat'][:]
    G_lons = g05_data.variables['lon'][:]
    latmin_ind = np.argmin(np.abs(G_lats - latmin))
    latmax_ind = np.argmin(np.abs(G_lats - latmax))
    lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
    lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
    # load lat&lon of G5NR
    G_lats = g05_data.variables['lat'][latmin_ind - 1:latmax_ind + 1]
    G_lons = g05_data.variables['lon'][lonmin_ind:lonmax_ind + 2]
    return G_lats, G_lons


def get_countryshape_latlon():
    file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
    file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
    file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'

    file_path_country_AFG = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp']

    file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                         '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']
    shape_AFG = read_shape(file_path_country_AFG)
    shape_G = read_shape(file_path_country)

    g05_data = nc.Dataset(file_path_g_05)
    # get outer bound of country shape
    lats_AFG, lons_AFG = get_lat_lon(shape_AFG, g05_data)
    lats_G, lons_G = get_lat_lon(shape_G, g05_data)
    return shape_G, lats_G, lons_G, shape_AFG, lats_AFG, lons_AFG


def get_semivariogram(data, lats, lons, title):
    flat_data = pd.DataFrame({'z': data.reshape(np.prod(data.shape)),
                              'x': np.repeat(lats, len(lons)),
                              'y': list(lons) * len(lats)})
    flat_data.dropna(inplace=True)
    V1 = Variogram(flat_data[['x', 'y']].values, flat_data.z.values, normalize=False)
    V1.plot()
    plt.savefig(title)


def get_empty_raster():
    data_processor2 = data_processer()
    target_var = 'DUEXTTAU'

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

    # load input data
    _, _, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor2.load_data(target_var,
                                                                                file_path_g_05,
                                                                                file_path_g_06,
                                                                                file_path_m,
                                                                                file_path_ele,
                                                                                file_path_country)
    temp_array = np.zeros_like(ele_data)
    temp_array[:] = np.NaN
    data = xr.DataArray(temp_array, dims=('y', 'x'), coords={'y': G_lats, 'x': G_lons})
    return data

if __name__ == "__main__":
    # define parameters
    data_cache_path = sys.argv[1]
    n_lag = 40
    n_pred = 1
    task_dim = [5, 5]
    target_var = 'DUEXTTAU'
    latent_space_dim = 10
    n_est = 1
    cut_by_country = True


    # in-data evaluation
    for season in [1, 2, 3, 4]:
        # read test days
        season_path = os.path.join(data_cache_path, 'Season'+str(season))
        test_set = np.load(os.path.join(season_path, 'test_days.npy'))
        if 'season_downscaled_mean.npy' in os.listdir(season_path) \
                and 'season_downscaled_mean_AFG.npy' in os.listdir(season_path):
            season_downscaled_mean = np.load(os.path.join(season_path, 'season_downscaled_mean.npy'))
            season_downscaled_mean_AFG = np.load(os.path.join(season_path, 'season_downscaled_mean_AFG.npy'))
            print('Skipped in-data downscale, loaded downscaled data!')
        else:
            mean_list = []
            for area in [0, 1]:
                area_path = os.path.join(season_path, 'Area' + str(area))
                downscaled_mean = np.load(os.path.join(area_path, 'downscaled_mean.npy'))
                #downscaled_var = np.load(os.path.join(area_path, 'downscaled_var.npy'))
                mean_list.append(downscaled_mean)
                #var_list.append(downscaled_var)
            # TODO: reconstruct downscaled data to large image and save
            start = time.time()
            season_downscaled_mean_AFG, season_downscaled_mean  = mean_list
            #season_downscaled_var, season_downscaled_var_AFG = reconstruct_season_data(var_list)
            np.save(os.path.join(season_path, 'season_downscaled_mean.npy'), season_downscaled_mean)
            #np.save(os.path.join(season_path, 'season_downscaled_var.npy'), season_downscaled_var)
            np.save(os.path.join(season_path, 'season_downscaled_mean_AFG.npy'), season_downscaled_mean_AFG)
            #np.save(os.path.join(season_path, 'season_downscaled_var_AFG.npy'), season_downscaled_var_AFG)
            print('Reconstruct Season Data time:', (time.time() - start) / 60, 'mins')

        # TODO: evaluate and save
        # get true data
        start = time.time()
        g_data, g_data_AFG = get_true_data(test_set)
        shape_G, lats_G, lons_G, shape_AFG, lats_AFG, lons_AFG = get_countryshape_latlon()

        if cut_by_country:
            data_processor = data_processer()
            true_img = get_empty_raster()
            img_G,_, _, _ = data_processor.country_filter(g_data[0], lats_G, lons_G, shape_G, return_obj=True)
            img_AFG, _, _, _ = data_processor.country_filter(g_data_AFG[0], lats_AFG, lons_AFG, shape_AFG, return_obj=True)
            true_img = true_img.combine_first(img_G)
            true_img = true_img.combine_first(img_AFG)
            true_img.rio.set_crs(4326)
            true_img.rio.to_raster(os.path.join(data_cache_path, 'Season'+str(season)+'_true_' + ".tif"))

            pred_img = get_empty_raster()
            D_img_G,_, _, _ = data_processor.country_filter(season_downscaled_mean[0], lats_G, lons_G, shape_G, return_obj=True)
            D_img_AFG, _, _, _ = data_processor.country_filter(season_downscaled_mean_AFG[0], lats_AFG, lons_AFG, shape_AFG, return_obj=True)
            pred_img = pred_img.combine_first(D_img_AFG)
            pred_img = pred_img.combine_first(D_img_G)
            pred_img.rio.set_crs(4326)
            pred_img.rio.to_raster(os.path.join(data_cache_path, 'Season'+str(season)+'_pred_' + ".tif"))







