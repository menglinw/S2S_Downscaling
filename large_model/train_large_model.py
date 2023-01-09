import numpy as np
import sys
import os
import tensorflow as tf
import time
if '/scratch1/menglinw/S2S_Downscaling' not in sys.path:
    sys.path.append('/scratch1/menglinw/S2S_Downscaling')
from util_tools.data_loader import data_processer
import pandas as pd
from util_tools import downscale
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


def id_to_boundary(iid):
    if iid == 1:
        return 0, 120
    elif iid == 2:
        return 120, 240
    else:
        return 240, 360


def get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, train_set, AFG_only=False, stride=2):
    start = time.time()
    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path, exist_ok=True)
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
    g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor.load_data(target_var,
                                                                                          file_path_g_05,
                                                                                          file_path_g_06,
                                                                                          file_path_m,
                                                                                          file_path_ele,
                                                                                          file_path_country)

    # unify the spatial dimension of low resolution data to high resolution data
    match_m_data = data_processor.unify_m_data(g_data[:10], m_data, G_lats, G_lons, M_lats, M_lons)
    # only keep the range that is the same as G5NR
    match_m_data = match_m_data[1826:(1826+730), :, :]
    print('m_data shape:', match_m_data.shape)
    print('g_data shape: ', g_data.shape)
    days = list(range(1, 731))

    # subset the data
    # area subset
    '''
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
    # flatten train data (day by day)
    if 'X_high.npy' not in os.listdir(data_cache_path) or \
            'X_low.npy' not in os.listdir(data_cache_path) or \
            'X_ele.npy' not in os.listdir(data_cache_path) or \
            'X_other.npy' not in os.listdir(data_cache_path) or \
            'Y.npy' not in os.listdir(data_cache_path):
        '''
        X_high = np.zeros((1, n_lag, task_dim[0], task_dim[1], 1))
        X_low = np.zeros((1, n_lag, task_dim[0], task_dim[1], 1))
        X_ele = np.zeros((1, task_dim[0], task_dim[1], 1))
        X_other = np.zeros((1, 3))
        Y =  np.zeros((1, 1, task_dim[0], task_dim[1]))
        for day in train_set:
            # get flattened data of target day
            X_high1, X_low1, X_ele1, X_other1, Y1 = data_processor.flatten(g_data[day-n_lag:day+1],
                                                                           match_m_data[day-n_lag:day+1],
                                                                           ele_data,
                                                                           [G_lats, G_lons],
                                                                           days[day-n_lag:day+1],
                                                                           n_lag=n_lag,
                                                                           n_pred=n_pred,
                                                                           task_dim=task_dim,
                                                                           is_perm=True,
                                                                           return_Y=True,
                                                                           stride=stride)
            X_high = np.concatenate([X_high, X_high1], axis=0)
            X_low = np.concatenate([X_low, X_low1], axis=0)
            X_ele = np.concatenate([X_ele, X_ele1], axis=0)
            X_other = np.concatenate([X_other, X_other1], axis=0)
            Y = np.concatenate([Y, Y1], axis=0)
        # cache train set
        np.save(os.path.join(data_cache_path, 'X_high.npy'), X_high[1:])
        np.save(os.path.join(data_cache_path, 'X_low.npy'), X_low[1:])
        np.save(os.path.join(data_cache_path, 'X_ele.npy'), X_ele[1:])
        np.save(os.path.join(data_cache_path, 'X_other.npy'), X_other[1:])
        np.save(os.path.join(data_cache_path, 'Y.npy'), Y[1:])
        '''
        X_high, X_low, X_ele, X_other, Y = data_processor.flatten(g_data,
                                                                       match_m_data,
                                                                       ele_data,
                                                                       [G_lats, G_lons],
                                                                       days,
                                                                       n_lag=n_lag,
                                                                       n_pred=n_pred,
                                                                       task_dim=task_dim,
                                                                       is_perm=True,
                                                                       return_Y=True,
                                                                       stride=stride)
        # cache train set
        np.save(os.path.join(data_cache_path, 'X_high.npy'), X_high)
        np.save(os.path.join(data_cache_path, 'X_low.npy'), X_low)
        np.save(os.path.join(data_cache_path, 'X_ele.npy'), X_ele)
        np.save(os.path.join(data_cache_path, 'X_other.npy'), X_other)
        np.save(os.path.join(data_cache_path, 'Y.npy'), Y)
    else:
        print('Data is processed and saved, skipped data processing!')
    print('Data Processing Time: ', (time.time()-start)/60, 'mins')

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
        reconstruction_loss_factor = 500
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


def get_area_data(area):
    target_var = 'DUEXTTAU'
    area = int(area)
    AFG_only = False if area != 0 else True
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
    g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor.load_data(target_var,
                                                                                          file_path_g_05,
                                                                                          file_path_g_06,
                                                                                          file_path_m,
                                                                                          file_path_ele,
                                                                                          file_path_country)

    # unify the spatial dimension of low resolution data to high resolution data
    match_m_data_all = data_processor.unify_m_data(g_data[:10], m_data, G_lats, G_lons, M_lats, M_lons)
    # only keep the range that is the same as G5NR
    match_m_data = match_m_data_all[1826:(1826+730), :, :]
    match_m_data_all = match_m_data_all[1826:(1826+365*3), :, :]
    print('m_data shape:', match_m_data.shape)
    print('g_data shape: ', g_data.shape)
    days = list(range(1, 731))

    # subset the data
    # area subset
    '''
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
    return g_data, match_m_data, ele_data, [G_lats, G_lons], days, match_m_data_all


if __name__ == '__main__':
    start = time.time()
    # define parameters
    data_cache_path = sys.argv[1]
    head, tail = os.path.split(data_cache_path)
    area = tail[-1]
    AFG_only = True if int(area) == 0 else False
    train_set = np.load(os.path.join(head, 'train_days.npy'))

    n_lag = 40
    n_pred = 1
    stride = 5
    task_dim = [5, 5]
    target_var = 'DUEXTTAU'
    test_ratio = 0.1
    epochs = 100
    latent_space_dim = 10
    n_est = 5

    # process data
    # create area dataset
    get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, train_set, AFG_only=AFG_only,
             stride=stride)

    X_high = np.load(os.path.join(data_cache_path, 'X_high.npy'))
    X_low = np.load(os.path.join(data_cache_path, 'X_low.npy'))
    X_ele = np.load(os.path.join(data_cache_path, 'X_ele.npy'))
    X_other = np.load(os.path.join(data_cache_path, 'X_other.npy'))
    Y = np.load(os.path.join(data_cache_path, 'Y.npy'))

    generator = get_generator(n_lag, n_pred, task_dim, latent_space_dim)
    # define callbacks
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    best_save = tf.keras.callbacks.ModelCheckpoint(os.path.join(data_cache_path, 's2s_model'), save_best_only=True,
                                                   save_weights_only=True, monitor='val_loss', mode='min')
    callbacks = [lr_scheduler, early_stopping, best_save]
    if 's2s_model.index' not in os.listdir(data_cache_path):
        history = generator.fit([X_high, X_low, X_ele, X_other], Y, epochs=epochs, callbacks=callbacks, validation_split=0.2)
        pd.DataFrame(history.history).to_csv(os.path.join(data_cache_path, 'history.csv'))
        print('Training Time: ', (time.time() - start) / 60, 'mins')
    else:
        print('Model is trained, skip model training!')



    # in-data downscale
    # load model
    generator.load_weights(os.path.join(data_cache_path, 's2s_model'))
    dscler = downscale.downscaler(generator)
    area_path = data_cache_path
    # load data
    g_data, match_m_data, ele_data, [G_lats, G_lons], days, match_m_data_all = get_area_data(area)
    if 'downscaled_mean.npy' not in os.listdir(data_cache_path) \
            or 'downscaled_var.npy' not in os.listdir(data_cache_path):
        test_set = np.load(os.path.join(head, 'test_days.npy'))


        start = time.time()

        # downscale each test day
        downscaled_mean = np.zeros((1, g_data.shape[1], g_data.shape[2]))
        #downscaled_var = np.zeros((1, g_data.shape[1], g_data.shape[2]))
        for t_day in test_set:
            # downscale 1 day
            d_day_mean = dscler.downscale(g_data[t_day - n_lag:t_day + 1],
                                          match_m_data[t_day - n_lag:t_day + 2],
                                          ele_data,
                                          [G_lats, G_lons, None, None],
                                          days[t_day - n_lag:t_day + 2],
                                          n_lag,
                                          n_pred,
                                          task_dim,
                                          n_est=n_est)
            downscaled_mean = np.concatenate([downscaled_mean, d_day_mean], axis=0)
            #downscaled_var = np.concatenate([downscaled_var, d_day_var], axis=0)
        downscaled_mean = downscaled_mean[1:].filled(0)
        #downscaled_var = downscaled_var[1:].filled(0)
        np.save(os.path.join(area_path, 'downscaled_mean.npy'), downscaled_mean)
        #np.save(os.path.join(area_path, 'downscaled_var.npy'), downscaled_var)
        print('In Data Downscale Time:', (time.time() - start) / 60, 'mins')
    else:
        print('In Data Downsclae Skipped!')

    # out data downscale
'''    days = list(range(1, match_m_data_all.shape[0]+1))
    d_day_mean, d_day_var = dscler.downscale(g_data[-n_lag:],
                                             match_m_data_all[-(n_lag+365):],
                                             ele_data,
                                             [G_lats, G_lons, None, None],
                                             days[730-n_lag:],
                                             n_lag,
                                             n_pred,
                                             task_dim,
                                             n_est=n_est)
    np.save(os.path.join(area_path, 'out_downscaled_mean.npy'), d_day_mean.filled(0))
    np.save(os.path.join(area_path, 'out_downscaled_var.npy'), d_day_var.filled(0))'''