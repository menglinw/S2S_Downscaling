import util_tools
import numpy as np
import sys
import os
import tensorflow as tf
import time

start = time.time()
# define necessary parameters
n_lag = 10
n_pred = 3
task_dim = [5, 5]

data_cache_path = '/scratch1/menglinw/S2S_temp_data'
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

# load input data
data_processor = util_tools.data_loader.data_processer()
g_data, m_data, [G_lats, G_lons, M_lats, M_lons], ele_data = data_processor.load_data(target_var,
                                                                                      file_path_g_05,
                                                                                      file_path_g_06,
                                                                                      file_path_m,
                                                                                      file_path_ele,
                                                                                      file_path_country)
# unify the spatial dimension of low resolution data to high resolution data
match_m_data = data_processor.unify_m_data(g_data[:10], m_data, G_lats, G_lons, M_lats, M_lons)
# only keep the range that is the same as G5NR
match_m_data = m_data[range(1826, 1826+730), :, :]
days = list(range(1, 731))

# train/test split
test_ratio = 0.1
train_g_data = g_data[:int(g_data.shape[0]*(1-test_ratio))]
train_m_data = match_m_data[:int(match_m_data.shape[0]*(1-test_ratio))]
train_days = days[:int(730*(1- test_ratio))]

test_g_data = g_data[int(g_data.shape[0]*(1-test_ratio)):(int(g_data.shape[0]*(1-test_ratio))+n_lag)]
test_m_data = match_m_data[int(match_m_data.shape[0]*(1-test_ratio)):]
test_days = days[int(730*(1- test_ratio)):]

# train set flatten
if 'X_high.npy' not in os.listdir(data_cache_path) or \
        'X_low.npy' not in os.listdir(data_cache_path) or \
        'X_ele.npy' not in os.listdir(data_cache_path) or \
        'X_other.npy' not in os.listdir(data_cache_path) or \
        'Y.npy' not in os.listdir(data_cache_path):
    X_high, X_low, X_ele, X_other, Y = data_processor.flatten(train_g_data, train_m_data, ele_data,
                                                              [G_lats, G_lons], train_days, n_lag=n_lag,
                                                              n_pred=n_pred, task_dim=task_dim,
                                                              is_perm=True, return_Y=True)
    # cache train set
    np.save(os.path.join(data_cache_path, 'X_high.npy'), X_high)
    np.save(os.path.join(data_cache_path, 'X_low.npy'), X_low)
    np.save(os.path.join(data_cache_path, 'X_ele.npy'), X_ele)
    np.save(os.path.join(data_cache_path, 'X_other.npy'), X_other)
    np.save(os.path.join(data_cache_path, 'Y.npy'), Y)
else:
    X_high = np.load(os.path.join(data_cache_path, 'X_high.npy'))
    X_low = np.load(os.path.join(data_cache_path, 'X_low.npy'))
    X_ele = np.load(os.path.join(data_cache_path, 'X_ele.npy'))
    X_other = np.load(os.path.join(data_cache_path, 'X_other.npy'))
    Y = np.load(os.path.join(data_cache_path, 'Y.npy'))
print('Data Processing Time: ', (time.time()-start)/60, 'mins')
start = time.time()
# define model
def nnelu(input):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

high_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
x1 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3,3), return_sequences=True, activation=tf.keras.layers.LeakyReLU())(high_input)
x1 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3,3), return_sequences=True, activation=tf.keras.layers.LeakyReLU())(x1)
x1 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(1,1), activation=tf.keras.layers.LeakyReLU())(x1)
x1 = tf.keras.layers.Flatten()(x1)

low_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
x2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3,3), activation=tf.keras.layers.LeakyReLU())(low_input)
x2 = tf.keras.layers.Flatten()(x1)

ele_input = tf.keras.Input(shape=(task_dim[0], task_dim[1], 1))
x3 = tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation=tf.keras.layers.LeakyReLU())(ele_input)
x3 = tf.keras.layers.Flatten()(x3)

other_input =  tf.keras.Input(shape=(3))
x4 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU())(other_input)

x = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3, x4])
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU())(x)
x = tf.keras.layers.Dense(n_pred*np.prod(task_dim), activation=nnelu)(x)
x = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(x)
generator = tf.keras.Model([high_input, low_input, ele_input, other_input], x)
generator.compile(optimizer='adam', loss='mean_absolute_error')

# define callbacks
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
callbacks = [lr_scheduler, early_stopping]

generator.fit([X_high, X_low, X_ele, X_other], Y, epochs=100, callbacks=callbacks, validation_split=0.25)
generator.save_weights(data_cache_path)
print('Training Time: ', (time.time()-start)/60, 'mins')
start = time.time()