import numpy as np
import sys
import os
import tensorflow as tf
import time
if '..' not in sys.path:
    sys.path.append('..')
import util_tools
from util_tools import downscale

# define parameters
data_cache_path = sys.argv[1]
n_lag = 20
n_pred = 10
task_dim = [1, 1]
start = time.time()

# load data
test_g_data = np.load(os.path.join(data_cache_path, 'test_g_data.npy'))
test_m_data = np.load(os.path.join(data_cache_path, 'test_m_data.npy'))
test_days = np.load(os.path.join(data_cache_path, 'test_days.npy'))
ele_data = np.load(os.path.join(data_cache_path, 'test_ele.npy'))
G_lats = np.load(os.path.join(data_cache_path, 'test_lats.npy'))
G_lons = np.load(os.path.join(data_cache_path, 'test_lons.npy'))
M_lats = None
M_lons = None

# load model
generator = tf.keras.models.load_model(os.path.join(data_cache_path, 's2s_model'))
#generator_fine = tf.keras.models.load_model(os.path.join(data_cache_path, 's2s_model_fine'))

# run downscale
dscler = downscale.downscaler(generator)
downscaled_data = dscler.downscale(test_g_data[:n_lag], test_m_data, ele_data,  [G_lats, G_lons, M_lats, M_lons],
                                   test_days, n_lag, n_pred, task_dim)

#dscler_fine = downscale.downscaler(generator_fine)
#downscaled_data_fine = dscler_fine.downscale(test_g_data[:n_lag], test_m_data, ele_data,  [G_lats, G_lons, M_lats, M_lons],
#                                   test_days, n_lag, n_pred, task_dim)
# save downscale data
np.save(os.path.join(data_cache_path, 'downscaled_data.npy'), downscaled_data)
#np.save(os.path.join(data_cache_path, 'downscaled_data_fine.npy'), downscaled_data_fine)

print('Downscaling Time: ', (time.time() - start) / 60, 'mins')