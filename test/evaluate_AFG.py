import numpy as np
import os
import sys
from scipy import stats


def read_sub_data(path, fine=False):
    if fine:
        downscaled_data1 = np.load(os.path.join(path, 'downscaled_data_fine.npy'))
    else:
        downscaled_data1 = np.load(os.path.join(path, 'downscaled_data.npy'))

    true_data1 = np.load(os.path.join(path, 'test_g_data.npy'))
    return downscaled_data1, true_data1


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return((r_value**2, p_value))


def evaluate(d_data, t_data):
    n_lag = t_data.shape[0] - d_data.shape[0]
    print('Day: ', 'R2', 'RMSE')
    r2_list = []
    rmse_list = []
    for i in range(d_data.shape[0]):
        r2, p = rsquared(t_data[n_lag+i].reshape(np.prod(t_data.shape[1:])),
                         d_data[i].reshape(np.prod(d_data.shape[1:])))
        rmse = np.sqrt(np.mean(np.square(t_data[n_lag+i] - d_data[i])))
        #r2_f, p = rsquared(t_data[n_lag+i].reshape(np.prod(t_data.shape[1:])),
        #                 df_data[i].reshape(np.prod(df_data.shape[1:])))
        print('Day', i+1, ': ',  r2, rmse)
        r2_list.append(r2)
        rmse_list.append(rmse)
    print('Avg:', np.mean(r2_list), np.mean(rmse_list))

if __name__=='__main__':
    data_cache_path = sys.argv[1]
    d_data, t_data = read_sub_data(data_cache_path, fine=False)
    evaluate(d_data, t_data)
    print('Fine Tune Results')
    d_data, t_data = read_sub_data(data_cache_path, fine=True)
    evaluate(d_data, t_data)
