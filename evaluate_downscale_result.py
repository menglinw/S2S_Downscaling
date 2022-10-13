import numpy as np
import os
import sys
from scipy import stats


def reconstruc_downscale(data1, data2, data3, data4):
    data_left = np.concatenate([data1, data2], axis=1)
    data_right = np.concatenate([data3, data4], axis=1)
    data = np.concatenate([data_left, data_right], axis=2)
    return data

def read_sub_data(path):
    downscaled_data1 = np.load(os.path.join(path, 'downscaled_data.npy'))
    #downscaled_data1f = np.load(os.path.join(data_cache_path, season, 'Area1', 'downscaled_data_fine.npy'))
    true_data1 = np.load(os.path.join(path, 'test_g_data.npy'))
    return downscaled_data1, true_data1


def read_lon_data(path, lat_id):
    d_data1, t_data1 = read_sub_data(os.path.join(path, 'Area' + str(((lat_id - 1) * 5) + 1)))
    d_data2, t_data2 = read_sub_data(os.path.join(path, 'Area' + str(((lat_id - 1) * 5) + 2)))
    d_data3, t_data3 = read_sub_data(os.path.join(path, 'Area' + str(((lat_id - 1) * 5) + 3)))
    d_data4, t_data4 = read_sub_data(os.path.join(path, 'Area' + str(((lat_id - 1) * 5) + 4)))
    d_data5, t_data5 = read_sub_data(os.path.join(path, 'Area' + str(((lat_id - 1) * 5) + 5)))
    d_data = np.concatenate([d_data1, d_data2, d_data3, d_data4, d_data5], axis=2)
    t_data = np.concatenate([t_data1, t_data2, t_data3, t_data4, t_data5], axis=2)
    return d_data, t_data


def read_downscale_data(data_cache_path, season):
    # read 28 area downscaled data
    d_data1, t_data1 = read_lon_data(os.path.join(data_cache_path, season), 1)
    d_data2, t_data2 = read_lon_data(os.path.join(data_cache_path, season), 2)
    d_data3, t_data3 = read_lon_data(os.path.join(data_cache_path, season), 3)
    d_data = np.concatenate([d_data1, d_data2, d_data3], axis=1)
    t_data = np.concatenate([t_data1, t_data2, t_data3], axis=1)
    return d_data, t_data


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


def read_and_save(data_cache_path):
    d_data1, t_data1 = read_downscale_data(data_cache_path, 'Season1')
    np.save(os.path.join(data_cache_path, 'Season1_d_data.npy'), d_data1)
    #np.save(os.path.join(data_cache_path, 'Season1_df_data.npy'), df_data1)
    np.save(os.path.join(data_cache_path, 'Season1_t_data.npy'), t_data1)
    print('-------------Season 1---------------------')
    evaluate(d_data1, t_data1)

    d_data2,  t_data2 = read_downscale_data(data_cache_path, 'Season2')
    np.save(os.path.join(data_cache_path, 'Season2_d_data.npy'), d_data2)
    #np.save(os.path.join(data_cache_path, 'Season2_df_data.npy'), df_data2)
    np.save(os.path.join(data_cache_path, 'Season2_t_data.npy'), t_data2)
    print('-------------Season 2---------------------')
    evaluate(d_data2, t_data2)

    d_data3, t_data3 = read_downscale_data(data_cache_path, 'Season3')
    np.save(os.path.join(data_cache_path, 'Season3_d_data.npy'), d_data3)
    #np.save(os.path.join(data_cache_path, 'Season3_df_data.npy'), df_data3)
    np.save(os.path.join(data_cache_path, 'Season3_t_data.npy'), t_data3)
    print('-------------Season 3---------------------')
    evaluate(d_data3, t_data3)

    d_data4, t_data4 = read_downscale_data(data_cache_path, 'Season4')
    np.save(os.path.join(data_cache_path, 'Season4_d_data.npy'), d_data4)
    #np.save(os.path.join(data_cache_path, 'Season4_df_data.npy'), df_data4)
    np.save(os.path.join(data_cache_path, 'Season4_t_data.npy'), t_data4)
    print('-------------Season 4---------------------')
    evaluate(d_data4, t_data4)



if __name__ == '__main__':
    data_cache_path = sys.argv[1]
    read_and_save(data_cache_path)
