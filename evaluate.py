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


def combine_lon_data(a1, a2, a3):
    a12 = a1[:, :, -30:]
    a1 = a1[:, :, :-30]

    a21 = a2[:, :, :30]
    a23 = a2[:, :, -30:]
    a2 = a2[:, :, 30:-30]

    a32 = a3[:, :, :30]
    a3 = a3[:, :, 30:]
    a = np.concatenate([a1, np.mean([a12, a21], axis=0), a2, np.mean([a23, a32], axis=0), a3], axis=2)
    return a


def combine_lat_data(a1, a2, a3):
    a12 = a1[:, -30:, :]
    a1 = a1[:, :-30, :]

    a21 = a2[:, :30, :]
    a23 = a2[:, -30:, :]
    a2 = a2[:, 30:-30, :]

    a32 = a3[:, :30, :]
    a3 = a3[:, 30:, :]
    a = np.concatenate([a1, np.mean([a12, a21], axis=0), a2, np.mean([a23, a32], axis=0), a3], axis=1)
    return a


def read_season_data(season_path, fine=False):
    a1_d, a1_t = read_sub_data(os.path.join(season_path, 'Area1'), fine)
    a2_d, a2_t = read_sub_data(os.path.join(season_path, 'Area2'), fine)
    a3_d, a3_t = read_sub_data(os.path.join(season_path, 'Area3'), fine)
    a123_d = combine_lon_data(a1_d, a2_d, a3_d)
    a123_t = combine_lon_data(a1_t, a2_t, a3_t)

    a4_d, a4_t = read_sub_data(os.path.join(season_path, 'Area4'), fine)
    a5_d, a5_t = read_sub_data(os.path.join(season_path, 'Area5'), fine)
    a6_d, a6_t = read_sub_data(os.path.join(season_path, 'Area6'), fine)
    a456_d = combine_lon_data(a4_d, a5_d, a6_d)
    a456_t = combine_lon_data(a4_t, a5_t, a6_t)

    a7_d, a7_t = read_sub_data(os.path.join(season_path, 'Area7'), fine)
    a8_d, a8_t = read_sub_data(os.path.join(season_path, 'Area8'), fine)
    a9_d, a9_t = read_sub_data(os.path.join(season_path, 'Area9'), fine)
    a789_d = combine_lon_data(a7_d, a8_d, a9_d)
    a789_t = combine_lon_data(a7_t, a8_t, a9_t)
    d_data = combine_lat_data(a123_d, a456_d, a789_d)
    t_data = combine_lat_data(a123_t, a456_t, a789_t)

    AFG_d, AFG_t = read_sub_data(os.path.join(season_path, 'AreaAFG'), fine)
    return d_data, t_data, AFG_d, AFG_t

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


def evaluate_season(data_cache_path, season, fine=False):
    season_path = os.path.join(data_cache_path, season)
    d_data, t_data, AFG_d, AFG_t = read_season_data(season_path, fine)
    n_lag = t_data.shape[0] - d_data.shape[0]
    print('Day: ', 'R2', 'RMSE')
    r2_list = []
    rmse_list = []
    for i in range(d_data.shape[0]):
        t_all = np.concatenate([t_data[n_lag+i].reshape(np.prod(t_data.shape[1:])),
                                AFG_t[n_lag+1].reshape(np.prod(AFG_t.shape[1:]))])
        d_all = np.concatenate([d_data[i].reshape(np.prod(d_data.shape[1:])),
                                AFG_d[i].reshape(np.prod(AFG_d.shape[1:]))])
        r2, p = rsquared(t_all, d_all)
        rmse = np.sqrt(np.mean(np.square(t_all - d_all)))
        #r2_f, p = rsquared(t_data[n_lag+i].reshape(np.prod(t_data.shape[1:])),
        #                 df_data[i].reshape(np.prod(df_data.shape[1:])))
        print('Day', i+1, ': ',  r2, rmse)
        r2_list.append(r2)
        rmse_list.append(rmse)
    print('Avg:', np.mean(r2_list), np.mean(rmse_list))
    if fine:
        np.save(os.path.join(data_cache_path, season+'_d_data_fine.npy'), d_data)
        np.save(os.path.join(data_cache_path, season + '_d_data_AFG_fine.npy'), AFG_d)
    else:
        np.save(os.path.join(data_cache_path, season + '_d_data.npy'), d_data)
        np.save(os.path.join(data_cache_path, season + '_d_data_AFG.npy'), AFG_d)
    np.save(os.path.join(data_cache_path, season + '_t_data.npy'), t_data)
    np.save(os.path.join(data_cache_path, season + '_t_data_AFG.npy'), AFG_t)



def read_and_save(data_cache_path):
    evaluate_season(data_cache_path, 'Season1', fine=False)
    #evaluate_season(data_cache_path, 'Season1', fine=True)

    evaluate_season(data_cache_path, 'Season2', fine=False)
    #evaluate_season(data_cache_path, 'Season2', fine=True)

    evaluate_season(data_cache_path, 'Season3', fine=False)
    #evaluate_season(data_cache_path, 'Season3', fine=True)

    evaluate_season(data_cache_path, 'Season4', fine=False)
    #evaluate_season(data_cache_path, 'Season4', fine=True)

if __name__ == '__main__':
    data_cache_path = sys.argv[1]
    read_and_save(data_cache_path)


