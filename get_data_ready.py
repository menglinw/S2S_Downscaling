import numpy as np
import random
import sys
import os
import time
if '..' not in sys.path:
    sys.path.append('..')
import util_tools
from util_tools.data_loader import data_processer


def id_to_boundary(id):
    if id == 1:
        return 0, 135
    elif id == 2:
        return 105, 255
    else:
        return 225, 360


def get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, train_set, area, AFG_only=False, stride=2):
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

    # flatten train data (day by day)
    if 'X_high.npy' not in os.listdir(data_cache_path) or \
            'X_low.npy' not in os.listdir(data_cache_path) or \
            'X_ele.npy' not in os.listdir(data_cache_path) or \
            'X_other.npy' not in os.listdir(data_cache_path) or \
            'Y.npy' not in os.listdir(data_cache_path):
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
    else:
        print('Data is processed and saved, skipped data processing!')
    print('Data Processing Time: ', (time.time()-start)/60, 'mins')

if __name__ == '__main__':
    # define necessary parameters
    n_lag = 20
    n_pred = 1
    stride = 2
    task_dim = [5, 5]
    target_var = 'DUEXTTAU'
    test_ratio = 0.1
    data_cache_path = sys.argv[1]
    #season = int(sys.argv[2])
    #area = int(sys.argv[3])
    #AFG_only = True if sys.argv[4] == 'AFG' else False

    for season in [1, 2, 3, 4]:
        # season subset
        year1_avlb_days = list(
            range((season - 1) * 91 + n_lag if (season - 1) * 91 - 45 < 0 else (season - 1) * 91 + n_lag - 45,
                  season * 91 if season != 4 else season * 91 + 1))
        year2_avlb_days = list(range(365 + (season - 1) * 91 + n_lag - 45,
                                     365 + season * 91 if season != 4 else 365 + season * 91 + 1))
        avlb_days = year1_avlb_days + year2_avlb_days

        # train/test split
        # get all available days
        # random split into train/test sets
        avlb_days = set(avlb_days)
        test_set = set(random.sample(avlb_days, int(len(avlb_days) * test_ratio)))
        train_set = avlb_days - test_set

        # save test date
        np.save(os.path.join(data_cache_path, 'test_days.npy'), np.array(list(test_set)))

        # create season dir if not exist
        season_path = os.path.join(data_cache_path, 'Season'+str(season))
        if not os.path.exists(season_path):
            os.mkdir(season_path)

        # create area dir and data
        for area in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            # create area dir
            area_path = os.path.join(season_path, 'Area'+str(area))
            if not os.path.exists(area_path):
                os.mkdir(area_path)

            AFG_only = True if area == 0 else False
            # create area dataset
            get_data(area_path, target_var, n_lag, n_pred, task_dim, train_set, area, AFG_only=AFG_only,
                     stride=stride)



