import numpy as np
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


def get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, test_ratio, season, area, AFG_only=False, stride=2):
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
    '''
    if area == 1:
        g_data = g_data[:, :61, :103]
        match_m_data = match_m_data[:, :61, :103]
        ele_data = ele_data[:61, :103]
        G_lats = G_lats[:61]
        G_lons = G_lons[:103]
    elif area == 2:
        g_data = g_data[:, 61:, :103]
        match_m_data = match_m_data[:, 61:, :103]
        ele_data = ele_data[61:, :103]
        G_lats = G_lats[61:]
        G_lons = G_lons[:103]
    elif area == 3:
        g_data = g_data[:, :61, 103:]
        match_m_data = match_m_data[:, :61, 103:]
        ele_data = ele_data[:61, 103:]
        G_lats = G_lats[:61]
        G_lons = G_lons[103:]
    else:
        g_data = g_data[:, 61:, 103:]
        match_m_data = match_m_data[:, 61:, 103:]
        ele_data = ele_data[61:, 103:]
        G_lats = G_lats[61:]
        G_lons = G_lons[103:]
    '''

    # season subset
    if season == 1:
        g_data1 = g_data[:91]
        match_m_data1 = match_m_data[:91]
        g_data2 = g_data[365:(365+91)]
        match_m_data2 = match_m_data[365:(365+91)]
        days1 = days[:91]
        days2 = days[365:(365+91)]
    elif season == 2:
        g_data1 = g_data[91:(91+91)]
        match_m_data1 = match_m_data[91:(91+91)]
        g_data2 = g_data[(365+91):(365+91*2)]
        match_m_data2 = match_m_data[(365+91):(365+91*2)]
        days1 = days[91:(91+91)]
        days2 = days[(365+91):(365+91*2)]
    elif season==3:
        g_data1 = g_data[91*2:91*3]
        match_m_data1 = match_m_data[91*2:91*3]
        g_data2 = g_data[(365+91*2):(365+91*3)]
        match_m_data2 = match_m_data[(365+91*2):(365+91*3)]
        days1 = days[91*2:91*3]
        days2 = days[(365+91*2):(365+91*3)]
    else:
        g_data1 = g_data[91*3:(91*4+1)]
        match_m_data1 = match_m_data[91*3:(91*4+1)]
        g_data2 = g_data[(365+91*3):730]
        match_m_data2 = match_m_data[(365+91*3):730]
        days1 = days[91*3:(91*4+1)]
        days2 = days[(365+91*3):730]
    print('m_data shape:', match_m_data1.shape, match_m_data2.shape)
    print('g_data shape: ', g_data1.shape, g_data2.shape)

    # train/test split
    n = g_data1.shape[0]-(n_pred-1)-(n_lag) + g_data2.shape[0]-(n_pred-1)-n_lag
    test_n = int(n*test_ratio)
    test_g_data = g_data2[-(test_n+n_lag):]
    test_m_data = match_m_data2[-(test_n+n_lag):]
    test_days = days2[-(test_n+n_lag):]

    g_data2 = g_data2[:-test_n]
    match_m_data2 = match_m_data2[:-test_n]
    days2 = days2[:-test_n]

    # train set flatten
    if 'X_high.npy' not in os.listdir(data_cache_path) or \
            'X_low.npy' not in os.listdir(data_cache_path) or \
            'X_ele.npy' not in os.listdir(data_cache_path) or \
            'X_other.npy' not in os.listdir(data_cache_path) or \
            'Y.npy' not in os.listdir(data_cache_path):
        X_high1, X_low1, X_ele1, X_other1, Y1 = data_processor.flatten(g_data1, match_m_data1, ele_data,
                                                                  [G_lats, G_lons], days1, n_lag=n_lag,
                                                                  n_pred=n_pred, task_dim=task_dim,
                                                                  is_perm=True, return_Y=True, stride=stride)
        X_high2, X_low2, X_ele2, X_other2, Y2 = data_processor.flatten(g_data2, match_m_data2, ele_data,
                                                                  [G_lats, G_lons], days2, n_lag=n_lag,
                                                                  n_pred=n_pred, task_dim=task_dim,
                                                                  is_perm=True, return_Y=True, stride=stride)
        X_high = np.concatenate([X_high1, X_high2], axis=0)
        X_low = np.concatenate([X_low1, X_low2], axis=0)
        X_ele = np.concatenate([X_ele1, X_ele2], axis=0)
        X_other = np.concatenate([X_other1, X_other2], axis=0)
        Y = np.concatenate([Y1, Y2], axis=0)

        # cache train set
        np.save(os.path.join(data_cache_path, 'X_high.npy'), X_high)
        np.save(os.path.join(data_cache_path, 'X_low.npy'), X_low)
        np.save(os.path.join(data_cache_path, 'X_ele.npy'), X_ele)
        np.save(os.path.join(data_cache_path, 'X_other.npy'), X_other)
        np.save(os.path.join(data_cache_path, 'Y.npy'), Y)

        np.save(os.path.join(data_cache_path, 'test_g_data.npy'), np.array(test_g_data))
        np.save(os.path.join(data_cache_path, 'test_m_data.npy'), np.array(test_m_data))
        np.save(os.path.join(data_cache_path, 'test_days.npy'), np.array(test_days))
        np.save(os.path.join(data_cache_path, 'test_ele.npy'), np.array(ele_data))
        np.save(os.path.join(data_cache_path, 'test_lats.npy'), np.array(G_lats))
        np.save(os.path.join(data_cache_path, 'test_lons.npy'), np.array(G_lons))
    else:
        print('Data is processed and saved, skipped data processing!')
    print('Data Processing Time: ', (time.time()-start)/60, 'mins')

if __name__ == '__main__':
    # define necessary parameters
    n_lag = 20
    n_pred = 3
    stride = 4
    task_dim = [10, 10]
    target_var = 'DUEXTTAU'
    test_ratio = 0.1
    data_cache_path = sys.argv[1]
    season = int(sys.argv[2])
    area = int(sys.argv[3])
    AFG_only = True if sys.argv[4] == 'AFG' else False

    get_data(data_cache_path, target_var, n_lag, n_pred, task_dim, test_ratio, season, area, AFG_only=AFG_only,
             stride=stride)