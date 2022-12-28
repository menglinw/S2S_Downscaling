import numpy as np
import sys
import os
import tensorflow as tf
import time
if '..' not in sys.path:
    sys.path.append('..')
from util_tools.data_loader import data_processer
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

from skgstat import Variogram, OrdinaryKriging
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
print('Successfully loaded packages!')

target_var = 'DUEXTTAU'
data_cache_path = sys.argv[1]


# TODO: load G5NR data and MERRA2 data

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
g_data, m_data, [G_lats, G_lons, M_lats, M_lons], _ = data_processor.load_data(target_var,
                                                                               file_path_g_05,
                                                                               file_path_g_06,
                                                                               file_path_m,
                                                                               file_path_ele,
                                                                               file_path_country)

g_data_AFG, m_data_AFG, [G_lats_AFG, G_lons_AFG, M_lats_AFG, M_lons_AFG], _ = data_processor.load_data(target_var,
                                                                                                       file_path_g_05,
                                                                                                       file_path_g_06,
                                                                                                       file_path_m,
                                                                                                       file_path_ele,
                                                                                                       file_path_country_AFG)
# TODO: load downscaled data
down_AFG_data = np.load(os.path.join(data_cache_path, 'out_downscaled_AFG_data.npy'))
down_data = np.load(os.path.join(data_cache_path, 'out_downscaled_g_data.npy'))

# TODO: single day data semivariogram
def get_semivariogram(data, lats, lons, title):
    flat_data = pd.DataFrame({'z': data.reshape(np.prod(data.shape)),
                         'x': np.repeat(lats, len(lons)),
                         'y': list(lons) * len(lats)})
    V1 = Variogram(flat_data[['x', 'y']].values, flat_data.z.values, normalize=False)
    V1.plot()
    plt.savefig(title)

# TODO: choose day and plot
