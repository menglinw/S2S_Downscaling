import numpy as np
import tensorflow as tf
import sys
if '..' not in sys.path:
    sys.path.append('..')
from util_tools import data_loader


class downscaler():
    def __init__(self, model):
        self.model = model

    def downscale(self, h_data, l_data, ele_data, lat_lon, days, n_lag, n_pred, task_dim):
        '''

        :param h_data: initalization
        :param l_data: range same as downscaled
        :param ele_data: fixed
        :param lat_lon: fixed
        :param days: range same as l_data
        :param n_lag:
        :param n_pred:
        :param task_dim:
        :return:
        '''
        G_lats, G_lons, M_lats, M_lons = lat_lon
        data_processor = data_loader.data_processer()
        if l_data.shape[1:] != h_data.shape[1:]:
            l_data = data_processor.unify_m_data(h_data, l_data, G_lats, G_lons, M_lats, M_lons)
        X_high, X_low, X_ele, X_other = data_processor.flatten(h_data[-n_lag:], l_data, ele_data, [G_lats, G_lons],
                                                               days,
                                                               n_lag=n_lag, n_pred=n_pred, task_dim=task_dim,
                                                               is_perm=False, return_Y=False)
        temp_matrix = np.zeros((n_pred, n_pred, h_data.shape[1], h_data.shape[2]))
        for i in range(l_data.shape[0]-(n_lag-1)-1): #or l_data.shape[0]-(n_lag-1)-(n_pred-1)
            # TODO: prediction
            pred_Y = self.model.predict([X_high, X_low, X_ele, X_other])
            # TODO: reconstruct predictions at time t back to large image(define a separate function)
            pred_list = [self._reconstruct(pred_Y[:, j], h_data.shape[1:], task_dim=task_dim) for j in range(n_pred)]
            # TODO: cache predictions
            temp_matrix = np.concatenate([temp_matrix[1:], np.expand_dims(np.array(pred_list), 0)], axis=0)
            # TODO: get current estimation from different predictions
            current_est = np.sum(np.array([temp_matrix[i, n_pred-i-1] for i in range(n_pred)]), axis=0)
            # TODO: update high resolution initialization
            h_data = np.concatenate([h_data, np.expand_dims(current_est, 0)], axis=0)
            # TODO: flatten to input data
            X_high, X_low, X_ele, X_other = data_processor.flatten(h_data[-n_lag:], l_data[i+1:],
                                                                   ele_data, [G_lats, G_lons], days[i+1:],
                                                                   n_lag=n_lag, n_pred=n_pred, task_dim=task_dim,
                                                                   is_perm=False, return_Y=False)
        return h_data[-(l_data.shape[0]-n_lag):]

    def _reconstruct(self, pred_Y, org_dim, task_dim):
        '''
        reconstruct a list of small images back to a large image
        :param pred_Y:
        :param org_dim:
        :param task_dim:
        :return:
        '''
        rec_Y = dict()
        lat_dim = org_dim[0] + 1 - pred_Y.shape[-2]
        lon_dim = org_dim[1] + 1 - pred_Y.shape[-1]
        for i, lat_corner in enumerate(range(pred_Y.shape[-2], org_dim[0] + 1), 1):
            for j, lon_corner in enumerate(range(pred_Y.shape[-1], org_dim[1] + 1), 1):
                current_index = (i - 1) * lon_dim + j - 1
                current_mtx = pred_Y[current_index]
                for k, lat in enumerate(range(lat_corner - task_dim[0], lat_corner)):
                    for h, lon in enumerate(range(lon_corner - task_dim[1], lon_corner)):
                        if (lat, lon) not in rec_Y:
                            rec_Y.setdefault((lat, lon), [current_mtx[k, h]])
                        else:
                            rec_Y[(lat, lon)].append(current_mtx[k, h])
        out = np.zeros(org_dim)
        for a in range(org_dim[0]):
            for b in range(org_dim[1]):
                out[a, b] = np.mean(rec_Y[(a, b)])
        return out

