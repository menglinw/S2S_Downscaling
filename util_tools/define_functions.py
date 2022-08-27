import os
import sys
from MetaTrain import MetaSGD
from TaskExtractor import TaskExtractor
from Downscaler import Downscaler
import utils.data_processing as data_processing
import math
import numpy as np
import netCDF4 as nc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from matplotlib import pyplot as plt
from math import exp, sqrt, log
import time
import geopandas as gpd
import pandas as pd


class run_metadownscale():
    def __init__(self, task_dim, test_proportion, n_lag, components, save_path, target_var, data_part, use_beta=True,
                 use_meta=True):
        # task dimension 3*3
        self.task_dim = task_dim

        # proportion of test data
        self.test_proportion = test_proportion

        # number of lagging steps
        self.n_lag = n_lag

        # number of components of MDN
        self.components = components

        # save path
        self.save_path = save_path

        # target variable
        self.target_var = target_var

        # data part
        # lat: 1 - 4, lon: 1-7
        # example: 13-lat1, lon3
        self.data_part = str(data_part)

        # use beta function for training
        self.use_beta = use_beta

        # use meta train or not
        self.use_meta = use_meta

        # load data
        self.data, self.lats_lons, self.test_g_data, self.test_m_data = self._load_data()

    def _normalize(self, data):
        return (data - data.min()) / (np.quantile(data, 0.95) - data.min())

    def _load_data(self):
        file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
        file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
        file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']
        # read data
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        m_data_nc = nc.Dataset(file_path_m)

        # define lat&lon of MERRA, G5NR and mete
        M_lons = m_data_nc.variables['lon'][:]
        # self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        M_lats = m_data_nc.variables['lat'][:]
        # self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        G_lons = g05_data.variables['lon'][:]
        # self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        G_lats = g05_data.variables['lat'][:]

        # extract target data
        g_data = np.concatenate((g05_data.variables[self.target_var], g06_data.variables[self.target_var]), axis=0)
        m_data = m_data_nc.variables[self.target_var][5 * 365:7 * 365, :, :]

        # load country file
        country_shape = gpd.read_file(file_path_country[0])
        for country_path in file_path_country[1:]:
            country_shape = pd.concat([country_shape, gpd.read_file(country_path)])

        # get outer bound
        latmin, lonmin, latmax, lonmax = country_shape.total_bounds
        latmin_ind = np.argmin(np.abs(G_lats - latmin))
        latmax_ind = np.argmin(np.abs(G_lats - latmax))
        lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
        # 123 * 207
        g_data = g_data[:, latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]

        G_lats = G_lats[latmin_ind - 1:latmax_ind + 1]
        G_lons = G_lons[lonmin_ind:lonmax_ind + 2]

        # take part of the data
        data_part_lat = int(self.data_part[0])
        data_part_lon = int(self.data_part[1])
        length_lat = 30 if data_part_lat != 4 else 33
        g_data = g_data[:, (data_part_lat-1)*30:((data_part_lat-1)*30+length_lat), (data_part_lon-1)*30:data_part_lon*30]
        G_lats = G_lats[(data_part_lat-1)*30:((data_part_lat-1)*30+length_lat)]
        G_lons = G_lons[(data_part_lon-1)*30:data_part_lon*30]

        print('Data part:', self.data_part)
        print('Data shape:', g_data.shape)

        # log and normalize data
        g_data = self._normalize(np.log(g_data))
        m_data = self._normalize(np.log(m_data))

        # split data into traing and test
        train_g_data, test_g_data = g_data[:657], g_data[657:]
        train_m_data, test_m_data = m_data[:657], m_data[657:]
        data = [train_g_data, train_m_data]
        lats_lons = [G_lats, G_lons, M_lats, M_lons]
        return data, lats_lons, test_g_data, test_m_data


    def _model_generator(self, prob):

        def nnelu(input):
            return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

        input1 = layers.Input(shape=(self.n_lag, self.task_dim, self.task_dim, 1), dtype='float32')
        input1 = layers.BatchNormalization()(input1)
        input2 = layers.Input(shape=(self.task_dim, self.task_dim, 1), dtype='float32')
        input2 = layers.BatchNormalization()(input2)
        input3 = layers.Input(shape=(1,), dtype='float32')
        input3 = layers.BatchNormalization()(input3)

        X = layers.ConvLSTM2D(filters=100, kernel_size=(2, 2), activation=layers.LeakyReLU(), padding='same',
                              return_sequences=True)(input1)
        X = layers.ConvLSTM2D(filters=100, kernel_size=(2, 2), activation=layers.LeakyReLU(), padding='same',
                              return_sequences=True)(X)
        X = layers.ConvLSTM2D(filters=100, kernel_size=(2, 2), activation=layers.LeakyReLU())(X)
        X = layers.Flatten()(X)

        X1 = layers.Conv2D(20, (2, 2), activation='relu')(input2)
        X1 = layers.Flatten()(X1)
        X2 = layers.BatchNormalization()(input3)
        X = layers.Concatenate()([X, X1, X2])

        X3 = layers.Dense(128, kernel_initializer="he_normal", use_bias=True)(X)
        X3 = layers.LeakyReLU(alpha=0.05)(X3)
        X3 = layers.BatchNormalization()(X3)
        #X3 = layers.Dropout(0.5)(X3)

        for nodes in [128, 256, 128, 128, 128, 128, 128]:
            X3 = layers.Dense(nodes, kernel_initializer="he_normal", use_bias=True)(X3)
            X3 = layers.LeakyReLU(alpha=0.05)(X3)
            X3 = layers.BatchNormalization()(X3)
            #X3 = layers.Dropout(0.5)(X3)

        alphas1 = layers.Dense(self.components, activation="softmax")(X3)
        mus1 = layers.Dense(self.components * self.task_dim * self.task_dim)(X3)
        sigmas1 = layers.Dense(self.components * self.task_dim * self.task_dim, activation=nnelu)(X3)
        output1 = layers.Concatenate()([alphas1, mus1, sigmas1])

        X4 = layers.Dense(128, kernel_initializer="he_normal", use_bias=True)(X)
        X4 = layers.LeakyReLU(alpha=0.05)(X4)
        X4 = layers.BatchNormalization()(X4)
        #X4 = layers.Dropout(0.5)(X4)

        for nodes in [128, 256, 128, 128, 128, 128, 128, 64, 64, 64, 64]:
            X4 = layers.Dense(nodes, kernel_initializer="he_normal", use_bias=True)(X4)
            X4 = layers.LeakyReLU(alpha=0.05)(X4)
            X4 = layers.BatchNormalization()(X4)
            #X4 = layers.Dropout(0.5)(X4)

        output2 = layers.Dense(self.task_dim * self.task_dim, activation='relu')(X4)
        output2 = layers.Reshape((self.task_dim, self.task_dim))(output2)

        if prob:
            model = Model([input1, input2, input3], output1)
        else:
            model = Model([input1, input2, input3], output2)
        return model

    def slice_parameter_vectors(self, parameter_vector):
        alphas = parameter_vector[:, :self.components]
        mus = parameter_vector[:, self.components:(self.components * (self.task_dim * self.task_dim + 1))]
        sigmas = parameter_vector[:, (self.components * (self.task_dim * self.task_dim + 1)):]
        return alphas, mus, sigmas

    def res_loss(self, y, parameter_vector):
        alphas, mus, sigmas = self.slice_parameter_vectors(parameter_vector)
        mus = tf.reshape(mus, (tf.shape(mus)[0], self.components, self.task_dim, self.task_dim))
        sigmas = tf.reshape(sigmas, (tf.shape(sigmas)[0], self.components, self.task_dim, self.task_dim))
        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alphas),
            components_distribution=tfd.Independent(tfd.Normal(loc=mus, scale=sigmas), reinterpreted_batch_ndims=2))
        log_likelihood = tf.clip_by_value(gm.log_prob(tf.cast(y, tf.float32)), clip_value_min=-10000,
                                          clip_value_max=0)
        return -tf.experimental.numpy.nanmean(log_likelihood)

    def meta_train(self, epochs, batch_size, meta_lr, prob):


        # Prob Model Training
        meta_model = self._model_generator(prob=prob)

        # define TaskExtractor
        taskextractor = TaskExtractor(self.data, self.lats_lons, self.task_dim, self.test_proportion, self.n_lag)

        # define meta learner
        meta_optimizer = tf.keras.optimizers.Adam(meta_lr)
        inner_step = 3
        inner_optimizer = tf.keras.optimizers.Adam(meta_lr)
        if prob:
            self.meta_learner = MetaSGD(meta_model, self.res_loss, meta_optimizer, inner_step, inner_optimizer, taskextractor,
                                   meta_lr=meta_lr)
        else:
            self.meta_learner = MetaSGD(meta_model, tf.keras.losses.MeanAbsoluteError(), meta_optimizer,
                                   inner_step, inner_optimizer, taskextractor, meta_lr=meta_lr)
        start = time.time()
        # meta train with beta
        # define beta function
        def covariance_function(h, phi=0.5):
            return exp(-(h / phi)**2)

        def distance_function(loc1, loc2):
            return sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

        def beta_function(meta_rate, batch_locations, seen_locations, covariance_function, distance_function):
            batch_size = len(batch_locations)
            seen_size = len(seen_locations.items())
            if seen_size == 0:
                return meta_rate
            temp = 0
            for b_loc in batch_locations:
                for s_loc, n in seen_locations.items():
                    cov = covariance_function(distance_function(b_loc, s_loc))
                    temp += cov * (1 + log(n))
            mean_cov = temp / (batch_size * sum(list(seen_locations.values())))
            cov_factor = -log(mean_cov)
            bsize_factor = exp((batch_size / seen_size) ** 0.5) - 1
            print('mean cov:', mean_cov)
            print('covariance factor:', cov_factor)
            print('batch size factor:', bsize_factor)
            lr = meta_rate * bsize_factor * cov_factor
            return lr
        if self.use_beta:
            meta_beta_history = self.meta_learner.meta_fit(epochs, batch_size=batch_size, basic_train=True,
                                                      bootstrap_train=False,
                                                      randomize=True, beta_function=beta_function,
                                                      covariance_function=covariance_function,
                                                      distance_function=distance_function)
        else:
            meta_beta_history = self.meta_learner.meta_fit(epochs, batch_size=batch_size, basic_train=True,
                                                      bootstrap_train=False,
                                                      randomize=True)
        print('Meta Training:', (time.time() - start) / 60, ' mins')

        use_beta = '_beta' if self.use_beta else ''
        use_meta = '_meta' if self.use_meta else ''

        # save weights and history
        if prob:
            self.meta_learner.save_meta_weights(os.path.join(self.save_path, "meta_weights_prob_" + str(self.data_part)+use_meta+use_beta))
            np.save(os.path.join(self.save_path, 'meta_history_prob_' + str(self.data_part) +use_meta+use_beta),
                    np.array(meta_beta_history))
        else:
            self.meta_learner.save_meta_weights(os.path.join(self.save_path, "meta_weights_" + str(self.data_part)+use_meta+use_beta))
            np.save(os.path.join(self.save_path, 'meta_history_' + str(self.data_part)+use_meta+use_beta),
                    np.array(meta_beta_history))


        # save history plot
        plt.figure()
        plt.plot(meta_beta_history[0], "-r", label="with beta loss")
        plt.plot(meta_beta_history[1], "--r", label="with beta val loss")
        plt.legend(loc="upper left")
        if prob:
            plt.title('MDN Meta Training History')
            plt.show()
            plt.savefig(os.path.join(self.save_path, 'Meta_train_prob_compare_'+str(self.data_part)+use_meta+use_beta+'.jpg'))
        else:
            plt.title('NN Meta Training History')
            plt.show()
            plt.savefig(os.path.join(self.save_path, 'Meta_train_compare_'+str(self.data_part)+use_meta+use_beta+'.jpg'))

    def downscale(self, epochs, prob_use_meta=False, reg_use_meta=False):
        # define prob meta SGD
        meta_model_prob = self._model_generator(prob=True)

        taskextractor = TaskExtractor(self.data, self.lats_lons, self.task_dim, self.test_proportion, self.n_lag)

        # define meta learner
        meta_optimizer = tf.keras.optimizers.Adam(0.001)
        inner_step = 1
        inner_optimizer = tf.keras.optimizers.Adam(0.001)

        meta_learner_prob = MetaSGD(meta_model_prob, self.res_loss, meta_optimizer, inner_step, inner_optimizer,
                                    taskextractor, meta_lr=0.001)
        # meta_learner.load_meta_weights('../../Result/meta_weights_wob_prob')

        # Define reg meta SGD
        meta_model_reg = self._model_generator(prob=False)

        meta_learner_reg = MetaSGD(meta_model_reg, tf.keras.losses.MeanAbsoluteError(), meta_optimizer, inner_step,
                                   inner_optimizer,
                                   taskextractor, meta_lr=0.001)

        use_beta = '_beta' if self.use_beta else ''
        use_meta = '_meta' if self.use_meta else ''
        if prob_use_meta:
            meta_learner_prob.load_meta_weights(os.path.join(self.save_path, "meta_weights_prob_"+str(self.data_part)+use_meta+use_beta))
        if reg_use_meta:
            meta_learner_reg.load_meta_weights(os.path.join(self.save_path, "meta_weights_" + str(self.data_part)+use_meta+use_beta))

        downscaler = Downscaler(meta_learner_prob, meta_learner_reg, self.components, self.test_m_data)

        # define prob call backs
        optimizer = tf.keras.optimizers.Adam()

        def scheduler_reg(epoch, lr):
            if epoch <= 20:
                return 0.0001
            else:
                return 0.00001

        def scheduler_prob(epoch, lr):
            if epoch <= 10:
                return 0.0001
            elif epoch <= 20 and epoch > 10:
                return 0.00001
            else:
                return 0.000001

        lr_scheduler_prob = tf.keras.callbacks.LearningRateScheduler(scheduler_prob)
        lr_scheduler_reg = tf.keras.callbacks.LearningRateScheduler(scheduler_reg)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        callbacks_prob = [lr_scheduler_prob, early_stopping]
        callbacks_reg = [lr_scheduler_reg, early_stopping]

        start = time.time()
        downscaled_data, epochs_data = downscaler.downscale(epochs, optimizer, callbacks_prob=callbacks_prob, callbacks_reg=callbacks_reg)

        use_beta = '_beta' if self.use_beta else ''
        use_meta = '_meta' if self.use_meta else ''
        np.save(os.path.join(self.save_path, 'downscaled_data_'+str(self.data_part)+use_meta+use_beta), downscaled_data)
        np.save(os.path.join(self.save_path, 'epochs_data_'+str(self.data_part)+use_meta+use_beta), epochs_data)
        print('Prob Downscale Time:', (time.time() - start) / 60, 'mins')















