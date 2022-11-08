import numpy as np
import sys
import os
import tensorflow as tf
import time
if '..' not in sys.path:
    sys.path.append('..')
import util_tools
from util_tools.cGAN_model import Condition_GAN
import pandas as pd
from util_tools import downscale

# define helper function
def mapping_tanh(x):
    x02 = tf.keras.backend.tanh(x) + 1  # x in range(0,2)
    scale = 1 / 2.
    return x02 * scale


def mapping_abs(x):
    x02 = x / (1 + tf.keras.backend.abs(x)) + 1
    scale = 1 / 2.
    return x02 * scale


def nnelu(x):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(x))


def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
    return random_sample


def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_predict), axis=[1, 2, 3])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) -
                                              tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) -
                                              tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss

def get_generator(n_lag, n_pred, task_dim, latent_space_dim):
    high_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))

    x1 = tf.keras.layers.ConvLSTM2D(32, kernel_size=(3, 3), return_sequences=False,
                                    activation=tf.keras.layers.LeakyReLU())(high_input)
    x1 = tf.keras.layers.Flatten()(x1)

    low_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
    x2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3, 3), return_sequences=False,
                                    activation=tf.keras.layers.LeakyReLU())(low_input)
    x2 = tf.keras.layers.Flatten()(x2)

    ele_input = tf.keras.Input(shape=(task_dim[0], task_dim[1], 1))
    x3 = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU())(ele_input)
    x3 = tf.keras.layers.Flatten()(x3)

    other_input = tf.keras.Input(shape=(3))
    x4 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU())(other_input)

    e1 = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3, x4])
    e2 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(e1)
    e3 = tf.keras.layers.BatchNormalization()(e2)
    e3 = tf.keras.layers.Dropout(0.5)(e3)
    e4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(e3)
    e5 = tf.keras.layers.BatchNormalization()(e4)
    e5 = tf.keras.layers.Dropout(0.5)(e5)
    e6 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(e5)
    e7 = tf.keras.layers.BatchNormalization()(e6)
    e7 = tf.keras.layers.Dropout(0.5)(e7)
    est_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(e7)
    est_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(e7)
    est_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([est_mu, est_log_variance])

    d1 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(est_output)
    d2 = tf.keras.layers.BatchNormalization()(d1)
    d3 = tf.keras.layers.Concatenate(axis=1)([d2, e7])
    d3 = tf.keras.layers.Dropout(0.5)(d3)
    d4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(d3)
    d5 = tf.keras.layers.BatchNormalization()(d4)
    d6 = tf.keras.layers.Concatenate(axis=1)([d5, e5])
    d6 = tf.keras.layers.Dropout(0.5)(d6)
    d7 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                               kernel_regularizer=tf.keras.regularizers.l2(0.01),
                               activation=tf.keras.layers.LeakyReLU())(d6)
    d8 = tf.keras.layers.BatchNormalization()(d7)
    d9 = tf.keras.layers.Concatenate(axis=1)([d8, e3])
    d9 = tf.keras.layers.Dropout(0.5)(d9)
    d10 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                activation=tf.keras.layers.LeakyReLU())(d9)
    d11 = tf.keras.layers.BatchNormalization()(d10)
    d12 = tf.keras.layers.Dense(n_pred * np.prod(task_dim), activation=mapping_abs)(d11)
    d13 = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(d12)
    generator = tf.keras.Model([high_input, low_input, ele_input, other_input], d13)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    generator.compile(optimizer=opt, loss=loss_func(est_mu, est_log_variance))
    return generator

# clip model weights to a given hypercube
class ClipConstraint(tf.keras.constraints.Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)


def define_discriminator(n_pred, task_dim):
    const = ClipConstraint(0.01)
    pred_input = tf.keras.Input(shape=(n_pred, task_dim[0], task_dim[1]))
    y1 = tf.keras.layers.Flatten()(pred_input)
    y = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU(), kernel_constraint=const)(y1)
    #y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.Dense(1)(y)
    discriminator = tf.keras.Model([pred_input], y)
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    discriminator.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])
    return discriminator

if __name__ == '__main__':
    start = time.time()
    # define parameters
    data_cache_path = sys.argv[1]
    n_lag = 20
    n_pred = 1
    task_dim = [5, 5]
    latent_space_dim = 20

    # load data
    X_high = np.load(os.path.join(data_cache_path, 'X_high.npy'))
    X_low = np.load(os.path.join(data_cache_path, 'X_low.npy'))
    X_ele = np.load(os.path.join(data_cache_path, 'X_ele.npy'))
    X_other = np.load(os.path.join(data_cache_path, 'X_other.npy'))
    Y = np.load(os.path.join(data_cache_path, 'Y.npy'))
    if 's2s_model' in os.listdir(data_cache_path):
        generator = tf.keras.models.load_model(os.path.join(data_cache_path, 's2s_model'))
    else:
        generator = get_generator(n_lag, n_pred, task_dim, latent_space_dim)
        # define callbacks
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        best_save = tf.keras.callbacks.ModelCheckpoint(os.path.join(data_cache_path, 's2s_model'), save_best_only=True,
                                                       save_weights_only=True, monitor='val_loss', mode='min')
        callbacks = [lr_scheduler, early_stopping, best_save]

        history = generator.fit([X_high, X_low, X_ele, X_other], Y, epochs=100, callbacks=callbacks, validation_split=0.2)
        pd.DataFrame(history.history).to_csv(os.path.join(data_cache_path, 'history.csv'))


    print('Training Time: ', (time.time() - start) / 60, 'mins')

    # Downscale
    start = time.time()

    generator.load_weights('s2s_model')
    test_g_data = np.load(os.path.join(data_cache_path, 'test_g_data.npy'))
    test_m_data = np.load(os.path.join(data_cache_path, 'test_m_data.npy'))
    test_days = np.load(os.path.join(data_cache_path, 'test_days.npy'))
    ele_data = np.load(os.path.join(data_cache_path, 'test_ele.npy'))
    G_lats = np.load(os.path.join(data_cache_path, 'test_lats.npy'))
    G_lons = np.load(os.path.join(data_cache_path, 'test_lons.npy'))
    M_lats = None
    M_lons = None

    dscler = downscale.downscaler(generator)
    downscaled_data = dscler.downscale(test_g_data[:n_lag], test_m_data, ele_data, [G_lats, G_lons, M_lats, M_lons],
                                       test_days, n_lag, n_pred, task_dim)
    np.save(os.path.join(data_cache_path, 'downscaled_data.npy'), downscaled_data)
    print('Downscaling Time: ', (time.time() - start) / 60, 'mins')
