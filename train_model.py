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



# train model
def nnelu(input):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def mapping_to_target_range( x, target_min=0, target_max=1 ) :
    x02 = tf.keras.backend.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min

def get_generator(n_lag, n_pred, task_dim):
    high_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
    x1 = tf.keras.layers.ConvLSTM2D(32, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU())(high_input)
    #x1 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3, 3), return_sequences=True,
    #                                activation=tf.keras.layers.LeakyReLU())(x1)
    #x1 = tf.keras.layers.Reshape([n_lag, 16])(x1)
    #x1 = tf.keras.layers.LSTM(32, activation=tf.keras.layers.LeakyReLU())(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    low_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
    x2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3, 3), return_sequences=False,
                                    activation=tf.keras.layers.LeakyReLU())(low_input)
    x2 = tf.keras.layers.Flatten()(x2)

    ele_input = tf.keras.Input(shape=(task_dim[0], task_dim[1], 1))
    x3 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU())(ele_input)
    x3 = tf.keras.layers.Flatten()(x3)

    other_input = tf.keras.Input(shape=(3))
    x4 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU())(other_input)

    x = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3, x4])
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(60, kernel_initializer="he_normal", use_bias=True,
                              activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(60, kernel_initializer="he_normal", use_bias=True,
                              activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                              activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                              activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                              activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                              activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(n_pred * np.prod(task_dim), activation=nnelu)(x)
    x = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(x)
    generator = tf.keras.Model([high_input, low_input, ele_input, other_input], x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    generator.compile(optimizer=opt, loss='mean_absolute_error')
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
    task_dim = [3, 3]

    # load data
    X_high = np.load(os.path.join(data_cache_path, 'X_high.npy'))
    X_low = np.load(os.path.join(data_cache_path, 'X_low.npy'))
    X_ele = np.load(os.path.join(data_cache_path, 'X_ele.npy'))
    X_other = np.load(os.path.join(data_cache_path, 'X_other.npy'))
    Y = np.load(os.path.join(data_cache_path, 'Y.npy'))
    if 's2s_model' in os.listdir(data_cache_path):
        generator = tf.keras.models.load_model(os.path.join(data_cache_path, 's2s_model'))
    else:
        generator = get_generator(n_lag, n_pred, task_dim)
        # define callbacks
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        best_save = tf.keras.callbacks.ModelCheckpoint(os.path.join(data_cache_path, 's2s_model'), save_best_only=True,
                                                       monitor='val_loss', mode='min')
        callbacks = [lr_scheduler, early_stopping, best_save]

        history = generator.fit([X_high, X_low, X_ele, X_other], Y, epochs=20, callbacks=callbacks, validation_split=0.2)
        pd.DataFrame(history.history).to_csv(os.path.join(data_cache_path, 'history.csv'))


    print('Training Time: ', (time.time() - start) / 60, 'mins')


    start = time.time()
    discriminator = define_discriminator(n_pred, task_dim)
    generator2 = get_generator(n_lag, n_pred, task_dim)

    cGAN = Condition_GAN(generator2, discriminator, lr=0.00005)
    history = cGAN.fit(20, 200, [X_high, X_low, X_ele, X_other], Y)
    history.to_csv(os.path.join(data_cache_path, 'loss_history.csv'))
    generator2.save('s2s_model_fine')
    print('cGAN Training Time: ', (time.time()-start)/60, 'mins')
