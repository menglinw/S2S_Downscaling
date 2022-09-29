import numpy as np
import sys
import os
import tensorflow as tf
import time
if '..' not in sys.path:
    sys.path.append('..')
import util_tools
import pandas as pd

# define parameters
data_cache_path = sys.argv[1]
n_lag = 10
n_pred = 3
task_dim = [5, 5]

# load data
X_high = np.load(os.path.join(data_cache_path, 'X_high.npy'))
X_low = np.load(os.path.join(data_cache_path, 'X_low.npy'))
X_ele = np.load(os.path.join(data_cache_path, 'X_ele.npy'))
X_other = np.load(os.path.join(data_cache_path, 'X_other.npy'))
Y = np.load(os.path.join(data_cache_path, 'Y.npy'))

# train model
start = time.time()


def nnelu(input):
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def mapping_to_target_range( x, target_min=0, target_max=1 ) :
    x02 = tf.keras.backend.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min

def get_generator(n_lag, n_pred, task_dim):
    high_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
    x1 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3,3), return_sequences=True, activation=tf.keras.layers.LeakyReLU())(high_input)
    x1 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3,3), activation=tf.keras.layers.LeakyReLU())(x1)
    x1 = tf.keras.layers.Flatten()(x1)

    low_input = tf.keras.Input(shape=(n_lag, task_dim[0], task_dim[1], 1))
    x2 = tf.keras.layers.ConvLSTM2D(16, kernel_size=(3,3), activation=tf.keras.layers.LeakyReLU())(low_input)
    x2 = tf.keras.layers.Flatten()(x1)

    ele_input = tf.keras.Input(shape=(task_dim[0], task_dim[1], 1))
    x3 = tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation=tf.keras.layers.LeakyReLU())(ele_input)
    x3 = tf.keras.layers.Flatten()(x3)

    other_input =  tf.keras.Input(shape=(3))
    x4 = tf.keras.layers.Dense(8, activation=tf.keras.layers.LeakyReLU())(other_input)

    x = tf.keras.layers.Concatenate(axis=1)([x1, x2, x3, x4])
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(30, activation=tf.keras.layers.LeakyReLU())(x)
    x = tf.keras.layers.Dense(n_pred*np.prod(task_dim), activation=nnelu)(x)
    x = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(x)
    model = tf.keras.Model([high_input, low_input, ele_input, other_input], x)
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    return model


generator = get_generator(n_lag, n_pred, task_dim)
# define callbacks
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
best_save = tf.keras.callbacks.ModelCheckpoint(os.path.join(data_cache_path, 's2s_model'), save_best_only=True, monitor='val_loss', mode='min')
callbacks = [lr_scheduler, early_stopping, best_save]

history = generator.fit([X_high, X_low, X_ele, X_other], Y, epochs=100, callbacks=callbacks, validation_split=0.25)
pd.DataFrame(history.history).to_csv(os.path.join(data_cache_path, 'history.csv'))
print('Training Time: ', (time.time()-start)/60, 'mins')
