# example of training an unconditional gan on the fashion mnist dataset
import numpy as np
import tensorflow as tf
import pandas as pd


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


class Condition_GAN():
    def __init__(self, n_lag, n_pred, task_dim, latent_space_dim, lr=0.0002):
        self.n_lag, self.n_pred, self.task_dim, self.latent_space_dim = n_lag, n_pred, task_dim, latent_space_dim
        self.Discriminator = self.define_discriminator(n_pred, task_dim)
        self.lr = lr
        self.Discriminator.trainable = False

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
                                   activation=tf.keras.layers.LeakyReLU())(e1)
        e3 = tf.keras.layers.BatchNormalization()(e2)
        e3 = tf.keras.layers.Dropout(0.5)(e3)
        e4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(e3)
        e5 = tf.keras.layers.BatchNormalization()(e4)
        e5 = tf.keras.layers.Dropout(0.5)(e5)
        e6 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(e5)
        e7 = tf.keras.layers.BatchNormalization()(e6)
        e7 = tf.keras.layers.Dropout(0.5)(e7)
        est_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(e7)
        est_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(e7)
        est_output = tf.keras.layers.Lambda(self.sampling, name="encoder_output")([est_mu, est_log_variance])

        d1 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(est_output)
        d2 = tf.keras.layers.BatchNormalization()(d1)
        d3 = tf.keras.layers.Concatenate(axis=1)([d2, e7])
        d3 = tf.keras.layers.Dropout(0.5)(d3)
        d4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(d3)
        d5 = tf.keras.layers.BatchNormalization()(d4)
        d6 = tf.keras.layers.Concatenate(axis=1)([d5, e5])
        d6 = tf.keras.layers.Dropout(0.5)(d6)
        d7 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(d6)
        d8 = tf.keras.layers.BatchNormalization()(d7)
        d9 = tf.keras.layers.Concatenate(axis=1)([d8, e3])
        d9 = tf.keras.layers.Dropout(0.5)(d9)
        d10 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                    activation=tf.keras.layers.LeakyReLU())(d9)
        d11 = tf.keras.layers.BatchNormalization()(d10)
        d12 = tf.keras.layers.Dense(n_pred * np.prod(task_dim), activation=self.mapping_abs)(d11)
        d13 = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(d12)
        self.Generator = tf.keras.Model([high_input, low_input, ele_input, other_input], d13)

        x_temp = self.Generator([high_input, low_input, ele_input, other_input])
        x = self.Discriminator(x_temp)

        self.GAN_model = tf.keras.Model([high_input, low_input, ele_input, other_input], [x, x_temp], name='cGAN')
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.GAN_model.compile(loss=[self.wasserstein_loss, self.loss_func(est_mu, est_log_variance)], optimizer=opt)


    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    # define helper function
    def mapping_tanh(self, x):
        x02 = tf.keras.backend.tanh(x) + 1  # x in range(0,2)
        scale = 1 / 2.
        return x02 * scale

    def mapping_abs(self, x):
        x02 = x / (1 + tf.keras.backend.abs(x)) + 1
        scale = 1 / 2.
        return x02 * scale

    def nnelu(self, x):
        return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(x))

    def sampling(self, mu_log_variance):
        mu, log_variance = mu_log_variance
        epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
        random_sample = mu + tf.keras.backend.exp(log_variance / 2) * epsilon
        return random_sample

    def loss_func(self, encoder_mu, encoder_log_variance):
        def vae_reconstruction_loss(y_true, y_predict):
            reconstruction_loss_factor = 1000
            reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.abs(y_true - y_predict), axis=[1, 2, 3])
            return reconstruction_loss_factor * reconstruction_loss

        def vae_kl_loss(y_true, y_predict):
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

    def define_discriminator(self, n_pred, task_dim):
        const = ClipConstraint(0.01)
        pred_input = tf.keras.Input(shape=(n_pred, task_dim[0], task_dim[1]))
        y1 = tf.keras.layers.Flatten()(pred_input)
        y = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(), kernel_constraint=const)(y1)
        y = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(), kernel_constraint=const)(y)
        # y = tf.keras.layers.Dropout(0.5)(y)
        y = tf.keras.layers.Dense(1)(y)
        discriminator = tf.keras.Model([pred_input], y)
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        discriminator.compile(loss=self.wasserstein_loss, optimizer=opt, metrics=['accuracy'])
        return discriminator

    def define_GAN(self, D, n_lag, n_pred, task_dim, latent_space_dim, lr=0.00005):
        D.trainable = False
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
                                   activation=tf.keras.layers.LeakyReLU())(e1)
        e3 = tf.keras.layers.BatchNormalization()(e2)
        e3 = tf.keras.layers.Dropout(0.5)(e3)
        e4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(e3)
        e5 = tf.keras.layers.BatchNormalization()(e4)
        e5 = tf.keras.layers.Dropout(0.5)(e5)
        e6 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(e5)
        e7 = tf.keras.layers.BatchNormalization()(e6)
        e7 = tf.keras.layers.Dropout(0.5)(e7)
        est_mu = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_mu")(e7)
        est_log_variance = tf.keras.layers.Dense(units=latent_space_dim, name="encoder_log_variance")(e7)
        est_output = tf.keras.layers.Lambda(self.sampling, name="encoder_output")([est_mu, est_log_variance])

        d1 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(est_output)
        d2 = tf.keras.layers.BatchNormalization()(d1)
        d3 = tf.keras.layers.Concatenate(axis=1)([d2, e7])
        d3 = tf.keras.layers.Dropout(0.5)(d3)
        d4 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(d3)
        d5 = tf.keras.layers.BatchNormalization()(d4)
        d6 = tf.keras.layers.Concatenate(axis=1)([d5, e5])
        d6 = tf.keras.layers.Dropout(0.5)(d6)
        d7 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                   activation=tf.keras.layers.LeakyReLU())(d6)
        d8 = tf.keras.layers.BatchNormalization()(d7)
        d9 = tf.keras.layers.Concatenate(axis=1)([d8, e3])
        d9 = tf.keras.layers.Dropout(0.5)(d9)
        d10 = tf.keras.layers.Dense(30, kernel_initializer="he_normal", use_bias=True,
                                    activation=tf.keras.layers.LeakyReLU())(d9)
        d11 = tf.keras.layers.BatchNormalization()(d10)
        d12 = tf.keras.layers.Dense(n_pred * np.prod(task_dim), activation=self.mapping_abs)(d11)
        x_temp = tf.keras.layers.Reshape([n_pred, task_dim[0], task_dim[1]])(d12)
        G = tf.keras.Model([high_input, low_input, ele_input, other_input], x_temp)

        x = D(x_temp)
        model1 = tf.keras.Model([high_input, low_input, ele_input, other_input], [x, x_temp], name='cGAN')
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        model1.compile(loss=[self.wasserstein_loss, self.loss_func(est_mu, est_log_variance)], optimizer=opt)
        return model1, G

    def fit(self, epochs, batch_size, X, Y):
        n = Y.shape[0]
        loss = []
        for i in range(epochs):
            d1 = []
            d2 = []
            g = []
            mae = []
            sum_list = []
            for j in range(int(n/batch_size)):
                batch_X = [d[j*batch_size:(j+1)*batch_size] for d in X]
                batch_Y = Y[j*batch_size:(j+1)*batch_size]
                fake_Y = self.Generator.predict(batch_X)

                d1_loss, _ = self.Discriminator.train_on_batch(batch_Y, tf.cast(-np.ones((batch_size, 1)), tf.float32))
                d2_loss, _ = self.Discriminator.train_on_batch(fake_Y, tf.cast(np.ones((batch_size, 1)), tf.float32))
                # print('finish discriminator training')
                sum_loss, g_loss, mae_loss = self.GAN_model.train_on_batch(batch_X,
                                                       [tf.cast(np.ones((batch_size, 1)), tf.float32),
                                                        batch_Y])
                #print(g_loss)
                d1.append(d1_loss)
                d2.append(d2_loss)
                sum_list.append(sum_loss)
                mae.append(mae_loss)
                g.append(g_loss)
            print('Epoch:%d, real_loss=%.3f, fake_loss=%.3f, sum_loss=%.3f, mae_loss=%.3f, g_loss=%.3f' %
                  (i + 1, np.mean(d1), np.mean(d2), np.mean(sum_list), np.mean(mae), np.mean(g)))
            loss.append([i + 1, np.mean(d1), np.mean(d2), np.mean(sum_list), np.mean(mae), np.mean(g)])
        loss = pd.DataFrame(loss, columns=['epochs', 'real_loss', 'fake_loss', 'sum_loss', 'mae_loss', 'g_loss'])
        return loss

