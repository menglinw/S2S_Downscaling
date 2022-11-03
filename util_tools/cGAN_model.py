# example of training an unconditional gan on the fashion mnist dataset
import numpy as np
import tensorflow as tf
import pandas as pd


class Condition_GAN():
    def __init__(self, Generator, Discriminator, loss, lr=0.0002):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.lr = lr
        self.loss = loss
        self.GAN_model = self.define_GAN(self.Generator, self.Discriminator, self.loss, lr=self.lr)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def define_GAN(self, G, D, loss, lr=0.00005):
        D.trainable = False
        high_shape, low_shape, ele_shape, other_shape = G.input_shape
        high_input = tf.keras.layers.Input(shape=high_shape[1:])
        low_input = tf.keras.layers.Input(shape=low_shape[1:])
        ele_input = tf.keras.layers.Input(shape=ele_shape[1:])
        other_input = tf.keras.layers.Input(shape=other_shape[1:])
        x_temp = G([high_input, low_input, ele_input, other_input])
        x = D(x_temp)
        model1 = tf.keras.Model([high_input, low_input, ele_input, other_input], [x, x_temp], name='cGAN')
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        model1.compile(loss=[self.wasserstein_loss, loss], loss_weights=[0.01, 0.99], optimizer=opt)
        return model1

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

