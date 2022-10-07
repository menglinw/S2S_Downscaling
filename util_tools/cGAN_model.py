# example of training an unconditional gan on the fashion mnist dataset
import numpy as np
import tensorflow as tf


class Condition_GAN():
    def __init__(self, Generator, Discriminator, lr=0.0002):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.lr = lr
        self.GAN_model = self.define_GAN(self.Generator, self.Discriminator, lr=self.lr)

    def define_GAN(self, G, D, lr):
        D.trainable = False
        high_shape, low_shape, ele_shape, other_shape = G.input_shape
        high_input = tf.keras.layers.Input(shape=high_shape[1:])
        low_input = tf.keras.layers.Input(shape=low_shape[1:])
        ele_input = tf.keras.layers.Input(shape=ele_shape[1:])
        other_input = tf.keras.layers.Input(shape=other_shape[1:])
        x = G([high_input, low_input, ele_input, other_input])
        x = D(x)
        model1 = tf.keras.Model([high_input, low_input, ele_input, other_input], x, name='cGAN')
        opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        model1.compile(loss='binary_crossentropy', optimizer=opt)
        return model1

    def fit(self, epochs, batch_size, X, Y):
        n = Y.shape[0]
        for i in range(epochs):
            for j in range(int(n/batch_size)):
                batch_X = [d[j*batch_size:(j+1)*batch_size] for d in X]
                batch_Y = Y[j*batch_size:(j+1)*batch_size]
                fake_Y = self.Generator.predict(batch_X)
                '''
                D_Input = np.concatenate([batch_Y, fake_Y], axis=0)
                D_Output = np.array([0]*batch_size+[1]*batch_size).reshape((2*batch_size, 1))
                perm = np.random.permutation(2*batch_size)
                '''
                self.Discriminator.trainable = True
                d1_loss, _ = self.Discriminator.train_on_batch(batch_Y[:batch_size//15], np.ones((batch_size//15, 1)))
                d2_loss, _ = self.Discriminator.train_on_batch(fake_Y[:batch_size//15], np.zeros((batch_size//15, 1)))
                self.Discriminator.trainable = False
                g_loss = self.GAN_model.train_on_batch(batch_X, np.array([0]*batch_size).reshape((batch_size, 1)))
                print('Epoch:%d, batch:%d/%d, real_loss=%.3f, fake_loss=%.3f, g_loss=%.3f' %
                      (i + 1, j + 1, int(n/batch_size), d1_loss, d2_loss, g_loss))

