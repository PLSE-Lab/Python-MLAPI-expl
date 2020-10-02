import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

import theano

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.discriminator = None
        self.generator = None
        self.adversarial_model = None
        self.discriminator_model = None
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel

    def discriminator_nn(self, depth=64, dropout=0.4):
        if self.discriminator:
            return self.discriminator
        self.discriminator = Sequential()

        in_shape = (self.channel,self.img_rows, self.img_cols)
        self.discriminator.add(Conv2D(depth*1, (5, 5), strides=(2, 2), padding="same", input_shape=in_shape, data_format='channels_first'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*2, (5, 5), strides=(2,2), padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*4, (5, 5), strides=(2,2), padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*8, (5, 5), strides=(1,1), padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(1))
        self.discriminator.add(Activation('sigmoid'))

        self.discriminator.summary()

        return self.discriminator

    def generator_nn(self, dropout = 0.4, dim = 7):
        if self.generator:
            return self.generator
        self.generator = Sequential()
        
        depth = 64+64+64+64
        # In: 100
        # Out: dim x dim x depth
        self.generator.add(Dense(dim*dim*depth, input_dim=100))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(Reshape((depth, dim, dim)))
        self.generator.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.generator.add(UpSampling2D(data_format='channels_first'))
        self.generator.add(Conv2DTranspose(int(depth/2), (5, 5), padding='same',output_shape=(None, int(depth/2), 2*dim, 2*dim), data_format='channels_first'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D(data_format='channels_first'))
        self.generator.add(Conv2DTranspose(int(depth/4), (5, 5), padding='same',output_shape=(None ,int(depth/4), 4*dim, 4*dim), data_format='channels_first'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(Conv2DTranspose(int(depth/8), (5, 5), padding='same',output_shape=(None ,int(depth/8), 4*dim, 4*dim), data_format='channels_first'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.generator.add(Conv2DTranspose(1, (5, 5), padding='same',output_shape=(None, 1, 4*dim, 4*dim), data_format='channels_first'))
        self.generator.add(Activation('sigmoid'))
        self.generator.summary()
        return self.generator

    def get_discriminator_model(self):
        if self.discriminator_model:
            return self.discriminator_model
        optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.discriminator_model = Sequential()
        self.discriminator_model.add(self.discriminator_nn())
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.discriminator_model

    def get_adversarial_model(self):
        if self.adversarial_model:
            return self.adversarial_model
        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.adversarial_model = Sequential()
        self.adversarial_model.add(self.generator_nn())
        self.adversarial_model.add(self.discriminator_nn())
        self.adversarial_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.adversarial_model

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = pd.read_csv("../input/train.csv").values
        self.x_train = self.x_train[:, 1:].reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols).astype(np.float32)

        print('train shape:', self.x_train.shape)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.get_discriminator_model()
        self.adversarial = self.DCGAN.get_adversarial_model()
        self.generator = self.DCGAN.generator_nn()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "step %d: [D loss: %f, acc: %f] [A loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1], a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        if fake:
            filename = 'generated_mnist.png'
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "generated_mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            filename = 'real_mnist.png'
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    mnist_dcgan.train(train_steps=20, batch_size=256, save_interval=1)
    mnist_dcgan.plot_images(fake=True, save2file=True) # fake final plots
    mnist_dcgan.plot_images(fake=False, save2file=True) # compare with real plots