#!/usr/bin/env python
# coding: utf-8

# # Goal
# Here the goal is to use cyclegan for going back and forth between low-resolution grey-scale eyes (generated in Unity) and high-resolution color eyes from real-images. It will have to learn color-ification as well as change of perspective. The idea is basically to see where these models work well and what problems frequently come up.
# 
# ## VAE?
# Here we add a variational component to the generators to make them possibly more diverse.

# In[ ]:


get_ipython().system('pip install -qq git+https://www.github.com/keras-team/keras-contrib.git')


# In[ ]:


import scipy
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from IPython.display import Image
from keras.utils.vis_utils import model_to_dot
from tqdm import tqdm_notebook


# # Setup CycleGAN Code
# The code below has been lightly adapted from the code at https://github.com/eriklindernoren/Keras-GAN/tree/master/cyclegan by @eriklindernoren

# In[ ]:


from keras import backend as K
from keras import layers
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def conv2d(layer_input, filters, f_size=4, strides=2, **kwargs):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', **kwargs)(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    return d
        
def sampling_2d(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim_x = K.int_shape(z_mean)[1]
    dim_y = K.int_shape(z_mean)[2]
    dim_c = K.int_shape(z_mean)[3]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim_x, dim_y, dim_c))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def z_log_loss(x):
    return 0.5*K.mean(K.exp(x)-x)


def z_mean_loss(x):
    return 0.5*K.mean(K.square(x))



def make_vae_layer(x, prefix=''):
    in_shape = x._keras_shape[1:]
    
    z_mean = conv2d(x, filters=in_shape[2], f_size=1, strides=1, activity_regularizer=z_mean_loss)
    z_log_var = conv2d(x, filters=in_shape[2], f_size=1, strides=1, activity_regularizer=z_log_loss)
    z = layers.Lambda(sampling_2d, output_shape=in_shape, name='z_{}'.format(prefix))([z_mean, z_log_var])
    return z


# In[ ]:


def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = Concatenate()([u, skip_input])
    return u

class CycleGAN():
    def __init__(self, img_rows, img_cols, channels_A, channels_B):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels_A = channels_A
        self.channels_B = channels_B
        self.img_shape_A = (self.img_rows, self.img_cols, self.channels_A)
        self.img_shape_B = (self.img_rows, self.img_cols, self.channels_B)
        # Calculate output shape of D (PatchGAN)
        patch_r = int(self.img_rows / 2**4)
        patch_c = int(self.img_cols / 2**4)
        self.disc_patch = (patch_r, patch_c, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator(self.img_shape_A)
        self.d_B = self.build_discriminator(self.img_shape_B)
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator(self.img_shape_A, self.img_shape_B)
        self.g_BA = self.build_generator(self.img_shape_B, self.img_shape_A)

        # Input images from both domains
        img_A = Input(shape=self.img_shape_A, name='ImageA')
        img_B = Input(shape=self.img_shape_B, name='ImageB')

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_B)
        img_B_id = self.g_AB(img_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self, in_img_shape, out_img_shape):
        """U-Net Generator"""
        # Image input
        d0 = Input(shape=in_img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        
        d4 = make_vae_layer(d4, prefix='{}-{}'.format(in_img_shape[-1], out_img_shape[-1]))
        
        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(out_img_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img, name='Gen_{}_{}_{}-{}'.format(*in_img_shape, out_img_shape[-1]))

    def build_discriminator(self, img_shape):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity, name='Disc_{}_{}_{}'.format(*img_shape))


# # Build Models

# In[ ]:


cg = CycleGAN(32, 64, 3, 1)


# In[ ]:


Image(model_to_dot(cg.combined, show_shapes=True).create_png())


# ## Generators
# We just show one and the other can be inferred

# In[ ]:


Image(model_to_dot(cg.g_AB, show_shapes=True).create_png())


# ## Discriminator

# In[ ]:


Image(model_to_dot(cg.d_A, show_shapes=True).create_png())


# # Data Loaders
# Here we setup the data-loaders

# In[ ]:


import h5py
from skimage.util import montage as montage2d
norm_stack = lambda x: np.clip((x-127.0)/127.0, -1, 1)
def norm_stack(x):
    # calculate statistics on first 20 points
    mean = np.mean(x[:20])
    std = np.std(x[:20])
    return (1.0*x-mean)/(2*std)


# In[ ]:


data_dir = os.path.join('..', 'input', 'eye-gaze')
helen_eye_dir = '../input/getting-all-the-eye-balls/'


# ## Low-Res Unity Eyes
# Here we take the low-resolution unity eyes as the $A$ input to the models

# In[ ]:


# load the data file and extract dimensions
with h5py.File(os.path.join(data_dir,'gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    assert 'look_vec' in t_file, "Look vector is missing"
    look_vec = t_file['look_vec'][()]
    assert 'path' in t_file, "Paths are missing"
    print('Images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_width, img_height = ival.shape
    syn_image_stack = norm_stack(np.expand_dims(np.stack([a for a in t_file['image'].values()],0), -1))
    print(syn_image_stack.shape, 'loaded')
plt.matshow(montage2d(syn_image_stack[0:9, :, :, 0]), cmap = 'gray')


# ## High-res RGB images
# Here we take images from the [Helen-Eye dataset](https://www.kaggle.com/kmader/helen-eye-dataset) as the B-stack where we want to convert to.

# In[ ]:


# load the data file and extract dimensions
montage_rgb = lambda x: np.clip(0.5*np.stack([montage2d(x[..., i]) for i in range(x.shape[-1])], -1)+0.5, 0, 1) 
with h5py.File(os.path.join(helen_eye_dir,'eye_balls_rgb.h5'),'r') as t_file:
    real_image_stack = norm_stack(t_file['image'][()])
plt.imshow(montage_rgb(real_image_stack[0:16, :, :]))


# In[ ]:


from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
class loader_class():
    def __init__(self, a_stack, b_stack, goal_size=None):
        if goal_size is not None:
            a_stack = zoom(a_stack, (1, goal_size[0]/a_stack.shape[1], goal_size[1]/a_stack.shape[2], 1), order=0)
            b_stack = zoom(b_stack, (1, goal_size[0]/b_stack.shape[1], goal_size[1]/b_stack.shape[2], 1), order=0)
        self.a_stack = train_test_split(a_stack, test_size=0.25, random_state=2019)
        self.b_stack = train_test_split(b_stack, test_size=0.25, random_state=2019)
        self.n_batches = 0
    def load_batch(self, batch_size, repeats=5):
        train_a = self.a_stack[0]
        train_b = self.b_stack[0]
        for _ in range(repeats): # make sure we go through all of both datasets (-ish)
            a_idx = np.random.permutation(np.arange(train_a.shape[0]))
            b_idx = np.random.permutation(np.arange(train_b.shape[0]))
            seq_len = min(a_idx.shape[0], b_idx.shape[0])//batch_size*batch_size
            for i in range(0, seq_len, batch_size):
                self.n_batches+=1
                c_len = min(batch_size, seq_len-i)
                yield train_a[a_idx[i:i+c_len]], train_b[b_idx[i:i+c_len]]
    def load_data(self, domain="A", batch_size=1, is_testing=False):
        if domain=="A":
            train_x, test_x = self.a_stack
        elif domain=="B":
            train_x, test_x = self.b_stack
        else:
            raise ValueError("Unknown domain")
        if is_testing:
            train_x = test_x
        out_idx = np.random.choice(range(train_x.shape[0]), size=batch_size)
        
        return train_x[out_idx]
loader_obj = loader_class(a_stack=real_image_stack, b_stack=syn_image_stack, goal_size=(32, 64))
loader_obj.load_data(domain="A", batch_size=1, is_testing=True).shape


# In[ ]:


# sanity check on the tool
for _, (a, b) in zip(range(2), loader_obj.load_batch(8)):
    print(a.shape, b.shape)


# ## Preview Model Output

# In[ ]:


def sample_images(cyc_gan, data_loader, epoch, batch_i):
    plt.close('all')
    r, c = 2, 3
    np.random.seed(batch_i)
    imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Translate images to the other domain
    fake_B = cyc_gan.g_AB.predict(imgs_A)
    fake_A = cyc_gan.g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = cyc_gan.g_BA.predict(fake_B)
    reconstr_B = cyc_gan.g_AB.predict(fake_A)

    gen_imgs = [imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B]

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c, figsize=(10, 5))
    cnt = 0
    for i in range(r):
        for j in range(c):
            c_img = np.clip(0.5 * gen_imgs[cnt][0]+0.5, 0, 1)
            if c_img.shape[-1]==1:
                c_img = c_img[:, :, 0]
            axs[i,j].imshow(c_img, cmap='gray', vmin=0, vmax=1)
            axs[i, j].set_title('{} {}'.format(titles[j], 'A' if i==0 else 'B'))
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("{:03d}_{:03d}.png".format(epoch, batch_i))
sample_images(cg, loader_obj, 0, 0)


# In[ ]:


sample_images(cg, loader_obj, 0, 0) # show if there is some variation


# # Train Model

# In[ ]:


BATCH_SIZE = 256
EPOCHS = 30


# In[ ]:


start_time = datetime.datetime.now()

# Adversarial loss ground truths
valid = np.ones((BATCH_SIZE,) + cg.disc_patch)
fake = np.zeros((BATCH_SIZE,) + cg.disc_patch)

for epoch in tqdm_notebook(range(EPOCHS), desc='Epochs'):
    for batch_i, (imgs_A, imgs_B) in tqdm_notebook(enumerate(loader_obj.load_batch(BATCH_SIZE)), desc='Batch'):
        #  Train Discriminators

        # Translate images to opposite domain
        fake_B = cg.g_AB.predict(imgs_A)
        fake_A = cg.g_BA.predict(imgs_B)

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = cg.d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = cg.d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = cg.d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = cg.d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)
        
        #  Train Generators
        # Train the generators
        g_loss = cg.combined.train_on_batch([imgs_A, imgs_B],
                                                [valid, valid,
                                                imgs_A, imgs_B,
                                                imgs_A, imgs_B])

        elapsed_time = datetime.datetime.now() - start_time
    
    # Plot the progress at each epoch
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s "                                                             % ( epoch, EPOCHS,
                                                                batch_i, loader_obj.n_batches,
                                                                d_loss[0], 100*d_loss[1],
                                                                g_loss[0],
                                                                np.mean(g_loss[1:3]),
                                                                np.mean(g_loss[3:5]),
                                                                np.mean(g_loss[5:6]),
                                                                elapsed_time))
    
    sample_images(cg, loader_obj, epoch, 0)


# In[ ]:


sample_images(cg, loader_obj, EPOCHS, 1)


# In[ ]:


sample_images(cg, loader_obj, EPOCHS, 2)


# In[ ]:


sample_images(cg, loader_obj, EPOCHS, 3)


# In[ ]:


sample_images(cg, loader_obj, EPOCHS, 4)


# In[ ]:


sample_images(cg, loader_obj, EPOCHS, 5)


# In[ ]:


sample_images(cg, loader_obj, EPOCHS, 5)


# In[ ]:




