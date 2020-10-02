#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd /kaggle/input/anime-sketch-colorization-pair/data/


# In[ ]:


from keras.preprocessing.image import load_img, img_to_array, array_to_img
import numpy as np
import matplotlib.pyplot as plt
import os


# In[ ]:


def load_imgs(dir):
  src_images, tar_images = [], []
  for filename in os.listdir(dir)[:3000]:
    img = load_img(dir + filename, target_size = (256,512,3))
    pixels = img_to_array(img)
    p1 = (pixels[:,:256] - 127.5) / 127.5
    src_images.append(p1)
    p2 = (pixels[:,256:] - 127.5) / 127.5
    tar_images.append(p2)
  return [np.asarray(src_images), np.asarray(tar_images)]
  


# In[ ]:


data = load_imgs("train/")


# In[ ]:


from keras.initializers import RandomNormal
from keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, BatchNormalization, LeakyReLU, Activation, Input, Concatenate
from keras.models import Model
import keras


# In[ ]:


def discriminator(inp_shape = (256, 256,3)):

  in_src_img = Input(inp_shape)
  in_tar_img = Input(inp_shape)
  init = RandomNormal(0.02)
  merge = Concatenate()([in_src_img, in_tar_img])

  g = Conv2D(64, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(merge)
  g = LeakyReLU(0.2)(g)

  g = Conv2D(128, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = LeakyReLU(0.2)(g)

  g = Conv2D(256, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = LeakyReLU(0.2)(g)

  g = Conv2D(512, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = LeakyReLU(0.2)(g)

  g = Conv2D(512, (4,4),  padding = "same", kernel_initializer=init)(g)
  g = BatchNormalization()(g)
  g = LeakyReLU(0.2)(g)

  g = Conv2D(1, (4,4), padding = "same", kernel_initializer=init)(g)
  patch_out = Activation("sigmoid")(g)

  model = Model([in_src_img, in_tar_img], patch_out)
  model.compile(optimizer = keras.optimizers.Adam(0.0002, 0.5), loss = "binary_crossentropy", loss_weights = [0.5])
  return model


# In[ ]:


d_model = discriminator()


# In[ ]:


def encoder_unit(in_layer, n_filters, batch_norm = True):
  init = RandomNormal(0.02)
  g = Conv2D(n_filters, (4,4), strides = (2,2), kernel_initializer=init, padding = "same")(in_layer)
  if batch_norm:
    g = BatchNormalization()(g, training = True)
  g = LeakyReLU(0.2)(g)
  return g


# In[ ]:


def decoder_unit(in_layer, merge_layer, n_filters, dropout = True):
  init = RandomNormal(0.02)
  g = Conv2DTranspose(n_filters, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(in_layer)
  g = BatchNormalization()(g, training = True)
  if dropout:
    g = Dropout(0.5)(g)
  g = Concatenate()([g, merge_layer])
  g = Activation("relu")(g)

  return g


# In[ ]:


def generator(inp_shape = (256,256,3)):
  init = RandomNormal(0.02)
  in_img = Input(inp_shape)
  g1 = encoder_unit(in_img, 64, batch_norm = False)
  g2 = encoder_unit(g1, 128)
  g3 = encoder_unit(g2, 256) 
  g4 = encoder_unit(g3, 512)
  g5 = encoder_unit(g4, 512)
  g6 = encoder_unit(g5, 512)
  g7 = encoder_unit(g6, 512)

  bottleneck = Conv2D(512, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(g7)
  bottleneck = Activation("relu")(bottleneck)

  e1 = decoder_unit(bottleneck, g7, 512)
  e2 = decoder_unit(e1, g6, 512)
  e3 = decoder_unit(e2, g5, 512)
  e4 = decoder_unit(e3, g4, 512, dropout = False)
  e5 = decoder_unit(e4, g3, 256, dropout = False)
  e6 = decoder_unit(e5, g2, 128, dropout = False)
  e7 = decoder_unit(e6, g1, 64, dropout = False)

  patch = Conv2DTranspose(3, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(e7)
  patch_out = Activation("tanh")(patch)

  model = Model(in_img, patch_out)
  return model


# In[ ]:


g_model = generator()


# In[ ]:


g_model.summary()


# In[ ]:


def gan(d_model, g_model, inp_shape = (256,256,3)):
  d_model.trainable = False
  in_img = Input(inp_shape)
  g_out = g_model(in_img)
  d_out = d_model([in_img, g_out])
  model = Model(in_img, [d_out, g_out])
  model.compile(keras.optimizers.Adam(0.0002, 0.5), loss = ['binary_crossentropy', "mae"], loss_weights = [1,100])
  return model


# In[ ]:


gan_model = gan(d_model, g_model)


# In[ ]:


gan_model.summary()


# In[ ]:


def generate_real_samples(data, batch_size, patch_size):
  src_imgs = data[0]
  tar_imgs = data[1]
  ind = np.random.randint(0, src_imgs.shape[0], batch_size)
  X_real, X_fake = src_imgs[ind], tar_imgs[ind]
  y = np.ones((batch_size, patch_size, patch_size, 1))
  return [X_real, X_fake], y


# In[ ]:


o = generate_real_samples(data, 2, 70)


# In[ ]:


o[0][0].shape


# In[ ]:


def generate_fake_samples(g_model,samples, patch_size):
  pred = g_model.predict(samples)
  y = np.zeros((len(pred), patch_size, patch_size, 1))
  return pred, y


# In[ ]:


def summarize_performance(g_model, dataset,step, n_samples = 3):
    [X_real_src, X_real_tar], y_real = generate_real_samples(dataset, 3,1)
    X_fake_tar, _ = generate_fake_samples(g_model, X_real_src, 1)
    X_real_src = (X_real_src + 1) / 2.0
    X_real_tar = (X_real_tar + 1) / 2.0
    X_fake_tar = (X_fake_tar + 1) / 2.0
	# plot real source images
    plt.figure(figsize = (12,8))
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_real_src[i])
	# plot generated target image
  
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_fake_tar[i])
	# plot real target image
#     for i in range(n_samples):
#         plt.subplot(3, n_samples, 1 + n_samples*2 + i)
#         plt.axis('off')
#         plt.imshow(X_real_tar[i])
        # save plot to file
    plt.plot()
#     filename1 = 'images/plot_%06d.png' % (step+1)
#     plt.savefig(filename1)
#     plt.close()
#     #save the generator model
#     filename2 = 'models/model_%06d.h5' % (step+1)
#     g_model.save(filename2)


# In[ ]:


cd 


# In[ ]:


cd /kaggle/working


# In[ ]:


get_ipython().system('rm -rf images')


# In[ ]:


get_ipython().system('rm -rf models')


# In[ ]:



mkdir images


# In[ ]:


mkdir models


# In[ ]:


def train(d_model, g_model, gan_model, dataset, epochs = 3, batch_size = 1):
  steps = dataset[0].shape[0] * epochs
  patch_size = d_model.output_shape[1]
  for i in range(steps):
    [X_real_src, X_real_tar], y_real = generate_real_samples(dataset, batch_size, patch_size)
    X_fake, y_fake = generate_fake_samples(g_model, X_real_src, patch_size)
    d_loss1 = d_model.train_on_batch([X_real_src, X_real_tar], y_real)
    d_loss2 = d_model.train_on_batch([X_real_src, X_fake], y_fake)
    g_loss, _,_ = gan_model.train_on_batch(X_real_src, [y_real, X_real_tar])
    if((i+1)%500 == 0):
        print(">%d  d1 = [%.3f]  d2 = [%0.3f]  g = [%.3f]" % (i+1, d_loss1, d_loss2, g_loss))
    if((i+1)%7000 == 0):
        summarize_performance(g_model, dataset, i + 1)


# In[ ]:


train(d_model, g_model, gan_model, data)


# In[ ]:


summarize_performance(g_model, data, 9999)


# In[ ]:




