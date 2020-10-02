#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os, os.path
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from xml.etree import ElementTree as ET

def parse_annotation(fname):
    objects = []
    for child in ET.parse(fname).findall("object"):
        dog = {}
        dog['name'] = child.find('name').text
        dog['pose'] = child.find('pose').text
        dog['difficult'] = int(child.find('difficult').text)
        dog['truncated'] = int(child.find('truncated').text)
        
        bbox = child.find('bndbox')
        dog['bbox'] = [
            int(bbox.find('xmin').text),
            int(bbox.find('ymin').text),
            int(bbox.find('xmax').text),
            int(bbox.find('ymax').text)
        ]
        objects.append(dog)
    return objects


# In[ ]:


IMAGE_DIR = '../input/all-dogs/all-dogs'
dog_imgs = pd.DataFrame(os.listdir(IMAGE_DIR), columns=['filename'])
dog_imgs['basename'] = dog_imgs['filename'].str.split('.').apply(lambda x: x[0])
dog_imgs[['class','id']] = dog_imgs['basename'].str.split("_",expand=True,)
dog_imgs = dog_imgs.set_index('basename').sort_index()

ANNOTATION_DIR = '../input/annotation/Annotation'
dog_breeds = pd.DataFrame(os.listdir(ANNOTATION_DIR), columns=['dirname'])
dog_breeds[['class', 'breedname']] = dog_breeds['dirname'].str.split("-",1,expand=True)
dog_breeds = dog_breeds.set_index('class').sort_index()

dog_imgs['annotation_filename'] = dog_imgs.apply(lambda x: os.path.join(ANNOTATION_DIR, dog_breeds.loc[x['class']]['dirname'], x.name), axis=1)
dog_imgs['objects'] = dog_imgs['annotation_filename'].apply(parse_annotation)

display(dog_imgs.head())
display(dog_breeds.head())


# In[ ]:


doggo = dog_imgs.sample(1).iloc[0]

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

import imgaug.augmenters as iaa

pil_im = Image.open(os.path.join(IMAGE_DIR, doggo['filename']))
im = np.asarray(pil_im)

fig,ax = plt.subplots(1)

ax.imshow(im)

h,w,c = im.shape
for dog in doggo['objects']:
    xmin, ymin, xmax, ymax = dog['bbox']
    print(h,w,":",xmin,ymin,xmax,ymax)
    bbox = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)

plt.show()

fig,ax = plt.subplots(1)

dog = doggo.objects[0]

h,w,c = im.shape
xmin, ymin, xmax, ymax = dog['bbox']

#im = im[ymin:ymax,xmin:xmax]
pil_crop = pil_im.crop((xmin, ymin, xmax, ymax)).resize((64, 64))
im2 = np.asarray(pil_crop)

ax.imshow(im2)

plt.show()


# In[ ]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image

import imgaug.augmenters as iaa
from tqdm import tqdm, tqdm_notebook

def get_truth_images():
    all_imgs = []
    
    for _,doggo in tqdm_notebook(dog_imgs.iterrows(), total=len(dog_imgs)):
        pil_im = Image.open(os.path.join(IMAGE_DIR, doggo['filename']))
        h,w,c = im.shape
        
        for dog in doggo['objects']:
            border = 4#int(min(h,w)*.1)
            
            xmin, ymin, xmax, ymax = dog['bbox']
            
            xmin = max(0, xmin-border)
            ymin = max(0, ymin-border)
            xmax = min(w, xmax+border)
            ymax = min(h, ymax+border)

            pil_crop = pil_im.crop((xmin, ymin, xmax, ymax)).resize((64, 64))
            all_imgs.append(np.asarray(pil_crop))

    return np.stack(all_imgs)

truth_imgs = get_truth_images()


# In[ ]:


truth_nrm_imgs = (truth_imgs-127.5)/127.5


# In[ ]:


from keras.optimizers import Adam
from keras import backend as K

# adapted from keras.optimizers.Adam
class AdamWithWeightnorm(Adam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations, K.floatx())))

        t = K.cast(self.iterations + 1, K.floatx())
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Adam
            ps = K.get_variable_shape(p)
            if len(ps)>1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(p, g)

                # Adam containers for the 'g' parameter
                V_scaler_shape = K.get_variable_shape(V_scaler)
                m_g = K.zeros(V_scaler_shape)
                v_g = K.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1. - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1. - self.beta_2) * K.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / (K.sqrt(v_g_t) + self.epsilon)
                self.updates.append(K.update(m_g, m_g_t))
                self.updates.append(K.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(grad_V)
                new_V_param = V - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                # if there are constraints we apply them to V, not W
                if getattr(p, 'constraint', None) is not None:
                    new_V_param = p.constraint(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(self.updates, new_V_param, new_g_param, p, V_scaler)

            else: # do optimization normally
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                new_p = p_t
                # apply constraints
                if getattr(p, 'constraint', None) is not None:
                    new_p = p.constraint(new_p)
                self.updates.append(K.update(p, new_p))
        return self.updates

import tensorflow as tf
    
def get_weightnorm_params_and_grads(p, g):
    ps = K.get_variable_shape(p)

    # construct weight scaler: V_scaler = g/||V||
    V_scaler_shape = (ps[-1],)  # assumes we're using tensorflow!
    V_scaler = K.ones(V_scaler_shape)  # init to ones, so effective parameters don't change

    # get V parameters = ||V||/g * W
    norm_axes = [i for i in range(len(ps) - 1)]
    V = p / tf.reshape(V_scaler, [1] * len(norm_axes) + [-1])

    # split V_scaler into ||V|| and g parameters
    V_norm = tf.sqrt(tf.reduce_sum(tf.square(V), norm_axes))
    g_param = V_scaler * V_norm

    # get grad in V,g parameters
    grad_g = tf.reduce_sum(g * V, norm_axes) / V_norm
    grad_V = tf.reshape(V_scaler, [1] * len(norm_axes) + [-1]) *              (g - tf.reshape(grad_g / V_norm, [1] * len(norm_axes) + [-1]) * V)

    return V, V_norm, V_scaler, g_param, grad_g, grad_V

def add_weightnorm_param_updates(updates, new_V_param, new_g_param, W, V_scaler):
    ps = K.get_variable_shape(new_V_param)
    norm_axes = [i for i in range(len(ps) - 1)]

    # update W and V_scaler
    new_V_norm = tf.sqrt(tf.reduce_sum(tf.square(new_V_param), norm_axes))
    new_V_scaler = new_g_param / new_V_norm
    new_W = tf.reshape(new_V_scaler, [1] * len(norm_axes) + [-1]) * new_V_param
    updates.append(K.update(W, new_W))
    updates.append(K.update(V_scaler, new_V_scaler))

# data based initialization for a given Keras model
def data_based_init(model, input):
    # input can be dict, numpy array, or list of numpy arrays
    if type(input) is dict:
        feed_dict = input
    elif type(input) is list:
        feed_dict = {tf_inp: np_inp for tf_inp,np_inp in zip(model.inputs,input)}
    else:
        feed_dict = {model.inputs[0]: input}

    # add learning phase if required
    if model.uses_learning_phase and K.learning_phase() not in feed_dict:
        feed_dict.update({K.learning_phase(): 1})

    # get all layer name, output, weight, bias tuples
    layer_output_weight_bias = []
    for l in model.layers:
        trainable_weights = l.trainable_weights
        if len(trainable_weights) == 2:
            W,b = trainable_weights
            assert(l.built)
            layer_output_weight_bias.append((l.name,l.get_output_at(0),W,b)) # if more than one node, only use the first

    # iterate over our list and do data dependent init
    sess = K.get_session()
    for l,o,W,b in layer_output_weight_bias:
        print('Performing data dependent initialization for layer ' + l)
        m,v = tf.nn.moments(o, [i for i in range(len(o.get_shape())-1)])
        s = tf.sqrt(v + 1e-10)
        updates = tf.group(W.assign(W/tf.reshape(s,[1]*(len(W.get_shape())-1)+[-1])), b.assign((b-m)/s))
        sess.run(updates, feed_dict)


# In[ ]:


from keras.models import Model, Sequential
from keras.layers import (
    Dense, Conv2D, Flatten, Concatenate, UpSampling2D,
    Dropout, LeakyReLU, ReLU, Reshape, Input, Conv2DTranspose
)
from keras.initializers import RandomNormal

from keras import backend as K
import tensorflow as tf

def make_discriminator_model(input_shape=(64, 64, 3)):
    init = RandomNormal(mean=0.0, stddev=0.02)
    
    model = Sequential()
    # 64x64 => in
    model.add(Conv2D(32, 
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=init,
                     input_shape=input_shape,))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    # out => 32x32
    
    # 32x32 => in
    model.add(Conv2D(64, 
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=init,))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    # out => 16x16
    
    # 16x16 => in
    model.add(Conv2D(128, 
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=init,))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    # out => 8x8
    
    # 8x8 => in
    model.add(Conv2D(256, 
                     kernel_size=4,
                     strides=2,
                     padding='same',
                     kernel_initializer=init,))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    # out => 4x4
    
    model.add(Flatten())
    model.add(Dense(1, 
                    activation='linear', 
                    kernel_initializer=init))
    
    return model


# In[ ]:


def make_generator_model(random_dim=128, start_shape=(4, 4, 64)):
    init = RandomNormal(mean=0.0, stddev=0.02)
    
    model = Sequential()
    
    a,b,c = start_shape; start_dim = a*b*c
    
    model.add(Dense(start_dim,
                    kernel_initializer=init,
                    input_dim=random_dim))
    model.add(Reshape(start_shape))
    
    #model.add(Conv2DTranspose(512, kernel_size=4, padding='same', 
    #                          kernel_initializer=init))
    
    # => 8x8
    model.add(UpSampling2D(interpolation='bilinear'))
    model.add(Conv2D(512, 
                     kernel_size=4,
                     padding='same',
                     kernel_initializer=init))
    model.add(ReLU())
    
    # => 16x16
    model.add(UpSampling2D(interpolation='bilinear'))
    model.add(Conv2D(256, 
                     kernel_size=4,
                     padding='same',
                     kernel_initializer=init))
    model.add(ReLU())
    
    # => 32x32
    #model.add(Conv2DTranspose(128,
    #                          kernel_size=3,
    #                          strides=2,
    #                          padding='same',
    #                          kernel_initializer=init))
    model.add(UpSampling2D(interpolation='bilinear'))
    model.add(Conv2D(128, 
                     kernel_size=4,
                     padding='same',
                     kernel_initializer=init))
    model.add(ReLU())
    
    # => 64x64
    #model.add(Conv2DTranspose(128,
    #                          kernel_size=3,
    #                          strides=2,
    #                          padding='same',
    #                          kernel_initializer=init))
    model.add(UpSampling2D(interpolation='bilinear'))
    model.add(Conv2D(64, 
                     kernel_size=4,
                     padding='same',
                     kernel_initializer=init))
    model.add(ReLU())
    
    model.add(Conv2D(3, 
                     kernel_size=3,
                     activation='tanh',
                     padding='same',
                     kernel_initializer=init))
    model.summary()
    
    return model


# In[ ]:


def make_gan_model(dis_model, gen_model, random_dim=128):
    dis_model.trainable = False
    gan_input = Input(shape=(random_dim,))
    gen_output = gen_model(gan_input)
    gan_output = dis_model(gen_output)
    
    gan_model = Model(inputs=gan_input, outputs=gan_output)
    
    return gan_model


# In[ ]:


def gen_input(random_dim, n_samples):
    noise = np.random.randn(random_dim * n_samples)
    noise = noise.reshape((n_samples, random_dim))
    
    return noise

def plot_gen_noise(gen_model, random_dim=128, examples=25, dim=(5,5)):
    gen_imgs = gen_model.predict(gen_input(128, 25))
    gen_imgs = ((gen_imgs + 1)*127.5).astype('uint8')
    
    plt.figure(figsize=(12,8))
    for i, img in enumerate(gen_imgs):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(img, interpolation='bilinear')
        plt.axis('off')
    plt.tight_layout()


# In[ ]:


RANDOM_DIM = 128
RAW_BATCH_SIZE = 32
DIS_TRAIN_RATIO = 2

MINI_BATCH_SIZE = RAW_BATCH_SIZE//DIS_TRAIN_RATIO

dis_model = make_discriminator_model()
gen_model = make_generator_model()

batch_count = truth_nrm_imgs.shape[0] // RAW_BATCH_SIZE

adam_nrm_op = AdamWithWeightnorm(lr=0.0002, beta_1=0.5, beta_2=0.999)

real_inp = Input(shape=truth_nrm_imgs.shape[1:])
nois_inp = Input(shape=(RANDOM_DIM,))
fake_inp = gen_model(nois_inp)

disc_r = dis_model(real_inp)
disc_f = dis_model(fake_inp)

# Relative GAN
def rel_dis_loss(_y_real, _y_pred):
    epsilon = K.epsilon()
    return -(
        K.mean(K.log(  K.sigmoid(disc_r - K.mean(disc_f, axis=0)) + epsilon), axis=0) +
        K.mean(K.log(1-K.sigmoid(disc_f - K.mean(disc_r, axis=0)) + epsilon), axis=0)
    )

def rel_gen_loss(_y_real, _y_pred):
    epsilon = K.epsilon()
    return -(
        K.mean(K.log(  K.sigmoid(disc_f - K.mean(disc_r, axis=0)) + epsilon), axis=0) +
        K.mean(K.log(1-K.sigmoid(disc_r - K.mean(disc_f, axis=0)) + epsilon), axis=0)
    )

# RaLSGAN
REAL_LABEL = 0.8
def rals_dis_loss(_y_real, _y_pred):
    return K.mean(
        K.pow(disc_r - K.mean(disc_f, axis=0) - REAL_LABEL, 2) +
        K.pow(disc_f - K.mean(disc_r, axis=0) + REAL_LABEL, 2)
    )

def rals_gen_loss(_y_real, _y_pred):
    return K.mean(
        K.pow(disc_r - K.mean(disc_f, axis=0) + REAL_LABEL, 2) +
        K.pow(disc_f - K.mean(disc_r, axis=0) - REAL_LABEL, 2)
    )

gen_train = Model([nois_inp, real_inp], [disc_r, disc_f])
dis_model.trainable = False
gen_train.compile(adam_nrm_op, loss=[rals_gen_loss, None])
gen_train.summary()

dis_train = Model([nois_inp, real_inp], [disc_r, disc_f])
gen_model.trainable = False
dis_model.trainable = True
dis_train.compile(adam_nrm_op, loss=[rals_dis_loss, None])
dis_train.summary()

gen_loss = []
dis_loss = []

dummy_y = np.zeros((RAW_BATCH_SIZE, 1), dtype=np.float32)
dummy_mini_y = np.zeros((MINI_BATCH_SIZE, 1), dtype=np.float32)


# In[ ]:


import imgaug.augmenters as iaa

aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-8,8)),
    iaa.Fliplr(0.5),
], random_order=True)

def augment(images):
    return aug_seq.augment_images(images=images)


# In[ ]:


for epoch_i in range(300):
    epoch_idx = np.arange(truth_nrm_imgs.shape[0])
    np.random.shuffle(epoch_idx)

    for batch_i in tqdm_notebook(range(batch_count)):

        # train the discriminator
        dis_model.trainable = True
        gen_model.trainable = False
        
        loss = 0
        for mini_j in range(DIS_TRAIN_RATIO):
            mini_oset = batch_i*DIS_TRAIN_RATIO + mini_j
            
            z_fake = gen_input(RANDOM_DIM, MINI_BATCH_SIZE)
            X_real = augment(truth_nrm_imgs[epoch_idx[
                mini_oset*MINI_BATCH_SIZE:(mini_oset+1)*MINI_BATCH_SIZE]])

            loss += dis_train.train_on_batch([z_fake, X_real], dummy_mini_y)[0]
        dis_loss.append(loss/DIS_TRAIN_RATIO)

        dis_model.trainable = False
        gen_model.trainable = True
            
        z_fake = gen_input(RANDOM_DIM, RAW_BATCH_SIZE)
        X_real = augment(truth_nrm_imgs[epoch_idx[batch_i*RAW_BATCH_SIZE:(batch_i+1)*RAW_BATCH_SIZE]])
        
        gen_loss.append(gen_train.train_on_batch([z_fake, X_real], dummy_y)[0])

    if epoch_i%5 == 0:
        plot_gen_noise(gen_model)
        plt.suptitle('Epoch {}'.format(epoch_i+1), x=0.5, y=1.0)
        plt.savefig('dog_at_epoch_{}.png'.format(epoch_i+1))
        plt.show()


# In[ ]:


plt.plot(gen_loss, label='Generator loss')
plt.plot(dis_loss, label="Discriminator loss")
plt.legend()
plt.ylim([0, 15])
plt.show()


# In[ ]:


plot_gen_noise(gen_model)
plt.show()


# In[ ]:


import zipfile
from PIL import Image

N_OUTPUT = 10000
z = zipfile.PyZipFile('images.zip', mode='w')
generated_images = gen_model.predict(gen_input(RANDOM_DIM, N_OUTPUT))

for k, img_arr in tqdm_notebook(enumerate(generated_images), total=N_OUTPUT):
    image = Image.fromarray(((img_arr+1)*127.5).astype('uint8'))
    
    fname = str(k)+'.png'
    image.save(fname, 'PNG')
    z.write(fname)
    os.remove(fname)
z.close()


# In[ ]:




