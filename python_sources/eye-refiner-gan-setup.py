#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook implements a simpler version of the model discussed in [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/abs/1612.07828). 
# ### The initial focus is to 
# - load the datasets correctly
# - create the refiner and discriminator models
# - use data augmentation on the real and fake images
# - train for a few epochs
# - use the simpler training approach
# 
# ### Training
# - Unity Images - $x$
# - Real images $y$
# - Refiner Model $\mathcal{R}$
# - Discriminator Model $\mathcal{D}$
# ### Training Loop (one epoch)
# 1. Improve Generator: minimize $-\log(\mathcal{D}(\mathcal{R}(x)))+||\mathcal{R}(x)-x||_1$ by updating parameters in $\mathcal{R}$
# 1. Improve Discriminator: maximize $-\log(\mathcal{D}(y)+\log(1-\mathcal{D}(\mathcal{R}(x)))$ by updating parameters in $\mathcal{D}$

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import keras
import keras.backend as K
from skimage.util.montage import montage2d
data_dir = os.path.join('..', 'input')
norm_stack = lambda x: np.clip((x-127.0)/127.0, -1, 1)
def norm_stack(x):
    # calculate statistics on first 20 points
    mean = np.mean(x[:20])
    std = np.std(x[:20])
    return (1.0*x-mean)/(2*std)


# # Load Real Data

# In[ ]:


# load the data file and extract dimensions
with h5py.File(os.path.join(data_dir,'real_gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    print('Images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_width, img_height = ival.shape
    real_image_stack = norm_stack(np.expand_dims(np.stack([a for a in t_file['image'].values()],0), -1))
    print(real_image_stack.shape, 'loaded')
plt.matshow(montage2d(real_image_stack[0:9, :, :, 0]), cmap = 'gray')


# # Load Synthetic Data
# Generated using Unity and UnityEyes Tools

# In[ ]:


# load the data file and extract dimensions
with h5py.File(os.path.join(data_dir,'gaze.h5'),'r') as t_file:
    print(list(t_file.keys()))
    assert 'image' in t_file, "Images are missing"
    assert 'look_vec' in t_file, "Look vector is missing"
    look_vec = t_file['look_vec'].value
    assert 'path' in t_file, "Paths are missing"
    print('Images found:',len(t_file['image']))
    for _, (ikey, ival) in zip(range(1), t_file['image'].items()):
        print('image',ikey,'shape:',ival.shape)
        img_width, img_height = ival.shape
    syn_image_stack = norm_stack(np.expand_dims(np.stack([a for a in t_file['image'].values()],0), -1))
    print(syn_image_stack.shape, 'loaded')
plt.matshow(montage2d(syn_image_stack[0:9, :, :, 0]), cmap = 'gray')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
ax1.hist(syn_image_stack[::10].ravel());
ax1.set_title('Synthetic Data')
ax2.hist(real_image_stack[::10].ravel());
ax2.set_title('Real Data')


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, test_X = train_test_split(syn_image_stack, 
                                   test_size = 0.25, 
                                   random_state = 2018)
train_Y, test_Y = train_test_split(real_image_stack,
                                   test_size = 0.25,
                                   random_state = 2018)
print('Fake Images', train_X.shape, test_X.shape, train_X.max(), train_X.min(), train_X.mean(), train_X.std())
print('Real Images', train_Y.shape, test_Y.shape, train_Y.max(), train_Y.min(), train_Y.mean(), train_Y.std())


# # Build Models

# In[ ]:


from keras.layers import Input, concatenate, Conv2D, MaxPool2D, UpSampling2D, Flatten, Dense, Dropout, GaussianNoise, add, ZeroPadding2D, Cropping2D, Conv2DTranspose
from keras import models, layers
from collections import defaultdict
gauss_noise_level = 1e-3
leakiness = 0.1
def make_gen(depth=16, layer_count=2, use_dilation=False, use_add=False):
    in_lay = Input(shape = (train_X.shape[1:4]), name = 'Generator_Input')
    padding_size = ((2, 3), (2,3))
    padding_size = ((6, 7), (4, 5))
    gn = ZeroPadding2D(padding_size)(in_lay)
    gn = GaussianNoise(gauss_noise_level)(gn)
    
    c1 = Conv2D(depth, (3,3), padding = 'same')(gn)
    out_layers = []
    # dilation
    if use_dilation:
        for i in range(layer_count):
            out_layers += [Conv2D(depth, (3,3), padding = 'same', dilation_rate=(2**i, 2**i))(c1)]
            out_layers += [Conv2D(depth, (1,3), padding = 'same', dilation_rate=(1, 2**i))(c1)]
        c2 = concatenate(out_layers)
    else:
        layer_db = defaultdict(lambda : [])
        x = c1
        layer_db[c1._keras_shape[1:3]] += [c1]
        for i in range(layer_count):
            x = Conv2D(depth*2**i, (3,3), padding = 'same', activation='linear')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(leakiness)(x)
            x = MaxPool2D((2, 2))(x)
            layer_db[x._keras_shape[1:3]] += [x]
        for idx, i in enumerate(reversed(range(layer_count))):
            if idx>0:
                x = Conv2D(depth*2**i, (1,1), padding = 'same', activation='linear')(x)
                x = layers.BatchNormalization()(x)
                x = layers.LeakyReLU(leakiness)(x)
            x = Conv2DTranspose(depth, (2, 2), strides = (2,2), padding = 'same')(x)
            x = concatenate([x] + layer_db.get(x._keras_shape[1:3]))
        c2 = x
    
    if use_add:
        c_out = Conv2D(1, (1,1), padding = 'same', activation = 'tanh')(c2)
        c_out = add([gn, c_out])
    else:
        c_out = Conv2D(1, (1,1), padding = 'same', activation = 'tanh')(c2)
    c_out = Cropping2D(padding_size)(c_out)
    return models.Model(inputs = [in_lay], outputs = [c_out], name = 'Generator')

def make_disc(depth=4, layer_count=3):
    in_lay = Input(shape = (train_X.shape[1:4]), name = 'Disc_Input')
    gn = GaussianNoise(gauss_noise_level)(in_lay)
    x = Conv2D(depth, (3,3), padding = 'valid', activation='linear')(gn)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(leakiness)(x)
    for i in range(layer_count):
        x = Conv2D(depth*2**i, (3,3), strides=(1, 1), padding = 'same', activation='linear')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(leakiness)(x)
        x = Conv2D(depth*2**i, (3,3), strides=(2, 2), padding = 'same', activation='linear')(x)
        x = layers.LeakyReLU(leakiness)(x)
    
    c_out = layers.concatenate([layers.GlobalMaxPool2D()(x), layers.GlobalAvgPool2D()(x)])
    c_out = Dropout(0.5)(c_out)
    c_out = Dense(2, activation = 'softmax')(c_out)
    return models.Model(inputs = [in_lay], outputs = [c_out], name = 'Discriminator')


# In[ ]:


simple_gen = make_gen(32, layer_count=3)
simple_disc = make_disc(32)
def make_full(gen_mod, disc_model):
    raw_img_in = Input(shape = (train_X.shape[1:4]), name = 'Image_In')
    ref_img_out = simple_gen(raw_img_in)
    ref_disc_score = simple_disc(ref_img_out)
    return models.Model(inputs=[raw_img_in], outputs=[ref_disc_score, ref_img_out])
full_gen_model = make_full(simple_gen, simple_disc)


# In[ ]:


from IPython.display import SVG, Image
from keras.utils.vis_utils import model_to_dot
d = model_to_dot(simple_gen, show_shapes=True)
d.set_rankdir('UD')
#SVG(d.create_svg())
Image(d.create_png())


# In[ ]:


# show the discriminator
Image(model_to_dot(simple_disc, show_shapes=True).create_png())


# In[ ]:


from keras.optimizers import Adam
GLOBAL_LR = 1e-3
def compile_generator(lr = 4e-3): 
    simple_disc.trainable = False
    full_gen_model.compile(optimizer=Adam(lr=lr), 
                           loss = ['categorical_crossentropy', 'mean_absolute_error'], 
                           loss_weights = [1.0, 0.3],
                           metrics = ['accuracy'])

def compile_discriminator(lr = 1e-4): 
    simple_disc.trainable = True
    simple_disc.compile(optimizer=Adam(lr=lr), 
                           loss = 'categorical_crossentropy', 
                           metrics = ['accuracy'])


# In[ ]:


compile_generator(GLOBAL_LR)
assert all([x in simple_gen.trainable_weights 
            for x in full_gen_model.trainable_weights]), "Only generator should be trainable"
full_gen_model.summary()


# In[ ]:


compile_discriminator(GLOBAL_LR)
simple_disc.summary()


# In[ ]:


fake_score, fake_images = full_gen_model.predict(train_X[0:2])
print(fake_score)
plt.imshow(fake_images[0, :, :, 0], cmap='gray')


# # Prepare Training Data

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 5, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.8, 1.2],  
                  horizontal_flip = True, 
                  vertical_flip = False,
                  fill_mode = 'reflect',
               data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
def make_train_gen_batch(in_X, batch_size = 512):
    # improve generator
    for x in image_gen.flow(in_X, batch_size=batch_size):
        out_vec = np.zeros((x.shape[0], 2))
        out_vec[:, 1] = 1.0
        yield x, [out_vec, x]


# In[ ]:


gen_train = make_train_gen_batch(train_X)
gen_valid = make_train_gen_batch(test_X)
a, (b, c) = next(gen_train)
print(a.shape, b.shape, c.shape)
fig, (ax1) = plt.subplots(1, 1, figsize = (20, 10))
ax1.imshow(montage2d(a[:, :, :, 0]), cmap = 'bone')
ax1.set_title('Synth Images')


# In[ ]:


def show_status(seed = None, img_cnt = 9):
    if seed is not None:
        np.random.seed(seed)
    syn_block = np.random.permutation(syn_image_stack)[0:img_cnt]
    real_block = np.random.permutation(real_image_stack)[0:img_cnt]
    bins = np.linspace(-1, 1, 30)
    fig, ((ax1, ax2, ax3), (ax1h, ax2h, ax3h))  = plt.subplots(2, 3, figsize = (24, 12))
    ax1.imshow(montage2d(syn_block[:, :, :, 0]), cmap = 'gray')
    ax1h.hist(syn_block[:, :, :, 0].flatten(), bins)
    ax1.set_title('Simulated Images\nReal: %2.2f%%' % (np.mean(simple_disc.predict(syn_block)[:, 1])*100))
    gen_stack = simple_gen.predict(syn_block)
    ax2.imshow(montage2d(gen_stack[: , :, :, 0]), cmap = 'gray')
    ax2h.hist(gen_stack[:, :, :, 0].flatten(), bins)
    ax2.set_title('Generated Images\nReal: %2.2f%%' % (np.mean(simple_disc.predict(gen_stack)[:, 1])*100))
    realness = np.mean(simple_disc.predict(syn_block)[:, 1])
    ax3.imshow(montage2d(real_block[:, :, :, 0]), cmap = 'gray')
    ax3h.hist(real_block[:, :, :, 0].flatten(), bins)
    ax3.set_title('Real Images\nReal: %2.2f%%' % (np.mean(simple_disc.predict(real_block)[:, 1])*100))
    return fig
show_status();


# In[ ]:


def make_train_disc_batch(in_fake, in_real, batch_size = 256, refine_images=True):
    """we create batches consisting of a 50/50 split between
    fake and real images. The fake images are processed using the refiner (refine_images=True), but
    in future we plan to provide fake images from many different generations of
    the generator model to 'stabilize training'  """
    while True:
        real_img = image_gen.flow(in_real, batch_size=batch_size)
        fake_img = image_gen.flow(in_fake, batch_size=batch_size)
        for (c_real, c_fake) in zip(real_img, fake_img):
            real_cat = np.zeros((c_real.shape[0], 2))
            real_cat[:, 1] = 1.0 # real
            refined_cat = np.zeros((c_fake.shape[0], 2))
            refined_cat[:, 0] = 1.0 # learn that they are fake

            if refine_images:
                c_refined = simple_gen.predict(c_fake)
            else:
                c_fake = c_fake
            yield np.concatenate([c_real, c_refined], 0), np.concatenate([real_cat, refined_cat])
disc_train = make_train_disc_batch(train_X, train_Y)
disc_valid = make_train_disc_batch(test_X, test_Y)


# In[ ]:


compile_discriminator(GLOBAL_LR)
print('Improving Discriminator')
simple_disc.fit_generator(disc_train, steps_per_epoch=100)


# In[ ]:


compile_generator(GLOBAL_LR)
print('Improving Generator')
full_gen_model.fit_generator(gen_train,
                             steps_per_epoch=200)


# In[ ]:





# In[ ]:


show_status(2002, 25).savefig('pretraining_image_gen.png', dpi = 300)


# # Big Training
# Here we run a number of loops 
# - improve the generator
# - improve the discriminator
# - decrease the learning rate of both
# - show results on fixed images
# - repeat

# In[ ]:


from IPython.display import clear_output, display
t_steps = 200
v_steps = 0
epochs = 20
from keras.callbacks import EarlyStopping
es_callback = lambda : EarlyStopping(monitor="val_loss", mode="min", patience=1)
for i in range(epochs):
    cur_lr = GLOBAL_LR*(0.8**(i))
    print('Improving Generator ({:2.2g})'.format(cur_lr))
    compile_generator(cur_lr)
    if v_steps>0:
        v_args = dict(validation_data=gen_valid, validation_steps=v_steps)
    else:
        v_args = {}
    full_gen_model.fit_generator(gen_train, steps_per_epoch=t_steps,**v_args)
    
    plt.close('all')
    clear_output()
    display(show_status(2018, 4))
    
    # we might be required to precompute images at some point here
    disc_train = make_train_disc_batch(train_X, train_Y)
    disc_valid = make_train_disc_batch(test_X, test_Y)
    compile_discriminator(cur_lr)
    print('Improving Discriminator')
    if v_steps>0:
        v_args = dict(validation_data=disc_valid, validation_steps=v_steps)
    else:
        v_args = {}
    simple_disc.fit_generator(disc_train, steps_per_epoch=t_steps, **v_args)
    display(show_status(2018, 4))


# In[ ]:


show_status(2002, 25).savefig('image_gen.png', dpi = 300)


# In[ ]:


show_status(2003, 25);

