#!/usr/bin/env python
# coding: utf-8

# This notebook is my current attempt at creating and optimizing a DCGAN using Tensorflow and Keras. My methods are mostly based on Chad Malla's DCGAN [kernel](https://www.kaggle.com/cmalla94/dcgan-generating-dog-images-with-tensorflow) and the DCGAN hacks [post](https://www.kaggle.com/c/generative-dog-images/discussion/98595). Will provide more details once I actually generate some decent results.

# ## References

# [1]. My previous kernel on [EDA and image preprocessing](https://www.kaggle.com/jadeblue/dog-generator-starter-eda-preprocessing).
# 
# [2]. [Xml parsing and cropping to specified bounding box](https://www.kaggle.com/paulorzp/show-annotations-and-breeds).
# 
# [3]. [Image cropping method with interpolation](https://www.kaggle.com/amanooo/wgan-gp-keras).
# 
# [4]. [Another great Keras-based DCGAN approach](https://www.kaggle.com/cmalla94/dcgan-generating-dog-images-with-tensorflow).
# 
# [5]. [DCGAN hacks for improving your model performance](https://www.kaggle.com/c/generative-dog-images/discussion/98595).
# 
# [6]. [Tensorflow DCGAN tutorial](https://www.tensorflow.org/beta/tutorials/generative/dcgan).

# ## Importing libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt, zipfile
import os
import glob
import math
import random
import time
import datetime
from tqdm import tqdm, tqdm_notebook

import xml.etree.ElementTree as ET 

import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Reshape,Conv2DTranspose, Conv2D, Flatten, Dropout, Embedding
from tensorflow.keras.optimizers import Adam

#from IPython import display

# libraries for SpectralNorm
from tensorflow.keras import backend as K
from keras.engine import *
from keras.legacy import interfaces
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.utils import conv_utils

print(os.listdir("../input"))


# In[ ]:


tf.enable_eager_execution()


# ## Setting input variables

# In[ ]:


image_width = 64
image_height = 64
image_channels = 3
image_sample_size = 10000
image_output_dir = '../output_images/'
image_input_dir = '../input/all-dogs/all-dogs/'
image_ann_dir = "../input/annotation/Annotation/"


# ## Creating the image features

# In[ ]:


dog_breed_dict = {}
for annotation in os.listdir(image_ann_dir):
    annotations = annotation.split('-')
    dog_breed_dict[annotations[0]] = annotations[1]


# In[ ]:


def read_image(src):
    img = cv2.imread(src)
    if img is None:
        raise FileNotFoundError
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# #### Crop the images and apply scaling

# In[ ]:


def load_cropped_images(dog_breed_dict=dog_breed_dict, image_ann_dir=image_ann_dir, sample_size=25000, 
                        image_width=image_width, image_height=image_height, image_channels=image_channels):
    curIdx = 0
    breeds = []
    dog_images_np = np.zeros((sample_size,image_width,image_height,image_channels))
    for breed_folder in os.listdir(image_ann_dir):
        for dog_ann in tqdm(os.listdir(image_ann_dir + breed_folder)):
            try:
                img = read_image(os.path.join(image_input_dir, dog_ann + '.jpg'))
            except FileNotFoundError:
                continue
                
            tree = ET.parse(os.path.join(image_ann_dir + breed_folder, dog_ann))
            root = tree.getroot()
            
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                xmin = max(0, xmin - 4)        # 4 : margin
                xmax = min(width, xmax + 4)
                ymin = max(0, ymin - 4)
                ymax = min(height, ymax + 4)

                w = np.min((xmax - xmin, ymax - ymin))
                w = min(w, width, height)                     # available w

                if w > xmax - xmin:
                    xmin = min(max(0, xmin - int((w - (xmax - xmin))/2)), width - w)
                    xmax = xmin + w
                if w > ymax - ymin:
                    ymin = min(max(0, ymin - int((w - (ymax - ymin))/2)), height - w)
                    ymax = ymin + w
                
                img_cropped = img[ymin:ymin+w, xmin:xmin+w, :]      # [h,w,c]
                # Interpolation method
                if xmax - xmin > image_width:
                    interpolation = cv2.INTER_AREA          # shrink
                else:
                    interpolation = cv2.INTER_CUBIC         # expansion
                    
                img_cropped = cv2.resize(img_cropped, (image_width, image_height), 
                                         interpolation=interpolation)  # resize
                    
                dog_images_np[curIdx,:,:,:] = np.asarray(img_cropped)
                dog_breed_name = dog_breed_dict[dog_ann.split('_')[0]]
                breeds.append(dog_breed_name)
                curIdx += 1
                
    return dog_images_np, breeds


# In[ ]:


start_time = time.time()
dog_images_np, breeds = load_cropped_images(sample_size=22125)
est_time = round(time.time() - start_time)
print("Feature loading time: {}.".format(str(datetime.timedelta(seconds=est_time))))


# In[ ]:


print('Loaded features shape: ', dog_images_np.shape)
print('Loaded labels: ', len(breeds))


# In[ ]:


def plot_features(features, labels, image_width=image_width, image_height=image_height, 
                image_channels=image_channels,
                examples=25, disp_labels=True): 
  
    if not math.sqrt(examples).is_integer():
        print('Please select a valid number of examples.')
        return
    
    imgs = []
    classes = []
    for i in range(examples):
        rnd_idx = np.random.randint(0, len(labels))
        imgs.append(features[rnd_idx, :, :, :])
        classes.append(labels[rnd_idx])
    
    
    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)),figsize=(15,15),
    subplot_kw = {'xticks':[], 'yticks':[]},
    gridspec_kw = dict(hspace=0.3, wspace=0.01))
    
    for i, ax in enumerate(axes.flat):
        if disp_labels == True:
            ax.title.set_text(classes[i])
        ax.imshow(imgs[i])


# In[ ]:


print('Plotting cropped images by their specified coordinates..')
plot_features(dog_images_np / 255., breeds, examples=25, disp_labels=True)


# ### Normalize the pixel values of the images

# In[ ]:


dog_images_np = (dog_images_np - 127.5) / 127.5  # normalize the pixel range to [-1, 1] ((image - 127.5) / 127.5) or [0, 1] (image / 255.) alternatively


# In[ ]:


print('Plotting cropped images by their specified coordinates..')
plot_features(dog_images_np, breeds, examples=25, disp_labels=True)


# In[ ]:


print(np.max(dog_images_np[3,:,:,:]), np.min(dog_images_np[3,:,:,:]))


# ### Deprocessing back to the original values

# In[ ]:


plot_features((dog_images_np * 127.5 + 127.5) / 255., breeds, examples=25, disp_labels=True)


# ## Tensorflow-based preprocessing

# In[ ]:


print("Dog features shape:", dog_images_np.shape)


# In[ ]:


dog_features_tf = tf.cast(dog_images_np, 'float32')


# ### Set model hyperparameters

# In[ ]:


sample_size = 22125
batch_size = 128
weight_init_std = 0.02
weight_init_mean = 0.0
leaky_relu_slope = 0.2
downsize_factor = 3
dropout_rate = 0.3
scale_factor = 2 ** downsize_factor
BN_MOMENTUM = 0.1
BN_EPSILON  = 0.00002


# ## Create tensorflow-type dataset

# In[ ]:


dog_features_data = tf.data.Dataset.from_tensor_slices(dog_features_tf).shuffle(sample_size).batch(batch_size)


# In[ ]:


print(dog_features_data)


# ## Creating the generator

# In[ ]:


weight_initializer = tf.keras.initializers.RandomNormal(mean=weight_init_mean, stddev=weight_init_std)


# In[ ]:


class DenseSN(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output 
    
class ConvSN2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# In[ ]:


def transposed_conv(model, out_channels):
    model.add(Conv2DTranspose(out_channels, (4, 4), strides=(2, 2), padding='same', 
                              kernel_initializer=weight_initializer, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    return model


def convSN(model, out_channels, ksize, stride_size):
    model.add(ConvSN2D(out_channels, (ksize, ksize), strides=(stride_size, stride_size), padding='same',
                     kernel_initializer=weight_initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(Dropout(dropout_rate))
    return model

def conv(model, out_channels, ksize, stride_size):
    model.add(Conv2D(out_channels, (ksize, ksize), strides=(stride_size, stride_size), padding='same',
                     kernel_initializer=weight_initializer))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(Dropout(dropout_rate))
    return model


# #### Confirm that input dimensions are correct

# In[ ]:


print(image_height // scale_factor, image_width // scale_factor, 512)


# In[ ]:


def DogGenerator():
    model = Sequential()
    model.add(Dense(image_width // scale_factor * image_height // scale_factor * 512,
                    input_shape=(100,), kernel_initializer=weight_initializer, use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_relu_slope))
    model.add(Reshape((image_height // scale_factor, image_width // scale_factor, 512)))
    
    model = transposed_conv(model, 256)
    model = transposed_conv(model, 128)
    model = transposed_conv(model, 64)
    
    model.add(Dense(3, activation='tanh', kernel_initializer=weight_initializer, use_bias=False))
    return model


# In[ ]:


dog_generator = DogGenerator()
print(dog_generator.summary())


# In[ ]:


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape((n_samples, latent_dim))
    return x_input


# In[ ]:


# random noise vector
noise = tf.random.normal([1,100])
#sample = generate_latent_points(100, 50)
# run the generator model with the noise vector as input
generated_image = dog_generator(noise, training=False)
# display output
plt.imshow(generated_image[0, :, :, :])
print(generated_image.shape)
#print(sample.shape, sample.mean(), sample.std())
print(noise.shape, tf.math.reduce_mean(noise).numpy(), tf.math.reduce_std(noise).numpy())


# ## Creating the discriminator

# In[ ]:


def DogDiscriminator(spectral_normalization=True):
    model = Sequential()
    if spectral_normalization:
        model.add(ConvSN2D(64, (4, 4), strides=(2,2), padding='same',
                         input_shape=[image_height, image_width, image_channels], 
                         kernel_initializer=weight_initializer))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=leaky_relu_slope))
        model.add(Dropout(dropout_rate))

        model = convSN(model, 64, ksize=4, stride_size=2)
        model = convSN(model, 128, ksize=3, stride_size=1)
        model = convSN(model, 128, ksize=4, stride_size=2)
        model = convSN(model, 256, ksize=3, stride_size=1)
        model = convSN(model, 256, ksize=4, stride_size=2)
        model = convSN(model, 512, ksize=4, stride_size=2)

        model.add(Flatten())
        model.add(DenseSN(1, activation='sigmoid'))
    else:
        model.add(Conv2D(64, (4, 4), strides=(2,2), padding='same',
                         input_shape=[image_height, image_width, image_channels], 
                         kernel_initializer=weight_initializer))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=leaky_relu_slope))
        model.add(Dropout(dropout_rate))

        model = conv(model, 64, ksize=4, stride_size=2)
        model = conv(model, 128, ksize=3, stride_size=1)
        model = conv(model, 128, ksize=4, stride_size=2)
        model = conv(model, 256, ksize=3, stride_size=1)
        model = conv(model, 256, ksize=4, stride_size=2)
        model = conv(model, 512, ksize=4, stride_size=2)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
    return model


# In[ ]:


dog_discriminator = DogDiscriminator(spectral_normalization=True)
print(dog_discriminator.summary())


# In[ ]:


decision = dog_discriminator(generated_image)
print(decision)


# ### Provide label smoothing (Thanks to Chad Malla).

# In[ ]:


# Label smoothing -- technique from GAN hacks, instead of assigning 1/0 as class labels, we assign a random integer in range [0.7, 1.0] for positive class
# and [0.0, 0.3] for negative class

def smooth_positive_labels(y):
    return y - 0.3 + (np.random.random(y.shape) * 0.5)

def smooth_negative_labels(y):
    return y + np.random.random(y.shape) * 0.3


# ### Randomly flip labels to introduce more noise to the discriminator

# In[ ]:


# randomly flip some labels
def noisy_labels(y, p_flip):
    # determine the number of labels to flip
    n_select = int(p_flip * int(y.shape[0]))
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(int(y.shape[0]))], size=n_select)
    
    op_list = []
    # invert the labels in place
    #y_np[flip_ix] = 1 - y_np[flip_ix]
    for i in range(int(y.shape[0])):
        if i in flip_ix:
            op_list.append(tf.subtract(1, y[i]))
        else:
            op_list.append(y[i])
    
    outputs = tf.stack(op_list)
    return outputs


# In[ ]:


'''
# generate 'real' class labels (1)
n_samples = 1000
y = np.ones((n_samples, 1))
# flip labels with 5% probability
y = noisy_labels(y, 0.05)
# summarize labels
print(y.sum())
 
# generate 'fake' class labels (0)
y = np.zeros((n_samples, 1))
# flip labels with 5% probability
y = noisy_labels(y, 0.05)
# summarize labels
print(y.sum())
'''


# ### Optimizers and loss functions

# In[ ]:


generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator_optimizer = Adam(lr=0.0002, beta_1=0.5)
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


# In[ ]:


def discriminator_loss(real_output, fake_output, apply_label_smoothing=True, label_noise=True):
    if label_noise and apply_label_smoothing:
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)
        real_output_smooth = smooth_positive_labels(real_output_noise)
        fake_output_smooth = smooth_negative_labels(fake_output_noise)
        real_loss = cross_entropy(real_output_smooth, real_output)
        fake_loss = cross_entropy(fake_output_smooth, fake_output)
    elif label_noise and not apply_label_smoothing:
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)
        real_loss = cross_entropy(real_output_noise, real_output)
        fake_loss = cross_entropy(fake_output_noise, fake_output)
    elif apply_label_smoothing and not label_noise:
        real_output_smooth = smooth_positive_labels(tf.ones_like(real_output))
        fake_output_smooth = smooth_negative_labels(tf.zeros_like(fake_output))
        real_loss = cross_entropy(real_output_smooth, real_output)
        fake_loss = cross_entropy(fake_output_smooth, fake_output)
    else:
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# In[ ]:


def generator_loss(fake_output, apply_label_smoothing=True):
    if apply_label_smoothing:
        fake_output_smooth = smooth_negative_labels(tf.ones_like(fake_output))
        return cross_entropy(fake_output_smooth, fake_output)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# ### Define a checkpointer

# In[ ]:


checkpoint_dir = '/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=dog_generator,
                                 discriminator=dog_discriminator)


# ### Main training loop

# In[ ]:


EPOCHS = 250
noise_dim = 100
num_examples_to_generate = 8
seed = tf.random.normal([num_examples_to_generate, noise_dim])


# In[ ]:


def train_step(images, G_loss, D_loss):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = dog_generator(noise, training=True)
        
        real_output = dog_discriminator(images, training=True)
        fake_output = dog_discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output, apply_label_smoothing=False)
        disc_loss = discriminator_loss(real_output, fake_output, apply_label_smoothing=False, label_noise=False)

    G_loss.append(gen_loss.numpy())
    D_loss.append(disc_loss.numpy())
    
    gradients_of_generator = gen_tape.gradient(gen_loss, dog_generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, dog_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, dog_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, dog_discriminator.trainable_variables))


# In[ ]:


# function by Nanashi
def plot_loss(G_losses, D_losses, epoch):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss - EPOCH {}".format(epoch))
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[ ]:


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(10,10))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 2, i+1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.)
        plt.axis('off') 
    #plt.savefig('image_at_epoch_{}.png'.format(epoch))
    plt.show()


# In[ ]:


def generate_test_image(model, noise_dim=100):
    test_input = tf.random.normal([1, noise_dim])
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(5,5))
    plt.imshow((predictions[0, :, :, :] * 127.5 + 127.5) / 255.)
    plt.axis('off') 
    plt.show()


# In[ ]:


def train(dataset, epochs):
    G_loss = []
    D_loss = []
    for epoch in tqdm(range(epochs)):
        
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch, G_loss, D_loss)
         
        #display.clear_output(wait=True)
        if (epoch + 1) % 25 == 0:
            generate_and_save_images(dog_generator, epoch + 1, seed)
            plot_loss(G_loss, D_loss, epoch + 1)
        
        G_loss = []
        D_loss = []           

        print ('Epoch: {} computed for {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    #display.clear_output(wait=True)
    generate_and_save_images(dog_generator, epochs, seed)
    checkpoint.save(file_prefix = checkpoint_prefix)
    print('Final epoch.')


# In[ ]:


#%%time
train(dog_features_data, EPOCHS)


# ### Generate a test image

# In[ ]:


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


# In[ ]:


generate_test_image(dog_generator)


# ### Generate 10000 images for submission and save them to a zip file

# In[ ]:


# SAVE TO ZIP FILE NAMED IMAGES.ZIP
z = zipfile.PyZipFile('images.zip', mode='w')
for k in tqdm(range(image_sample_size)):
    generated_image = dog_generator(tf.random.normal([1, noise_dim]), training=False)
    f = str(k)+'.png'
    img = np.array(generated_image)
    img = (img[0, :, :, :] + 1.) / 2.
    img = Image.fromarray((255*img).astype('uint8').reshape((image_height,image_width,image_channels)))
    
    #cv2.imwrite(f, img)
    img.save(f,'PNG')
    z.write(f)
    os.remove(f)
    #if k % 1000==0: print(k)
z.close()

