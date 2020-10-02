#!/usr/bin/env python
# coding: utf-8

# Imagine if we had access to the true data distribution $P_{data}(x)$ we could sample from that distribution in order to generate new samples, however there is no direct way to do this as typically this distribution is complex and high-dimensional. What if we could instead sample from a random noise (e.g. Normal distribution) and then learn to transform that to $P_{data}(x)$. Neural networks are a prime candidate to capture functions with high complexity and we can use to to capture this transformation. This is exactly what the do. They train the transformer network or Generator along with another network, called the Discriminator, in a game theoretic way. Going back to our image generation example:
# 
# The Generator network ($G$), tries to fool the discriminator in thinking that the generated images are real,meaning that they are taken from $P_{data}$, and
# The Discriminator network ($D$), tries to differentiate between real ($x\sim P_{data}$) and fake images.
# 
# Random noise is fed into the Generator that transforms it into a "fake image". The Discriminator is fed both from the training set images ($p_{data}(x)$) and the fake images coming from the Generator and it has to tell them apart. The idea behind GAN, is to train both of these networks alternatively to do the best they can in generating and discriminating images. The intuition is that by improving one of these networks, in this game theoretic manner, the other network has to do a better job to win the game, and that in turn improves its performance and this loop continues.

# <img src="https://4.bp.blogspot.com/-YTGLQjhch-Q/Wz475NU2TSI/AAAAAAAAsu4/zaC_wfZBX80dePLflgQUaAaxE72od3VCgCEwYBhgL/s1600/Figura_2.png" height="500" width="500">

# <pre><b>Credits to Chris Deotte's Kernel</b>
# https://www.kaggle.com/cdeotte/dog-memorizer-gan</pre>
# 
# <pre><b>Credits to Robin Smits Kernel</b>
# https://www.kaggle.com/rsmits/keras-dcgan-with-weight-normalization</pre>

# # Import Libraries

# In[ ]:


import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import zipfile
import tensorflow as tf
import xml.etree.ElementTree as ET
from tqdm import tqdm
from keras.models import *
from keras.layers.core import *
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from PIL import Image


# # Constants and Directories

# In[ ]:


NOISE_SIZE = 64
LR_D = 0.0001
LR_G = 0.0002
#BATCH_SIZE = 128 # 128 is for final
BATCH_SIZE = 64 # pre-final test only
EPOCHS = 650 # For better results increase this value 
BETA1 = 0.5
SEED = 4250
random_dim = 128
img_rows = 64
img_cols = 64
channels = 3
SEED = 4250
np.random.seed(SEED)
random_dim = 128


# In[ ]:


# Constants and Directories

ROOT_DIR = '../input/'
IMAGES_DIR = ROOT_DIR + 'all-dogs/all-dogs/'
BREEDS_DIR = ROOT_DIR + 'annotation/Annotation/'

# File Lists
IMAGES = os.listdir(IMAGES_DIR)
BREEDS = os.listdir(BREEDS_DIR) 

# Summary
print('Total Images: {}'.format(len(IMAGES)))
print('Total Annotations: {}'.format(len(BREEDS)))


# # Load and Process Images

# In[ ]:


def load_images():
    # Place holder for output 
    all_images = np.zeros((22250, 64, 64, 3))
    
    # Index
    index = 0
    
    for breed in BREEDS:
        for dog in os.listdir(BREEDS_DIR + breed):
            try: img = Image.open(IMAGES_DIR + dog + '.jpg') 
            except: continue  
                
            tree = ET.parse(BREEDS_DIR + breed + '/' + dog)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bndbox = o.find('bndbox') 
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Determine each side
                xdelta = xmax - xmin
                ydelta = ymax - ymin
                
                # Take the mean of the sides
                #w = int((xdelta + ydelta) / 2)
                
                # Filter out images where bounding box is below 64 pixels.
                # This filters out a couple of 100 images but prevents using low resolution images.
                if xdelta >= 64 and ydelta >= 64:
                    img2 = img.crop((xmin, ymin, xmax, ymax))
                    img2 = img2.resize((64, 64), Image.ANTIALIAS)
                    image = np.asarray(img2)
                    
                    #    # Normalize to range[-1, 1]
                    all_images[index,:] = (image.astype(np.float32) - 127.5)/127.5
                    
                    index += 1
        
                # Plot Status
                if index % 1000 == 0:
                    print('Processed Images: {}'.format(index))

    print('Total Processed Images: {}'.format(index))

    return all_images


# # Weight Normalization
# 
# Using the class AdamWithWeightnorm from [github repository](https://github.com/krasserm/weightnorm/tree/master/keras_2).

# In[ ]:


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


# # Generator

# In[ ]:


def create_generator_model():
    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    # Model
    model = Sequential()

    # Start at 4 * 4
    start_shape = NOISE_SIZE * 4 * 4
    model.add(Dense(start_shape, kernel_initializer = init, input_dim = random_dim))
    model.add(Reshape((4, 4, 64)))
    
    # Upsample => 8 * 8 
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # Upsample => 16 * 16 
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # Upsample => 32 * 32
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # Upsample => 64 * 64
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))
    model.add(ReLU())
    
    # output
    model.add(Conv2D(3, kernel_size = 3, activation = 'tanh', padding = 'same', kernel_initializer=init))
    model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = LR_G, beta_1 = BETA1))
    print(model.summary())

    return model


# # Discriminator

# In[ ]:


def create_discriminator_model():
    input_shape = (img_rows, img_cols, channels)

    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    # Define Model
    model = Sequential()

    # Downsample ==> 32 * 32
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init, input_shape = input_shape))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))

    # Downsample ==> 16 * 16
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    # Downsample => 8 * 8
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    # Downsample => 4 * 4
    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.25))
    
    # Final Layers
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = init))

    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = LR_D , beta_1 = BETA1))
    
    print(model.summary())
    
    return model


# # GAN Model
# Next the code to create the GAN model.

# In[ ]:


def create_gan_model(discriminator, random_dim, generator):
    # Set trainable to False initially
    discriminator.trainable = False
    
    # Gan Input
    gan_input = Input(shape = (random_dim,))
    
    # Generator Output...an image
    generator_output = generator(gan_input)
    
    # Output of the discriminator is the probability of an image being real or fake
    gan_output = discriminator(generator_output)
    gan_model = Model(inputs = gan_input, outputs = gan_output)
    gan_model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = LR_G, beta_1 = BETA1))
    print(gan_model.summary())
    
    return gan_model


# # Noise function

# In[ ]:


def generator_input(latent_dim, n_samples):
    # Generate points in latent space
    input = np.random.randn(latent_dim * n_samples)

    # Reshape to input batch for the network
    input = input.reshape((n_samples, latent_dim))

    return input


# # Plot functions
# Next we define some functions to plot the images to give an impression of how the training progressed and one to plot the loss during training.

# In[ ]:


def plot_generated_images(epoch, generator, examples = 25, dim = (5, 5)):
    generated_images = generator.predict(np.random.normal(0, 1, size = [examples, random_dim]))
    generated_images = ((generated_images + 1) * 127.5).astype('uint8')
        
    plt.figure(figsize = (12, 8))
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation = 'nearest')
        plt.axis('off')
    plt.suptitle('Epoch %d' % epoch, x = 0.5, y = 1.0)
    plt.tight_layout()
    plt.savefig('dog_at_epoch_%d.png' % epoch)
    
def plot_loss(d_f, d_r, g):
    plt.figure(figsize = (18, 12))
    plt.plot(d_f, label = 'Discriminator Fake Loss')
    plt.plot(d_r, label = 'Discriminator Real Loss')
    plt.plot(g, label = 'Generator Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()


# # Train Model

# In[ ]:


def train_model(epochs = 1, batch_size = 128):
    # Get the Dog images
    x_train = load_images()
    
    # Calculate amount of batches
    batch_count = x_train.shape[0] / batch_size

    # Create Generator and Discriminator Models
    generator = create_generator_model()
    discriminator = create_discriminator_model()
    
    # Create GAN Model
    gan_model = create_gan_model(discriminator, random_dim, generator)
    
    # Lists for Loss History
    discriminator_fake_hist, discriminator_real_hist, generator_hist = [], [], []
    
    for e in range(epochs):
        
        # Script Stop Counter
        script_stopper_counter = 0
        
        print('======================== Epoch {} ============================='.format(e))
        for _ in tqdm(range(int(batch_count))):
            
            # Discriminator Loss
            discriminator_fake_loss, discriminator_real_loss = [], []
            
            # Train the Discriminator more than the Generator
            for _ in range(2):
                # Train discriminator on Fake Images
                X_fake = generator.predict(generator_input(random_dim, batch_size))
                y_fake = np.zeros(batch_size)
                y_fake[:] = 0
                discriminator.trainable = True
                d_fake_loss = discriminator.train_on_batch(X_fake, y_fake)
                
                # Train discriminator on Real Images
                X_real = x_train[np.random.randint(0, x_train.shape[0], size = batch_size)]
                y_real = np.zeros(batch_size)
                y_real[:] = 0.9  # label smoothing
                discriminator.trainable = True
                d_real_loss = discriminator.train_on_batch(X_real, y_real)

                # Store Loss each iteration
                discriminator_fake_loss.append(d_fake_loss)
                discriminator_real_loss.append(d_real_loss)

            # Train generator
            noise = generator_input(random_dim, batch_size)
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            generator_loss = gan_model.train_on_batch(noise, y_gen)

            # Summarize Batch Loss
            # Uncomment Lines below if you want per batch update Loss statistics
            #print('\nd_fake_loss = %.4f, d_real_loss = %.4f g_loss = %.4f' % \
            #      (np.mean(discriminator_fake_loss), np.mean(discriminator_real_loss), generator_loss))

            # Store Loss in Loss History lists
            discriminator_fake_hist.append(np.mean(discriminator_fake_loss))
            discriminator_real_hist.append(np.mean(discriminator_real_loss)) 
            generator_hist.append(generator_loss)
            
            # Stop script preliminary Counter
            # Occasionally the Discriminator Fake Loss explodes and remains high...in that case we stop the script
            if np.mean(discriminator_fake_loss) > 10:
                script_stopper_counter += 1
        
        # Summarize Image Quality for epochs during training
        if e % 100 == 0:
            plot_generated_images(e, generator)
            
        # Stop Script? If almost 1 epoch with exploded Loss...then Yes.
        if script_stopper_counter > 160:
            plot_generated_images(e, generator)
            break
            
    # Plot Loss during Training
    plot_loss(discriminator_fake_hist, discriminator_real_hist, generator_hist)

    # Create Images.zip
    z = zipfile.PyZipFile('images.zip', mode = 'w')
    for k in range(10000):
        # Generate new dogs
        generated_images = generator.predict(np.random.normal(0, 1, size = [1, random_dim]))
        image = Image.fromarray(((generated_images + 1) * 127.5).astype('uint8').reshape(64, 64, 3))

        # Save to zip file  
        f = str(k)+'.png'
        image.save(f, 'PNG')
        z.write(f)
        os.remove(f)
        
        # Plot Status Counter
        if k % 1000 == 0: 
            print(k)
    z.close()


# In[ ]:


train_model(EPOCHS, BATCH_SIZE)

