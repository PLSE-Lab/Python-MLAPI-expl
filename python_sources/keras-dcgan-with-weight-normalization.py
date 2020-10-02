#!/usr/bin/env python
# coding: utf-8

# In this notebook I will create and run a Keras DCGAN. With some experimenting I've created a model with a simple architecture that still provides in a nice LB score. I thought it would be a nice one to share with you and allow you to further experiment with it.
# 
# I won't be doing a whole lecture on how a GAN is working and why it is working. There are a whole bunch of kernels available in this competition which give a very nice explanation about GAN's.
# 
# Let start with importing all the needed libraries.

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
from keras.models import Model 
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
from PIL import Image


# # Constants and Directories
# Next lets define some constants and directories.

# In[ ]:


# Constants and Directories
SEED = 4250
np.random.seed(SEED)
random_dim = 128
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
# Next we define a method to load and process all images. From this kernel version and onwards I'am using the dog annotations for the images as provided. The bounding box code is based on earlier kernels from [cdeotte](https://www.kaggle.com/cdeotte/dog-memorizer-gan) and [paulorzp](https://www.kaggle.com/paulorzp/show-annotations-and-breeds).
# 
# I made some changes however that improved the score even further by about 10 points (at least it did on a private forked kernel of this notebook ;-) ).
# 
# I basically filter out images where one of the bounding box sides is smaller than 64 pixels. 
# (UPDATED in this version) I use the entire bounding box and just resize that to a square image. This does give some image distortion..however after a visual inspection of multiple images it does not look that bad. So lets try to use the whole bounding box as input and ignore the distortion. In multiple local runs this gave a slightly higher score. So lets see and wait what a run in a public kernel does. Each image is also normalized to have its values between -1 and 1.

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
# When I started with this competition I followed a lot of the tips and tricks on this [site](https://github.com/soumith/ganhacks). One of the tips is to use BatchNormalization. I however noticed this very nice [paper](https://arxiv.org/pdf/1704.03971.pdf). It discusses the usage and effects of Batch and Weights Normalization in GAN's. After reading it I thought I could give my GAN a try with Weights Normalization.
# 
# I haven't been able to find an official implementation of a Keras Weight Normalization layer. I was however able to find the code at this [github repository](https://github.com/krasserm/weightnorm/tree/master/keras_2). From that I will use the class AdamWithWeightnorm.

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
# Next define the code to create the Generator model. Note the usage of the AdamWithWeightnorm optimizer. I don't use any Batch Normalization layers. I also experienced that using Dropout layers hurts the performance and doesn't improve the creation of nice dog images. 

# In[ ]:


def create_generator_model():
    # Random Normal Weight Initialization
    init = RandomNormal(mean = 0.0, stddev = 0.02)

    # Model
    model = Sequential()

    # Start at 4 * 4
    start_shape = 64 * 4 * 4
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
    model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))
    print(model.summary())

    return model


# # Discriminator
# Next the code to define the Discriminator model. Again note that I don't use BatchNormalization layers as we use the Weight Normalization.
# 
# I do use Dropout Layers here as it improves the performance...but only a Dropout of about 25%.

# In[ ]:


def create_discriminator_model():
    input_shape = (64, 64, 3)

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
    model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))
    
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
    gan_model.compile(loss = 'binary_crossentropy', optimizer = AdamWithWeightnorm(lr = 0.0002, beta_1 = 0.5))
    print(gan_model.summary())
    
    return gan_model


# # Noise function
# Next we define a method to generate the input for the generator.

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
# Next we define the method to train the model and generate the submission 'images.zip'. Note that for each batch update we train the generator once on a batch and the discriminator 4 times on a batch (2 * fake and 2 * real images.)

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


# And finally we trigger the function to start the training for a number of epochs.

# In[ ]:


train_model(676, 128)


# I hope you enjoyed this notebook and that it provides some more insights in how to create a DCGAN with Keras.
# 
# Let me know if you have questions or feedback.
