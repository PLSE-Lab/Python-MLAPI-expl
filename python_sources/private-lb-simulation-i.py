#!/usr/bin/env python
# coding: utf-8

# # Private LB Simulation
# ## PART I: Private Dataset [5 min read]
# 
# 
# <img src="https://i.ibb.co/t4cCsqF/Captura-de-pantalla-de-2019-07-17-13-58-42.png" alt="Captura-de-pantalla-de-2019-07-17-13-58-42" border="0">
# 
# ----
# 
# I'm back! no more discussions about memorizers or rules, it's boring.
# 
# **I think this can help everyone, no matter what kind of GAN you use: memorizer, GAN, CGAN, RaLSGAN etc**
# 
# **In this kernel I simulate the private dataset!** And we'll see if the dataset we use to generate matters when it comes to getting a good result.
# 
# > This took me some time, so I hope you'll appreciate it if it works for you. **Remember, the <span style="color:blue">upvote</span> button is near the fork button.** And as you see, the title says *part I*, the *part II* is about the ```mysterious pretrained NN``` ;)
# 
# 
# ### Datasets
# 
# 
# 1. [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction/data)
# 2. [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
# 3. [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)  
# 4. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
# 
# 
# **Why this datasets could be the private dataset aka "mysterious private dog images" ?** 
# 
# We are using [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), it's public, like these others datasets. For example, CIFAR-10 is also a well known dataset and has **dogs** images (among others). But keep reading, you'll see the best part soon ;)
# 
# 
# ### My test
# 
# 1. Train using **these datasets** and generate ```User Generate N Dogs Images ```
# 2. Use as ```Mysterious NN``` the inception model (risky hypothesis)
# 3. Use as ```Mysterious Dog Dataset``` (private) the *Generative Dogs Dataset*
# 
# In other words, I do the opposite of what kaggle does.
# 
# **Let's see if the dog dataset we use to generate images is important when it comes to getting a great LB score!**
# > If the results do not vary much, then perhaps the dataset is not important
# 
# ----
# 
# **NOTE** 
# 1. I can't import all these datasets (the kernel doesn't start properly) so I only import **2** of them (*my winning horses*)
# 2. You'll see this kernel has many versions, probably I'll try one different dataset in each version.
# 
# <br>

# # [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
# 
# The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. Obviously, I'm going to use only **dog** images.
# 
# Here are the classes in the dataset, as well as 10 random images from each:
# 
# <img src="https://i.ibb.co/02xnm2X/Captura-de-pantalla-de-2019-07-18-01-27-11.png" alt="Captura-de-pantalla-de-2019-07-18-01-27-11" border="0">
# 
# 

# In[ ]:


import numpy as np
import keras
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# ```x_train, x_test:``` uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3) based on the image_data_format backend setting of either channels_first or channels_last respectively.
# 
# ```y_train, y_test:``` uint8 array of category labels (integers in range 0-9) with shape (num_samples,).

# In[ ]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# **Labels** are encoded, the magic value is **dogs = 5**  (airplane=0 ...truck=9)

# In[ ]:


y_train[0:5]


# In[ ]:


idx_dogs = [i for i in range(x_train.shape[0]) if y_train[i]==5]
len(idx_dogs), idx_dogs[0:5]


# In[ ]:


imagesIn = x_train[idx_dogs]
imagesIn.shape


# Now you have **5000 new dogs!** Resize the pictures to 64x64 and try your GAN with this dataset.

# In[ ]:


del x_train, x_test, y_train, y_test


# # [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)
# 
# **Kaggle description**
# 
# > In this playground competition, you are provided a strictly canine subset of ImageNet in order to practice fine-grained image categorization. How well you can tell your Norfolk Terriers from your Norwich Terriers? With 120 breeds of dogs and a limited number training images per class, you might find the problem more, err, ruff than you anticipated.
# 
# >**Acknowledgments**
# We extend our gratitude to the creators of the Stanford Dogs Dataset for making this competition possible: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li.
# 
# 
# **This dataset is very similar to out dataset!**
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/3333/media/border_collies.png)

# In[ ]:


get_ipython().system('ls ../input/dog-breed-identification/')


# **Load and crop**

# In[ ]:


ComputeLB = True
DogsOnly = False

import numpy as np, pandas as pd, os
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt, zipfile 
from PIL import Image 

ROOT = '../input/dog-breed-identification/train/'
if not ComputeLB: ROOT = '../input/'
IMAGES = os.listdir(ROOT + 'train/')
#breeds = os.listdir(ROOT + 'annotation/Annotation/') 

idxIn = 0; namesIn = []
imagesIn = np.zeros((25000,64,64,3))

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY
# https://www.kaggle.com/paulorzp/show-annotations-and-breeds
if DogsOnly:
    pass
    
# RANDOMLY CROP FULL IMAGES
else:
    IMAGES = np.sort(IMAGES)
    np.random.seed(810)
    x = np.random.choice(np.arange(len(IMAGES)),10000)
    np.random.seed(None)
    for k in range(len(x)):
        #print (ROOT + 'train/' + IMAGES[x[k]])
        img = Image.open(ROOT + 'train/' + IMAGES[x[k]])
        w = img.size[0]; h = img.size[1];
        if (k%2==0)|(k%3==0):
            w2 = 100; h2 = int(h/(w/100))
            a = 18; b = 0          
        else:
            a=0; b=0
            if w<h:
                w2 = 64; h2 = int((64/w)*h)
                b = (h2-64)//2
            else:
                h2 = 64; w2 = int((64/h)*w)
                a = (w2-64)//2
        img = img.resize((w2,h2), Image.ANTIALIAS)
        img = img.crop((0+a, 0+b, 64+a, 64+b))    
        imagesIn[idxIn,:,:,:] = np.asarray(img)
        namesIn.append(IMAGES[x[k]])
        #if idxIn%1000==0: print(idxIn)
        idxIn += 1


# **Display cropped images**
# > **NOTE:** probably this crop is not the best, but this is just a quick test.

# In[ ]:


x = np.random.randint(0,idxIn,25)
for k in range(5):
    plt.figure(figsize=(37,3))
    for j in range(5):
        plt.subplot(1,5,j+1)
        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )
        plt.axis('off')
        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)
        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)
        plt.imshow(img)
    plt.show()


# ### <span style="color:red"> DISCLAIMER </span>
# 
# The following code is from: [Dog Memorizer GAN](https://www.kaggle.com/cdeotte/dog-memorizer-gan) , I use this code because **it's fast!** (I'm not 100% if it's allowed or not) you should generate the 10k images using **your own gan** and check **[LB Score]()**
# 

# ### Build Discriminator

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam


# In[ ]:


# BUILD DISCRIMINATIVE NETWORK
dog = Input((12288,))
dogName = Input((10000,))
x = Dense(12288, activation='sigmoid')(dogName) 
x = Reshape((2,12288,1))(concatenate([dog,x]))
x = Conv2D(1,(2,1),use_bias=False,name='conv')(x)
discriminated = Flatten()(x)

# COMPILE
discriminator = Model([dog,dogName], discriminated)
discriminator.get_layer('conv').trainable = False
discriminator.get_layer('conv').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# DISPLAY ARCHITECTURE
discriminator.summary()


# ### Train Discriminator

# In[ ]:


# TRAINING DATA
train_y = (imagesIn[:10000,:,:,:]/255.).reshape((-1,12288))
train_X = np.zeros((10000,10000))
for i in range(10000): train_X[i,i] = 1
zeros = np.zeros((10000,12288))

# TRAIN NETWORK
lr = 0.5
for k in range(5):
    annealer = LearningRateScheduler(lambda x: lr)
    h = discriminator.fit([zeros,train_X], train_y, epochs = 10, batch_size=256, callbacks=[annealer], verbose=0)
    print('Epoch',(k+1)*10,'/30 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1]<0.533: lr = 0.1


# ### Delete Training Images

# In[ ]:


del imagesIn


# **Discriminator Recalls from Memory Dogs**

# In[ ]:


for k in range(5):
   plt.figure(figsize=(15,3))
   for j in range(5):
       xx = np.zeros((10000))
       xx[np.random.randint(10000)] = 1
       plt.subplot(1,5,j+1)
       img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))
       img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))
       plt.axis('off')
       plt.imshow(img)
   plt.show()


# ### Build Generator and GAN

# In[ ]:


# BUILD GENERATOR NETWORK
seed = Input((10000,))
generated = Dense(12288, activation='linear')(seed)

# COMPILE
generator = Model(seed, [generated,Reshape((10000,))(seed)])

# DISPLAY ARCHITECTURE
generator.summary()


# **BUILD GENERATIVE ADVERSARIAL NETWORK**

# In[ ]:


discriminator.trainable=False    
gan_input = Input(shape=(10000,))
x = generator(gan_input)
gan_output = discriminator(x)

# COMPILE GAN
gan = Model(gan_input, gan_output)
gan.get_layer('model_1').get_layer('conv').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])
gan.compile(optimizer=Adam(5), loss='mean_squared_error')

# DISPLAY ARCHITECTURE
gan.summary()


# ### Discriminator Coaches Generator

# In[ ]:


# TRAINING DATA
train = np.zeros((10000,10000))
for i in range(10000): train[i,i] = 1
zeros = np.zeros((10000,12288))

# TRAIN NETWORKS
lr = 5.
for k in range(50):  

    # BEGIN DISCRIMINATOR COACHES GENERATOR
    annealer = LearningRateScheduler(lambda x: lr)
    h = gan.fit(train, zeros, epochs = 1, batch_size=256, callbacks=[annealer], verbose=0)
    if (k<10)|(k%5==4):
        print('Epoch',(k+1)*10,'/500 - loss =',h.history['loss'][-1] )
    if h.history['loss'][-1] < 25: lr = 1.
    if h.history['loss'][-1] < 1.5: lr = 0.5
        
    # DISPLAY GENERATOR LEARNING PROGRESS
    if k<10:        
        plt.figure(figsize=(15,3))
        for j in range(5):
            xx = np.zeros((10000))
            xx[np.random.randint(10000)] = 1
            plt.subplot(1,5,j+1)
            img = generator.predict(xx.reshape((-1,10000)))[0].reshape((-1,64,64,3))
            img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))
            plt.axis('off')
            plt.imshow(img)
        plt.show()  


# ### Build Generator Class

# In[ ]:


class DogGenerator:
    index = 0   
    def getDog(self,seed):
        xx = np.zeros((10000))
        xx[self.index] = 0.999
        xx[np.random.randint(10000)] = 0.001
        img = generator.predict(xx.reshape((-1,10000)))[0].reshape((64,64,3))
        self.index = (self.index+1)%10000
        return Image.fromarray( img.astype('uint8') ) 


# ### Submit to Kaggle

# In[ ]:


# SAVE TO ZIP FILE NAMED IMAGES.ZIP
z = zipfile.PyZipFile('images.zip', mode='w')
d = DogGenerator()
for k in range(10000):
    img = d.getDog(np.random.normal(0,1,100))
    f = str(k)+'.png'
    img.save(f,'PNG'); z.write(f); os.remove(f)
    #if k % 1000==0: print(k)
z.close()


# # [LB Score]()
# 
# 1. We didn't use ```ImageNet Dog Images``` to generate ```User Generated dog Images```, **we used Dog Breed Identification dataset!**
# 2. We use as ```Mysterious Dog Dataset``` (**private dataset**) the ```ImageNet Dog Images``` to generate ```User Generated dog Images```
# 3. We use as ```Mysterious Pre-trained dataset``` the same inception model (a risky hypothesis). But, do you see the title, it says *part I*, if there is a lot of people interested I can do *part 2* and show my **private NN test**
# 
# That means **we do the opposite of kaggle**
# 
# ### The question is:
# 
# - ```private dataset = all dogs```, generate using ```Dog Breed Identification``` gives us a LB around 7.5 - 8.5
# 
# - **```private dataset = Dog Breed Identification```, generate using ```all dogs``` gives us a LB around 7.5 - 8.5 ? This is the real case, and I think the answer is yes!**
# 
# You can try with the others datasets too! I checked ```Dog Breed Identification``` and ```Cifar-10```, and it's pretty similar (using cifar-10 the score is around 10).
# 
# <img src="https://i.ibb.co/DVn018V/Captura-de-pantalla-de-2019-07-18-02-55-09.png" alt="Captura-de-pantalla-de-2019-07-18-02-55-09" border="0">
# 

# In[ ]:


from __future__ import absolute_import, division, print_function
import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from scipy import linalg
import pathlib
import urllib
import warnings
from tqdm import tqdm
from PIL import Image

class KernelEvalException(Exception):
    pass

model_params = {
    'Inception': {
        'name': 'Inception', 
        'imsize': 64,
        'output_layer': 'Pretrained_Net/pool_3:0', 
        'input_layer': 'Pretrained_Net/ExpandDims:0',
        'output_shape': 2048,
        'cosine_distance_eps': 0.1
        }
}

def create_model_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')

def _get_model_layer(sess, model_name):
    # layername = 'Pretrained_Net/final_layer/Mean:0'
    layername = model_params[model_name]['output_layer']
    layer = sess.graph.get_tensor_by_name(layername)
    ops = layer.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims != []:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return layer

def get_activations(images, sess, model_name, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_model_layer(sess, model_name)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images//batch_size + 1
    pred_arr = np.empty((n_images,model_params[model_name]['output_shape']))
    for i in tqdm(range(n_batches)):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
                    
        batch = images[start:end]
        pred = sess.run(inception_layer, {model_params[model_name]['input_layer']: batch})
        pred_arr[start:end] = pred.reshape(-1,model_params[model_name]['output_shape'])
    if verbose:
        print(" done")
    return pred_arr


# def calculate_memorization_distance(features1, features2):
#     neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')
#     neigh.fit(features2) 
#     d, _ = neigh.kneighbors(features1, return_distance=True)
#     print('d.shape=',d.shape)
#     return np.mean(d)

def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def cosine_distance(features1, features2):
    # print('rows of zeros in features1 = ',sum(np.sum(features1, axis=1) == 0))
    # print('rows of zeros in features2 = ',sum(np.sum(features2, axis=1) == 0))
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))
    print('d.shape=',d.shape)
    print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)
    mean_min_d = np.mean(np.min(d, axis=1))
    print('distance=',mean_min_d)
    return mean_min_d


def distance_thresholding(d, eps):
    if d < eps:
        return d
    else:
        return 1

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    # covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1,sigma2))

    print('covmean.shape=',covmean.shape)
    # tr_covmean = tf.linalg.trace(covmean)

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    # return diff.dot(diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, model_name, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, model_name, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act
    
def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):
    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    imsize = model_params[model_name]['imsize']

    # In production we don't resize input images. This is just for demo purpose. 
    x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])
    m, s, features = calculate_activation_statistics(x, sess, model_name)
    del x #clean up memory
    return m, s, features

# check for image size
def img_read_checks(filename, resize_to, is_checksize=False, check_imsize = 64, is_check_png = False):
    im = Image.open(str(filename))
    if is_checksize and im.size != (check_imsize,check_imsize):
        raise KernelEvalException('The images are not of size '+str(check_imsize))
    
    if is_check_png and im.format != 'PNG':
        raise KernelEvalException('Only PNG images should be submitted.')

    if resize_to is None:
        return im
    else:
        return im.resize((resize_to,resize_to),Image.ANTIALIAS)

def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None, mm=[], ss=[], ff=[]):
    ''' Calculates the KID of two paths. '''
    tf.reset_default_graph()
    create_model_graph(str(model_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1, features1 = _handle_path_memorization(paths[0], sess, model_name, is_checksize = True, is_check_png = True)
        if len(mm) != 0:
            m2 = mm
            s2 = ss
            features2 = ff
        elif feature_path is None:
            m2, s2, features2 = _handle_path_memorization(paths[1], sess, model_name, is_checksize = False, is_check_png = False)
        else:
            with np.load(feature_path) as f:
                m2, s2, features2 = f['m'], f['s'], f['features']

        print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))
        print('starting calculating FID')
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('done with FID, starting distance calculation')
        distance = cosine_distance(features1, features2)        
        return fid_value, distance, m2, s2, features2


# In[ ]:


if ComputeLB:
  
    # UNCOMPRESS OUR IMGAES
    with zipfile.ZipFile("../working/images.zip","r") as z:
        z.extractall("../tmp/images2/")

    # COMPUTE LB SCORE
    m2 = []; s2 =[]; f2 = []
    user_images_unzipped_path = '../tmp/images2/'
    images_path = [user_images_unzipped_path,'../input/generative-dog-images/all-dogs/all-dogs/']
    public_path = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'

    fid_epsilon = 10e-15

    fid_value_public, distance_public, m2, s2, f2 = calculate_kid_given_paths(images_path, 'Inception', public_path, mm=m2, ss=s2, ff=f2)
    distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])
    print("FID_public: ", fid_value_public, "distance_public: ", distance_public, "multiplied_public: ",
            fid_value_public /(distance_public + fid_epsilon))


# In[ ]:


# REMOVE FILES TO PREVENT KERNEL ERROR OF TOO MANY FILES
get_ipython().system(' rm -r ../tmp')

