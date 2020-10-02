#!/usr/bin/env python
# coding: utf-8

# ## Code modules, functions, & links
# [Generate Artificial Faces with CelebA Progressive GAN Model](https://github.com/tensorflow/hub/blob/master/examples/colab/tf_hub_generative_image_module.ipynb)

# In[ ]:


get_ipython().system('pip install git+https://github.com/tensorflow/docs')


# In[ ]:


import numpy as np,pylab as pl,pandas as pd
import imageio,h5py
import tensorflow as tf
tf.random.set_seed(23)
from tensorflow_docs.vis import embed
import tensorflow_hub as th


# In[ ]:


def timg(img):
    img=tf.constant(img)
    img=tf.image.convert_image_dtype(img,tf.uint8)
    pl.imshow(img.numpy()); pl.title(img.shape);
def tanimate(images):
    converted_images=np.clip(images*255,0,255)    .astype(np.uint8)
    imageio.mimsave('animation.gif',converted_images)
    return embed.embed_file('animation.gif')


# ## Symbol images & hypersphere interpolation

# In[ ]:


fpath='../input/classification-of-handwritten-letters/'
f='LetterColorImages_123.h5'
f=h5py.File(fpath+f,'r')
keys=list(f.keys()); print(keys)
x=np.array(f[keys[1]],dtype='float32')/255
x=np.array(tf.image.resize(x,[128,128]),dtype='float32')
y=np.array(f[keys[2]],dtype='int32').reshape(-1,1)-1
N=len(y); n=int(.1*N)
shuffle_ids=np.arange(N)
np.random.RandomState(23).shuffle(shuffle_ids)
x,y=x[shuffle_ids],y[shuffle_ids]
x.shape,y.shape


# In[ ]:


def interpolate_hypersphere(v1,v2,steps):
    v1norm=tf.norm(v1)
    v2norm=tf.norm(v2)
    v2normalized=v2*(v1norm/v2norm)
    vectors=[]
    for step in range(steps):
        interpolated=v1+(v2normalized-v1)*step/(steps-1)
        interpolated_norm=tf.norm(interpolated)
        interpolated_normalized=        interpolated*(v1norm/interpolated_norm)
        vectors.append(interpolated_normalized)
    return tf.stack(vectors)


# In[ ]:


imgs=interpolate_hypersphere(x[7],x[8],120)
tanimate(imgs)


# In[ ]:


timg(np.concatenate([x[7],imgs[40],
                     imgs[80],x[8]],axis=1))


# ## Latent space interpolation using a pre-trained Progressive GAN

# In[ ]:


handle='https://tfhub.dev/google/progan-128/1'
progan=th.load(handle).signatures['default']
latent_dim=512


# In[ ]:


def interpolate_between_vectors(steps,latent_dim=512):
    tf.random.set_seed(1)
    v1=tf.random.normal([latent_dim])
    v2=tf.random.normal([latent_dim])
    vectors=interpolate_hypersphere(v1,v2,steps)
    interpolated_images=progan(vectors)['default']
    return interpolated_images


# In[ ]:


imgs2=interpolate_between_vectors(30)
tanimate(imgs2)


# In[ ]:


timg(np.concatenate([imgs2[0],imgs2[10],
                     imgs2[-10],imgs2[-1]],axis=1))


# In[ ]:


tf.random.set_seed(23)
initial_vector=tf.random.normal([1,latent_dim])
initial_img=progan(initial_vector)['default'][0]
tf.random.set_seed(123)
target_vector=tf.random.normal([1,latent_dim])
target_img=progan(target_vector)['default'][0]
timg(np.concatenate([initial_img,target_img],axis=1))


# In[ ]:


def find_closest_latent_vector(initial_vector,target_img,
                               steps,steps_per_image):
    images=[]; losses=[]
    vector=tf.Variable(initial_vector)  
    optimizer=tf.optimizers.Adam(learning_rate=.01)
    loss_fn=tf.losses.MeanAbsoluteError(reduction="sum")
    for step in range(steps):
        if (step%100)==0: print()
        print('.',end='')
        with tf.GradientTape() as tape:
            image=progan(vector.read_value())['default'][0]
            if (step%steps_per_image)==0:
                images.append(image.numpy())
            target_image_difference=loss_fn(image,target_img[:,:,:3])
            regularizer=tf.abs(tf.norm(vector)-np.sqrt(latent_dim))     
            loss=target_image_difference+regularizer
            losses.append(loss.numpy())
        grads=tape.gradient(loss,[vector])
        optimizer.apply_gradients(zip(grads,[vector]))
    return images,losses
steps=200; steps_per_image=5
images,loss=find_closest_latent_vector(initial_vector,target_img,
                                       steps,steps_per_image)


# In[ ]:


tanimate(np.stack(images))


# In[ ]:


timg(np.concatenate([images[0],images[10],
                     images[-10],target_img],axis=1))

