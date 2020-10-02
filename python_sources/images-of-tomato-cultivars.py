#!/usr/bin/env python
# coding: utf-8

# ## Code Modules & Functions

# In[ ]:


get_ipython().system('pip install git+https://github.com/tensorflow/docs')


# In[ ]:


import warnings; warnings.filterwarnings('ignore')
import pandas as pd,numpy as np,tensorflow as tf
import h5py,imageio,os
import seaborn as sn,pylab as pl
from keras.preprocessing import image as kimage
from tensorflow_docs.vis import embed
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True 
fpath='../input/tomato-cultivars/'


# In[ ]:


def path_to_tensor(img_path,fpath=fpath):
    img=kimage.load_img(fpath+img_path, 
                        target_size=(160,160))
    x=kimage.img_to_array(img)
    return np.expand_dims(x,axis=0)
def paths_to_tensor(img_paths):
    tensor_list=[path_to_tensor(img_path) 
                 for img_path in tqdm(img_paths)]
    return np.vstack(tensor_list)
def animate(images):
    converted_images=np.clip(images*255,0,255)    .astype(np.uint8)
    imageio.mimsave('animation.gif',converted_images)
    return embed.embed_file('animation.gif')
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


def plcmap(cmap,n):
    return [pl.cm.get_cmap(cmap)(i/n)[:3] 
            for i in range(1,n+1)]
plcmap('Reds',5)


# ## Data Processing

# In[ ]:


names=['Kumato','Beefsteak','Tigerella',
       'Roma','Japanese Black Trifele',
       'Yellow Pear','Sun Gold','Green Zebra',
       'Cherokee Purple','Oxheart','Blue Berries',
       'San Marzano','Banana Legs',
       'German Orange Strawberry','Supersweet 100']
flist=sorted(os.listdir(fpath))
labels=np.array([int(el[:2]) for el in flist],
               dtype='int32')-1
images=np.array(paths_to_tensor(flist),
                dtype='float32')/255
N=labels.shape[0]; n=int(.2*N)
shuffle_ids=np.arange(N)
np.random.RandomState(12).shuffle(shuffle_ids)
images,labels=images[shuffle_ids],labels[shuffle_ids]
x_test,x_train=images[:n],images[n:]
y_test,y_train=labels[:n],labels[n:]


# In[ ]:


pd.DataFrame([[x_train.shape,x_test.shape],
              [x_train.dtype,x_test.dtype],
              [y_train.shape,y_test.shape],
              [y_train.dtype,y_test.dtype]],               
             columns=['train','test'])


# In[ ]:


with h5py.File('TomatoCultivarImages.h5','w') as f:
    f.create_dataset('train_images',data=x_train)
    f.create_dataset('train_labels',data=y_train)
    f.create_dataset('test_images',data=x_test)
    f.create_dataset('test_labels',data=y_test)
os.stat('TomatoCultivarImages.h5')


# ## Data Representation

# In[ ]:


set(labels)


# In[ ]:


pl.figure(figsize=(10,5))
sn.countplot(x=labels,facecolor=(0,0,0,0),
             linewidth=7,linestyle='-.',
             edgecolor=plcmap('Reds',15))
pl.title('Cultivar Distribution',fontsize=20);


# In[ ]:


n=np.random.randint(40)
print('Label: ',y_test[n],
      names[y_test[n]])
pl.figure(figsize=(3,3))
pl.imshow((x_test[n]));


# In[ ]:


imgs=interpolate_hypersphere(x_train[0],x_train[1],180)
animate(imgs)

