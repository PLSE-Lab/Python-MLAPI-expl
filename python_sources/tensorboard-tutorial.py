#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

import pandas as pd
import numpy as np
from skimage.io import imread_collection,imread,imsave
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder

from tensorboard import summary as summary_lib
import warnings 
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.


# **TENSORBOARD USAGE:**
# 
# This kernel is mainly to showcase how tensorboard can be used in a kaggle kernel. We take a CNN to showcase it. The dataset we will be using is 'Natural Images' dataset.
# 
# **Tensorboard:** It is tool used to visualise your TF models, it is mainly used for debugging model performances.
# 
# **WORKFLOW OF TENSORBOARD:** 
# 
# 1. We compute what needs to summarized using tf.summary.
# 1. We combine all the summary record using tf.summary.merge_all().
# 1. We create a directory to store the logs i.e is done by tf.summary.FileWriter(log_dir).
# 1. Finally we keep updating the summary in the disk by calling tf.summary.merge_all() in different instances of the training step.
# 
# Please make sure the internet is connected.

# > We will resize the images to the same resolution of 50x50.
# It is done batch-wise.

# In[ ]:


height=50
width =50
def Resize_Images(images,h=height,w=width):
    new_images = []
    for image in images:
        new_images.append(resize(image, (h, w),anti_aliasing=True))
    return np.array(new_images)


# Note that we are deleting the data to free up the RAM because if comes close to its threshold, the kernel crashes.

# In[ ]:


images_dir = '../input/data/natural_images/'
airplanes = Resize_Images(imread_collection(images_dir+'airplane/*.jpg'))
plt.imshow(airplanes[0])
n_of_obj1 = np.shape(airplanes)[0]


# In[ ]:


cars = Resize_Images(imread_collection(images_dir+'car/*.jpg'))
n_of_obj2 = np.shape(cars)[0]
plt.imshow(cars[0])
final_data = np.concatenate([airplanes,cars],axis=0)
del airplanes,cars


# In[ ]:


dogs = Resize_Images(imread_collection(images_dir+'dog/*.jpg'))
n_of_obj3 = np.shape(dogs)[0]
plt.imshow(dogs[0])
final_data = np.concatenate([final_data,dogs],axis=0)
del dogs


# In[ ]:


fruits = Resize_Images(imread_collection(images_dir+'fruit/*.jpg'))
n_of_obj4 = np.shape(fruits)[0]
plt.imshow(fruits[0])
final_data = np.concatenate([final_data,fruits],axis=0)
del fruits


# In[ ]:


people = Resize_Images(imread_collection(images_dir+'person/*.jpg'))
n_of_obj5 = np.shape(people)[0]
plt.imshow(people[0])
final_data = np.concatenate([final_data,people],axis=0)
final_data = np.array(final_data,np.float32)
del people


# In[ ]:


N = final_data.shape[0]


# 1. We will be creating sprite image im the next cell.
# 1. There will be 7x32 images imported in case of 'airplanes' and 'cars' and 6x32 images in case of the rest.
# 1. This is mainly to fulfill the criterea of having equal AxA shaped image, as the tensorboard projecter only accepts square images.
# 1. Else it keeps copying the contents of the first square formed in the rectangular image.
# 1. We create a 'label.tsv' file to create the metadata label.
# 1. We also create a test_array to input into the model to create its embeddings, which will be masked by its sprite image in the tensorboard projection
# 

# In[ ]:


def create_sprite_image(data,per_data=32):
    test_array = []
    test_label = []
    for i in range(6): #Concatenation of images horizontally then vertically
        airplane = data[i*per_data:(i+1)*per_data]
        now = n_of_obj1
        test_array.extend(airplane)
        test_label.extend([1 for i in range(per_data)])
        
        cars = data[now+(i*per_data):now+(i+1)*per_data]
        now += n_of_obj2
        test_array.extend(cars)
        test_label.extend([2 for i in range(per_data)])
        
        dogs = data[now+(i*per_data):now+(i+1)*per_data]
        now += n_of_obj3
        test_array.extend(dogs)
        test_label.extend([3 for i in range(per_data)])
        
        fruits = data[now+(i*per_data):now+(i+1)*per_data]
        now += n_of_obj4
        test_array.extend(fruits)
        test_label.extend([4 for i in range(per_data)])
        
        people = data[now+(i*per_data):now+(i+1)*per_data]
        test_array.extend(people)
        test_label.extend([5 for i in range(per_data)])
        
        airplane = np.reshape(airplane,(per_data*50,50,3))
        cars = np.reshape(cars,(per_data*50,50,3))
        dogs = np.reshape(dogs,(per_data*50,50,3))
        fruits = np.reshape(fruits,(per_data*50,50,3))
        people = np.reshape(people,(per_data*50,50,3))
        
        
        if(i == 0): 
            final_array =  np.concatenate([airplane,cars,people],axis=1)
            continue
        final_array =  np.concatenate([final_array,airplane,cars,people],axis=1)
        if(i == 5):
            airplane = data[i*per_data:(i+1)*per_data]
            now = n_of_obj1
            test_array.extend(airplane)
            test_label.extend([1 for i in range(per_data)])
            
            cars = data[now+(i*per_data):now+(i+1)*per_data]
            now += n_of_obj2
            test_array.extend(cars)
            test_label.extend([2 for i in range(per_data)])
            airplane = np.reshape(airplane,(per_data*50,50,3))
            cars = np.reshape(cars,(per_data*50,50,3))
            final_array = np.concatenate([final_array,airplane,cars],axis=1)

    
    tsv_file = []
    #Creating the metadata label
    options = ['Airplane','Car','Dog','Fruit','People']
    for row in range(per_data):
        for i in range(6):
            tsv_file.extend(options)
        tsv_file.extend(['Airplane','Car'])
    tsv_file = np.reshape(tsv_file,(-1,1))
    tsv_file = pd.DataFrame(tsv_file)
    tsv_file.to_csv('label.tsv',header=False, index=False) #The tensorboard metadata parser considers headers as one of its contents

    return np.array(test_array),np.reshape(test_label,(-1,1)),resize(final_array,(32*50,32*50),anti_aliasing=True)

#####
test_array,test_label,sprite_image = create_sprite_image(final_data)    
print(test_array.shape,test_label.shape,sprite_image.shape)
imsave('sprite.png',sprite_image)
fig, ax = plt.subplots(figsize=(50, 50))
plt.imshow(imread('sprite.png'))


# In[ ]:


#Creating the label to be OneHot Encoded
labels = [1 for i in range(n_of_obj1)]
labels.extend([2 for i in range(n_of_obj2)])
labels.extend([3 for i in range(n_of_obj3)])
labels.extend([4 for i in range(n_of_obj4)])
labels.extend([5 for i in range(n_of_obj5)])
labels = np.array(labels)
labels = np.reshape(labels,(labels.shape[0],1),np.float32)


# **OneHotEncoding Step:**

# In[ ]:


oh_model = OneHotEncoder(sparse=False)
labels = oh_model.fit_transform(labels)
test_label = oh_model.transform(test_label)


# In[ ]:


category_numbers = 5
batch_size = N


#  **The Model Details:**
# > LAYER 1 : 
# * Convulution Layer 
# * Convulution Layer
# * Max-pooling layer
# 
# >LAYER 2:
# * Convulution Layer 
# * Convulution Layer
# * Max-pooling Layer
# 
# >FINAL-LAYER:
# * [](http://)Flatten the output from Max-pooling Layer
# * Dense layer with 16 neurons
# * Softmax-layer
# 

# In[ ]:


tf.reset_default_graph()
sess = tf.Session()

with tf.name_scope('INPUT'):
    X = tf.placeholder(tf.float32,[None,height,width,3])
    Y = tf.placeholder(tf.float32,[None,category_numbers])
    tf.summary.image('image_test',X,10)

with tf.name_scope('LAYER_1'):
    conv1 = tf.keras.layers.Conv2D(32,kernel_size=[3,3],padding='same',activation='relu')(X)
    conv1 = tf.keras.layers.Conv2D(32,kernel_size=[3,3],padding='same',activation='relu')(conv1)
    conv1 = tf.keras.layers.MaxPooling2D((2,2))(conv1)
    tf.summary.histogram('layer_conv1',conv1)

with tf.name_scope('LAYER_2'):
    conv2 = tf.keras.layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu')(conv1)
    conv2 = tf.keras.layers.Conv2D(64,kernel_size=[3,3],padding='same',activation='relu')(conv2)
    conv2 = tf.keras.layers.MaxPooling2D((2,2))(conv2)
    tf.summary.histogram('layer_conv2',conv2)
    
with tf.name_scope('FINAL_LAYER'):
    flat = tf.keras.layers.Flatten()(conv2)
    dense = tf.keras.layers.Dense(16,activation='relu')(flat)
    embed_input = dense #We take this as the embedding vector for our Tensorboard projection
    embedding_input_size = 16
    logits = tf.keras.layers.Dense(category_numbers,activation='softmax')(dense)
    tf.summary.histogram('final_layer_logits',logits)

with tf.name_scope('LOSS'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=Y,logits=logits)
    tf.summary.scalar('loss',loss)
    lab = tf.reshape(Y,[1,-1])
    lab = tf.dtypes.cast(lab,tf.bool)
    pred = tf.reshape(logits,[1,-1])
    
with tf.name_scope('TRAIN'):
    opt = tf.train.AdamOptimizer(10E-4)
    train = opt.minimize(loss)

#We assemble all the summaries together else we would have to call it indivisually in the training step    
summary = tf.summary.merge_all()
embedding = tf.Variable(tf.zeros([1024, embedding_input_size]), name="Embedding")
assignment = embedding.assign(embed_input)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('tb_folder/log/',sess.graph)

config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add(tensor_name=embedding.name)
# embedding_config.tensor_name = embedding.name
embedding_config.sprite.image_path = 'sprite.png'
embedding_config.metadata_path = 'label.tsv'

embedding_config.sprite.single_image_dim.extend([height, width])
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


# In[ ]:


#We move the sprite and the metadat to the working directory
get_ipython().system('cp sprite.png tb_folder/log')
get_ipython().system('cp label.tsv tb_folder/log')


# **THE TRAINING STEP**
# We need to keep updating the summary, here we are updating it after every epoch and we are updating the embeddings after every 50 epochs, as the weights changes as the epochs increases.

# In[ ]:


epochs = 300

for i in range(epochs):    
    L,_,S = sess.run([loss,train,summary],feed_dict={X:final_data,Y:labels})
    if i%50 == 0:
        sess.run([assignment],feed_dict={X:test_array,Y:test_label})
        saver.save(sess, "tb_folder/log/model.ckpt", i)
    writer.add_summary(S,0) #Updating summary
    print('Loss:%0.5f '%L,"Epoch: ",i)


# ngrok provides us with a port (a web server). The Beholder plugin doesn't work very well with the server, hence I have not implemented it but if you want to try beholder add this to your code
# >>from tensorboard.plugins.beholder import Beholder
# 
# >>beholder = Beholder("tb_folder/log/")
# 
# >**Add the next line to your training step**
# 
# >>beholder.update(session=sess)

# In[ ]:


get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip')

LOG_DIR = 'tb_folder/log/' 
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# In[ ]:


sess.close()


# In[ ]:





# In[ ]:




