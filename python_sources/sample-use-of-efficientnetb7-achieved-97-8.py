#!/usr/bin/env python
# coding: utf-8

# ### Before the trip begin
# 
# #####  If you like my work, please hit upvote since it will keep me motivated

# * **1. Introduction**   
# * **2. Data Preparation**
#    * 2.1 Import Libaries
#    * 2.2 Load Data
#    * 2.3 Check the Data
#    * 2.4 Split to train and test
# * **3. Data Augmentation**
# * **4. CNN**
#    * 4.1 Define Callbacks
#    * 4.2 Efficientnet-B7
#    * 4.3 Evaluate the Model

# # 1. Introduction
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Black_rot_lifecycle.tif/lossy-page1-1200px-Black_rot_lifecycle.tif.jpg)
# 
# ### Problem Statement
# Misdiagnosis of the many diseases impacting agricultural crops can lead to misuse of chemicals leading to the emergence of resistant pathogen strains, increased input costs, and more outbreaks with significant economic loss and environmental impacts. Current disease diagnosis based on human scouting is time-consuming and expensive, and although computer-vision based models have the promise to increase efficiency, the great variance in symptoms due to age of infected tissues, genetic variations, and light conditions within trees decreases the accuracy of detection.
# 
# ### Importance of Plant Pathology
# Plant Pathology has advanced techniques to protect crops from losses due to diseases. The science of plant pathology has contributed disease free certified seed production. Most of the diseases with known disease cycle can now be avoided by the modification of cultural practices

# # 2. Data Preparation

# ## 2.1 Import Libraries

# In[ ]:


import seaborn as sns
import numpy as np
import pandas as pd
import random
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt 
import glob as gb
from kaggle_datasets import KaggleDatasets
get_ipython().system('pip install -q efficientnet')
import efficientnet.tfkeras as efn
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential


# ## 2.2 Load Data

# In[ ]:


train_df = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"


# ## 2.3 Check Data

# This one below is to check the most common resolution of pictures among all Train data

# In[ ]:


size = []
files = gb.glob(pathname= str("../input/plant-pathology-2020-fgvc7/images/*.jpg"))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()
# Because it take looong time the output is below
# (1365, 2048, 3)    3620
# (2048, 1365, 3)      22


# In[ ]:


train_df.head(10)
# train_df.shape


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(14,14))
sns.barplot(x=train_df.healthy.value_counts().index,y=train_df.healthy.value_counts(),ax=ax[0,0])
ax[0,0].set_xlabel('Healthy',size=9)
ax[0,0].set_ylabel('Count',size=9)

sns.barplot(x=train_df.multiple_diseases.value_counts().index,y=train_df.multiple_diseases.value_counts(),ax=ax[0,1])
ax[0,1].set_xlabel('Multiple Diseases',size=9)
ax[0,1].set_ylabel('Count',size=9)

sns.barplot(x=train_df.rust.value_counts().index,y=train_df.rust.value_counts(),ax=ax[1,0])
ax[1,0].set_xlabel('Rust',size=9)
ax[1,0].set_ylabel('Count',size=9)

sns.barplot(x=train_df.scab.value_counts().index,y=train_df.scab.value_counts(),ax=ax[1,1])
ax[1,1].set_xlabel('Scab',size=9)
ax[1,1].set_ylabel('Count',size=9)


# ### Let's check all the categories we have 

# In[ ]:


healthy = list(train_df[train_df["healthy"]==1].image_id)
multiple_diseases = list(train_df[train_df["multiple_diseases"]==1].image_id)
rust = list(train_df[train_df["rust"]==1].image_id)
scab = list(train_df[train_df["scab"]==1].image_id)


# In[ ]:


def load_image(filenames):
    sample = random.choice(filenames)
    image = load_img("../input/plant-pathology-2020-fgvc7/images/"+sample+".jpg")
    plt.imshow(image) 


# In[ ]:


load_image(healthy)


# In[ ]:


load_image(multiple_diseases)


# In[ ]:


load_image(rust)


# In[ ]:


load_image(scab)


# Everything look fine !

# ## 2.4 Split to Train and Validaiton

# In[ ]:


GCS_DS_PATH = KaggleDatasets().get_gcs_path()
#to verify your dir
get_ipython().system('gsutil ls $GCS_DS_PATH')


# In[ ]:


def format_path_gcs(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'
#===============================================================
#===============================================================
X = train_df.image_id.apply(format_path_gcs).values
y = np.float32(train_df.loc[:, 'healthy':'scab'].values)

X_train, X_val, y_train, y_val =train_test_split(X, y, test_size=0.1, random_state=43)
print('done!')


# In[ ]:


print('Shape of X_train : ',X_train.shape)
print('Shape of y_train : ',y_train.shape)
print('===============================================')
print('Shape of X_val : ',X_val.shape)
print('Shape of y_val : ',y_val.shape)


# # 3. Create Dataset

# But first let's setup TPU

# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


BATCH_SIZE = 4 * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = y_train.shape[0] // BATCH_SIZE


# In[ ]:


def decode_image(filename, label=None, image_size=(1024,1024)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    
    
    if label is None:
        return image
    else:
        return image, label


# In[ ]:


train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(decode_image,num_parallel_calls=AUTO)
    .map(data_augment,num_parallel_calls=AUTO)
    .repeat()
    .shuffle(256)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(decode_image,num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


# # 4. CNN

# ## 4.1. Define Callbacks

# In[ ]:


def build_lrfn(lr_start=0.00001, lr_max=0.00005,lr_min=0.00001, lr_rampup_epochs=5,lr_sustain_epochs=0, lr_exp_decay=.8):
    
    lr_max = lr_max * strategy.num_replicas_in_sync
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *                 lr_exp_decay**(epoch - lr_rampup_epochs- lr_sustain_epochs) + lr_min
        return lr
    return lrfn


# In[ ]:


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
EarlyStopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,verbose=True, mode="min")


# In[ ]:


# reduce_lr =  tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10,
#   verbose = 0, mode = "auto", epsilon = 1e-04, cooldown = 0,
#   min_lr = 1e-5)
# es = tf.keras.callbacks.EarlyStopping(monitor = "val_loss" , verbose = 1 , mode = 'min' , patience = 10 )
#weights='noisy-student',


# ## 4.2. Efficientnet-B7

# ![](https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/efficientnet/g3doc/params.png)

# EfficientNets rely on AutoML and compound scaling to achieve superior performance without compromising resource efficiency. The AutoML Mobile framework has helped develop a mobile-size baseline network, EfficientNet-B0, which is then improved by the compound scaling method to obtain EfficientNet-B1 to B7.

# In[ ]:


def Eff_B7_NS():
    model_EfficientNetB7_NS = Sequential([efn.EfficientNetB7(input_shape=(1024,1024,3),weights='noisy-student',include_top=False),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(128,activation='relu'),
                                 tf.keras.layers.Dense(64,activation='relu'),
                                 tf.keras.layers.Dense(4,activation='softmax')])               
    model_EfficientNetB7_NS.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['categorical_accuracy'])
    
    
    return model_EfficientNetB7_NS


# In[ ]:


with strategy.scope():
    model_Eff_B7_NS=Eff_B7_NS()
    
model_Eff_B7_NS.summary()
#del model_Eff_B7_NS


# In[ ]:


EfficientNetB7_NS = model_Eff_B7_NS.fit(train_dataset,
                    epochs=50,
                    callbacks=[lr_schedule,EarlyStopping],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


# ## 4.3. Evaluate Model

# In[ ]:


plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(EfficientNetB7_NS.history['loss'])
ax1.plot(EfficientNetB7_NS.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('loss')
ax1.set_xlabel('epoch')

ax2.plot(EfficientNetB7_NS.history['categorical_accuracy'])
ax2.plot(EfficientNetB7_NS.history['val_categorical_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epoch')


# As you can see after no improvment EarlyStop did its job ,and for this model I think is the best to choose between 14-17 epochs . 

# 
# 
# 
# ### If you like my work, please hit upvote since it will keep me motivated
