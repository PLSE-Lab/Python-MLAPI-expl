#!/usr/bin/env python
# coding: utf-8

# In[46]:


# import the libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization


# In[47]:


# for re-producible keras results
import numpy as np
import tensorflow as tf
import random as rn
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# In[48]:


train_df = pd.read_csv('../input/gameofdl/train.csv')
test_df = pd.read_csv('../input/gameofdl/test.csv')


# In[49]:


train_df.head(2)
train_df['category'] = train_df['category'].astype('str')


# In[50]:


from keras.preprocessing.image import ImageDataGenerator


# In[51]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.20,
                                   zoom_range=0.20,
                                   #validation_split=0.20,   
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


# In[57]:


img_sz = 350


# In[58]:


train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory='../input/gameofdl/train/',
                                                    x_col='image',
                                                    y_col='category',
                                                    has_ext=True,
                                                    seed=42,
                                                    target_size=(img_sz, img_sz),
                                                    batch_size=16,
                                                    #subset='training',    
                                                    shuffle=True,
                                                    class_mode='categorical')


# valid_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
#                                                     directory='../input/gameofdl/train',
#                                                     x_col='image',
#                                                     y_col='category',
#                                                     has_ext=True,
#                                                     seed=42,
#                                                     target_size=(299, 299),
#                                                     batch_size=16,
#                                                     subset='validation',    
#                                                     shuffle=True,
#                                                     class_mode='categorical')

# In[59]:


test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                  directory='../input/gameofdl/test/test_images/',
                                                  x_col='image',
                                                  y_col=None,
                                                  has_ext=True,
                                                  target_size=(img_sz, img_sz),
                                                  class_mode=None,
                                                  batch_size=1,
                                                  shuffle=False, 
                                                  seed=42)


# In[60]:


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
train_generator.n, STEP_SIZE_TRAIN, 
#valid_generator.n, STEP_SIZE_VALID


# In[71]:


from keras.applications import InceptionResNetV2, Xception 
conv_base = Xception(include_top=False, input_shape=(img_sz,img_sz,3))


# import keras
# import keras_applications
# conv_base = keras_applications.resnext.ResNeXt101(include_top = False, weights = 'imagenet', input_shape=(224,224,3),
#                                                  backend = keras.backend, layers = keras.layers, models = keras.models, 
#                                                  utils = keras.utils)

# In[72]:


import keras
from keras.models import Sequential
from keras import regularizers, initializers
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tqdm import TQDMCallback, TQDMNotebookCallback

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(64, activation='relu', 
                kernel_initializer=initializers.he_uniform(seed=None),
                kernel_regularizer=regularizers.l2(0.01)))  
model.add(Dense(5, activation='softmax'))


# In[73]:


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0001,rho=0.9, epsilon=None, decay=0.0),
              metrics=['acc'])


# In[74]:


history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples/train_generator.batch_size,
                              epochs=7,
                              shuffle=True,
                              #callbacks=[earlystopper,checkpointer,reduce_lr],
                              #validation_data=valid_generator,
                              #validation_steps=valid_generator.samples/valid_generator.batch_size,
                              verbose=1)


# ### Predict on test set

# In[83]:


test_generator.reset()
pred = model.predict_generator(test_generator, steps=2680, verbose=1)


# In[84]:


predicted_class_indices = np.argmax(pred, axis=1)

labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[85]:


test_df['category'] = pd.DataFrame(data=predictions)


# In[86]:


test_df.head()


# In[87]:


# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

test_df['category'] = pd.DataFrame(data=predictions)

# create a link to download the dataframe
create_download_link(test_df)


# In[ ]:





# In[43]:


import cv2
import os

height = []
weight = []
for filename in os.listdir('../input/gameofdl/test/test_images/'):
    img = cv2.imread(os.path.join('../input/gameofdl/test/test_images/',filename))
    height.append(img.shape[0])
    weight.append(img.shape[1])


# In[44]:


max(height), max(weight)


# In[45]:


min(height), min(weight)


# In[ ]:




