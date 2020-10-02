#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###########################################IMPORTS AND PREPROCESSING ##################################################
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import seaborn as sns
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas.util.testing as tm
import PIL  
from PIL import Image  




AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

#*Updating to the final Train_paths_images and Test_paths_images*



path='../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')
sub = pd.read_csv(path + '/sample_submission.csv')


train_labels = train.loc[:,'healthy':].values
train_labels_healthy = train.loc[:,'healthy'].values
train_labels_multiple_diseases = train.loc[:,'multiple_diseases'].values
train_labels_rust = train.loc[:,'rust'].values
train_labels_scab = train.loc[:,'scab'].values




path_gc_train = '../input/kaggle-plant-pathology-1/images_gc_train-20200524T112401Z-001/images_gc_train/'
path_gc_test = '../input/kaggle-plant-pathology-1/images_gc_test-20200524T112222Z-001/images_gc_test/'


gcs_path_train = 'gs://plant-pathology-bhavesh/plant-pathology-2020-fgvc7/images_gc_train/'
gcs_path_test = 'gs://plant-pathology-bhavesh/plant-pathology-2020-fgvc7/images_gc_test/'



Train_paths_gc = train.image_id.apply(lambda x: gcs_path_train + str(x) +'.jpg').values
Test_paths_gc = test.image_id.apply(lambda x: gcs_path_test + str(x) +'.jpg').values

train_paths_gc = train.image_id.apply(lambda x: path + '/images/' + str(x) +'.jpg').values
test_paths_gc = test.image_id.apply(lambda x: path + '/images/' + str(x) +'.jpg').values

path2 = 'gs://plant-pathology-bhavesh/plant-pathology-2020-fgvc7/images_withborder_train/'
path3 = 'gs://plant-pathology-bhavesh/plant-pathology-2020-fgvc7/images_withborder_test/'

filenames = get_ipython().getoutput('gsutil ls -r gs://plant-pathology-bhavesh/plant-pathology-2020-fgvc7/images_withborder_train/')

int_files = []
for id in range(len(filenames)):
  int_files.append(filenames[id].split('/')[-1].split('.')[0])

for id,img in enumerate(Train_paths_gc):
  if img.split('.')[0].split('/')[-1] in int_files:
    Train_paths_gc[id] = os.path.join(path2,'Train_'+str(id)+'.jpg')


filenames = get_ipython().getoutput('gsutil ls -r gs://plant-pathology-bhavesh/plant-pathology-2020-fgvc7/images_withborder_test/')

int_files = []
for id in range(len(filenames)):
  int_files.append(filenames[id].split('/')[-1].split('.')[0])

for id,img in enumerate(Test_paths_gc):
  if img.split('.')[0].split('/')[-1] in int_files:
    Test_paths_gc[id] = os.path.join(path2,'Train_'+str(id)+'.jpg')



#################### 2
get_ipython().system('pip install -U git+https://github.com/qubvel/efficientnet')
import efficientnet.tfkeras as efn 


import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D , Input
from tensorflow.keras.optimizers import Adam , RMSprop 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler

import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras import optimizers
print(tf.__version__)
print(tf.keras.__version__)

from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet
#for reproducible results
#import random
#seed_value = 13
#random.seed(seed_value)
#np.random.seed(seed_value)
#tf.random.set_seed(seed_value)


#################### 5
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
train_labels_healthy_one_hot = ohe.fit_transform(np.array(train_labels_healthy).reshape(-1,1)).toarray()
train_labels_multiple_diseases = train.loc[:,'multiple_diseases'].values
train_labels_rust = train.loc[:,'rust'].values
train_labels_scab = train.loc[:,'scab'].values

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
train_labels_multiple_diseases_one_hot = ohe.fit_transform(np.array(train_labels_multiple_diseases).reshape(-1,1)).toarray()
train_labels_rust_one_hot = ohe.fit_transform(np.array(train_labels_rust).reshape(-1,1)).toarray()
train_labels_scab_one_hot = ohe.fit_transform(np.array(train_labels_scab).reshape(-1,1)).toarray()


IMG_SIZE = 256

def decode_image(filename, label=None, image_size=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)

    
    #convert to numpy and do some cv2 staff mb?
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None, seed=5050):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
           
    if label is None:
        return image
    else:
        return image, label
    
y_train = np.array(train.loc[:,'healthy':])
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(Train_paths_gc,y_train,test_size=0.2,shuffle=True)
    

BATCH_SIZE = 8*strategy.num_replicas_in_sync
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train,y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )

val_dataset = (tf.data.Dataset
               .from_tensor_slices((x_val,y_val))
               .map(decode_image,num_parallel_calls=AUTO)
               .batch(BATCH_SIZE)
               .cache()
               .prefetch(AUTO)
              )

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(Test_paths_gc)
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )
IMG_SIZE = 256
BATCH_SIZE = 8*strategy.num_replicas_in_sync
nb_classes = 4


# In[ ]:


y_train.astype(float)


# In[ ]:


##############################Binary Classification followed by FunctionalAPI#############################################
Train_images_gc = []
for id,img in enumerate(Train_paths_gc):
  image = cv2.imread(img)
    #image = cv2.rotate(image, cv2.ROTATE_180)
  Train_images_gc.append(image/255)
  if id%100==0:
     print(id)
print('0')



#print('0')

#history = model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/content/drive/My Drive/Kaggle - Plant Pathology/binaryhealthy_7.h5')
print('2')

#############################################################BINARY MODEL#################################################

#model.load_weights('/content/drive/My Drive/Kaggle - Plant Pathology/binaryhealthy_7.h5')
#history = model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/content/drive/My Drive/Kaggle - Plant Pathology/binaryhealthy_14.h5')
print('3')

#model = efn.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(256,256, 3))
#model = Sequential([model])
#model.add(Dense(2,activation='softmax'))
#adam = Adam(learning_rate=0.0003)
#model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
#print('4')
#model.load_weights('/content/drive/My Drive/Kaggle - Plant Pathology/binaryhealthy_.h5')
#history = model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=4,batch_size=64,validation_split=0.1)



#Latest Weights binary classification
#model.save_weights('/content/drive/My Drive/Kaggle - Plant Pathology/binaryhealthy_25.h5')
#print('5')



####################################################################Multiclass Model###############################################


X1 = efn.EfficientNetB0(weights='imagenet',include_top = False,input_shape=(256,256,3),pooling='avg')
X0 = Dense(1024,activation='sigmoid')(X1.layers[-1].output)
X02 = Dense(128,activation='relu')(X0)
X2 = Dense(2,activation='softmax',name='x2_1')(X02)
X3 = Dense(2,activation='softmax',name='x2_2')(X02)
X4 = Dense(2,activation='softmax',name='x2_3')(X02)

model = Model(X1.inputs,[X2,X3,X4])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('1')
#model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
#          epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/multiclass_7_extralayers.h5')

#print('2')
#model.load_weights('/kaggle/working/multiclass_7_extralayers.h5')

#model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
#          epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/multiclass_14_extralayers.h5')

#print('3')
#model.load_weights('/kaggle/working/multiclass_14_extralayers.h5')

#model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
#          epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/multiclass_21_extralayers.h5')

#print('4')
adam = Adam(learning_rate=0.0003)
#model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

#model.load_weights('/kaggle/working/multiclass_21_extralayers.h5')

#model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
#          epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/multiclass_28_extralayers.h5')

print('5')
#model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
#model.load_weights('/kaggle/working/multiclass_28_extralayers.h5')

#model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
#          epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/multiclass_35_extralayers.h5')

print('6')

model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
model.load_weights('/kaggle/working/multiclass_35_extralayers.h5')

model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
          epochs=5,batch_size=64,validation_split=0.1)
model.save_weights('/kaggle/working/multiclass_40_extralayers.h5')


# In[ ]:


Train_images_gc = []
for id,img in enumerate(Train_paths_gc):
  image = cv2.resize(cv2.imread(img),(450,450),cv2.INTER_AREA)
    #image = cv2.rotate(image, cv2.ROTATE_180)
  Train_images_gc.append(image/255)
  if id%100==0:
     print(id)
print('0')

X1 = efn.EfficientNetB0(weights='imagenet',include_top = False,input_shape=(256,256,3),pooling='avg')
X0 = Dense(1024,activation='sigmoid')(X1.layers[-1].output)
X02 = Dense(128,activation='relu')(X0)
X2 = Dense(2,activation='softmax',name='x2_1')(X02)
X3 = Dense(2,activation='softmax',name='x2_2')(X02)
X4 = Dense(2,activation='softmax',name='x2_3')(X02)

model = Model(X1.inputs,[X2,X3,X4])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(np.array(Train_images_gc),{'x2_1':train_labels_multiple_diseases_one_hot,'x2_2':train_labels_rust_one_hot,'x2_3':train_labels_scab_one_hot},
          epochs=40,batch_size=64,validation_split=0.1)


plt.plot(history.history['val_x2_1_accuracy'])
plt.xlabel('cross val acc')
plt.ylabel('epochs')
plt.show()
plt.plot(history.history['val_x2_2_accuracy'])
plt.xlabel('cross val acc')
plt.ylabel('epochs')
plt.show()
plt.plot(history.history['val_x2_3_accuracy'])
plt.xlabel('cross val acc')
plt.ylabel('epochs')
plt.show()


# In[ ]:


#multiclass_28_extralayers.h5 are the best multiclass weights


# In[ ]:


Train_images_gc = []
for id,img in enumerate(Train_paths_gc):
  image = cv2.imread(img)
    #image = cv2.rotate(image, cv2.ROTATE_180)
  Train_images_gc.append(image/255)
  if id%100==0:
     print(id)
print('0')

model = efn.EfficientNetB0(weights='imagenet',include_top = False, input_shape = (256,256,3),pooling = 'avg')
model = Sequential([model])
model.add(Dense(1024,activation = 'sigmoid'))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(2,activation = 'softmax'))
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])




print('0')

#model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/binaryhealthy_7.h5')
print('2')

#############################################################BINARY MODEL#################################################

#model.load_weights('/kaggle/working/binaryhealthy_7.h5')
#model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=7,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/binaryhealthy_14.h5')
#print('3')


print('4')
#model.load_weights('/kaggle/working/binaryhealthy_14.h5')
#model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=4,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/binaryhealthy_21.h5')
adam = Adam(learning_rate = 0.0003)
#print('5')
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])


#model.load_weights('/kaggle/working/binaryhealthy_21.h5')
#model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=3,batch_size=64,validation_split=0.1)
#model.save_weights('/kaggle/working/binaryhealthy_24.h5')


model.load_weights('/kaggle/working/binaryhealthy_24.h5')
model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=3,batch_size=64,validation_split=0.1)
model.save_weights('/kaggle/working/binaryhealthy_27.h5')

model.load_weights('/kaggle/working/binaryhealthy_27.h5')
model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=3,batch_size=64,validation_split=0.1)
model.save_weights('/kaggle/working/binaryhealthy_30.h5')


# In[ ]:


Train_images_gc = []
for id,img in enumerate(Train_paths_gc):
  image = cv2.imread(img)
    #image = cv2.rotate(image, cv2.ROTATE_180)
  Train_images_gc.append(image/255)
  if id%100==0:
     print(id)
print('0')

model = efn.EfficientNetB0(weights='imagenet',include_top = False, input_shape = (256,256,3),pooling = 'avg')
model = Sequential([model])
model.add(Dense(1024,activation = 'sigmoid'))
model.add(Dense(128,activation = 'relu'))
model.add(Dense(2,activation = 'softmax'))
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])

history = model.fit(np.array(Train_images_gc),train_labels_healthy_one_hot,epochs=40,batch_size=64,validation_split=0.1)


plt.plot(history.history['val_accuracy'])
plt.xlabel('cross val acc')
plt.ylabel('epochs')
plt.show()


# In[ ]:


#binaryhealthy27 are the best binary model weights


# In[ ]:


############################################################MAKING PREDICTIONS#################################################################


###################BINARY MODEL###################################

model_1 = efn.EfficientNetB0(weights='imagenet',include_top = False, input_shape = (256,256,3),pooling = 'avg')
model_1 = Sequential([model_1])
model_1.add(Dense(1024,activation = 'sigmoid'))
model_1.add(Dense(128,activation = 'relu'))
model_1.add(Dense(2,activation = 'softmax'))
model_1.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])

model_1.load_weights('../input/weights/binaryhealthy_27.h5')

print('0')
#################MULTICLASS MODEL##################################

X1 = efn.EfficientNetB0(weights='imagenet',include_top = False,input_shape=(256,256,3),pooling='avg')
X0 = Dense(1024,activation='sigmoid')(X1.layers[-1].output)
X02 = Dense(128,activation='relu')(X0)
X2 = Dense(2,activation='softmax',name='x2_1')(X02)
X3 = Dense(2,activation='softmax',name='x2_2')(X02)
X4 = Dense(2,activation='softmax',name='x2_3')(X02)

model_2 = Model(X1.inputs,[X2,X3,X4])
model_2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model_2.load_weights('../input/weights/multiclass_28_extralayers.h5')

print('2')
####################################################################TEST DATASET#############################################
Test_images_gc = []
for id,img in enumerate(Test_paths_gc):
  image = cv2.imread(img)
    #image = cv2.rotate(image, cv2.ROTATE_180)
  Test_images_gc.append(image/255)
  if id%100==0:
     print(id)
print('3')



################################################Looping Predictions############################################### NOT WORKING FOR SINGLE IMAGE
#predictions = pd.DataFrame(np.zeros((1821,4)).astype(int),columns=['healthy','multiple_diseases','rust','scab'])

#for id,image in enumerate(Test_images_gc):
#    predict = model_1.predict(np.array(image))
#    Predict = ohe.inverse_transform(predict)
#    predictions['healthy'][id] = Predict
#    if predictions['healthy'][id] == 1:
#        continue
#    else:
#        [predict1,predict2,predict3] = model_2.predict(np.array(image))
#        Predict1 = ohe.inverse_transform(predict1)
#        Predict2 = ohe.inverse_transform(predict2)
#        Predict3 = ohe.inverse_transform(predict3)

#        predictions['multiple_diseases'][id] = Predict1
#        predictions['rust'][id] = Predict2
#        predictions['scab'][id] = Predict3
#    if id%100 ==0:
#        print(id)
#predictions.to_csv('/kaggle/working/Submission23:39.csv',index=False)

##################################################

predict = model_1.predict(np.array(Test_images_gc))
Predict = ohe.inverse_transform(predict)
Predictions = pd.concat([test,pd.DataFrame(Predict)],axis=1)



[predict1,predict2,predict3] = model_2.predict(np.array(Test_images_gc))
Predict1 = ohe.inverse_transform(predict1)
Predict2 = ohe.inverse_transform(predict2)
Predict3 = ohe.inverse_transform(predict3)
Predictions = pd.concat([Predictions,pd.DataFrame(Predict1),pd.DataFrame(Predict2),pd.DataFrame(Predict3)],axis=1)
Predictions.rename(columns = {0:'image_id',1:'healthy',2:'multiple_diseases',3:'rust',4:'scab'}).to_csv('/kaggle/working/SubmissionNonLoop.csv',index=False)


# In[ ]:


predictions = pd.DataFrame(np.zeros((1821,4)).astype(int),columns=['healthy','multiple_diseases','rust','scab'])
predictions['healthy'][1] = 2
predictions


# In[ ]:



for id in range(Predictions.shape[0]):
    if Predictions.iloc[id,1] == 1:
        Predictions.iloc[id,2] = 0
        Predictions.iloc[id,3] = 0
        Predictions.iloc[id,4] = 0


# In[ ]:


Predictions.rename(columns = {0:'image_id',1:'healthy',2:'multiple_diseases',3:'rust',4:'scab'}).to_csv('/kaggle/working/SubmissionNonLoop2.csv',index=False)


# In[ ]:


#######################SEPARATE PREDICTIONS ARE FAILING#####################################
Train_images_gc = []
for id,img in enumerate(train_paths_gc):
  image = cv2.resize(cv2.imread(img),(256,256),cv2.INTER_AREA)
    #image = cv2.random_rotate(image, cv2.ROTATE_180)
  Train_images_gc.append(image/255)
  if id%100==0:
     print(id)
print('0')



###########################APPLY THEM IN TOGETHER FORMAT##################################
X1 = efn.EfficientNetB2(weights='imagenet',include_top = False,input_shape=(256,256,3),pooling='avg')
X0 = Dense(1024,activation='sigmoid')(X1.layers[-1].output)
X02 = Dense(128,activation='relu')(X0)
X2 = Dense(2,activation='softmax',name='x2_1')(X02)
X3 = Dense(2,activation='softmax',name='x2_2')(X02)
X4 = Dense(2,activation='softmax',name='x2_3')(X02)
X5 = Dense(2,activation='softmax',name='x2_4')(X02)

model = Model(X1.inputs,[X2,X3,X4,X5])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



print('1')
#model.fit(np.array(Train_images_gc),{'x2_1':train_labels_healthy_one_hot,'x2_2':train_labels_multiple_diseases_one_hot,'x2_3':train_labels_rust_one_hot,'x2_4':train_labels_scab_one_hot},
#          epochs=20,batch_size=64,validation_split=0.1,shuffle=True)
#model.save_weights('/kaggle/working/multiclass_20_extralayers.h5')

print('2')
model.load_weights('/kaggle/working/multiclass_20_extralayers.h5')

model.fit(np.array(Train_images_gc),{'x2_1':train_labels_healthy_one_hot,'x2_2':train_labels_multiple_diseases_one_hot,'x2_3':train_labels_rust_one_hot,'x2_4':train_labels_scab_one_hot},
          epochs=5,batch_size=64,validation_split=0.1,shuffle=True)
model.save_weights('/kaggle/working/multiclass_25_extralayers.h5')

print('3')
adam = Adam(learning_rate=0.0003)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
model.load_weights('/kaggle/working/multiclass_25_extralayers.h5')

model.fit(np.array(Train_images_gc),{'x2_1':train_labels_healthy_one_hot,'x2_2':train_labels_multiple_diseases_one_hot,'x2_3':train_labels_rust_one_hot,'x2_4':train_labels_scab_one_hot},
          epochs=5,batch_size=64,validation_split=0.1,shuffle=True)
model.save_weights('/kaggle/working/multiclass_30_extralayers.h5')

print('4')
adam = Adam(learning_rate=0.0003)
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

model.load_weights('/kaggle/working/multiclass_30_extralayers.h5')

model.fit(np.array(Train_images_gc),{'x2_1':train_labels_healthy_one_hot,'x2_2':train_labels_multiple_diseases_one_hot,'x2_3':train_labels_rust_one_hot,'x2_4':train_labels_scab_one_hot},
          epochs=3,batch_size=64,validation_split=0.1,shuffle=True)
model.save_weights('/kaggle/working/multiclass_33_extralayers.h5')

print('5')
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
model.load_weights('/kaggle/working/multiclass_33_extralayers.h5')

model.fit(np.array(Train_images_gc),{'x2_1':train_labels_healthy_one_hot,'x2_2':train_labels_multiple_diseases_one_hot,'x2_3':train_labels_rust_one_hot,'x2_4':train_labels_scab_one_hot},
          epochs=3,batch_size=64,validation_split=0.1,shuffle=True)
model.save_weights('/kaggle/working/multiclass_36_extralayers.h5')

print('6')

model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
model.load_weights('/kaggle/working/multiclass_36_extralayers.h5')

model.fit(np.array(Train_images_gc),{'x2_1':train_labels_healthy_one_hot,'x2_2':train_labels_multiple_diseases_one_hot,'x2_3':train_labels_rust_one_hot,'x2_4':train_labels_scab_one_hot},
          epochs=3,batch_size=64,validation_split=0.1,shuffle=True)
model.save_weights('/kaggle/working/multiclass_39_extralayers.h5')


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

model_name = 'effNetPlants.h5'

#good callbacks
best_model = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True,mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,min_lr=0.000001,patience=6)

BATCH_SIZE = 64



def get_model():
    base_model = efn.EfficientNetB7(weights='imagenet',
                          include_top=False,
                          input_shape=(IMG_SIZE,IMG_SIZE, 3),
                          pooling='avg')
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)

with strategy.scope():
    model = get_model()

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['categorical_accuracy'])
print('2')

import tensorflow as tf, tensorflow.keras.backend as K


history = model.fit(train_dataset,
                    steps_per_epoch = train.shape[0]//BATCH_SIZE,
                    epochs=40,
                    verbose=1,
                    validation_data=val_dataset,
                    callbacks=[reduce_lr,best_model]
                    )

model.save_weights('/kaggle/working/tpu40.h5')
plt.title('model accuracy')
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



# In[ ]:


train.shape[0]


# In[ ]:


plt.plot(history.history['val_accuracy'])
plt.xlabel('cross val acc sigmoid 40 epochs 450 img size')
plt.ylabel('epochs')
plt.show()



# In[ ]:


model.save_weights('/kaggle/working/MultiClassWeights.h5')


# In[ ]:








#model.load_weights('../input/qwerty/categorical.h5')    #######################BEST WEIGHTS

predict1 = model.predict(test_dataset,verbose=1)
#Predict1 = ohe.inverse_transform(predict1)
#Predict2 = ohe.inverse_transform(predict2)
#Predict3 = ohe.inverse_transform(predict3)
#Predict4 = ohe.inverse_transform(predict4)


# In[ ]:


predict1 = model.predict(test_dataset,steps=test.shape[0]//64,verbose=1)
Predictions = pd.concat([test,pd.DataFrame(predict1)],axis=1)
#for id in range(Predictions.shape[0]):
#    if Predictions.iloc[id,1] == 1:
#        Predictions.iloc[id,2] = 0
#        Predictions.iloc[id,3] = 0
#        Predictions.iloc[id,4] = 0
#    else:
#        continue

Predictions.rename(columns = {0:'image_id',1:'healthy',2:'multiple_diseases',3:'rust',4:'scab'}).to_csv('/kaggle/working/SubmissionMultiClass.csv',index=False)
print('1')
###########################Predictions on which i am working are downloaded as SubmissionMultiClass.csv###################################################3
Predictions_readable = pd.DataFrame(np.zeros((1821,4)),columns=['healthy','multiple_diseases','rust','scab'])
for id in range(Predictions.shape[0]):
    if Predictions.iloc[id,1] >= 0.125:
        Predictions_readable['healthy'][id] = 1
    else:
        Predictions_readable['healthy'][id] = 0
        
        
    if Predictions.iloc[id,2] >= 0.125:
        Predictions_readable['multiple_diseases'][id] = 1
    else:
        Predictions_readable['multiple_diseases'][id] = 0
        
        
    if Predictions.iloc[id,3] >= 0.125:
        Predictions_readable['rust'][id] = 1
    else:
        Predictions_readable['rust'][id] = 0
        
        
    if Predictions.iloc[id,4] >= 0.125:
        Predictions_readable['scab'][id] = 1
    else:
        Predictions_readable['scab'][id] = 0
    if id%100 == 0:
        print(id)
Predictions_readable = pd.concat([test,Predictions_readable],axis=1)
Predictions_readable.to_csv('/kaggle/working/SubSubSub.csv',index=False)
print('2')


# In[ ]:


model.save_weights('/kaggle/working/multiclass50epochs.h5')


# In[ ]:


test_dataset


# In[ ]:


for id,val in enumerate(Predictions):
    if Predictions.iloc[id,1] >= 0.5:
        Predictions.iloc[id,1] = 1
    else:
        Predictions.iloc[id,1] = 0
    
    if Predictions.iloc[id,2] >= 0.5:
        Predictions.iloc[id,2] = 1
    else:
        Predictions.iloc[id,2] = 0
        
    if Predictions.iloc[id,3] >= 0.5:
        Predictions.iloc[id,3] = 1
    else:
        Predictions.iloc[id,3] = 0 
        
    if Predictions.iloc[id,4] >= 0.5:
        Predictions.iloc[id,4] = 1
    else:
        Predictions.iloc[id,4] = 0


# In[ ]:


Predictions = pd.concat([test,pd.DataFrame(predict1)],axis=1)

Predictions


# In[ ]:


Predictions.rename(columns = {0:'image_id',1:'healthy',2:'multiple_diseases',3:'rust',4:'scab'}).to_csv('/kaggle/working/SubmissionMultiClass.csv',index=False)


# In[ ]:


Predictions.iloc[2,1]


# In[ ]:


###########################Predictions on which i am working are downloaded as SubmissionMultiClass.csv###################################################3
Predictions_readable = pd.DataFrame(np.zeros((1821,4)),columns=['healthy','multiple_diseases','rust','scab'])
for id in range(Predictions.shape[0]):
    if Predictions.iloc[id,1] >= 0.125:
        Predictions_readable['healthy'][id] = 1
    else:
        Predictions_readable['healthy'][id] = 0
        
        
    if Predictions.iloc[id,2] >= 0.125:
        Predictions_readable['multiple_diseases'][id] = 1
    else:
        Predictions_readable['multiple_diseases'][id] = 0
        
        
    if Predictions.iloc[id,3] >= 0.125:
        Predictions_readable['rust'][id] = 1
    else:
        Predictions_readable['rust'][id] = 0
        
        
    if Predictions.iloc[id,4] >= 0.125:
        Predictions_readable['scab'][id] = 1
    else:
        Predictions_readable['scab'][id] = 0
    if id%100 == 0:
        print(id)
Predictions_readable = pd.concat([test,Predictions_readable],axis=1)


# In[ ]:


Predictions_readable


# In[ ]:


Predictions_readable.to_csv('/kaggle/working/SubSubSub.csv',index=False)


# In[ ]:





# In[ ]:


image = cv2.imread(Train_paths_gc[1])
image


# In[ ]:


path2 = '../input/kaggle-plant-pathology-1/images_withborder_train-20200524T112202Z-001/images_withborder_train'
path3 = '../input/kaggle-plant-pathology-1/images_withborder_test-20200524T113820Z-001/images_withborder_test'

filenames = os.listdir(path2)

int_files = []
for id in range(len(filenames)):
  int_files.append(filenames[id].split('.')[0])

for id,img in enumerate(Train_paths_gc):
  if img.split('.')[0].split('/')[-1] in int_files:
    Train_paths_gc[id] = os.path.join(path2,'Train_'+str(id)+'.jpg')


# In[ ]:


plt.imshow(cv2.imread(Train_paths_gc[1]))
plt.show()


# In[ ]:




