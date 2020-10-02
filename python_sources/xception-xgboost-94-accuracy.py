#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import HTML

HTML("""
    <video alt="test" controls height="300px"
        <source src="https://drive.google.com/file/d/1mEFMHTaSxP1Hk8nKO4ZdS5ecWdAaQPjc/view" type="video/mp4">
    </video>
""")


# Flask App:
# Used this model for image recognition and Bert for question answering
# 
# Check out video of my flask app : https://rb.gy/dvvkm8
# 
# Google places api to get the nearby dermatologist

# # Importing the required Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
from tqdm import tqdm, tqdm_notebook
import random
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed


# # Importing the data

# In[ ]:


print(os.listdir('../input/skin-cancer-mnist-ham10000/'))


# In[ ]:


path = '../input/skin-cancer-mnist-ham10000/'
images_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*', '*.jpg'))}


# In[ ]:


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


skin_df = pd.read_csv(os.path.join(path,'HAM10000_metadata.csv'))


# In[ ]:


skin_df['path'] = skin_df['image_id'].map(images_path.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


# In[ ]:


skin_df.info()


# In[ ]:


skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((71,71))))


# now we have one extra column as image which has image as an array of shape (71,71,3)

# In[ ]:


skin_df.head()


# In[ ]:


# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()


# Taking X as image and y as cell_type_idx

# In[ ]:


X = skin_df['image']
y = skin_df['cell_type_idx']


# Data Normalization

# In[ ]:


X = X.values
X = X/255
X.shape


# In[ ]:


lst = []
for _ in X:
    lst.append(_)
X = np.array(lst)
print(X.shape)


# # Splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=28)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.20,random_state=28)


# ## One hot encoding of the y labels

# In[ ]:


from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 7)
y_val = to_categorical(y_val , num_classes=7)


# In[ ]:


batch_size = 256
train_input_shape = (71, 71, 3)
n_classes = 7


# In[ ]:


from tensorflow.keras.layers import Input


# # Building a model

# ## Used both ResNet and Xception 
# ## Xception performed better than ResNet50

# In[ ]:


# Load pre-trained model
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)
# input_tensor = Input(shape=(50,50,3))
base_model = Xception(include_top = False , input_shape = train_input_shape)


for layer in base_model.layers:
    layer.trainable = True


# In[ ]:


# Add layers at the end
model = base_model.output
model = Flatten()(model)

model = Dense(512, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Dense(128, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Dense(52, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Dense(16, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

output = Dense(n_classes, activation='softmax')(model)

model = Model(inputs=base_model.input, outputs=output)


# In[ ]:


optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])


# In[ ]:


n_epoch = 10

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                              verbose=1, mode='auto')


# #### for random upsampling using the class weights

# In[ ]:


# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.keras import balanced_batch_generator
# from tensorflow.keras.utils import Sequence

# class BalancedDataGenerator(Sequence):
#     """ImageDataGenerator + RandomOversampling"""
#     def __init__(self, x, y, datagen, batch_size=256):
#         self.datagen = datagen
#         self.batch_size = batch_size
#         self._shape = x.shape        
#         datagen.fit(x)
#         self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)

#     def __len__(self):
#         return self._shape[0] // self.batch_size

#     def __getitem__(self, idx):
#         x_batch, y_batch = self.gen.__next__()
#         x_batch = x_batch.reshape(-1, *self._shape[1:])
#         return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()


# In[ ]:


# datagen = ImageDataGenerator(
#         featurewise_center=True,# set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#         zoom_range = 0.1, # Randomly zoom image 
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=True)


# In[ ]:


# bgen = BalancedDataGenerator(x_train,y_train, datagen, batch_size=256)
# vgen = BalancedDataGenerator(x_val,y_val,datagen,batch_size=256)
# steps_per_epoch = bgen.steps_per_epoch


# # Training the Model

# In[ ]:


history = model.fit(x_train,y_train,epochs=60,
                              callbacks=[reduce_lr,early_stop],validation_data=(x_val,y_val))


# In[ ]:


fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# In[ ]:


model.summary()


# # Model to feature Extractor
# 

# In[ ]:


model2 = Model(model.input, model.layers[-7].output)
model2.summary()


# In[ ]:


predictions = model2.predict(X)


# In[ ]:


predictions[0]


# In[ ]:


predictions[10011]


# In[ ]:


len(predictions[0])


# from feature extractor we will get the output as np array of length 52

# In[ ]:


data_df = skin_df


# In[ ]:


complete_data = pd.concat([data_df, pd.DataFrame(predictions)], axis=1)


# In[ ]:


complete_data.head()


# # Saving the model

# Saving both the complete model and the feature extractor

# In[ ]:


### saving a model!!!!!

model_json = model2.to_json()
with open("model_v2.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:


### saving a model!!!!!

completemodel_json = model.to_json()
with open("completemodel1.json", "w") as json_file:
    json_file.write(completemodel_json)


# In[ ]:


model2.save_weights("model_v2_weights.h5")
print("Saved model to disk")


# In[ ]:


model.save_weights("completeweights.h5")
print("Saved model to disk")


# # Preparing a new Dataset

# In this new Dataset the image is changed to the 52 new features

# In[ ]:


complete_data.head()


# In[ ]:


complete_data.columns


# # One hot-encoding 

# In[ ]:


dxtype_df=pd.get_dummies(complete_data['dx_type'],drop_first=False)
complete_data=pd.concat([dxtype_df,complete_data],axis=1)
# complete_data.drop(['dx_type'],axis=1,inplace=True)
complete_data.head()


# In[ ]:


localization_df=pd.get_dummies(complete_data['localization'],drop_first=False)
complete_data=pd.concat([localization_df,complete_data],axis=1)
# complete_data.drop(['dx_type'],axis=1,inplace=True)
complete_data.head()


# In[ ]:


sex_df=pd.get_dummies(complete_data['sex'],drop_first=False)
sex_df.drop(['unknown'],axis=1,inplace=True)
complete_data=pd.concat([sex_df,complete_data],axis=1)
# complete_data.drop(['dx_type'],axis=1,inplace=True)
complete_data.head()


# In[ ]:


complete_data.columns


# In[ ]:


X_labels = complete_data.drop(['lesion_id','image_id','dx_type','dx','path','cell_type','cell_type_idx','sex','path','localization','image'],axis=1,inplace=False)
y_label = complete_data['cell_type_idx']


# In[ ]:


X_labels.head()


# # Saving a new Dataset

# In[ ]:


complete_data.to_csv('skin_data_v2.csv')


# In[ ]:


preds = model.predict(x_test)


# In[ ]:


lst = []
for a in preds:
    lst.append(np.argmax(a))


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# # CNN Model evaluation 

# In[ ]:


confusion_matrix(lst,y_test)


# In[ ]:


accuracy_score(lst,y_test)


# In[ ]:


print(classification_report(y_test,lst))


# In[ ]:


skin_df = pd.read_csv('skin_data_v2.csv')


# In[ ]:


skin_df.drop(['Unnamed: 0'],axis=1,inplace=True)
skin_df.head()


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret


# # Importing a new dataset

# In[ ]:


skin_df.head()


# In[ ]:


X_labels = skin_df.drop(['lesion_id','image_id','dx','dx_type','sex','localization','path','cell_type','cell_type_idx','image'],axis=1,inplace=False)
y_label = skin_df['cell_type_idx']


# In[ ]:


data_classification = pd.concat([X_labels,y_label],axis=1)
data_classification.fillna(data_classification['age'].mean(),inplace=True)


# In[ ]:


# import the classification module 
from pycaret import classification
# setup the environment 
classification_setup = classification.setup(data= data_classification, target='cell_type_idx')


# In[ ]:


# build the decision tree model
classification_dt = classification.create_model('dt')


# In[ ]:


# build the xgboost model
classification_xgb = classification.create_model('xgboost')


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_labels,y_label,train_size=0.8,random_state=40)


# In[ ]:


import xgboost as xgb


# In[ ]:


clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
                                            colsample_bylevel=1,
                                            colsample_bynode=1,
                                            colsample_bytree=1, gamma=0,
                                            learning_rate=0.1, max_delta_step=0,
                                            max_depth=3, min_child_weight=1,
                                            missing=None, n_estimators=100,
                                            n_jobs=-1, nthread=None,
                                            objective='binary:logistic',
                                            random_state=1855, reg_alpha=0,
                                            reg_lambda=1, scale_pos_weight=1,
                                            seed=None, silent=None, subsample=1,
                                            verbosity=0)


# In[ ]:


clf.fit(X_train,y_train)


# # CNN + Xgboost evaluation

# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# It gave accuracy of 94% and has the precision and recall of each label more than 0.76

# In[ ]:




