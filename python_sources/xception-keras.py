#!/usr/bin/env python
# coding: utf-8

# **Transfer learning from pretrained model using Keras.**

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 


# ### Train data

# In[ ]:


ann_file = '../input/inaturalist-2019-fgvc6/train2019.json'
with open(ann_file) as data_file:
        train_anns = json.load(data_file)


# In[ ]:


train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)
df_train_file_cat.head()


# In[ ]:


len(df_train_file_cat['category_id'].unique())


# In[ ]:


# Example of images for category_id = 400
img_names = df_train_file_cat[df_train_file_cat['category_id']=='400']['file_name'][:30]

plt.figure(figsize=[15,15])
i = 1
for img_name in img_names:
    img = cv2.imread("../input/inaturalist-2019-fgvc6/train_val2019/%s" % img_name)[...,[2, 1, 0]]
    plt.subplot(6, 5, i)
    plt.imshow(img)
    i += 1
plt.show()


# ### Validation data

# In[ ]:


valid_ann_file = '../input/inaturalist-2019-fgvc6/val2019.json'
with open(valid_ann_file) as data_file:
        valid_anns = json.load(data_file)


# In[ ]:


valid_anns_df = pd.DataFrame(valid_anns['annotations'])[['image_id','category_id']]
valid_anns_df.head()


# In[ ]:


valid_img_df = pd.DataFrame(valid_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
valid_img_df.head()


# In[ ]:


df_valid_file_cat = pd.merge(valid_img_df, valid_anns_df, on='image_id')
df_valid_file_cat['category_id']=df_valid_file_cat['category_id'].astype(str)
df_valid_file_cat.head()


# In[6]:


nb_classes = 1010
batch_size = 64
img_size = 80
nb_epochs = 30


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_datagen=ImageDataGenerator(rescale=1./255, \n    validation_split=0.25,\n    horizontal_flip = True,    \n    zoom_range = 0.3,\n    width_shift_range = 0.3,\n    height_shift_range=0.3\n    )\n\ntrain_generator=train_datagen.flow_from_dataframe(\n    dataframe=df_train_file_cat,\n    directory="../input/inaturalist-2019-fgvc6/train_val2019",\n    x_col="file_name",\n    y_col="category_id",\n    batch_size=batch_size,\n    shuffle=True,\n    class_mode="categorical",    \n    target_size=(img_size,img_size))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_datagen = ImageDataGenerator(rescale=1./255)\n\nvalid_generator=test_datagen.flow_from_dataframe(\n    dataframe=df_valid_file_cat,\n    directory="../input/inaturalist-2019-fgvc6/train_val2019",\n    x_col="file_name",\n    y_col="category_id",\n    batch_size=batch_size,\n    shuffle=True,\n    class_mode="categorical",    \n    target_size=(img_size,img_size))')


# ### Model

# In[7]:


model = applications.Xception(weights='imagenet', 
                              include_top=False, 
                              input_shape=(img_size, img_size, 3))
#model.load_weights('../input/NASNet-large-no-top/NASNet-large-no-top.h5')
#model.summary()


# In[ ]:


# Freeze last 5 layers
for layer in model.layers[:-5]:
    layer.trainable = False


# In[ ]:


#Adding custom layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)

model_final.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


# Callbacks

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=7, verbose=2, mode='auto')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model_final.fit_generator(generator=train_generator,                   \n                                    steps_per_epoch=500,\n                                    validation_data=valid_generator,                    \n                                    validation_steps=200,\n                                    epochs=nb_epochs,\n                                    callbacks = [checkpoint, early],\n                                    verbose=2)')


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# ### Test data

# In[ ]:


test_ann_file = '../input/inaturalist-2019-fgvc6/test2019.json'
with open(test_ann_file) as data_file:
        test_anns = json.load(data_file)


# In[ ]:


test_img_df = pd.DataFrame(test_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
test_img_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntest_generator = test_datagen.flow_from_dataframe(      \n    \n        dataframe=test_img_df,    \n    \n        directory = "../input/inaturalist-2019-fgvc6/test2019",    \n        x_col="file_name",\n        target_size = (img_size,img_size),\n        batch_size = 1,\n        shuffle = False,\n        class_mode = None\n        )')


# ### Prediction

# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npredict=model_final.predict_generator(test_generator, steps = len(test_generator.filenames))')


# In[ ]:


len(predict)


# In[ ]:


predicted_class_indices=np.argmax(predict,axis=1)


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:


sam_sub_df = pd.read_csv('../input/inaturalist-2019-fgvc6/kaggle_sample_submission.csv')
sam_sub_df.head()


# In[ ]:


filenames=test_generator.filenames
results=pd.DataFrame({"file_name":filenames,
                      "predicted":predictions})
df_res = pd.merge(test_img_df, results, on='file_name')[['image_id','predicted']]    .rename(columns={'image_id':'id'})

df_res.head()


# In[ ]:


df_res.to_csv("submission.csv",index=False)

