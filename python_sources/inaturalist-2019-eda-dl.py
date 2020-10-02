#!/usr/bin/env python
# coding: utf-8

# # iNaturalist 2019 EDA + DL
# 
# As part of the FGVC6 workshop at CVPR 2019, Kaggle is conducting the iNat Challenge 2019, the large scale species classification competition, sponsored by Microsoft. It is estimated that the natural world contains several million species of plants and animals. Without expert knowledge, many of these species are extremely difficult to accurately classify due to their visual similarity. The goal of this competition is to push the state of the art in automatic image classification for real world data that features a large number of fine-grained categories.
# 
# This Kernel will use the idea of "Transfer Learning", various pre-trained model will be used be used for the problem of multiclassification.
# 
# Kudos and main ideas / reference: 
# 
# - [hsinwenchang/keras-data-augmentation-visualize](https://www.kaggle.com/hsinwenchang/keras-data-augmentation-visualize/notebook)
# - [ateplyuk/inat2019-starter-keras-efficientnet](https://www.kaggle.com/ateplyuk/inat2019-starter-keras-efficientnet)

# ![](https://www.bahai.org/chrome/img/beliefs/nature-feature-img.jpg?f0550045)
# 
# [image-source](https://www.bahai.org/chrome/img/beliefs/nature-feature-img.jpg?f0550045)

# ### Loading the necessary libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

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


# ### Train data

# In[ ]:


ann_file = '../input/train2019.json'
with open(ann_file) as data_file:
        train_anns = json.load(data_file)


# In[ ]:


train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)
df_train_file_cat.head()


# In[ ]:


df_train_file_cat.shape


# In[ ]:


len(df_train_file_cat['category_id'].unique())


# In[ ]:


# Example of images for category_id = 400
img_names = df_train_file_cat[df_train_file_cat['category_id']=='400']['file_name'][:30]

plt.figure(figsize=[15,15])
i = 1
for img_name in img_names:
    img = cv2.imread("../input/train_val2019/%s" % img_name)[...,[2, 1, 0]]
    plt.subplot(6, 5, i)
    plt.imshow(img)
    i += 1
plt.show()


# ### Validation data

# In[ ]:


valid_ann_file = '../input/val2019.json'
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


# In[ ]:


nb_classes = 1010
batch_size = 128
img_size = 150
nb_epochs = 5


# ### Oversampling

# In[ ]:


#from imblearn.over_sampling import RandomOverSampler

#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_resample(df_train_file_cat[["image_id", "file_name"]], df_train_file_cat["category_id"])

#train_df = pd.DataFrame(X_resampled, columns=["image_id", "file_name"])
#train_df["category_id"] = y_resampled


# here I applied Data Augmentation technic from [Udacity](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb#scrollTo=UOoVpxFwVrWy) as following:
# - random 45 degree rotation
# - random zoom of up to 50%
# - random horizontal flip
# - width shift of 0.15
# - height shfit of 0.15

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=45, \n                    width_shift_range=.15, \n                    height_shift_range=.15, \n                    horizontal_flip=True, \n                    zoom_range=0.5)\n\ntrain_generator=train_datagen.flow_from_dataframe(\n    dataframe=df_train_file_cat,\n    directory="../input/train_val2019",\n    x_col="file_name",\n    y_col="category_id",\n    batch_size=batch_size,\n    shuffle=True,\n    class_mode="categorical",    \n    target_size=(img_size,img_size))')


# In[ ]:


# udacity_intro_to_tensorflow_for_deep_learning/l05c04_exercise_flowers_with_data_augmentation_solution.ipynb#scrollTo=jqb9OGoVKIOi
# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
    
    
augmented_images = [train_generator[0][0][0] for i in range(5)]
plotImages(augmented_images)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_datagen = ImageDataGenerator(rescale=1./255)\n\nvalid_generator=test_datagen.flow_from_dataframe(    \n    dataframe=df_valid_file_cat,    \n    directory="../input/train_val2019",\n    x_col="file_name",\n    y_col="category_id",\n    batch_size=batch_size,\n    shuffle=False,\n    class_mode="categorical",    \n    target_size=(img_size,img_size))')


# In[ ]:


import gc
gc.collect();


# ### Model
# 
# I have applied various transfer learning models in order to see which has the best performance, since the numerous categorical data and the little number of cases 

# In[ ]:


#from keras.applications.vgg16 import VGG16
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.nasnet import NASNetLarge
#from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception

#model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
model = Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
model_name = "Xception"


# In[ ]:


#Adding custom layers 
model_final = Sequential()
model_final.add(model)
model_final.add(Flatten())
model_final.add(Dense(1024, activation='relu'))
model_final.add(Dropout(0.5))
model_final.add(Dense(nb_classes, activation='softmax'))

model_final.compile(optimizers.rmsprop(lr=0.0001, decay=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


#Callbacks

checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')


# In[ ]:


model_final.summary()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model_final.fit_generator(generator=train_generator, \n                    steps_per_epoch=80,\n                    validation_data=valid_generator,\n                    validation_steps=40,\n                    epochs=nb_epochs,\n                    callbacks = [checkpoint, early],                \n                    verbose=1)')


# In[ ]:


import gc
gc.collect();


# In[ ]:


with open('history.json', 'w') as f:
    json.dump(history.history, f)

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['acc', 'val_acc']].plot()


# In[ ]:


test_ann_file = '../input/test2019.json'
with open(test_ann_file) as data_file:
        test_anns = json.load(data_file)


# In[ ]:


test_img_df = pd.DataFrame(test_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
test_img_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntest_generator = test_datagen.flow_from_dataframe(      \n    \n        dataframe=test_img_df,    \n    \n        directory = "../input/test2019",    \n        x_col="file_name",\n        target_size = (img_size,img_size),\n        batch_size = 1,\n        shuffle = False,\n        class_mode = None\n        )')


# ### Prediction

# In[ ]:


gc.collect();


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict_valid=model_final.predict_generator(valid_generator, steps = np.ceil(valid_generator.samples / valid_generator.batch_size), verbose=1)')


# In[ ]:


predict_valid_class=np.argmax(predict_valid,axis=1)


# In[ ]:


len(predict_valid_class)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#print(classification_report(valid_generator.classes, predict_valid_class))
print(accuracy_score(valid_generator.classes, predict_valid_class))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npredict=model_final.predict_generator(test_generator, steps = len(test_generator.filenames), verbose=1)')


# In[ ]:


predicted_class_indices=np.argmax(predict,axis=1)


# In[ ]:


gc.collect();


# In[ ]:


labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:


sam_sub_df = pd.read_csv('../input/kaggle_sample_submission.csv')
sam_sub_df.head()


# In[ ]:


sam_sub_df.shape


# In[ ]:


filenames=test_generator.filenames
results=pd.DataFrame({"file_name":filenames,
                      "predicted":predictions})
df_res = pd.merge(test_img_df, results, on='file_name')[['image_id','predicted']]    .rename(columns={'image_id':'id'})

df_res.head()


# In[ ]:


df_res.to_csv("submission.csv",index=False)


# In[ ]:




