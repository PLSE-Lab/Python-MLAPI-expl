#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In this notebook we will try to make a classifier using pretrained Model**(Mobilnetv2)** trained on imagenet dataset and mould it to make a classifier to make predictions on this dataset

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os


# In[ ]:


data_dir = '../input/intel-image-classification'


# In[ ]:


os.listdir(data_dir)


# In[ ]:


test_dir = data_dir + '/seg_test/seg_test'


# In[ ]:


train_dir = data_dir + '/seg_train/seg_train'


# In[ ]:


pred_dir = data_dir + '/seg_pred/seg_pred'


# In[ ]:


plt.imshow(imread(train_dir+'/buildings/'+os.listdir(train_dir+'/buildings')[0])) # an example of a building


# In[ ]:


plt.imshow(imread(train_dir+'/forest/'+os.listdir(train_dir+'/forest')[0])) # an example of a forest


# In[ ]:


plt.imshow(imread(train_dir+'/glacier/'+os.listdir(train_dir+'/glacier')[0])) # an example of a glacier


# In[ ]:


plt.imshow(imread(train_dir+'/mountain/'+os.listdir(train_dir+'/mountain')[0])) # an example of a mountain


# In[ ]:


plt.imshow(imread(train_dir+'/sea/'+os.listdir(train_dir+'/sea')[0])) # an example of a sea


# In[ ]:


plt.imshow(imread(train_dir+'/street/'+os.listdir(train_dir+'/street')[0])) # an example of street


# In[ ]:


# this is done so if there are any image of variable shapes so we will reshape all of them to an average shape
dim1 = []
dim2 = []

for image_file in os.listdir(train_dir+'/buildings'):
    img = imread(train_dir+'/buildings/'+image_file)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)


# In[ ]:


height = int(np.average(d1))
height


# In[ ]:


width = int(np.average(d2))
width


# In[ ]:


img_shape = (height,width,3)


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[ ]:


# performing data augmentation and scaling on train set
image_gen_train = ImageDataGenerator(rescale=1/255,
                                    horizontal_flip=True,
                                    zoom_range=0.5,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10,
                                    fill_mode='nearest')


# In[ ]:


# performing scaling on test set
image_gen_test = ImageDataGenerator(rescale=1/255)


# In[ ]:


# returns an iterator of tuples of (x,y)
# here x is like our x_train and y like y_train
train_data_gen = image_gen_train.flow_from_directory(directory=train_dir,
                                                    class_mode='categorical',
                                                    batch_size=128,
                                                    color_mode='rgb',
                                                    shuffle=True,
                                                    target_size=img_shape[:2])


# In[ ]:


# returns an iterator of tuples of (x,y)
# here x is like our x_train and y like y_train
test_data_gen = image_gen_test.flow_from_directory(directory=test_dir,
                                                  class_mode='categorical',
                                                  color_mode='rgb',
                                                  batch_size=128,
                                                  target_size=img_shape[:2],
                                                  shuffle=False)


# In[ ]:


train_data_gen.class_indices


# In[ ]:


test_data_gen.class_indices


# In[ ]:


from tensorflow.keras.applications import MobileNetV2


# In[ ]:


# instantiating a base model which we will not be trained and we'll import this model along with trained weights and biases
# (actually this model was trained on imagenet dataset) we'll not include top of layers of that model because they are less generic
# instead we'll add on more layers later so that this model could make predictions on this dataset.
base_model = MobileNetV2(include_top=False,
                        weights='imagenet',
                        input_shape=img_shape)


# In[ ]:


base_model.trainable = False # freezing the base model layers to avoid it's retraining.


# In[ ]:


base_model.summary()


# In[ ]:


from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential


# In[ ]:


global_layer = GlobalAveragePooling2D() # this layer provides us a vetor of features from the just previous volume of base model.


# In[ ]:


pred_layer = Dense(6) # this layer makes raw predictions i.e, it returns numbers as logits.


# In[ ]:


model = Sequential([base_model,global_layer,pred_layer]) # our modified model.


# In[ ]:


model.summary() # see we have some trainable parameters these are due to the layers which we added on later.


# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


# In[ ]:


model.compile(optimizer=Adam(),
             loss=CategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[ ]:


history = model.fit(train_data_gen,
         validation_data=test_data_gen,
         epochs=15,
         callbacks=[early_stop])


# In[ ]:


# trend of losses
loss_metrics = pd.DataFrame(model.history.history)
loss_metrics


# In[ ]:


loss_metrics[['loss','val_loss']].plot(title='LOSS VS EPOCH COUNT')


# In[ ]:


loss_metrics[['accuracy','val_accuracy']].plot(title='ACCURACY VS EPOCH COUNT')


# In[ ]:


test_data_gen.classes


# In[ ]:


predictions = model.predict_classes(test_data_gen)
predictions


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(test_data_gen.classes,predictions))


# In[ ]:


print(confusion_matrix(test_data_gen.classes,predictions))


# from above classification report it is clear that our model performs pretty well on all classes except class 2 and class 3 which corresponds to glacier and mountain respectively. It's great reason which i found could be because many mountains have glaciers surrounding them so our model got confused between whether to say it a mountain or a glacier as in many pictures it is difficult to say whether it's a glacier or a mountain especially when the concentration of ice forming glacier is very less near a mountain.
# but still i'll try to make a new classifier soon with improved accuracy till then let's make predictions on some image from prediction dataset provided.

# In[ ]:


def predict_label(class_number):
    if class_number==0:
        return 'building'
    elif class_number==1:
        return 'forest'
    elif class_number==2:
        return 'glacier'
    elif class_number==3:
        return 'mountain'
    elif class_number==4:
        return 'sea'
    else:
        return 'street'


# In[ ]:


from tensorflow.keras.preprocessing import image


# In[ ]:


def predict_name(directory_to_img):
    pred_image = image.load_img(directory_to_img,target_size=img_shape)
    pred_image_array = image.img_to_array(pred_image)
    pred_image_array = pred_image_array/255
    pred_image_array = pred_image_array.reshape(1,150,150,3)
    prediction = model.predict_classes(pred_image_array)[0]
    plt.imshow(imread(directory_to_img))
    return predict_label(prediction)


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[0])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[1])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[9])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[3])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[4])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[10])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[67])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[33])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[37])


# In[ ]:


predict_name(data_dir+'/seg_pred/seg_pred/'+os.listdir(data_dir+'/seg_pred/seg_pred/')[72])


# In[ ]:


# that's it for this notebook soon updating with even better classifier.

