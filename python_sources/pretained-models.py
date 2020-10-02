#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


from keras.layers import Dense, Flatten, Dropout, Lambda, Input, Concatenate, concatenate
from keras.models import Model
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import regularizers


# In[3]:


filenames = os.listdir("../input/train/train")
labels = []
for file in filenames:
    category = file.split('.')[0]
    if category == 'cat':
        labels.append('cat')
    else:
        labels.append('dog')


# In[4]:


df = pd.DataFrame({
    'filename': filenames,
    'label': labels
})
train_df, validation_df = train_test_split(df, test_size=0.1, random_state = 42)
train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)


# In[5]:


batch_size = 64
train_num = len(train_df)
validation_num = len(validation_df)


# In[17]:


def two_image_generator(generator, df, directory, batch_size,
                        x_col = 'filename', y_col = None, model = None, shuffle = False,
                        img_size1 = (224, 224), img_size2 = (299,299)):
    gen1 = generator.flow_from_dataframe(
        df,
        directory,
        x_col = x_col,
        y_col = y_col,
        target_size = img_size1,
        class_mode = model,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = 1)
    gen2 = generator.flow_from_dataframe(
        df,
        directory,
        x_col = x_col,
        y_col = y_col,
        target_size = img_size2,
        class_mode = model,
        batch_size = batch_size,
        shuffle = shuffle,
        seed = 1)
    
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        if y_col:
            yield [X1i[0], X2i[0]], X1i[1]  #X1i[1] is the label
        else:
            yield [X1i, X2i]
        


# In[7]:


"""
#test if the generator generates same images with two different sizes

ex_df = pd.DataFrame()
ex_df['filename'] = filenames[:5]
ex_df['label'] = labels[:5]
ex_df.head()

train_aug_datagen = ImageDataGenerator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
e1 = two_image_generator(train_aug_datagen, ex_df, '../input/train/train/',
                                      batch_size = 2, y_col = 'label',
                                      model = 'binary', shuffle = True)

fig = plt.figure(figsize = (10,10))
batches = 0
rows = 4
cols = 5
i = 0
j = 0
indices_a = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
indices_b = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20]
for [x_batch, x_batch2], y_batch in e1:
    for image in x_batch:
        fig.add_subplot(rows, cols, indices_a[i])
        i += 1
        plt.imshow(image.astype('uint8'))
        
    for image in x_batch2:
        fig.add_subplot(rows, cols, indices_b[j])
        j += 1
        plt.imshow(image.astype('uint8'))
    
    batches += 1
    if batches >= 6:
        break
plt.show()

"""


# In[25]:


#add data_augmentation
train_aug_datagen = ImageDataGenerator(
    rotation_range = 20,
    shear_range = 0.1,
    zoom_range = 0.2,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)
train_generator = two_image_generator(train_aug_datagen, train_df, '../input/train/train/',
                                      batch_size = batch_size, y_col = 'label',
                                      model = 'binary', shuffle = True)


# In[26]:


validation_datagen = ImageDataGenerator()

validation_generator = two_image_generator(validation_datagen, validation_df,
                                           '../input/train/train/', batch_size = batch_size,
                                           y_col = 'label',model = 'binary', shuffle = True)


# In[10]:


def create_base_model(MODEL, img_size, lambda_fun = None):
    inp = Input(shape = (img_size[0], img_size[1], 3))
    x = inp
    if lambda_fun:
        x = Lambda(lambda_fun)(x)
    
    base_model = MODEL(input_tensor = x, weights = 'imagenet', include_top = False, pooling = 'avg')
        
    model = Model(inp, base_model.output)
    return model


# In[11]:


#define vgg + resnet50 + densenet
model1 = create_base_model(vgg16.VGG16, (224, 224), vgg16.preprocess_input)
model2 = create_base_model(resnet50.ResNet50, (224, 224), resnet50.preprocess_input)
model3 = create_base_model(inception_v3.InceptionV3, (299, 299), inception_v3.preprocess_input)
model1.trainable = False
model2.trainable = False
model3.trainable = False

inpA = Input(shape = (224, 224, 3))
inpB = Input(shape = (299, 299, 3))
out1 = model1(inpA)
out2 = model2(inpA)
out3 = model3(inpB)

x = Concatenate()([out1, out2, out3])                
x = Dropout(0.6)(x)
x = Dense(1, activation='sigmoid')(x)
multiple_pretained_model = Model([inpA, inpB], x)

multiple_pretained_model.compile(loss = 'binary_crossentropy',
                          optimizer = 'rmsprop',
                          metrics = ['accuracy'])

multiple_pretained_model.summary()


# In[12]:


checkpointer = ModelCheckpoint(filepath='dogcat.weights.best.hdf5', verbose=1, 
                               save_best_only=True, save_weights_only=True)


# In[27]:


multiple_pretained_model.fit_generator(
    train_generator,
    epochs = 5,
    steps_per_epoch = train_num // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_num // batch_size,
    verbose = 1,
    callbacks = [checkpointer]
)


# In[14]:


multiple_pretained_model.load_weights('dogcat.weights.best.hdf5')


# In[19]:


test_filenames = os.listdir("../input/test/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})
num_test = len(test_df)

test_datagen = ImageDataGenerator()

test_generator = two_image_generator(test_datagen, test_df, '../input/test/test/', batch_size = batch_size)


# In[20]:


prediction = multiple_pretained_model.predict_generator(test_generator, 
                                         steps=np.ceil(num_test/batch_size))
prediction = prediction.clip(min = 0.005, max = 0.995)


# In[23]:


submission_df = pd.read_csv('../input/sample_submission.csv')
for i, fname in enumerate(test_filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    submission_df.at[index-1, 'label'] = prediction[i]
submission_df.to_csv('submission.csv', index=False)


# In[24]:


submission_df.head()

