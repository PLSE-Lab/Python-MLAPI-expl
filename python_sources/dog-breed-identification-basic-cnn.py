#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


labels=pd.read_csv("../input/dog-breed-identification/labels.csv")


# In[ ]:


labels


# In[ ]:


filenames = os.listdir("../input/dog-breed-identification/train/train")


# In[ ]:


filenames.sort()


# In[ ]:


filenames


# In[ ]:


df=pd.DataFrame({
    'filename': filenames,
    'category': labels['breed']
})


# In[ ]:


df.tail()


# In[ ]:


labels


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import random


# In[ ]:


sample=random.choice(filenames)
image = load_img("../input/dog-breed-identification/train/train/"+sample)
plt.imshow(image)


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(120, activation='softmax'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


callbacks = [earlystop, learning_rate_reduction]


# In[ ]:


from sklearn.model_selection import train_test_split
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[ ]:


print(train_df.shape)
print(validate_df.shape)


# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe( 
    train_df,
    "../input/dog-breed-identification/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=8
)


# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../input/dog-breed-identification/train/train/", 
    x_col='filename',
    y_col='category',
    target_size=(256,256),
    class_mode='categorical',
    batch_size=8
)


# In[ ]:


total_validate=validate_df.shape[0]
total_train = train_df.shape[0]


# In[ ]:


history = model.fit_generator(
    train_generator, 
    epochs=15,
    validation_data=validation_generator,
    validation_steps=total_validate//8,
    steps_per_epoch=total_train//8,
    callbacks=callbacks
)


# In[ ]:


model.save_weights("model.h1")


# In[ ]:


test_filenames = os.listdir("../input/dog-breed-identification/test/test")
test_filenames.sort()
test_df = pd.DataFrame({
    'filename': test_filenames
})


# In[ ]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../input/dog-breed-identification/test/test/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=(256,256),
    batch_size=8,
    shuffle=False
)


# In[ ]:


total_test=test_df.shape[0]
predict = model.predict_generator(test_generator, steps=np.ceil(total_test/8))


# In[ ]:


predict


# In[ ]:


predict.shape


# In[ ]:


ans=pd.read_csv("../input/dog-breed-identification/sample_submission.csv")


# In[ ]:


ans.shape


# In[ ]:


ans.head()


# In[ ]:


labels = (train_generator.class_indices)
labels = list(labels.keys())
df = pd.DataFrame(data=predict,
                 columns=labels)

columns = list(df)
columns.sort()
df = df.reindex(columns=columns)

filenames = ans["id"]
df["id"]  = filenames

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.head(5)


# In[ ]:


df.to_csv("submission1.csv",index=False)


# In[ ]:


ans.head()

