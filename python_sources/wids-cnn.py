#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
from glob import glob 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from skimage.io import imread
import gc
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle

from IPython.display import clear_output
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


base_train_title_dir='../input/train_images/train_images/'
df=pd.DataFrame({'path': glob(os.path.join(base_train_title_dir,'*.jpg'))})
df['image_id']=df.path.map(lambda x: x.split('/')[4])
df.head()


# In[ ]:


labels =pd.read_csv( '../input/traininglabels.csv')
labels.head()


# In[ ]:


df_data = df.merge(labels, on = "image_id")
df_data.drop('score',axis=1,inplace=True)
df_data.head(3)
#df.drop(df[df.score < 0.75].index,inplace=True)


# In[ ]:


ax = sns.countplot(x="has_oilpalm", data=df_data)


# In[ ]:


df_data.has_oilpalm.value_counts()


# ## Train test split

# In[ ]:


SAMPLE_SIZE=942
#0
df_0 = df_data[df_data['has_oilpalm'] == 0].sample(SAMPLE_SIZE, random_state = 101)
#1
df_1 = df_data[df_data['has_oilpalm'] == 1].sample(SAMPLE_SIZE, random_state = 101)
# concat the dataframes
df_data = shuffle(pd.concat([df_0, df_1], axis=0).reset_index(drop=True))

#balance data
y = df_data['has_oilpalm']
df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

# Create directories
train_path = '..input/train'
valid_path = '..input/valid'
test_path = '../input/leaderboard_test_data'
for fold in [train_path, valid_path]:
    for subf in ["0", "1"]:
        os.makedirs(os.path.join(fold, subf),exist_ok=True)


# In[ ]:


# Set the id as the index in df_data
df_data.set_index('image_id', inplace=True)
df_data.head()


# In[ ]:


df_train.head()


# In[ ]:


for image in df_train['image_id'].values:
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image 
    label = str(df_data.loc[image,'has_oilpalm']) # get the label for a certain image
    src = os.path.join('../input/train_images/train_images/', fname)
    dst = os.path.join(train_path, label, fname)
    shutil.copyfile(src, dst)


# In[ ]:


for image in df_val['image_id'].values:
    fname = image 
    label = str(df_data.loc[image,'has_oilpalm']) # get the label for a certain image
    src = os.path.join('../input/train_images/train_images/', fname)
    dst = os.path.join(valid_path, label, fname)
    shutil.copyfile(src, dst)


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 256
num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 32
val_batch_size = 32

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)

datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                            horizontal_flip=True,
                            vertical_flip=True)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='binary')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='binary')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='binary',
                                        shuffle=False)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam

kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.3
dropout_dense = 0.5

model = Sequential()
model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(Conv2D(first_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size)) 
model.add(Dropout(dropout_conv))

model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(second_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Conv2D(third_filters, kernel_size, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = pool_size))
model.add(Dropout(dropout_conv))

#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(256, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation = "sigmoid"))

# Compile the model
model.compile(Adam(0.01), loss = "binary_crossentropy", metrics=["accuracy"])


# In[ ]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystopper = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
reducel = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.1)
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=2,#very low
                   callbacks=[reducel, earlystopper])


# In[ ]:


test_gen


# In[ ]:


from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

# make a prediction
y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_gen.classes, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras


# In[ ]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[ ]:


base_test_dir = '../input/leaderboard_test_data/leaderboard_test_data'
base_test_dir2 = '../input/leaderboard_holdout_data/leaderboard_data'
test_file1 = glob(os.path.join(base_test_dir,'*.jpg'))
test_file2 = glob(os.path.join(base_test_dir2,'*.jpg'))
test_files=test_file1 + test_file2
submission = pd.DataFrame()
file_batch = 5000
max_idx = len(test_files)


# In[ ]:


test_df = pd.DataFrame({'path': test_files})


# In[ ]:


test_df


# In[ ]:


for idx in range(0, max_idx):
    test_df['image_id'] = test_df.path.map(lambda x: x.split('/')[4])
    test_df['image'] = test_df['path'].map(imread)
    K_test = np.stack(test_df["image"].values)
    K_test = (K_test - K_test.mean()) / K_test.std()
    predictions = model.predict(K_test)
    test_df['has_oilpalm'] = predictions
    submission = pd.concat([submission, test_df[["image_id", "has_oilpalm"]]])
submission.head()


# In[ ]:


test_df

