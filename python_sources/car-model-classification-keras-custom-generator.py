#!/usr/bin/env python
# coding: utf-8

# # Prepare the Data

# ### Load library
# load basic libraries

# In[ ]:


import gc
import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from keras import backend as K

# check keras image data format
print(K.image_data_format())


# ### Fix random seed

# In[ ]:


np.random.seed(42)


# ### load Files
# 
# check files and load

# In[ ]:


DATA_PATH = '../input/2019-3rd-ml-month-with-kakr'
print(os.listdir(DATA_PATH))


# * **train.csv** - train set's image file name, bbox, class
# * **test.csv** - test set's image file name, bbox, class
# * **sample_submission.csv** - submission file corresponding to test.csv
# * **class.csv** - car label corresponding to dataset's class column
# * **train** - train image files
# * **test** - test image files

# In[ ]:


TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test')

# load .csv files
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
df_class = pd.read_csv(os.path.join(DATA_PATH, 'class.csv'))


# # Data Exploration
# 
# We try to solve the general question about the data.  
# such as whether the actual data matches the description, how the data is structured, and what kind of distribution it has by class.
# 

# Detailed description of each column in the Data Description.
# 
# * **img_file** - image file name associated with each row in the dataset
# * **bbox_x1** - Bounding box x1 coordinates (upper left x)
# * **bbox_y1** - Bounding box y1 coordinates (upper left y)
# * **bbox_x2** - Bounding box x2 coordinates (lower right x)
# * **bbox_y2** - Bounding box y2 coordinates (lower right y)
# * **class** - car model to predict(target)
# * **id** - car class id
# * **name** - actual car model label corresponding to class id

# In[ ]:


# Change the start value of the class to 0
df_train['class'] -= 1
df_class['id'] -= 1


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


# check missing data
if set(list(df_train['img_file'])) == set(os.listdir(TRAIN_IMG_PATH)):
    print('No Train file missing')
else:
    print('missing Train file')

if set(list(df_test['img_file'])) == set(os.listdir(TEST_IMG_PATH)):
    print('No Test file missing')
else:
    print('missing Test file')


# In[ ]:


# check number of Data
print("Number of Train Data : {}".format(df_train.shape[0]))
print("Number of Test Data : {}".format(df_test.shape[0]))


# In[ ]:


df_class.head()


# In[ ]:


print("Number of Target class : {}".format(df_class.shape[0]))
print("Number of Target class kinds of Training Data : {}".format(df_train['class'
].nunique()))


# ### Class Distribution
# 
# The first thing to doubt about the classification problem is the distribution of the Target Class.  
# You need to check the Target distribution of the Train set and check the balance.  
# 

# In[ ]:


plt.figure(figsize=(15, 6))
sns.countplot(df_train['class'], order=df_train["class"].value_counts(ascending=True).index)
plt.title('Number of data per each class')


# In[ ]:


cntEachClass = df_train['class'].value_counts(ascending=False)
print('Class with most count : {}'.format(cntEachClass.index[0]))
print("Most Count : {}".format(cntEachClass.max()))

print('Class with fewest count : {}'.format(cntEachClass.index[-1]))
print("Fewest Count : {}".format(cntEachClass.min()))

print('Mean : {}'.format(cntEachClass.mean()))


# In[ ]:


cntEachClass.describe()


# ### Image Visualization

# In[ ]:


import cv2

def load_image(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

tmp_imgs = df_train['img_file'][100:110]
plt.figure(figsize= (12, 20))

for num, f_name in enumerate(tmp_imgs, start=1):
    img = load_image(os.path.join(TRAIN_IMG_PATH, f_name))
    plt.subplot(5, 2, num)
    plt.title(f_name)
    plt.imshow(img)
    plt.axis('off')


# ### Bounding Box
# 
# What is bounding box?
# bounding box refer to the coordinates of a box that labled with a specific object inside the image.  
# Normally, the upper left corner and the lower right coordinates are given.

# In[ ]:


def draw_rect(img, pos, outline, width):
    p1 = tuple(pos[0:2])
    p2 = tuple(pos[2:4])
    img = cv2.rectangle(img, p1, p2, outline, width)
    return img

def make_boxing_img(img_name):
    if img_name.split('_')[0] == 'train':
        PATH = TRAIN_IMG_PATH
        data = df_train
    elif img_name.split('_')[0] == 'test':
        PATH = TEST_IMG_PATH
        data = df_test
    
    img = load_image(os.path.join(PATH, img_name))
    pos = data.loc[data['img_file'] == img_name, ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)
    img = draw_rect(img, pos, outline=(255, 0, 0), width=10)
    return img


# In[ ]:


f_name = "train_00102.jpg"

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)

# Original Image
origin_img = load_image(os.path.join(TRAIN_IMG_PATH, f_name))
plt.title('Original Image - {}'.format(f_name))
plt.imshow(origin_img)
plt.axis('off')

# Image included bounding box
plt.subplot(1, 2, 2)
boxing = make_boxing_img(f_name)
plt.title('Boxing Image - {}'.format(f_name))
plt.imshow(boxing)
plt.axis('off')

plt.show()


# # Model
# 
# Now let's create an image classification model in earnest.

# ### Train Valid Test dataset split

# In[ ]:


from sklearn.model_selection import train_test_split

its = np.arange(df_train.shape[0])
train_idx, val_idx = train_test_split(its, train_size = 0.8, random_state=42)

X_train = df_train.iloc[train_idx]
X_val = df_train.iloc[val_idx]

print(X_train.shape)
print(X_val.shape)
print(df_test.shape)


# In[ ]:


print(X_train.head())


# ### Generator
# 
# Generator can be really useful in a cloud environment   
# such as colab or kaggle kernel and in a typical local environment.
# because usually these environment don't have enough memory.

# ### Keras DataGenerator
# 
# Keras has a really nice generator function.
# Keras ImageDataGenerator allow you to define the generator and give the desired noise to the data at the same time.

# In[ ]:


from keras.applications.xception import Xception, preprocess_input

class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, dim, n_channels, n_classes, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.X = X.values
        self.y = y.values if y is not None else y
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X))
        self.shuffle_index()

    def shuffle_index(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        X_temp = [self.X[k] for k in indexes]

        if self.y is not None:
            y_temp = [self.y[k] for k in indexes]
            X, y = self.__data_generation(X_temp, y_temp)
            return X, y
        else:
            y_temp = None
            X = self.__data_generation(X_temp, y_temp)
            return X

    def __data_generation(self, X_temp, list_y_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)
        if list_y_temp is not None:
            for i, (img_path, label) in enumerate(zip(X_temp, list_y_temp)):
                img = cv2.imread(img_path)
                img = cv2.resize(img, dsize=self.dim)
                img = preprocess_input(img)

                X[i] = img
                y[i] = label

            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            for i, img_path in enumerate(X_temp):
                img = cv2.imread(img_path)
                img = cv2.resize(img, dsize=self.dim)
                img = preprocess_input(img)

                X[i] = img
                return X


# In[ ]:


# Parameter
img_size = (299, 299)
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
nb_test_samples = len(df_test)
epochs = 20
batch_size = 32

train_generator = CustomDataGenerator((TRAIN_IMG_PATH+'/')+X_train['img_file'], 
                                      X_train['class'], 
                                      dim = img_size, 
                                      batch_size=batch_size, 
                                      n_classes=df_class.shape[0], 
                                      n_channels=3, 
                                      shuffle=False)

validation_generator = CustomDataGenerator((TRAIN_IMG_PATH+'/')+X_val['img_file'], 
                                           X_val['class'], 
                                           dim = img_size, 
                                           batch_size=batch_size, 
                                           n_classes=df_class.shape[0], 
                                           n_channels=3, 
                                           shuffle=False)

test_generator = CustomDataGenerator((TEST_IMG_PATH+'/')+df_test['img_file'], 
                                     None, 
                                     dim = img_size, 
                                     batch_size=1, 
                                     n_classes=df_class.shape[0], 
                                     n_channels=3, 
                                     shuffle=False)


# ### Loading Pre-trained model - Xception
# 
# ![Xception](https://cdn-images-1.medium.com/max/1600/1*J8dborzVBRBupJfvR7YhuA.png)

# In[ ]:


from keras.models import Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D

xception_model = Xception(include_top=False, weights='imagenet', input_shape = (299, 299, 3))

model = Sequential()
model.add(xception_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(df_class.shape[0], activation='softmax', kernel_initializer='he_normal'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.summary()


# ### define evaluation metric

# In[ ]:


from sklearn.metrics import f1_score

def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


# ### Model Training

# In[ ]:


from math import ceil

def get_steps(num_samples, batch_size):
    return ceil(num_samples / batch_size)


# In[ ]:


'''
%%time
# running time profiling

from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath = "my_xception_model_{val_acc:.2f}_{val_loss:.4f}.h5"

ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
es = EarlyStopping(monitor='var_acc', patience=3, verbose=1)

callbackList = [ckpt, es]

hist = model.fit_generator(
    train_generator,
    steps_per_epoch = get_steps(nb_train_samples, batch_size),
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = get_steps(nb_validation_samples, batch_size),
    callbacks = callbackList
)

gc.collect()
'''


# ### Training result
# 
# * **default training** - acc : 0.33, loss : 3.3309
# * **imagenet weight + fine-tuning** - acc : 0.70, loss : 1.4957
# * **imagenet weight + fine-tuning + transfer-learning** - acc : 0.71, loss : 1.3773

# ### Load trained weights

# In[ ]:


WEIGHT_PATH = '../input/car-model-classification-weight'
print(os.listdir(WEIGHT_PATH))

MODEL_PATH = os.path.join(WEIGHT_PATH, 'fine-tuning-transfer-learning.h5')
print(MODEL_PATH)

model = load_model(MODEL_PATH)
model.summary()


# # Evaluate Predict & Make submission

# ### Model Evaluation

# In[ ]:


val_predict = model.predict_generator(
    generator = validation_generator,
    steps = get_steps(nb_validation_samples, batch_size),
    verbose = 1
)
val_predict = np.argmax(val_predict, axis=1)[:X_val.shape[0]]
f1_score = micro_f1(X_val['class'].values, val_predict)
print("f1_score : {:.3}".format(f1_score))


# ### Model predict

# In[ ]:


get_ipython().run_cell_magic('time', '', 'prediction = model.predict_generator(\n    generator = test_generator,\n    steps = nb_test_samples,\n    verbose = 1\n)')


# ### Make submission

# In[ ]:


predicted_class_indices = np.argmax(prediction, axis=1)[:df_test.shape[0]]
submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))

submission['class'] = predicted_class_indices + 1
print(submission['class'])
submission.to_csv('submission.csv', index=False)


# In[ ]:




