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
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import cv2 
import re
from tensorflow.python.platform import gfile

print(os.listdir("../input"))
print("Train data set:", len(os.listdir("../input/dog-breed-identification/train")))
print("Test data set:", len(os.listdir("../input/dog-breed-identification/test")))
print(os.listdir("../input/vgg16"))

# Any results you write to the current directory are saved as output.


# **Read the labels**

# In[ ]:


# all training images
train_dir = '../input/dog-breed-identification/train/'
model_dir = '../input/'
list_images = [train_dir+f for f in os.listdir(train_dir) if re.search('jpg|JPG', f)]

print(list_images[0:4])


# In[ ]:


train_dogs = pd.read_csv('../input/dog-breed-identification/labels.csv')
#train_dogs['image_path'] = list_images
train_dogs.head(5)


# **Count number of breeds**

# In[ ]:


br_labels = train_dogs.groupby("breed").count()
br_labels = br_labels.rename(columns = {"id" : "count"})
br_labels = br_labels.sort_values("count", ascending=False)
br_labels.head()


# **Distribution of the breeds**

# In[ ]:


yy = pd.value_counts(train_dogs['breed'])


fig, ax = plt.subplots()
fig.set_size_inches(15,9)
sns.set_style("whitegrid")

ax = sns.barplot(x = yy.index, y = yy, data = train_dogs)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 10)
ax.set(xlabel='Dog Breed', ylabel='Count')
ax.set_title('Distribution of the Dog Breeds')


# **One hot encoding the labels**

# In[ ]:


target_lables = train_dogs['breed']
one_hot = pd.get_dummies(target_lables, sparse = True)
one_hot_labels = np.asarray(one_hot)


# **Image parameters to be used**

# In[ ]:



IMG_SIZE = 224

IMG_SHAPE = (IMG_SIZE, IMG_SIZE)


n_classes = 120


# **Function to plot the images**

# In[ ]:


x_train = []
y_train = []
y_orig_label = []
orig_label = []
i = 0 

for f, breed in tqdm(train_dogs.values):
    img = cv2.imread('../input/dog-breed-identification/train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    orig_label = target_lables[i]
    x_train.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    y_train.append(label)
    y_orig_label.append(orig_label)
    i += 1

 


# In[ ]:


y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.


# In[ ]:


def plot_images(images, cls_true, cls_pred=None):
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.8, wspace=0.8)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        # Show true classes.
        xlabel = "True: {0}".format(cls_true[i])
        
        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[ ]:


train_img = x_train[0:6]
label_img = y_orig_label[0:6]
plot_images(images=train_img, cls_true=label_img)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
path_train = '../input/dog-breed-identification/train'
#Generator
def generator(df, path):
    
    while 1:
        for i in range(int(df.shape[0])):
            img_path = os.path.join(path, df.iloc[i]['id']+ '.jpg')
    
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            #y = df.iloc[i]['breed']
            #y = onehot.transform(y).toarray()
            #print(img.shape,np.array([y]).shape)
            yield (x)
                    
gen_1 = generator(train_dogs, path_train)


# In[ ]:


from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.optimizers import SGD, Adam
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.applications.vgg19 import VGG19



# In[ ]:


num_class = y_train_raw.shape[1]

print(num_class)


# In[ ]:


vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
base_model = VGG16(weights=vgg16_weights, input_shape=(224, 224, 3))


#base_model.summary()


# In[ ]:


optimizer = Adam(lr=0.0001)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


# In[ ]:


feature = model.predict_generator(gen_1,steps=10221, verbose=1)



# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feature, train_dogs.iloc[:10221]['breed'])

print(X_train[0:4])


# In[ ]:


#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

print("Incredible accuracy of : ",acc)


# In[ ]:


# Use the sample submission file to set up the test data - x_test
submit_data = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')


# In[ ]:


# Creae the x_test
x_test = []
for i in tqdm(submit_data['id'].values):
     img = cv2.imread('../input/dog-breed-identification/test/{}.jpg'.format(i))
     x_test.append(cv2.resize(img, (224, 224)))


# In[ ]:


path_test = '../input/dog-breed-identification/test/'
gen_test = generator(submit_data, path_test)


# In[ ]:


feature_test = model.predict_generator(gen_test,steps=10357, verbose=1)


# In[ ]:


predictions = clf.predict(feature_test)


# In[ ]:


col_names = one_hot.columns.values


# In[ ]:


submission_results = pd.DataFrame(predictions)


# In[ ]:


submission_results[0:5]


# In[ ]:


submission_results.insert(0, 'id', submit_data['id'])


# In[ ]:


submission_results.to_csv('submission.csv', index=False)


# 

# 
