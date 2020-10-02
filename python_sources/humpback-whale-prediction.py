#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, GlobalMaxPooling2D
from keras.models import Model
import glob
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization,     GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from tqdm import tqdm


# In[ ]:


training_dir = '../input/train/'
data = pd.read_csv('../input/train.csv')
train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=1337)
print('Checking training data head')
print(train.head())
print('Checking test data head')
print(test.head())


# In[ ]:


sampleImageFile1 = train.Image[2]
sampleImage = mpimg.imread(training_dir + sampleImageFile1)
plt.imshow(sampleImage)
plt.show()


# In[ ]:


sampleImageFile2 = train.Image[89]
sampleImage2 = mpimg.imread(training_dir + sampleImageFile2)
plt.imshow(sampleImage2)


# ## Read training CSV
# 
# 1. Checking duplicate IDs

# In[ ]:


vc = train.Id.value_counts().sort_values(ascending=False)
vc[:50].plot(kind='bar')
plt.show()


# # Train model

# In[ ]:


# PARAMETERS
# The parameters are not the final parameters and will be changed later.
k_size = (4,4)
drop_probability = 0.5
hidden_size = 256
batch_size = 64
input_shape = (batch_size, 128, 128)
pool_size = (2,2)
learning_rate = 0.07
num_of_epochs = 10
num_of_classes = 4251


# In[ ]:


# NETWORK
model = Sequential()
model.add(Convolution2D(32, kernel_size=k_size, activation="relu", input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2)))
model.add(Convolution2D(64, kernel_size=k_size, activation="relu"))
model.add(MaxPooling2D(pool_size=pool_size, strides=(1,1)))
model.add(Convolution2D(512, kernel_size=k_size, activation="relu"))
model.add(MaxPooling2D(pool_size=pool_size, strides=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(512, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(num_of_classes, activation="softmax"))


# In[ ]:


# COST AND OPTIMIZER
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])


# # Data preparation and training

# In[ ]:


def process(image):
    # resize
    image = np.resize(image, [128, 128])
    
    # convert to grayscale
    if image.shape == 3:
        image = np.dot([image[:,:,0],image[:,:,1],image[:,:,2]],[0.299,0.587,0.114])
    
    # return normalized
    return image / 255
    


# # Convert target variable to a one hot coded value

# In[ ]:


x = []
y = []


# In[ ]:


for path in tqdm(train.Image):
    image = mpimg.imread(training_dir + path)
    image = process(image)
    x.append(image)
    
    cod = 'Id_' + train[train.Image == path]['Id']
    y.append(cod)


# In[ ]:


model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=num_of_epochs, verbose=1)


# In[ ]:


test_preds = []
test_file_names = []
i = 1
test_files = glob.glob("../input/test/*.jpg")
for fnames, imgs in data_generator(test_files, batch=32):
    print(i * 32 / len(test_files) * 100)
    i += 1
    predicts = inference_model.predict(imgs)
    predicts = predicts.tolist()
    test_preds += predicts
    test_file_names += fnames

test_preds = np.array(test_preds)

