#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import PIL as pillow
from PIL import Image
import os 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[ ]:





# In[ ]:


base_image_dir = os.path.join("../input/")
train_dir = os.path.join(base_image_dir,'train_images/')
train = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))
train['path'] = train['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))


# In[ ]:


test_dir = os.path.join(base_image_dir, "test_images/")
test = pd.read_csv(os.path.join(base_image_dir, 'test.csv'))
test['path'] = test['id_code'].map(lambda x: os.path.join(test_dir, '{}.png'.format(x)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def preprocess_image(image_path, image_size = 224):
    im = Image.open(image_path)
    im = im.resize((image_size, image_size))
    
    return im


# In[ ]:


N = train.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i in range(len(train["path"])):
    x_train[i, :, :, :] = preprocess_image(train["path"][i])


# In[ ]:


N = test.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i in range(len(test["path"])):
    x_test[i, :, :, :] = preprocess_image(test["path"][i])


# In[ ]:





# In[ ]:


id_code = []
image = []
label = []
for i in x_train:
    image.append(i)

for i in train["id_code"]:
    id_code.append(i)

for i in train["diagnosis"]:
    label.append(i)
    


# In[ ]:


all_data = {"id_code": id_code, "image": image, "label":label}


# In[ ]:


from keras.models import Sequential, Model, Input
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50


# In[ ]:





# In[ ]:


x_train = np.array(all_data["image"])
y_train = np.array(all_data["label"])
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.2, random_state=2019, stratify = y_train
)


# In[ ]:


import keras.callbacks as callback

Terminate= callback.TerminateOnNaN()
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x1 = GlobalAveragePooling2D()(x)
x2 = Dense(2000, activation = "relu")(x1)
x3 = Dense(1000, activation = "relu")(x2)
x4 = Dense(500, activation = "relu")(x3)
final_output = Dense(5, activation = "softmax")(x4)
model = Model(inputs = base_model.input, outputs = final_output)

for i in model.layers[:44]:
    i.trainable = False
for i in model.layers[44:]:
    i.trainable = True

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
train_history = model.fit(x_train, y_train, epochs = 20, batch_size = 32)


# In[ ]:





# In[ ]:





# In[ ]:


import seaborn as sn
sn.countplot(train["diagnosis"])
plt.show()


# In[ ]:


prediction = model.predict(x_val)
predictions = []
for i in prediction:
    predictions.append(np.argmax(i))


# In[ ]:





# In[ ]:


from sklearn.metrics import cohen_kappa_score
predictions = np.array(predictions)
cohen_kappa_score(y_val, predictions, weights = "quadratic")


# In[ ]:





# In[ ]:




