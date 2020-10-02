#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('ls ../input/train/')


# In[ ]:


datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
        '../input/train/',  
        batch_size=1,
        class_mode='categorical')


# In[ ]:


# let's have a look at the images
x, y = train_generator.next()
plt.imshow((x[0]*255).astype('uint8'));
print(list(train_generator.class_indices.keys())[np.argmax(y)])


# In[ ]:


X_data, Y_data = [], []
for _ in tqdm(range(2750)):
    x, y = train_generator.next()
    X_data.append(x[0])
    Y_data.append(y[0])
X_data = np.asarray(X_data)
Y_data = np.asarray(Y_data)


# In[ ]:


def get_model():
    input_img = Input((256, 256, 3))
    X = BatchNormalization()(input_img)
    X = Convolution2D(16, (3, 3), activation='relu')(X)
    X = BatchNormalization()(X)
    X = Convolution2D(16, (3, 3), activation='relu')(X)
    X = MaxPooling2D()(X)
    X = Convolution2D(32, (3, 3), activation='relu')(X)
    X = BatchNormalization()(X)
    X = Convolution2D(32, (3, 3), activation='relu')(X)
    X = GlobalMaxPooling2D()(X)
#     X = Flatten()(X)
    X = BatchNormalization()(X)
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(10, activation='softmax')(X)
    model = Model(inputs=input_img, outputs=X)

    model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['acc'])
    model.summary()
    return model


# In[ ]:


model = get_model()


# In[ ]:


model_history = model.fit(X_data, Y_data, batch_size=10, epochs=3, validation_split=0.2,
                          callbacks=[EarlyStopping(monitor='val_acc', patience=3, verbose=1)])


# In[ ]:


# load test images
X_test = []
sub = pd.read_csv('../input/sample_submission.csv')

for fname in tqdm(sub['fname']):
    filepath = '../input/test/' + fname
    X_test.append(img_to_array(load_img(filepath, target_size=(256, 256))))
X_test = np.asarray(X_test)


# In[ ]:


preds = model.predict(X_test, verbose=1)
preds = np.argmax(preds, axis=1)
preds = [list(train_generator.class_indices.keys())[p] for p in tqdm(preds)]


# In[ ]:


sub['camera'] = preds
sub.to_csv('sub.csv', index=False)


# In[ ]:


sub.head()


# In[ ]:




