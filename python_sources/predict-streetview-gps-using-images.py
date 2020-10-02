#!/usr/bin/env python
# coding: utf-8

# Looking to predict GPS location (lat/lon) using camera images, still trying to see what is the best measure to identify performance of the model.

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


DIR = '../input/Street View Images/'
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
import warnings

warnings.filterwarnings("ignore", category=UserWarning) 


# Getting the data ready. Need to match up the data from   '../input/StreetViewGPS.csv'   with the images in the  
#   '../input/Street View Images/'  .

# In[ ]:


files = [os.path.join(DIR, fname)
        for fname in os.listdir(DIR) if '._' not in fname]
files = files[:]
files = sorted(files)


# In[ ]:


imgs = [plt.imread(fname)[..., :3] for fname in files]
imgs = [resize(img_i, (100, 100)) for img_i in imgs]
imgs = np.array(imgs).astype(np.float32)
print("Number of images", len(imgs))


# In[ ]:


gps_coords = pd.read_csv('../input/StreetViewGPS.csv', error_bad_lines=False)
print("Number of GPS coords", len(gps_coords))


# In[ ]:


imgs = np.delete(imgs, (64, 65, 66, 67, 68, 69, 70, 75, 80), axis=0)


# In[ ]:


print("Number of images", len(imgs))
print("Number of GPS coords", len(gps_coords))


# In[ ]:


print("GPS data and corresponding image:\n\n", gps_coords.iloc[-1, :].T)
plt.imshow(imgs[-1])


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(2))


# In[ ]:


from keras import optimizers
model.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(lr=1e-4))


# In[ ]:


targets = pd.concat([gps_coords.iloc[:,0], gps_coords.iloc[:,1]], axis=1)
inputs = imgs


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
target_scaler = scaler.fit(targets)
targets = target_scaler.transform(targets)
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, random_state=5,
                                                   test_size=0.25)


# In[ ]:


history = model.fit(x_train, y_train, verbose=0, epochs=50, batch_size=5,
                   validation_data=(x_test, y_test))


# In[ ]:


print(history.history.keys())


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
target_1 = targets[0].reshape(1,2)


# In[ ]:


pred = model.predict(np.reshape(imgs[-1], (1,100,100,3)))
pred = target_scaler.inverse_transform(pred)
target_1 = target_scaler.inverse_transform(target_1)
print("MSE: ", (mean_squared_error(target_1, pred)))


# In[ ]:


comparison = np.concatenate((pred, target_1), axis=1)
comparison = pd.DataFrame(comparison)
comparison.columns = ['Predicted latitude', 'Predicted longitude', 'True latitude', 'True longitude']
comparison = pd.concat([comparison['Predicted latitude'], comparison['True latitude'], comparison['Predicted longitude'], comparison['True longitude']], axis=1)                     
comparison.head()


# Still some room for improvement, in real world terms the predictions are still far off, but is a good start. To try and improve results we can increase input image size or increase epochs.

# In[ ]:




