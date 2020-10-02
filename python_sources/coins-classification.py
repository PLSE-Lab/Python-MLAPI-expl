#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


input_path = Path('/kaggle/input/br-coins/classification_dataset/all/')
im_size = 320


# In[ ]:


image_files = list(input_path.glob('*.jpg'))


# In[ ]:


def read_file(fname):
    # Read image
    im = Image.open(fname)

    # Resize
    im.thumbnail((im_size, im_size))

    # Convert to numpy array
    im_array = np.asarray(im)

    # Get target
    target = int(fname.stem.split('_')[0])

    return im_array, target


# In[ ]:


images = []
targets = []

for image_file in tqdm_notebook(image_files):
    image, target = read_file(image_file)
    
    images.append(image)
    targets.append(target)


# In[ ]:


X = (np.array(images).astype(np.float32) / 127.5) - 1
y_cls = np.array(targets)


# In[ ]:


X.shape, y_cls.shape


# In[ ]:


i = 555
plt.imshow(np.uint8((X[i] + 1) * 127.5))
plt.title(str(y_cls[i]));


# In[ ]:


coins_ids = {
    5: 0,
    10: 1,
    25: 2,
    50: 3,
    100: 4
}

ids_coins = [5, 10, 25, 50, 100]

y = np.array([coins_ids[coin] for coin in y_cls])


# In[ ]:


X_train, X_valid, y_train, y_valid, fname_train, fname_valid = train_test_split(
    X, y, image_files, test_size=0.2, random_state=42)


# In[ ]:


im_width = X.shape[2]
im_height = X.shape[1]

im_width, im_height


# # Keras

# In[ ]:


from keras.layers import Conv2D, MaxPool2D, Flatten, GlobalAvgPool2D, GlobalMaxPool2D, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


# In[ ]:


model = Sequential()

# CNN network
model.add( Conv2D(16, 3, activation='relu', padding='same', input_shape=(im_height, im_width, 3)) )
model.add( MaxPool2D(2) )

model.add( Conv2D(32, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(64, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(128, 3, activation='relu', padding='same') )
model.add( MaxPool2D(2) )

model.add( Conv2D(256, 3, activation='relu', padding='same') )

# Transition between CNN and MLP
model.add( GlobalAvgPool2D() )

# MLP network
model.add( Dense(256, activation='relu') )

model.add( Dense(5, activation='softmax') )

model.summary()


# In[ ]:


optim = Adam(lr=1e-3)
model.compile(optim, 'sparse_categorical_crossentropy', metrics=['acc'])


# In[ ]:


callbacks = [
    ReduceLROnPlateau(patience=5, factor=0.1, verbose=True),
    ModelCheckpoint('best.model', save_best_only=True),
    EarlyStopping(patience=12)
]

history = model.fit(X_train, y_train, epochs=2000, validation_data=(X_valid, y_valid), batch_size=32,
                   callbacks=callbacks)


# In[ ]:


df_history = pd.DataFrame(history.history)


# In[ ]:


ax = df_history[['acc', 'val_acc']].plot()
ax.set_ylim(0.9, 1)


# In[ ]:


df_history['val_acc'].max()


# In[ ]:


model.load_weights('best.model')


# In[ ]:


model.evaluate(X_valid, y_valid)


# # Evaluate results

# In[ ]:


y_pred = model.predict(X_valid)


# In[ ]:


y_pred_cls = y_pred.argmax(1)


# In[ ]:


errors = np.where(y_pred_cls != y_valid)[0]
errors


# In[ ]:


i = 55
plt.figure(figsize=(10, 10))
im = Image.open(fname_valid[i])
plt.imshow(np.uint8(im), interpolation='bilinear')
plt.title('Class: {}, Predicted: {}'.format(ids_coins[y_valid[i]], ids_coins[np.argmax(y_pred[i])]));


# In[ ]:




