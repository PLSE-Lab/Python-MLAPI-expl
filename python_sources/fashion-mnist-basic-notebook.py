#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(42)


# In[ ]:


print(tf.__version__)
print(keras.__version__)


# In[ ]:


mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[ ]:


(x_train, y_train), (x_test,y_test) = keras.datasets.fashion_mnist.load_data()


# In[ ]:


(x_train.shape), (y_train.shape), (x_test.shape), (y_test.shape)


# In[ ]:


plt.imshow(x_train[0])
plt.axis('off')
plt.show()


# In[ ]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[ ]:


#Normalizing the data

x_train, x_test = x_train/255, x_test/255


# In[ ]:


class_names[y_train[0]]


# In[ ]:


n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(x_train[index], interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()


# In[ ]:


keras.backend.clear_session()
tf.random.set_seed(42)


# In[ ]:


model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape = [28,28]),
                                 keras.layers.Dense(300, activation='relu'),
                                 keras.layers.Dense(100, activation='relu'),
                                 keras.layers.Dense(10, activation='softmax')
])


# In[ ]:


model.layers


# In[ ]:


model.summary()


# In[ ]:


keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


# code to visulize tensorboard callback
import os
root_logdir = os.path.join(os.curdir, 'my_logs')
def get_run_logdir():
    
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('my_model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)


# In[ ]:


history = model.fit(x_train,y_train, epochs=100, validation_split=0.2,
                    callbacks=[checkpoint_cb, early_stopping_cb,tensorboard_cb], batch_size=None, 
                    verbose=1)


# In[ ]:


history.params


# In[ ]:


pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir=./my_logs --port=6006')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




