#!/usr/bin/env python
# coding: utf-8

# This notebook is based on Peter Grenholm's MNIST kernel, but this version does not do validation.  (I presume you have already done validation to choose the number of epochs, so you can now run it with the whole training data set.  But you probably need more epochs than what seemed optimal when you were running with a partial training set in the validaiton run.  I'm too lazy to look up the formula right now, so I just did a seat-of-the-pants guess.  Really the optimal number of epochs is only a guess anyhow.)  To get good results you need to run it for a whole bunch of epochs (say 72) with a low learning rate (say 2e-4), which will be a very slow process unless you have a fast GPU, and which will not work at all on Kaggle.  Hence the <code>RUNNING_AS_KAGGLE_KERNEL</code> option for demonstration purposes.
# 
# Note that the <code>RANDOM_SEED</code> does NOT make results reproducible, at least not on my setup at home.  However, in theory, you would get better results on average by running it with multiple alternative random seeds and taking a majority vote of the results.  (I haven't tried that.)  Even better perhaps, write out the cross-entropy results from each run and take an average, and then classify based on that average rather than on individual cross-entropy results.  (I haven't tried that either yet.)

# In[ ]:


# Parameters

RUNNING_AS_KERAS_KERNEL = True

RANDOM_SEED = 0

if RUNNING_AS_KERAS_KERNEL:
   LEARNING_RATE = 1e-2
   N_EPOCHS = 1
   OUTPUT_FILE = "mnistGrenholm-nocv-1ep01seed" + str(RANDOM_SEED) + ".csv"
else:
   LEARNING_RATE = 2e-4
   N_EPOCHS = 72
   OUTPUT_FILE = "mnistGrenholm-nocv-72ep0002seed" + str(RANDOM_SEED) + ".csv"
    


# Peter Grenholm's Keras notebook for MNIST, with whatever changes I may have made

# In[ ]:


import keras


# In[ ]:


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[ ]:


import numpy as np # linear algebra

keras.__version__
from keras import backend as K
K.set_image_dim_ordering('tf')


# In[ ]:


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_file = "../input/train.csv"
test_file = "../input/test.csv"
output_file = OUTPUT_FILE


# In[ ]:


mnist_dataset = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')


# In[ ]:


n_train = mnist_dataset.shape[0]

np.random.seed(RANDOM_SEED)
np.random.shuffle(mnist_dataset)
x_train = mnist_dataset[:,1:]
y_train = mnist_dataset[:,0]

x_train = x_train.astype("float32")/255.0
y_train = np_utils.to_categorical(y_train)

n_classes = y_train.shape[1]
x_train = x_train.reshape(n_train, 28, 28, 1)


# Peter Grenholm copied this net from several examples he found online, and I may have modified it, but I don't remember. Feel free to modify the layers.

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (28, 28, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters = 32, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


# This Keras function simplifies augmentation, i.e. it randomly modifies the input for training to prevent overfitting. You can also normalize input with this function, just remember to adjust the validation and test sets accordingly in that case.

# In[ ]:


datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 20)


# Certain parameters were chosen for the Kaggle kernel. For better performance, try reducing the learning rate and increase the number of epochs. Peter was able to reach 99.5% which put him at place 75 on the leaderboard. With a GPU, it still takes less than an hour to train.

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=LEARNING_RATE), metrics = ["accuracy"])


# Model training. Note that the training loss is initially quite large, because of Dropout.

# In[ ]:


hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 64),
                           steps_per_epoch = n_train/20,
                           epochs = N_EPOCHS, 
                           verbose = 2  #verbose=1 outputs ETA, but doesn't work well in the cloud
                          )


# In[ ]:


mnist_testset = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')
x_test = mnist_testset.astype("float32")/255.0
n_test = x_test.shape[0]
x_test = x_test.reshape(n_test, 28, 28, 1)


# In[ ]:


y_test = model.predict(x_test, batch_size=64)


# y_test consists of class probabilities, I now select the class with highest probability

# In[ ]:


y_index = np.argmax(y_test,axis=1)


# In[ ]:


with open(output_file, 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(0,n_test) :
        f.write("".join([str(i+1),',',str(y_index[i]),'\n']))


# In[ ]:




