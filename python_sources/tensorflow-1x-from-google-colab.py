#!/usr/bin/env python
# coding: utf-8

# Easy steps to enable TensorFlow 1.15.2 in Kaggle notebooks

# In[ ]:


get_ipython().run_line_magic('set_env', 'PYTHONPATH=/kaggle/input/tensorflow-1x')


# In[ ]:


get_ipython().system('pip install virtualenv')

get_ipython().run_line_magic('cd', '/kaggle/working')
get_ipython().system('virtualenv --python=python3.6 env3.6')


# In[ ]:


get_ipython().run_cell_magic('writefile', '/kaggle/working/mnist_keras.py', "from __future__ import print_function\nimport keras\nfrom keras.datasets import mnist\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Dropout, Flatten\nfrom keras.layers import Conv2D, MaxPooling2D\nfrom keras import backend as K\n\nbatch_size = 128\nnum_classes = 10\nepochs = 12\n\n# input image dimensions\nimg_rows, img_cols = 28, 28\n\n# the data, split between train and test sets\n(x_train, y_train), (x_test, y_test) = mnist.load_data()\n\nif K.image_data_format() == 'channels_first':\n    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)\n    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)\n    input_shape = (1, img_rows, img_cols)\nelse:\n    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n    input_shape = (img_rows, img_cols, 1)\n\nx_train = x_train.astype('float32')\nx_test = x_test.astype('float32')\nx_train /= 255\nx_test /= 255\nprint('x_train shape:', x_train.shape)\nprint(x_train.shape[0], 'train samples')\nprint(x_test.shape[0], 'test samples')\n\n# convert class vectors to binary class matrices\ny_train = keras.utils.to_categorical(y_train, num_classes)\ny_test = keras.utils.to_categorical(y_test, num_classes)\n\nmodel = Sequential()\nmodel.add(Conv2D(32, kernel_size=(3, 3),\n                 activation='relu',\n                 input_shape=input_shape))\nmodel.add(Conv2D(64, (3, 3), activation='relu'))\nmodel.add(MaxPooling2D(pool_size=(2, 2)))\nmodel.add(Dropout(0.25))\nmodel.add(Flatten())\nmodel.add(Dense(128, activation='relu'))\nmodel.add(Dropout(0.5))\nmodel.add(Dense(num_classes, activation='softmax'))\n\nmodel.compile(loss=keras.losses.categorical_crossentropy,\n              optimizer=keras.optimizers.Adadelta(),\n              metrics=['accuracy'])\n\nmodel.fit(x_train, y_train,\n          batch_size=batch_size,\n          epochs=epochs,\n          verbose=1,\n          validation_data=(x_test, y_test))\nscore = model.evaluate(x_test, y_test, verbose=0)\nprint('Test loss:', score[0])\nprint('Test accuracy:', score[1])")


# In[ ]:


get_ipython().system('/kaggle/working/env3.6/bin/pip install keras==2.2.5 # for TensorFlow 1.x')
get_ipython().system('/kaggle/working/env3.6/bin/pip install protobuf absl-py wrapt gast astor termcolor')


# In[ ]:


import sys
sys.path.append("..") # to avoid ValueError: attempted relative import beyond top-level package

get_ipython().system('CUDA_VISIBLE_DEVICES=0 /kaggle/working/env3.6/bin/python /kaggle/working/mnist_keras.py')

