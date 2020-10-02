#!/usr/bin/env python
# coding: utf-8

# # Simple `load_data()` function for Keras
# 
# This kernel is extremely straightforward and very short. Run the command and copy the function below, and you can load the data in the exact format required for Keras.

# In[ ]:


get_ipython().system('tar xzvf ../input/cifar-10-python.tar.gz')

def load_data():
    """Loads CIFAR10 dataset.
    Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    import os
    import sys
    from six.moves import cPickle
    import numpy as np
    
    def load_batch(fpath):
        with open(fpath, 'rb') as f:
            d = cPickle.load(f, encoding='bytes')  
        data = d[b'data']
        labels = d[b'labels']
        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels
    
    path = 'cifar-10-batches-py'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
    
    x_test, y_test = load_batch(os.path.join(path, 'test_batch'))

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


# In[ ]:


(x_train, y_train), (x_test, y_test) = load_data()

