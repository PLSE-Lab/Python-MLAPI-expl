#!/usr/bin/env python
# coding: utf-8

# ## Summary
# 
# This is a little experiment to check if we can have reproducible Keras models when using a GPU.  [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development) provides some recommendations that we have put in practice here. We have used the famous MNIST dataset and built a deep learning model for it. We have repeated several times the training of this model and observed the accuracy for each test. This experiment has been done for two cases: ussing a gpu or only cpu. It has not been possible to get full reproducibility in case of using a gpu, but setting a seed helps.
# 
# *Note:* if you want to run this kernel in Kaggle you should attach a gpu to it in the configuration.

# In[1]:


import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import os
import multiprocessing
import numpy as np 
import pandas as pd 
import tensorflow as tf
import random as rn
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from keras.models import Model, Sequential


# ## Data loading
# we load the dataset and apply some transformations to use it in a deep learning model

# In[2]:


data_train = pd.read_csv("../input/train.csv")


# In[3]:


y = data_train['label'].astype('int32')
X = data_train.drop('label', axis=1).astype('float32')


# In[4]:


X = X.values.reshape(-1, 28, 28, 1)
y = to_categorical(y)


# In[5]:


SEED = 1
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train /= 255
X_val /= 255


# ## Deep Learning model
# We are going to build a model that we will use in the tests

# In[6]:


def create_model():
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(3, 3),
                     strides=2,
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Dropout(0.5))
    model.add(Conv2D(30, kernel_size=(3, 3), strides=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# ## Test funcion
# This function lets to test to test the deep learning model several times with the option of using a gpu or only a cpu

# In[47]:


def execute_test(mode='gpu', n_repeat=5, seed=1):
    n_epochs = 2
    batch_size = 128    
    num_cores=1  
    
    if type(seed)==int:
        seed_list = [seed]*n_repeat
    else:
        if (type(seed) in [list, tuple]) and (len(seed) >= n_repeat): 
            seed_list = seed
        else:
            raise ValueError('seed must be an integer or a list/tuple the lenght n_repeat')
        
    if mode=='gpu':
        num_GPU = 1
        num_CPU = 1
        gpu_name = tf.test.gpu_device_name()
        if (gpu_name != ''):
            gpu_message = gpu_name  
            print("Testing with GPU: {}".format(gpu_message))
        else:
            gpu_message = "ERROR <GPU NO AVAILABLE>"
            print("Testing with GPU: {}".format(gpu_message))
            return  
    else:    
        num_CPU = 1
        num_GPU = 0
        max_cores = multiprocessing.cpu_count()
        print("Testing with CPU: using {} core ({} availables)".format(num_cores, max_cores))

    results = []    
    for i in range(n_repeat):
        os.environ['PYTHONHASHSEED'] = '0'                      
        np.random.seed(seed_list[i])
        rn.seed(seed_list[i])
        tf.set_random_seed(seed_list[i])

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                      inter_op_parallelism_threads=num_cores, 
                                      allow_soft_placement=True,
                                      device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        model = create_model()

        model.fit(X_train, y_train, batch_size = batch_size, epochs=n_epochs, verbose=0)
        eval_acc = model.evaluate(x=X_val, y=y_val, batch_size=batch_size, verbose=0)[1]
        results.append(eval_acc)
        print("Accuracy Test {}: {}".format(i, eval_acc))
    K.clear_session()
    return results


# ## Test
# Let's test now!

# ### CPU test

# In[55]:


res_cpu_same_seed = execute_test(mode='cpu', n_repeat=5, seed=SEED)


# In[56]:


print("mean: {}".format(np.mean(res_cpu_same_seed)))
print("std: {}".format(np.std(res_cpu_same_seed)))    


# In the case of using a cpu we can see that the results are always the same if we use the same seed. If we change the seed the result can change a bit:

# In[59]:


_ = execute_test(mode='cpu', n_repeat=1, seed=SEED*2)


# In[60]:


_ = execute_test(mode='cpu', n_repeat=1, seed=SEED*10)


# ### GPU test
# We are going to check now if a gpu produces the same behaviour:

# In[61]:


res_gpu_same_seed = execute_test(mode='gpu', n_repeat=10, seed=SEED)


# In[62]:


print("mean: {}".format(np.mean(res_gpu_same_seed)))
print("std: {}".format(np.std(res_gpu_same_seed)))   


# We can see that with the same seed we have slightly different results. We are going to check now if these differences are bigger in case of using a different seed each time:

# In[63]:


res_gpu_diff_seed = execute_test(mode='gpu', n_repeat=10, seed=[i*10 for i in range(10)])


# In[64]:


print("mean: {}".format(np.mean(res_gpu_diff_seed)))
print("std: {}".format(np.std(res_gpu_diff_seed)))   


# We can see in this example that the standard deviation is bigger so, if we keep the same seed when training our neural networks using gpus, our results will be more reproducible. 
# This random behaviour affect only to the training process: when the weights of the neural network are obtained and fixed, all the predictions done with it produces the same result as we can see here:

# In[65]:


model = create_model()
model.fit(X_train, y_train, batch_size = 128, epochs=2, verbose=0)
for i in range(5):
    eval_acc = model.evaluate(x=X_val, y=y_val, batch_size=128, verbose=0)[1]
    print("Accuracy Test: {}".format(eval_acc))


# ## Conclusion
# With the configuration recomended in the Keras documentation it has not been possible to obtain full reproducibility when using a gpu although it has been possible when using only a cpu. The gpus used at the moment in kaggle are Nvidia K80. It looks like that Nvidia and proabably other brands use some kind of non deterministic internal process so, slightly different results can happen each time. Although it is not possible to have full reproducibility when using a gpu, if a seed is fixed, the results will be more similar. It makes sense, when using gpus, to train the same model with different seeds as the performance can be a bit better for some of them. Once the wights of a model have been obtained, predictions done with that model are completely deterministic. 
#  

# References:
# - https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# - https://stackoverflow.com/questions/46836857/results-not-reproducible-with-keras-and-tensorflow-in-python
# - https://www.twosigma.com/insights/article/a-workaround-for-non-determinism-in-tensorflow/
# - https://www.kaggle.com/dansbecker/dropout-and-strides-for-larger-models
# - https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
