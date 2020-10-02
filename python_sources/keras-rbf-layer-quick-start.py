#!/usr/bin/env python
# coding: utf-8

# [Welcome to GaussianKernel layer page! (keras RBF layer)](https://github.com/darecophoenixx/wordroid.sblo.jp/tree/master/lib/keras_ex/gkernel)

# In[ ]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import to_categorical


# In[ ]:


digits = load_digits()
X, y = digits.data, digits.target


# In[ ]:


X = X.reshape((X.shape[0], -1))


# In[ ]:


X_sc = X / 16.0
X_sc.shape


# In[ ]:


y_cat = to_categorical(y)


# In[ ]:


'''
chose landmarks from sample
pick 10 data each digit
'''
np.random.seed(0)
num_lm0 = 10
num_lm = num_lm0 * 10
init_list = []
for ii in range(10):
    init_wgt0 = X_sc[y==ii]
    init_wgt0 = init_wgt0[np.random.choice(range(init_wgt0.shape[0]), size=num_lm0, replace=False)] +                 np.random.normal(scale=0.01, size=num_lm0*64).reshape(num_lm0, 64)
    init_list.append(init_wgt0)
init_wgt = np.vstack(init_list)
init_wgt = init_wgt[np.random.permutation(range(init_wgt.shape[0]))]
init_wgt.shape


# In[ ]:


plt.imshow(X[0].reshape((8,8)))


# In[ ]:


plt.imshow(init_wgt[0].reshape((8,8)))


# In[ ]:


from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3

from keras.layers import Input, Dense
from keras.models import Model

np.random.seed(0)

inp = Input(shape=(64,), name='inp')
oup = GaussianKernel(num_lm, 64,
                     kernel_gamma=1./(2.*64*0.1), weights=[init_wgt],
                     name='gkernel1')(inp)
oup = Dense(10, activation='softmax')(oup)
model = Model(inp, oup)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[ ]:


model.fit(X_sc, y_cat, verbose=0,
          batch_size=32,
          epochs=150)


# In[ ]:





# In[ ]:




