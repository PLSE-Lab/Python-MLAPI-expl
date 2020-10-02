#!/usr/bin/env python
# coding: utf-8

# Example functions you can pass to keras to monitoring f2 scores during training.   

# In[ ]:


from keras import backend as K
import tensorflow as tf
import numpy as np


# In[ ]:


def f2_micro(y_true, y_pred):
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    total_true_positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    total_pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = agreement / (total_true_positive + K.epsilon())
    precision = agreement / (total_pred_positive + K.epsilon())
    return (1+2**2)*((precision*recall)/(2**2*precision+recall+K.epsilon()))


# In[ ]:


y_true = kvar = K.variable(np.array([[1,1,0], [0,1,1], [0,0,0]]), dtype='float32')
y_pred = kvar = K.variable(np.array([[0.6, 0.6, 0.4],[0.6, 0.2, 0.7], [0.2, 0.1, 1.1]]), dtype='float32')


# In[ ]:


K.eval(f2_micro(y_true, y_pred))


# In[ ]:


def f2_mean_example(y_true, y_pred):
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    true_positive = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)
    pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)
    recall = agreement / (true_positive + K.epsilon())
    precision = agreement / (pred_positive + K.epsilon())
    f2score = (1+2**2)*((precision*recall)/(2**2*precision+recall+K.epsilon()))
    return K.mean(f2score)


# In[ ]:


K.eval(f2_mean_example(y_true, y_pred))

