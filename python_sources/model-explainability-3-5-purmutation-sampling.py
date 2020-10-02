#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' conda install -y hvplot=0.5.2 bokeh==1.4.0')


# There are many very complicated methods in black-box model explainability and there are some simpler ones. The family of methods involving Permutation and Sampling are amoung the simplist. The main advantage of this family of methods in in its simplicity- sampling and permuation can be really easy to explain and really easy for domain experts to understand.  This approach also gives a great deal of flexibility in terms of the insights you provide.  The main challenge with black-box methods which rely solely on Purmutation and Sampling is that they can be inefficient and in some cases misleading based on the number of features in your dataset and the complexity of the model. 

# # Data

# For our examples in this notebook we am going to be looking at the Boston Housing Dataset, which is a simple, well understood dataset provided by default in the Scikit-learn API. The goal here is not find a good model, but to describe a model.  For this reason, we will not be discussing why we choose a particular model, or its hyperparameters, and we are not going to be looking into methods for cross-validation.  

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
import numpy as np
from toolz.curried import map, pipe, compose_left, partial
from typing import Union, Tuple, List, Dict
import tensorflow as tf
import tensorflow_probability as tfp
import warnings
from abc import ABCMeta
from itertools import chain
from operator import add
import holoviews as hv
import pandas as pd
import hvplot.pandas
from sklearn.datasets import load_digits, load_boston
import tensorflow as tf
from functools import reduce
from sklearn.inspection import permutation_importance, plot_partial_dependence, partial_dependence

hv.extension('bokeh')


# In[ ]:


data = load_boston()
print(data.DESCR)


# The only transformation I have opten to do is to take the log of our housing price target to make our assumption about our conditional distribution being symmetic, more realistic. 

# In[ ]:


pd.Series(data.target).hvplot.kde(xlabel='Log-Target Value')


# # Global

# I have opted to make use of the Dense Feed-forward Neural Network (DNN) with 4 hidden neurals and a relu activation function. This is a relatively contrained model, with the ability to model particular non-linearities in the data. 

# In[ ]:


estimator = MLPRegressor((4,)) 

X, y = data.data, np.log(data.target)

estimator.fit(X, y)
y_pred = estimator.predict(X)
(pd.Series(y - y_pred).hvplot.kde(xlabel='Model Errors', title='MLP Model') +pd.Series(y - y_pred).hvplot.box(ylabel='').opts(invert_axes=True, height=100)).cols(1)


# The model does present singificant bias in its estimates, as we might expect from such a flexible class of model.  

# ## Partial Dependence
# The Partial Dependence Plot, or Partial Dependence Curves, show how our average prediction changes as we substitute features with points from our dataset. While it may be difficult to visualize and interpret more than two interactions, these plots can be really easy to interpret for users trying to get a global overview of how the model responds to changes in the data. As with many of these methods which rely on permutation and sampling, we do run the risk of over-sampling highly improbably points based on the joint distribution of our model. 

# In[ ]:


plot_partial_dependence(estimator, X, [(1,2), 2, 1], feature_names=data.feature_names, n_cols=2)


# ## Permutive Importance
# Using Permutive Importance, we drop groups of features, fit a model and compare how our scores change without certain features.  For models which are rely on stochastic optimization, this approach can be expensive or misleading as we may converge on many different models with the same set of features and hyperparamters.  Another challenge with Feature Importance, is that it can eaily be misinterpretted. Feature Importances say little about the effect which changes in that feature have on predictions and must not be conflated the importance of a feature t all possible classes of models. 

# In[ ]:


imp = permutation_importance(estimator, X, y, scoring=None, n_repeats=1000, n_jobs=-1)

(pd.DataFrame(imp['importances'].T, columns=data.feature_names)
 .melt(var_name='Feature', value_name='Importance')
 .hvplot.violin(y='Importance', by='Feature'))


# # Local
# ## Sensitivity Analysis

# One simple approach to model explainability would be to evaluable how the output of our model changes if we permute the input space. This could be done using random sampling or by substituting our features with a univariate means, medians or a reference point.  This can be very computationally intensive and can suffer from many issues caused by sampling points which do not realistically come from our data generating process or failing to model interaction between substrituted variables.  
# 
# This is easy to implement but can be incredibly computationally expensive, for that reason I opted to write this in TensorFlow 2.0, so that we can benefit from end-to-end hardware Acceleration. 

# In[ ]:


EPOCHS = 50

class FFNN(tf.keras.Model):
    def __init__(self, layers = (4, )):
        super(FFNN, self).__init__()
        
        self.inputs = tf.keras.layers.InputLayer((3, 3))
        self.dense = list(map(lambda units: tf.keras.layers.Dense(units, activation='relu'), layers))
        self.final = tf.keras.layers.Dense(1, activation='linear')
        
    def call(self, inputs):
        
        return reduce(lambda x, f: f(x), [inputs, self.inputs, *self.dense, self.final])
    
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        
        loss = tf.keras.losses.mse(predictions, label)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
train_ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data.data.astype('float32')),
                                               tf.convert_to_tensor(np.log(data.target.astype('float32'))))).batch(32)

model = FFNN()
model.compile(loss='mse')

optimizer = tf.keras.optimizers.Adam()
for epoch in range(EPOCHS):
    for sample in train_ds:
        inputs, label = sample
        gradients = train_step(inputs, label)
        
y_pred = model(data.data.astype('float32')).numpy()

model.summary()


# In[ ]:


def sensitivity_importance(X: tf.Tensor, 
                                reference: tf.Tensor, 
                                model: tf.keras.Model,
                                sample=1000):
    """
    """
    length = tf.shape(X)[0]
    features = tf.shape(X)[1]
    all_subs = tf.dtypes.cast(tf.random.uniform((sample, features)) > 0.5, 'float32')
    
    f_mean = model(reference)[0]
    
    count_subs = tf.shape(all_subs)[0]
    
    @tf.function
    def apply(x):
        return tf.reduce_mean(model(tf.where(all_subs==1, 
                                     tf.ones((count_subs,features))*x, 
                                     tf.ones((count_subs,features))*reference_point)) - f_mean, axis=1)
    
    all_sub_float = tf.dtypes.cast(all_subs, 'float32')
    return tf.map_fn(apply, X) @ all_sub_float / tf.reduce_sum(all_sub_float, axis=0)

reference_point = data.data.mean(0).reshape(1, -1)
X = data.data.astype('float32')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'permutive_values = sensitivity_importance(X, reference_point, model)')


# In[ ]:


permutive_values = sensitivity_importance(X, reference_point, model)
pd.Series(permutive_values.numpy().mean(0) - permutive_values.numpy().mean(), index=data.feature_names).hvplot.bar(title='Average Sensitivity')


# After analyzing the effect of random substitutions of our features against our reference we can average across these explanations to look at the average effect changes in the input on our models predictions using senstitivity analysis.  
