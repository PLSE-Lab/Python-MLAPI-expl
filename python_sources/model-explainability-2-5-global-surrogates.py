#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' conda install -y hvplot=0.5.2 bokeh==1.4.0')
get_ipython().system(' conda install -y -c conda-forge sklearn-contrib-py-earth')


# # Global Surrogates Models
# Many classes of models can be difficult to explain. For Tree Ensembles, while it may be easy to describe the rationale for a single trees outputs, it may be much harder to describe how the prediction of many trees are combined by fitting on on errors and weighting thousands of threes. Similarly for neural networks, while the final layer may be linear, it may difficult to convey to domain experts how features- in some easily understood units of measurement- are scaled then combined and projected to make a prediction.  The challenge is that there may be a set of applications where we may benefit greatly from these styles of model but may look to or be required to explain our predictions to users based on regulations, a need for user feedback or for user buy-in.  
# 
# I the case of neural network models, the motivations may be most evident.  Neural network models can benefit from large distributed online training across petabytes of data. In the Federated Learning context, it may be the best-suited model for learning non-linear features for prediction as there is a well-understood body of research into how to train models in this complex environment.  The challenge we far may then face is how to extract explanations from this largely black-box model.
#   
# Using Global Surrogates Models, we try to 'imitate' a black-box model with a highly explainable model to provide explanations. In some cases, these highly non-linear explainable models may not scale well to the data or the learning environment and may be poorly suited to robustly fit the noise in the data. We may also have deployed black-box models historically, which we are now looking to explain and so need a way of understanding what is taking place on the decision surface of the black-box model for the purpose of prototyping and data collections in order to replace the model.  Using a Global Surrogates Model, we look to fit the predictions of the black-box model and analyze the properties of the explainable model to provide insight into the black-box model.  

# # Data

# For our examples in this notebook, we are going to be looking at the Boston Housing Dataset, which is a simple, well-understood dataset provided by default in the Scikit-learn API. The goal here is not to find a good model, but to be able to describe any chosen class of model.  For this reason, we will not be discussing why we choose a particular model or its hyperparameters, and we are not going to be looking into methods for cross-validation.  

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
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

hv.extension('bokeh')


# In[ ]:


data = load_boston()
print(data.DESCR)


# # Model

# I have opted to make use of the Dense Feed-forward Neural Network (DNN) with four hidden neurons and a Selu activation function. The actual properties of this black-box model are not necessary, and in fact, we are going to look to overfit to the data slightly, to provide a slightly greater challenge in our trying to explain this model's decision surface. 

# In[ ]:


EPOCHS = 50

class FFNN(tf.keras.Model):
    def __init__(self, layers = (4, )):
        super(FFNN, self).__init__()
        
        self.inputs = tf.keras.layers.InputLayer((3, 3))
        self.dense = list(map(lambda units: tf.keras.layers.Dense(units, activation='selu'), layers))
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


# The only transformation I have opted to do is to take the log of our housing price target to make our assumption about our conditional distribution being symmetric, more realistic. 

# In[ ]:


pd.Series(data.target).hvplot.kde(xlabel='Log-Target Value')


# In[ ]:


train_ds = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data.data.astype('float32')),
                                               tf.convert_to_tensor(np.log(data.target.astype('float32'))))).batch(32)


# In[ ]:


model = FFNN()
model.compile(loss='mse')

optimizer = tf.keras.optimizers.Adam()
for epoch in range(EPOCHS):
    for sample in train_ds:
        inputs, label = sample
        gradients = train_step(inputs, label)
        
y_pred = model(data.data.astype('float32')).numpy()


# In[ ]:


model.summary()


# ## Decision tree

# While Decision Tree's maybe poor surrogate models for many classes of Black-box model, they are highly interpretable and intuitive for domain experts. Post-hoc explanations can often face a trade-off between interpretability, compute and faithfulness, forcing us to choose approaches which best mirror the tradeoffs we are willing to make.  Many people have been exposed to similar, structured reasoning and while our decision tree may not approximate the reasoning process taken by our original model and be particularly faithful, the interpretability of our decision tree may form a good starting point in building trust with domain experts for complex black-box models. 

# In[ ]:


from sklearn import tree
import matplotlib.pyplot as plt
clf = tree.DecisionTreeRegressor(max_depth=4, min_weight_fraction_leaf=0.15)
clf = clf.fit(data.data, y_pred)


# In[ ]:


plt.figure(figsize=(30,7))
dot_data = tree.plot_tree(clf, max_depth=4, fontsize=12, feature_names=data.feature_names, filled=True, rounded=True)


# ## 'non-linear' Linear Model

# I have before written about my enthusiasm for Linear Models as an interpretable and flexible framework for modelling. What many people don't realize with linear models is that they can be super non-linear, you just need to be able to generate, select and constrain your feature-set in order to appropriately cope with the collinearity in your basis.  Here, we can have tremendous control over the explanations we provide, and while I would recommend starting with an explainable model rather than trying to do Post-hox explanations, Generalized Additive Models and similar classes of model can provide excellent surrogate models for describing the decision-space learned by a black-box model.  
# 
# Here I use a Multivariate Adaptive Regression Spline Model to learn features from the data which help describe the decision surface of my black-box DNN. 

# In[ ]:


from pyearth import Earth
earth = Earth(max_degree=2, allow_linear=True, feature_importance_type='gcv')

earth.fit(data.data, y_pred, xlabels=data.feature_names.tolist())


# In[ ]:


print(earth.summary())


# Here, I can get some notion of feature importances in approximating my model which may be valuable in data collections or feature engineering. 

# In[ ]:


print(earth.summary_feature_importances())


# In[ ]:


(pd.DataFrame([earth._feature_importances_dict['gcv']], columns=data.feature_names, index=['Features'])
 .hvplot.bar(title="'Non-linear' Linear Model Global Approximation Feature Importances")
 .opts(xrotation=45, ylabel='Importance'))


# The main application I may see this used in is scenario in which we believe we can benefit from stochastic optimization on a large noisy dataset using Deep Learning but would like to distill those insights using a subset of the data using our MARS model. One may opt, in some contexts, to improve stability of the fit using some spatial weighting matrix, to control for soem regions being poorly cpatured by the surrogate model due to mismatches in the learning capacity of particular surrogate models can cause entire regions of the decisions surface to have correlated errors. 

# # Conclusions
# One advantage of Suggorate Models is that you can quite easily sample any additional data you may need to describe the black-box model. This can be useful for subsampling the data but may be dangerous in regions where there is poor data coverage as the model may provide degenerate predictions due to overfitting.  
# 
# Global Surrogates are a blunt tool to model explainability, with some very specific use-cases. When using Global Surrogate models, it may be critical in planning a project to evaluate why a black-box model is being used at all if it can be well approximated by an explainable model.  The quality of the approximation and the distributional assumptions made when fitting the model are critical and must be tracked closely.  If you match poorly surrogates and black-box models, you may have very misleading results. That being said, this can be a fast and simple-to-implement heuristic to guide later methods. 
