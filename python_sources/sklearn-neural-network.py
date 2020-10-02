#!/usr/bin/env python
# coding: utf-8

# ### Introduction 
# 
# Multi-Layer Perceptron (MLP) is a  most common neural network. MLP is a function that maps input to output. MLP has a single input layer, a single output layer. In between, there can be one or more hidden layers. The input layer has the same set of neurons as that of features. Hidden layers can have more than one neuron as well. Each neuron is a linear function to which activation function is applied to solve complex problems. The output from each layer is given as input to all neurons of the next layers.

# `Scikit-learn` has **MLPRegressor** for regression problems and **MLPClassifier** for classification problems.

# In[ ]:


import numpy as np
import pandas as pd
import random
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits, load_boston
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split ,GridSearchCV
from sklearn.metrics import confusion_matrix,r2_score
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Classification

# ### Datasets
# 
# For classification we will use Digits dataset which has 8*8 size images for 0-9 digits. 

# In[ ]:


#Digits Dataset
digits = load_digits()
x_digits = digits.data
y_digits = digits.target
print("x_digits Datasets size",x_digits.shape,"\n y_digits Datasets size",y_digits.shape)


# In[ ]:


# Display Images
fig, axes = plt.subplots(2,5,figsize=(10,10),subplot_kw = {'xticks':[],'yticks':[]})
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray', interpolation='nearest') 
    ax.text(0.5,-0.2,str(digits.target[i]),transform = ax.transAxes)


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x_digits,y_digits,test_size=0.2,random_state=42,stratify= y_digits)
print("Train size: ",x_train.shape,y_train.shape,"Test Size: ",x_test.shape,y_test.shape )


# In[ ]:


mlp_clf = MLPClassifier(random_state=42)
mlp_clf.fit(x_train,y_train)


# In[ ]:


y_preds = mlp_clf.predict(x_test)
print(y_test[:20])
print(y_preds[:20])


# In[ ]:


print("Train Accuracy: ", mlp_clf.score(x_train,y_train))
print("Test Accuracy: ", mlp_clf.score(x_test,y_test))
print("Loss : ", mlp_clf.loss_)


# In[ ]:


con_max= confusion_matrix(y_test,y_preds)
plt.figure(figsize = (9,9))
sns.heatmap(con_max,annot= True,square= True,cbar= False,cmap='YlOrBr')
plt.xlabel("Predicted Value")
plt.ylabel('Actual_value')
plt.show()


# In[ ]:


print("Number of Iterations: ",mlp_clf.n_iter_)
print("Output Layer Activation Function :", mlp_clf.out_activation_)


# ### Finetuning Model By  Grid Search On Various Hyperparameters.
# 
# Below is a list of common hyperparameters that needs tuning for getting the best fit for our data. We'll try various hyperparameters settings to various splits of train/test data to find out best fit which will have almost the same accuracy for both train & test dataset or have quite less difference between accuracy.
# 
# - **hidden_layer_sizes** - It accepts tuple of integer specifying sizes of hidden layers in multi layer perceptrons. According to size of tuple, that many perceptrons will be created per hidden layer. `default=(100,)`
# 
# - **activation** - It specifies activation function for hidden layers. It accepts one of below strings as input. default=relu
#  - 'identity' - No Activation. f(x) = x
#  - 'logistic' - Logistic Sigmoid Function. f(x) = 1 / (1 + exp(-x))
#  - 'tanh' - Hyperbolic tangent function. f(x) = tanh(x)
#  - 'relu' - Rectified Linear Unit function. f(x) = max(0, x)
# 
# - **solve**r - It accepts one of below strings specifying which optimization solver to use for updating weights of neural network hidden layer perceptrons. default='adam'
# 'lbfgs'
# 'sgd'
# 'adam'
# 
# - **learning_rate_init** - It specifies initial learning rate to be used. Based on value of this parameter weights of perceptrons are updated.default=0.001
# 
# - **learning_rate** - It specifies learning rate schedule to be used for training. It accepts one of below strings as value and only applicable when solver='sgd'.
#  - 'constant' - Keeps learning rate constant through a learning process which was set in learning_rate_init.
#  - 'invscaling' - It gradually decreases learning rate. `effective_learning_rate = learning_rate_init / pow(t, power_t) ` 
#  - 'adaptive' - It keeps learning rate constant as long as loss is decreasing or score is improving. If consecutive epochs fails in decreasing loss according to tol parameter and early_stopping is on, then it divides current learning rate by 5.
# - **batch_size** - It accepts integer value specifying size of batch to use for dataset. default='auto'. The default auto batch size will set batch size to min(200, n_samples).
# - **tol** - It accepts float values specifying threshold for optimization. When training loss or score is not improved by at least tol for n_iter_no_change iterations, then optimization ends if learning_rate is constant else it decreases learning rate if learning_rate is adaptive. default=0.0001
# - **alpha** - It specifies L2 penalty coefficient to be applied to perceptrons. default=0.0001
# - **momentum** - It specifies momentum to be used for gradient descent and accepts float value between 0-1. It's applicable when solver is sgd.
# - **early_stopping** - It accepts boolean value specifying whether to stop training if training score/loss is not improving. default=False
# - **validation_fraction** It accepts float value between 0-1 specifying amount of training data to keep aside if early_stopping is set.default=0.1

# ## GridSearchCV

# In[ ]:


get_ipython().run_cell_magic('time', '', "params = {'activation': ['relu', 'tanh', 'logistic', 'identity','softmax'],\n          'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,)],\n          'solver': ['adam', 'sgd', 'lbfgs'],\n          'learning_rate' : ['constant', 'adaptive', 'invscaling']\n         }\n\nmlp_clf_grid = GridSearchCV(MLPClassifier(random_state=42), param_grid=params, n_jobs=-1, cv=5, verbose=5)\nmlp_clf_grid.fit(x_train,y_train)")


# In[ ]:


print('Train Accuracy : ',mlp_clf_grid.best_estimator_.score(x_train,y_train))
print('Test Accuracy : ',mlp_clf_grid.best_estimator_.score(x_test, y_test))
print('Grid Search Best Accuracy  :',mlp_clf_grid.best_score_)
print('Best Parameters : ',mlp_clf_grid.best_params_)
print('Best Estimators: ',mlp_clf_grid.best_estimator_)


# In[ ]:


y_preds = mlp_clf_grid.best_estimator_.predict(x_test)
con_max= confusion_matrix(y_test,y_preds)
plt.figure(figsize = (9,9))
sns.heatmap(con_max,annot= True,square= True,cbar= False,cmap='Pastel1')
plt.xlabel("Predicted Value")
plt.ylabel('Actual_value')
plt.show()


# In[ ]:


clf_model = MLPClassifier(activation = 'logistic', hidden_layer_sizes= (100,), learning_rate = 'constant', solver = 'adam')
clf_model.fit(x_train,y_train)
y_preds = clf_model.predict(x_test)
print("Loss: ",clf_model.loss_)
print(" Score is ",clf_model.score(x_test,y_test))


# ## Regression

# #### Dataset
# For regression we will use boston housing dataset.

# In[ ]:


from sklearn.datasets import load_boston
boston = load_boston()
x_boston = boston.data
y_boston = boston.target
print("Dataset Sizes ",x_boston.shape,y_boston.shape)


# In[ ]:


# Spliting dataset into train and test dataset
x_train,x_test,y_train,y_test = train_test_split(x_boston,y_boston,test_size = 0.25, random_state = 42)


# ### MLPRegressor
# `MLPRegressor`is an estimator available as a part of the `neural_network` module of sklearn for performing regression tasks using a multi-layer perceptron.

# In[ ]:


# import the regressor
from sklearn.neural_network import MLPRegressor
reg = MLPRegressor(random_state  = 42)
reg.fit(x_train,y_train)


# In[ ]:


y_preds = reg.predict(x_test)

print(y_preds[:5])
print(y_test[:5])

print("Train Score",reg.score(x_train,y_train))
print("Test Score" , reg.score(x_test,y_test))


# In[ ]:


print("Loss:",reg.loss_)


# In[ ]:


print("Number of Coefficents :", len(reg.coefs_))
[weights.shape for weights in reg.coefs_]


# In[ ]:


print("Number of intecepts :",len(reg.intercepts_))
[intercepts.shape for intercepts in reg.intercepts_]


# In[ ]:


print("Number of iterations estimators run: ", reg.n_iter_)
print("name of output layer activation function: ", reg.out_activation_)


# ### Finetuning Model By Doing Grid Search 
# 
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "reg= MLPRegressor(random_state = 42)\nparams= {'activation': ['relu','identity','tanh','logistic'],\n        'hidden_layer_sizes': [50,100,150] + list(itertools.permutations([50,100,150],2)) + list(itertools.permutations([50,100,150],3)),\n         'solver' : ['lbfgs','adam'],\n         'learning_rate': ['constant','adaptive','invscaling']\n        }\n\nreg_grid = GridSearchCV(reg,param_grid = params,n_jobs= -1,verbose = 10,cv=5)\nreg_grid.fit(x_train,y_train)\n")


# In[ ]:


print("Train score: ", reg_grid.score(x_train,y_train))
print("Test score: ", reg_grid.score(x_test,y_test))
print("Best R2 Score by grid search: ",reg_grid.best_score_)
print("Best Parameters: ", reg_grid.best_params_)
print("Best Estimators: ",reg_grid.best_estimator_)


# In[ ]:


reg_model = MLPRegressor(activation = 'relu', hidden_layer_sizes = (150, 50, 100), learning_rate= 'constant', solver= 'adam',random_state = 42)
reg_model.fit(x_train,y_train)
y_preds = reg_model.predict(x_test)
print("Loss: ",reg_model.loss_)
print("R2 Score is ",r2_score(y_test,y_preds))


# Updating.........

# In[ ]:





# In[ ]:




