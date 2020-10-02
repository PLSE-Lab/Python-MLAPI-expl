#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import torch

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


# # Load Dataset

# The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are:  
# 1. GRE Scores ( out of 340 )
# 2. TOEFL Scores ( out of 120 )
# 3. University Rating ( out of 5 ) 
# 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 )
# 5. Undergraduate GPA ( out of 10 )
# 6. Research Experience ( either 0 or 1 ) 
# 7. Chance of Admit ( ranging from 0 to 1 )

# In[ ]:


dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# # Dataset Analysis and Visualization

# In[ ]:


# display first 5 rows of dataset to see what we are looking at
dataset.head()


# * The **GRE Score** seems to have a high mean (close to the max GRE score) with an extremely low standard deviation
# * The **TOEFL Score** seems to have a high mean (close to the max TOEFL Score) with an extremely low standard deviation
#  * The **GRE** and **TOEFL** score makes sense, as top students apply for masters programs at UCLA
# * **SOP,LOR, and University rating** have more of a spread than other variables
#  * This makes sense, as these are subjective aspects of the student / university
# * **CPGA** seems to have a high mean with low standard deviation
# * **Research** a little over 50% of applicants do research

# In[ ]:


# show the distributions of data 
dataset.describe()


# In[ ]:


# The serial No. adds nothing to predicting the chance of admittance
dataset = dataset.drop('Serial No.',axis=1)


# Displaying the pairwise plots of all the variables (including the target variable chance of admission) it is clear that there is a linear correlation between the target variable **Chance of Admittance** and the data features (maybe the relationship is not clear between the binary categorical feature **Research** and **Chance of Admittance**
# * By observing the pairplot, it may be see that there is a fairly strong linear correlation between many variables, which may suggest redundant information

# In[ ]:


sns.pairplot(dataset,diag_kind='kde',plot_kws={'alpha': .2});


# The distribution for the students who participate in research have on average a higher chance of admittance than the students who do not participate in research, as shown by the histogram and box and whisker plots below. The mean chance of admit for students who participate is **.79**, whereas the mean chance of admit for students who do not partipate in research is **.63**

# In[ ]:


sns.factorplot(y="Chance of Admit ",x='Research',data=dataset,kind='box');
dataset.loc[:,['Research','Chance of Admit ']].groupby('Research').describe()


# In[ ]:


sns.distplot(dataset[dataset.loc[:,'Research'] == 1].loc[:,['Chance of Admit ']],kde=True);
sns.distplot(dataset[dataset.loc[:,'Research'] == 0].loc[:,['Chance of Admit ']],kde=True);
plt.xlabel('Chance of Admittance')
plt.ylabel('Count')
plt.title('Research vs No Research');
plt.legend(['Research','No Research']);


# As seen by the box and whisker plots below, there does seems to be a positive linear correlation between **University Rating , SOP, and LOR** with **Chance of Admit**

# In[ ]:


sns.factorplot(x='University Rating',y='Chance of Admit ',kind='box',data=dataset);
sns.factorplot(x='SOP',y='Chance of Admit ',kind='box',data=dataset);
sns.factorplot(x='LOR ',y='Chance of Admit ',kind='box',data=dataset);


# In[ ]:


target = dataset.pop('Chance of Admit ')


# # Split into train/test datasets
# * Split the dataset such that 80 percent of the data is the training set and 20 percent of the data is the test set. This step is done so that the when measuring the predictive power of the model, it is predicintg based on new data, and therefore the model is more generalizable (and not limited to predicting data only seen in the dataset).

# In[ ]:


# split data into train test 
X_train,X_test,y_train,y_test = train_test_split(dataset.values.astype(np.float32),
                                                 target.values.reshape(-1,1).astype(np.float32),
                                                 test_size=.2,
                                                random_state=42)


# # Standard Normalization Preprocess
# * Standard normalization makes all the features have zero-mean and unit-variance. 
# * The range of the raw values in the data vary widely, and in deep learning of the feature values are too large depending on the activation function of the neuron the neurons may become saturated (and not perform backpropogation to update the weights). If the feature values are too small, depending on the activation function of the neuron the neurons may output zero and be considered "dead neurons". These dead neurons will not train the model.

# In[ ]:


# normalize data to 0 mean and unit std
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Set up Neural Network
# * One Hidden Layer
# * The nonlinearity activation function of the neurons in the hidden layer is a LeakyReLU. This was done to avoid saturation for inputs to the activation function less than 0.
# * Dropout is employed as a regularization technique to prevent the Neural Network from overfitting to the data (the neural network is such a powerful model that it may find trends in the noise of the data and try to fit to the noise as well as the features of the dataset)
# * The loss function of the Neural Network is mean absolute error, as this is a common loss function employed in regression tasks.

# In[ ]:


import skorch
from skorch import NeuralNetRegressor

from sklearn.model_selection import RandomizedSearchCV

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[ ]:


class MyModule(nn.Module):
    def __init__(self,num_units=10,nonlin=F.relu,drop=.5):
        super(MyModule,self).__init__()
        
        self.module = nn.Sequential(
            nn.Linear(7,num_units),
            nn.LeakyReLU(),
            nn.Dropout(p=drop),
            nn.Linear(num_units,1),
        )
        
    def forward(self,X):
        X = self.module(X)
        return X


# # Wrap Pytorch Neural Network in Skorch Wrapper

# In[ ]:


net = NeuralNetRegressor(
    MyModule,
    criterion=nn.MSELoss,
    max_epochs=10,
    optimizer=optim.Adam,
    optimizer__lr = .005
)


# # Randomized Hyperparameter Search
# * Search the hyperparameter space of learning rate, epochs, number of hidden units, and dropout rate in the hidden layer to find the optimal hyperparameter combination that will minimize the expected loss.
# * 3 fold cross validation is used to get the mean validation loss and mean training loss over the training dataset.

# In[ ]:


lr = (10**np.random.uniform(-5,-2.5,1000)).tolist()
params = {
    'optimizer__lr': lr,
    'max_epochs':[300,400,500],
    'module__num_units': [14,20,28,36,42],
    'module__drop' : [0,.1,.2,.3,.4]
}

gs = RandomizedSearchCV(net,params,refit=True,cv=3,scoring='neg_mean_squared_error',n_iter=100)


# In[ ]:


get_ipython().run_cell_magic('capture', '', 'gs.fit(X_train_scaled,y_train);')


# In[ ]:


# Utility function to report best scores (found online)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


# review top 10 results and parameters associated
report(gs.cv_results_,10)


# # Display Learning curves to see if overfitting or underfitting data
# * By observing the learning curves, I can tell if the Neural Network overfitted or underfitted the data.
# * **Overfit** : if the training loss curve is significantly lower than the validation loss curve.
# * **Underfit**: if both the training loss curve and the validation loss curve are very high loss.
# * **Ideal**: both the training loss and validation loss curves have a minimal gap between them and converge to a very low loss.

# In[ ]:


# get training and validation loss
epochs = [i for i in range(len(gs.best_estimator_.history))]
train_loss = gs.best_estimator_.history[:,'train_loss']
valid_loss = gs.best_estimator_.history[:,'valid_loss']


# In[ ]:


plt.plot(epochs,train_loss,'g-');
plt.plot(epochs,valid_loss,'r-');
plt.title('Training Loss Curves');
plt.xlabel('Epochs');
plt.ylabel('Mean Squared Error');
plt.legend(['Train','Validation']);


# # See Regression Metrics to evaluate on test dataset

# In[ ]:


from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


# In[ ]:


# predict on test data
y_pred = gs.best_estimator_.predict(X_test_scaled.astype(np.float32))


# **Root mean squared error** (RMSE) says how far from the regression line the data points are on average. A lower RMSE is better.

# In[ ]:


# get RMSE
MSE(y_test,y_pred)**(1/2)


# Try to see how well the models predicted probability distribution overlaps with the actual probability distribution of the test set.

# In[ ]:


sns.kdeplot(y_pred.squeeze(), label='estimate', shade=True)
sns.kdeplot(y_test.squeeze(), label='true', shade=True)
plt.xlabel('Admission');


# In[ ]:


sns.distplot(y_test.squeeze()-y_pred.squeeze(),label='error');
plt.xlabel('Admission Error');


# * The **$R^{2}$** plot should have a positive slope of 1. The best possibe score is 1, whereas the worst score is 0.
# * **$R^{2}$** is a goodness of fit measure, and says how well the regression model explains the variability of the dataset

# In[ ]:


# show R^2 plot
print(r2_score(y_test,y_pred))
plt.plot(y_pred,y_test,'g*')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('$R^{2}$ visual');


# In[ ]:


# show where the big errors were
errors = np.where(abs(y_test-y_pred)>.2)
for tup in zip(y_test[errors],y_pred[errors]):
    print(tup)

