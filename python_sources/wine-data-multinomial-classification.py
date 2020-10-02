#!/usr/bin/env python
# coding: utf-8

#  # Multiclass Classification - Ensemble Method
#  
#  ## Introduction
#  
# These data are the results of a chemical analysis of **wines grown** in the same region in Italy but derived from three different cultivars. 
# The analysis determined the quantities of 13 constituents found in each of the three types of wines.
#  
#  

# ## 1. Import dependent libraries

# In[ ]:


# Basic libraries
import pandas as pd
import numpy as np
import os

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# SKlearn related libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, zero_one_loss, hamming_loss

# Boosting technique algorithm
import xgboost as xgb


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## 2.Load the dataset

# In[ ]:


# Dataset path
DATA_PATH = "/kaggle/input/wineuci/Wine.csv"

columns = ['class','alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'od280/od315_of_diluted_wines', 'proline']

wine_data = pd.read_csv(DATA_PATH, names=columns, header=0)

wine_data.info()
print("=="*40)
wine_data.head()


# In[ ]:


# transform the label into 0, 1, 2
def trans_class(class_label):
    return int(class_label) - 1


# In[ ]:


wine_data['class'] = wine_data['class'].apply(trans_class)


# In[ ]:


np.unique(wine_data['class'])


# ## 3.Exploratory Data Analysis

# In[ ]:


wine_data.describe().T


# In[ ]:


print(f"Is there any null values : {wine_data.isnull().sum().any()}")


# In[ ]:


sns.set(style='whitegrid', palette='muted')

# Pairplot to see the attribute comparison
fig, ax = plt.subplots(1,2,figsize=(12,6))

sns.distplot(wine_data['alcohol'], kde=True, hist=True, ax=ax[0])

sns.distplot(wine_data['malic_acid'], kde=True, hist=True, ax=ax[1])

plt.show()

g = sns.jointplot(x=wine_data['alcohol'], y=wine_data['malic_acid'], color='r')


# In[ ]:


wine_data['class'].value_counts().plot.bar()


# ## 4. Prepare dataset

# In[ ]:


# Making X and Y data from the dataset
X = wine_data.loc[:, wine_data.columns != 'class'].values
y = wine_data['class'].values


# In[ ]:


print(f"Train shape : {X.shape}, Label : { y.shape}")


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape : {X_train.shape}, Label : { y_train.shape}")


# ### DMatrix
# 
# XGB algorithm expects the data in 'DMatrix' form. We will use the data to construct Dmatrix object.

# In[ ]:


dm_train = xgb.DMatrix(data=X_train, label=y_train)
dm_test = xgb.DMatrix(data=X_test)


# In[ ]:


# Key parameters
params = {
    'max_depth': 6,
    'min_child_weight':1,
    'objective': 'multi:softmax',
    'subsample':1,
    'colsample_bytree':1,
    'num_class': 3,
    'n_gpus': 0
}


# ## 5. Model

# In[ ]:


xgb_clf = xgb.train(params, dm_train) # Train the model with dataset


# In[ ]:


# Prediction
predictions = xgb_clf.predict(dm_test)


# In[ ]:


predictions


# ## 6. Evaluation

# In[ ]:


print("Classification Report \n {}".format(classification_report(y_test, predictions)))


# ### Misclassification Rate
# 
# In multilabel classification, the zero_one_loss function corresponds to the subset zero-one loss: for each sample, the entire set of labels must be correctly predicted, otherwise the loss for that sample is equal to one.
# 
# If normalize is True, return the fraction of misclassifications (float), else it returns the number of misclassifications (int). The best performance is 0.

# In[ ]:


print("Misclassification rate {}".format(zero_one_loss(y_test, predictions, normalize=False)))


# ## Hamming Loss
# 
# In multiclass classification, the Hamming loss corresponds to the Hamming distance between y_true and y_pred which is equivalent to the subset zero_one_loss function, when normalize parameter is set to True.

# In[ ]:


print("Hamming rate {:.2f}".format(hamming_loss(y_test, predictions)))

