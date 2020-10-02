#!/usr/bin/env python
# coding: utf-8

# # Import libs

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# # Import dataset

# In[ ]:


file_train = '../input/train.csv'
file_valid = '../input/valid.csv'
file_test = '../input/test.csv'
file_exemplo = '../input/exemplo_resultado.csv'
data_train = pd.read_csv(file_train)
data_valid = pd.read_csv(file_valid)
data_test = pd.read_csv(file_test)
data_exemplo = pd.read_csv(file_exemplo)
print(len(data_train), len(data_valid), len(data_test))


# In[ ]:


corrmat = data_train.corr()
plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, cmap="YlGnBu", vmax=1, square=True, linewidths=0.1)


# In[ ]:


Y = data_train['default payment next month']
ID_predict = data_exemplo['ID']
data_train = data_train.drop('default payment next month', axis=1)

data_all = data_train.append(data_valid).append(data_test)
data_all = data_all.drop('ID', axis=1)

len(data_all)


# # Pre-processing

# ## Transform qualitative data in quantitative date with Dummies

# In[ ]:


data_all = pd.concat([data_all, pd.get_dummies(data_all['SEX'])], axis=1);
data_all = pd.concat([data_all, pd.get_dummies(data_all['EDUCATION'])], axis=1);
data_all = pd.concat([data_all, pd.get_dummies(data_all['MARRIAGE'])], axis=1);
data_all = pd.concat([data_all, pd.get_dummies(data_all['AGE'])], axis=1);

data_all = data_all.drop(['SEX', 'EDUCATION', 'MARRIAGE', 'AGE'], axis=1)


# ## Split data in training and valid

# In[ ]:


from sklearn.model_selection import train_test_split
# Create X
data_train = data_all[:(len(data_train))]
data_valid = data_all[(len(data_train)):((len(data_train))+(len(data_valid)))]
data_test = data_all[((len(data_train))+(len(data_valid))):]

print()

X = data_train
print(len(data_train), len(data_valid), len(data_test))


# # Processing

# ## Training Model

# ### Naive_bayes

# In[ ]:


# from sklearn.naive_bayes import GaussianNB


# model = GaussianNB()
# model.fit(X, Y)


# ### DecisionTree

# In[ ]:


from sklearn import tree

model = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 20, min_samples_leaf=0.2, min_samples_split=0.2, max_depth=3)
model.fit(X, Y)


# ## Model
# [DecisionTree](https://www.displayr.com/wp-content/uploads/2018/07/decision-tree-768x430.png)

# # Post processing

# ## Test using model with data valid

# In[ ]:


data_predict = data_valid.append(data_test)
Y_predict = model.predict(data_predict)

Y_predict
len(Y_predict)


# ## Generating output file.csv

# In[ ]:


Default_exemplo = data_exemplo['Default']
print(len(Y_predict), len(ID_predict))


# In[ ]:


data_to_submit = pd.DataFrame({
    'ID': ID_predict,
    'Default':Y_predict
})

data_to_submit.to_csv('csv_to_submit.csv', index = False)


# # Confusion matrix
# 
# ![ConfusionMatrix](http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix_files/confusion_matrix_1.png)

# # Metric
# ## Receiver Operating Characteristic Curve(ROC)
# ### True Positive Rate (TPR)
# ![TPR](https://latex.codecogs.com/gif.latex?TPR%20%3D%20%5Cfrac%7BTP%7D%20%7BTP%20&plus;%20FN%7D)
# 
# ### False Positive Rate (FPR)
# ![FPR](https://latex.codecogs.com/gif.latex?FPR%20%3D%20%5Cfrac%7BFP%7D%20%7BFP%20&plus;%20TN%7D)
# 
# <img src="https://i.stack.imgur.com/yn7ke.jpg" alt="drawing" width="400"/>
# 
# ## Area Under the ROC Curve(AUC)
# 
# <img src="https://developers.google.com/machine-learning/crash-course/images/AUC.svg" alt="drawing" width="400"/>
