#!/usr/bin/env python
# coding: utf-8

# This is my First Attempt in getting hands. I have tried using grid search with SVC classifier to improve the Accuracy on the undersampled data which I got reference from one of the kaggle solutions.
# Please share your valuable feedback. 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the dataset

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")
data.head()


# # Checking the target classes

# In[ ]:


count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")


# reshape will not work in new versions , hence we have to use pd.DataFrame

# In[ ]:


from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(pd.DataFrame(data['Amount']))
data = data.drop(['Time','Amount'],axis=1)
data.head()


# #### 2. Assigning X and Y. No resampling.

# In[ ]:


#x = data.loc[:, data.columns != 'Class']
#y = data.loc[:, data.columns == 'Class']
x =  data.drop(['Class'],axis = 1)
y =  data['Class']


# In[ ]:


fraud_indices = np.array(data[data.Class == 1].index)


# In[ ]:


len(data[data.Class == 0])


# In[ ]:


# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
#random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

#X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
#y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
X_undersample = under_sample_data.drop(['Class'],axis = 1)
Y_undersample = under_sample_data['Class']
# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# # Splitting data into train and test set. 

# In[ ]:


from sklearn.model_selection import train_test_split

# Whole dataset
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, Y_train_undersample, Y_test_undersample = train_test_split(X_undersample
                                                                                                   ,Y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))


# * Using Grid Search to find the best hyper-parameter value 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


param_grid = {'C' : [0.01 , 0.1 , 1 , 10 , 100],'gamma' : [1,0.1,0.01,0.001]}


# In[ ]:


grid = GridSearchCV(SVC(),param_grid ,refit = True,verbose = 2)


# In[ ]:


grid.fit(X_train_undersample,Y_train_undersample)


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print('Accuracy Score',accuracy_score(Y_test,grid_predictions))
print(confusion_matrix(Y_test,grid_predictions))
print(classification_report(Y_test,grid_predictions))


# In[ ]:




