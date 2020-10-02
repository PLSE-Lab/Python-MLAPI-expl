#!/usr/bin/env python
# coding: utf-8

# **Application of Once Class SVM to detect anomaly** 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import svm
cc =  pd.read_csv("../input/creditcard.csv")


# In[ ]:


#from pydataset import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
cc= pd.read_csv("../input/melanoma/melanoma.csv")


# In[ ]:


# Data check. 
cc.head()


# In[ ]:


#cc=cc[cc['status']!=3]


# In[ ]:





# In[ ]:


#I observed an conflict in the name 'class'. Therefore, I have changed the name from class to category

cc= cc.rename(columns={'status': 'Category'})
cc.Category[cc.Category == 2] = 0


# In[ ]:


cc.Category.value_counts()


# In[ ]:


# For convinience, divide the dataframe cc based on two labels. 

nor_obs = cc.loc[cc.Category==0]    #Data frame with normal observation
ano_obs = cc.loc[cc.Category==1]    #Data frame with anomalous observation


# The given dataframe 'cc' is divided into three sets 
# 
# Training set: train_features
# 
# Test observations/features: X_test
# 
# Test labels: Y_test

# Once class SVM is trained with the observations of only one class. In this case, the algorithm is trained with first 200,000 observation of normal transactions. The remaining observations are merged with the anomalous observation to create a test set. 
# 
# 

# In[ ]:


# The given dataframe 'cc' is divided into three sets 
# Training set: train_features
# Test observations/features: X_test
# Test labels: Y_test


# In[ ]:


# Once class SVM is trained with the observations of only one class. In this case, the algorithm is trained with 
# first 200,000 observation of normal transactions. The remaining observation is merged with the anomalous observation 
# to create a test set. 

train_feature = nor_obs.loc[0:120, :]
train_feature = train_feature.drop('Category', 1)
Y_1 = nor_obs.loc[120:, 'Category']
Y_2 = ano_obs['Category']


# In[ ]:


# Creatng test observations/features

X_test_1 = nor_obs.loc[120:, :].drop('Category',1)
X_test_2 = ano_obs.drop('Category',1)
X_test = X_test_1.append(X_test_2)


# In[ ]:


# Setting the hyperparameters for Once Class SVM
from sklearn import svm
oneclass = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.95)

# I have used various combination of hyperparameters like linear, rbf, poly, gamma- 0.001, 0.0001, nu- 0.25, 0.5, 0.75, 0.95
# This combination gave me the most satisfactory results.# The remain data set is (after 200,000 observations) are appended with anomalous observations

Y_1 = nor_obs.loc[120:, 'Category']
Y_2 = ano_obs['Category']

Y_test= Y_1.append(Y_2)

#Y_test is used to evaluste the model


# In[ ]:


#train_feature


# In[ ]:


from sklearn import svm
# Training the algorithm with the features. 
# This stage is very time consuming processes. In my laptop it took more than an hour to train for 200,000 observations. 
# For rbf, the time taken is even more.

oneclass.fit(train_feature)


# In[ ]:


# Test the algorithm on the test set

fraud_pred = oneclass.predict(X_test)


# In[ ]:


# Check the number of outliers predicted by the algorithm

unique, counts = np.unique(fraud_pred, return_counts=True)
print (np.asarray((unique, counts)).T)


# In[ ]:





# In[ ]:


#Convert Y-test and fraud_pred to dataframe for ease of operation

Y_test= Y_test.to_frame()
Y_test=Y_test.reset_index()
fraud_pred = pd.DataFrame(fraud_pred)
fraud_pred= fraud_pred.rename(columns={0: 'prediction'})


# In[ ]:


fraud_pred[fraud_pred['prediction']==1]=0
fraud_pred[fraud_pred['prediction']==-1]=1


# In[ ]:


print(fraud_pred['prediction'].value_counts())
print(sum(fraud_pred['prediction'])/fraud_pred['prediction'].shape[0])


# In[ ]:


print(Y_test['Category'].value_counts())
sum(Y_test['Category'])/Y_test['Category'].shape[0]


# In[ ]:





# In[ ]:


#let's built a ROC curve to validate the result
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate,  thresholds = roc_curve(Y_test['Category'],fraud_pred['prediction'])
roc_auc=auc(true_positive_rate,false_positive_rate)
print("true_positive_rate- ", true_positive_rate)
print("false_positive_rate- ", false_positive_rate)
print("roc_auc- ", roc_auc)


# In[ ]:


plt.title("Receiver Operating Curve")
plt.plot(false_positive_rate,true_positive_rate,'b',label=roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


##Performance check of the model

TP = FN = FP = TN = 0
for j in range(len(Y_test)):
    if Y_test['Category'][j]== 0 and fraud_pred['prediction'][j] == 1:
        TP = TP+1
    elif Y_test['Category'][j]== 0 and fraud_pred['prediction'][j] == -1:
        FN = FN+1
    elif Y_test['Category'][j]== 1 and fraud_pred['prediction'][j] == 1:
        FP = FP+1
    else:
        TN = TN +1
print (TP,  FN,  FP,  TN)


# In[ ]:


# Performance Matrix

accuracy = (TP+TN)/(TP+FN+FP+TN)
print (accuracy)
sensitivity = TP/(TP+FN)
print (sensitivity)
specificity = TN/(TN+FP)
print (specificity)


# Following results were obtained 
# 
# accuracy= 99.9%
# 
# sensitivity = 100%
# 
# specificity = 75%
# 
# Once class SVM has shown a very promising performance for this dataset with near 90% detection of anomaly and very few false alarm. This can be a starting point for fine tuning the algorthm to improve the specificity, keeping other things constant. Tuning the hyperparameters are very time consuming process and the Kaggle kernal stops after some time. Therefore, O couldnt run the code. I have just shown my codes in the cell. I am sure this code will run because i have ran it in my Jupyter note book. I have also isolation forest in my previous kernal. Once class SVM seems to outperform isolation forest in this case. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




