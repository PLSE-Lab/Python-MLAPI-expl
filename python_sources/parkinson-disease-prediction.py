#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import KFold


# In[ ]:


parkinson_df = pd.read_csv('../input/parkinsons2.csv')


# In[ ]:


parkinson_df.head().transpose()


# #### Description of the columns:
# MDVP:Fo(Hz) - Average vocal fundamental frequency 
# 
# MDVP:Fhi(Hz) - Maximum vocal fundamental frequency 
# 
# MDVP:Flo(Hz) - Minimum vocal fundamental frequency 
# 
# MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency 
# 
# MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude 
# 
# NHR,HNR - Two measures of ratio of noise to tonal components in the voice 
# 
# RPDE,D2 - Two nonlinear dynamical complexity measures 
# 
# DFA - Signal fractal scaling exponent 
# 
# spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation 
# 
# status - Health status of the subject (one) - Parkinson's, (zero) - healthy

# In[ ]:


parkinson_df.columns


# We observe that there are 195 rows and 23 columns in the given dataset.

# In[ ]:


#Since column names are big it will be easy to do plots and calculations if the column names are small
parkinson_df.columns = ['Fo','Fhi','Flo','Jitter(%)','Jitter(Abs)','RAP','PPQ','DDP','Shimmer','Shimmer(dB)','APQ3','APQ5','APQ','DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE','status']


# In[ ]:


parkinson_df.info()


# #### From the above information, we observe that all the given features are continuous except 'status'(since given in the description).

# In[ ]:


parkinson_df.describe().transpose()


# In[ ]:


parkinson_df[parkinson_df.isnull().any(axis=1)]


# #### From the above two cells,we observe that there are no missing values in the given dataset.

# In[ ]:


parkinson_df.boxplot(figsize=(24,8))


# #### From the above box plots, we observe that there are less outliers. So, the model will not be affected by the outliers.

# In[ ]:


parkinson_df.corr()


# #### From the above table,we observe that 'Fhi'(Maximum vocal fundamental frequency) and 'NHR'(Measure of ratio of noise to tonal components in the voice) are having less correlation(-0.166136 and 0.189429 respectively) with respect to status . So, we can drop these two features based on their correlations.

# In[ ]:


parkinson_df['status'].value_counts().sort_index()


# #### Most are having Parkinson disease. The ratio is almost 1:3 in favor of status 1. So, the model's ability to predict status 1 will be better than predicting status 0.

# In[ ]:


X = parkinson_df.drop(['Fhi','NHR','status'],axis=1)
Y = parkinson_df['status']


# In[ ]:


#Splitting the data into train and test in 70/30 ratio with random state as 2.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)


# In[ ]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)


# In[ ]:


Y1_predict = LR.predict(X_test)
Y1_predict


# In[ ]:


Y_acc = metrics.accuracy_score(Y_test,Y1_predict)
print("Accuracy of the model is {0:2f}".format(Y_acc*100))
Y_cm=metrics.confusion_matrix(Y_test,Y1_predict)
print(Y_cm)


# In[ ]:


#Sensitivity
TPR=Y_cm[1,1]/(Y_cm[1,0]+Y_cm[1,1])
print("Sensitivity of the model is {0:2f}".format(TPR))


# In[ ]:


#Specificity
TNR=Y_cm[0,0]/(Y_cm[0,0]+Y_cm[0,1])
print("Specificity of the model is {0:2f}".format(TNR))


# In[ ]:


Y_CR=metrics.classification_report(Y_test,Y1_predict)
print(Y_CR)


# #### So, for the above model, Accuracy is 81.35% , Sensitivity is 93.61% and Specificity is 33.33%

# In[ ]:


fpr,tpr, _ = roc_curve(Y_test, Y1_predict)
roc_auc = auc(fpr, tpr)

print("Area under the curve for the given model is {0:2f}".format(roc_auc))
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# #### Area under the curve is 0.6347 implies it is a good model.

# In[ ]:


X = parkinson_df.drop(['Fhi','NHR','status'],axis=1)
Y = parkinson_df['status']


# In[ ]:


# K-fold cross validation for the given model:
#Since the dataset contains 197 rows, we are taking the number of splits as 3
kf=KFold(n_splits=3,shuffle=True,random_state=2)
acc=[]
for train,test in kf.split(X,Y):
    M=LogisticRegression()
    Xtrain,Xtest=X.iloc[train,:],X.iloc[test,:]
    Ytrain,Ytest=Y[train],Y[test]
    M.fit(Xtrain,Ytrain)
    Y_predict=M.predict(Xtest)
    acc.append(metrics.accuracy_score(Ytest,Y_predict))
    print(metrics.confusion_matrix(Ytest,Y_predict))
    print(metrics.classification_report(Ytest,Y_predict))
print("Cross-validated Score:{0:2f} ".format(np.mean(acc)))


# #### So, for the above K-fold cross validation model, Precision for each fold is 0.82,0.89,0.88 respectively, Recall for each fold is 0.83,0.88,0.88 respectively and the overall Accuracy is 86.15%

# In[ ]:


#Accuracy for each fold
acc


# In[ ]:


#Error
error=1-np.array(acc)
error


# In[ ]:


# Variance Error of the model
np.var(error,ddof=1)


# In[ ]:


fpr,tpr, _ = roc_curve(Ytest, Y_predict)
roc_auc = auc(fpr, tpr)

print("Area under the curve for the given model is {0:2f}".format(roc_auc))
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# #### Area under the curve is 0.8416 implies it is a very good model.

# ### By Comparing the above two models, we observe that by doing K-fold cross validation, accuracy has been improved from 81.35% to 86.15% and area under the curve has been improved from 0.6347 to 0.8416. So, we can conclude that K-fold cross validation model will be the better model for this dataset.
