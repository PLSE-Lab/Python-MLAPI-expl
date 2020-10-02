#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
dataset.head()


# the different count of the counts of the classes

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
print("no of people with no diabestes ",dataset.Outcome.value_counts()[0])
print("no of people with diabestes ",dataset.Outcome.value_counts()[1])
sns.countplot(dataset.Outcome)
plt.savefig('no of values of the each datapoint.jpg')


# finding the correlation matrix and the heat map to find the relation of the features with each other. The correlation matrix values tell how much a feature is related to other feature. this helps in ruling out useless features and keeing the importtant ones

# In[ ]:


heat=dataset.corr()
sns.heatmap(heat)
plt.savefig('correlation_heatmap.jpg')


# In[ ]:


sns.pairplot(dataset,hue='Outcome')
plt.savefig('fig3.jpg')


# In[ ]:


sns.regplot(x=dataset.Pregnancies,y=dataset.Outcome)
plt.savefig('pregnancy vs outcome relation.jpg')


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# making the x dataste by dropping the outcome column and y dataset

# In[ ]:


x=dataset.drop(['Outcome'],axis=1)
x.head()
x=np.array(x)
print(x.shape)


# In[ ]:


y=dataset[['Outcome']]
y.head()
y=np.array(y).ravel()
print(y.shape)


# **performing anova test for feature selection and extraction. anova test is usefull when the output variable is categorical and input variable is numberical **

# In[ ]:


test=SelectKBest(score_func=f_classif,k='all')
fit=test.fit(x,y)
print("the signicance of the respective data features wrt to the output variable are \n",fit.scores_)


# **by the above anova test on the  input features among outcome variable we can see that the 5 features namely pregnancy , glucose , bmi , diabetes_pedigree_function and age are the most important factors in the prediction of the outcome **

# In[ ]:


test1=SelectKBest(score_func=f_classif,k=5)
fit1=test1.fit(x,y)
print(fit1.scores_)


# transforming the x dataset acc to the 5 most important features in the dataset

# In[ ]:


x=fit1.transform(x)


# distribution of pregnancies throughout the dataset

# In[ ]:


sns.distplot(x[:,0],rug=True,kde=True)
plt.savefig('distribution of the preganancies throughout the dataset.jpg')


# By the plot we can see the mostly the women who have been 0 or 1 time pregnant are wihtin the dataset cluster

# In[ ]:


sns.distplot(x[:,1],rug=True,kde=True)
plt.savefig('glusose distribution.jpg')


# In[ ]:


sns.distplot(x[:,2],rug=True,kde=True)
plt.savefig('BMI distribution.jpg')


# In[ ]:


g=sns.PairGrid(dataset)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6)
plt.savefig('fig7.jpg')


# In[ ]:


sns.boxplot(x='Pregnancies',y='Glucose',hue='Outcome',data=dataset)
plt.savefig('fig8.jpg')


# In[ ]:


sns.clustermap(heat)
plt.savefig('fig9.jpg')


# formulating the X_data and Y_data for training 

# In[ ]:


X_data=pd.DataFrame(x,columns=['Pregnancies','Glucose','BMI','diabetespedigreefunction','age'])
X_data.head()


# In[ ]:


Y_data=pd.DataFrame(y,columns=['Outcome'])
Y_data.head()


# peforming train and test split to keep both data points separate for effecive evaluation of the model after training" 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.25)


# In[ ]:


sns.countplot(np.array(Y_train).ravel())
plt.savefig('count of datapoints in y_train.jpg')


# since from the above countplot it is visible that there are diffent number of outcome points . this will lead to imbalanced classification. to avoid this SMOTE technique has been used to get equal number of datapoint . this technique does not lead to gain in datapoint

# In[ ]:


from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train,Y_train=sm.fit_sample(np.array(X_train),np.array(Y_train).ravel())


# In[ ]:


sns.countplot(Y_train.ravel())
plt.savefig('count of datapoint in the train data.jpg')


# making the logistic regression model for classification

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,confusion_matrix,roc_auc_score,accuracy_score


# fitting the model

# In[ ]:


model=LogisticRegression(solver='lbfgs',max_iter=200)
model.fit(X_train,Y_train)


# getting the probabilities of the each daatpoint in the x_test data which will then be used for threshold tuning

# In[ ]:


Y_predict_probs=model.predict_proba(X_test)
Y_predict_probs=Y_predict_probs[:,1]


# In[ ]:


lr_auc=roc_auc_score(Y_test,Y_predict_probs)
print("the area under the roc curve is ",lr_auc)


# getting the false positive rate , true positive rate and different threshold value from the roc curve to find the most optimum threshold value

# In[ ]:


fpr,tpr,thresholds=roc_curve(Y_test,Y_predict_probs)


# In[ ]:


plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
plt.savefig('roc_curve.jpg')


# gmeans tchniques is used to find the most optimumvalue of the threshold from the roc curve fpr and tpr values

# In[ ]:


g_means= np.sqrt(tpr*(1-fpr))
ix=np.argmax(g_means)
print("the optimum threshold value for the given dataset is ", thresholds[ix])


# In[ ]:


plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='Logistic')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
plt.savefig('roc_curve_optimum_value.jpg')


# predicting the values according the new threshold value

# In[ ]:


THRESHOLD=thresholds[ix]
y_predict=np.where(model.predict_proba(X_test)[:,1]>THRESHOLD ,1,0)


# In[ ]:


acc=accuracy_score(Y_test,y_predict)
print("the accuracy of the model is ",acc)


# In[ ]:


confusion_matrix(Y_test,y_predict)


# building the model without any feature engineering

# In[ ]:


model2=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
model2.fit(x_train,y_train)
preds=model2.predict(x_test)
print("the accuracy of the model without feature engineering is ",accuracy_score(y_test,preds))


# In[ ]:


confusion_matrix(y_test,preds)


# In[ ]:




