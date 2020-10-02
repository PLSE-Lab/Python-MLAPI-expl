#!/usr/bin/env python
# coding: utf-8

# A heart attack occurs when the flow of blood to the heart is blocked. The blockage is most often a buildup of fat, cholesterol and other substances, which form a plaque in the arteries that feed the heart (coronary arteries).
# 
# The plaque eventually breaks away and forms a clot. The interrupted blood flow can damage or destroy part of the heart muscle.
# 
# A heart attack, also called a myocardial infarction, can be fatal, but treatment has improved dramatically over the years, however the diagnosis for heart diseases are expensive and complex. 
# 
# The tests you'll need to diagnose your heart disease depend on what condition your doctor thinks you might have. No matter what type of heart disease you have, your doctor will likely perform a physical exam and ask about your personal and family medical history before doing any tests. Besides blood tests and a chest X-ray, tests to diagnose heart disease can include[1]:
# 
# Electrocardiogram (ECG).
# Holter monitoring. 
# Echocardiogram. 
# Stress test. 
# Cardiac catheterization.
# Cardiac computerized tomography (CT) scan. 
# Cardiac magnetic resonance imaging (MRI).
# 
# [1]https://www.mayoclinic.org/diseases-conditions/heart-disease/diagnosis-treatment/drc-20353124
# 
# Lets see how Machine Learning Models are used in predicting Heart Attack in human.
# 
# Context
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
# 
# Content
# 
# Attribute Information: 
# *  age 
# * sex 
# * chest pain type (4 values) 
# * resting blood pressure 
# * serum cholestoral in mg/dl 
# * fasting blood sugar > 120 mg/dl
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved 
# * exercise induced angina 
# * oldpeak = ST depression induced by exercise relative to rest 
# * the slope of the peak exercise ST segment 
# * number of major vessels (0-3) colored by flourosopy 
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/heart.csv")
dataset.head(10)


# * we need to check if there is any Null values in the dataset. 
# * If there are any Null then we need to impute the values.

# In[ ]:


dataset.isnull().values.any()


# * Check the datatypes to see if we need to perform encoding categorical data

# In[ ]:


dataset.dtypes


# * Feature selection is one of the most important step in machine learning.
# * Irrelevant Parameters will lower the performance of the model.
# * lets try out the first method, correlation heat map.

# In[ ]:


Corr = dataset.corr()
Corr


# * plotting heatmap to analyze the correlation of all the parameters.

# In[ ]:


sb.heatmap(Corr,vmin=0, vmax=1, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .5})


# * second method of feature selction
# * Univariate Selection

# In[ ]:


X = dataset.iloc[:,0:13]  
y = dataset.iloc[:,-1]
#apply SelectKBest class to extract best features
parameters = SelectKBest(score_func=chi2, k=13)
fit = parameters.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(13,'Score'))  #print 10 best features


# * Third Method of feature selection 
# * Feature Importance

# In[ ]:


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(13).plot(kind='barh')
plt.show()


# * Now let us build our model
# * we have already seperated x and y dataset
# * lets choose test and training set.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# lets scale the features
# 
# Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# * Fitting SVM to the Training set

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# * lets see how many corrct and in incorrect predictions through confusion matrix
# * Making the Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# * our model is making 65 correct predictions and 11 incorrect prediction

# * Lets find out the Recall and Specifity 

# In[ ]:


Recall = cm[0,0]/(cm[0,0]+cm[1,0])
print('Recall : ', Recall )

Sp = cm[1,1]/(cm[1,1]+cm[0,1])
print('Specifity : ', Sp)


# * Our Model is making good prediction with SVM linear kernel method
