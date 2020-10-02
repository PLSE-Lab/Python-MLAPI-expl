#!/usr/bin/env python
# coding: utf-8

# I understand Indians are most prone to Diabetes.Genetics,Lifestye and Food Habits can be attributed to it.I will be exploring the dataset and predicting Diabetes Paatients based on Body Parametes.Here I will demonstrate data exploration and how to build a neural network.This kernel is a work in Process.If you like it please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Importing Python Modules

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()


# ### The details of Features are as follows: 
# 
# 1.Pregnancies: Number of times Pregnant
# 
# 2.Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# 3.Blood Pressure: Diastolic blood pressure (mm Hg) 
# 
# 4.Skin Thickness: Triceps skin fold thickness (mm)
# 
# 5.Insulin: 2-Hour serum insulin (mu U/ml)
# 
# 6.BMI: Body mass index (weight in kg/(height in m)^2)
# 
# 7.Diabetes Pedigree Function: Diabetes pedigree function
# 
# 8.Age: Age (years)
# 

# ### Misssing Values

# In[ ]:


#There are 0 values in the dataset in the Glucose,BloodPressure,SkinThickness, Insulin and BMI, we need to replace them with the NAN 

df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0, np.NaN)


# In[ ]:


df.isnull().any()


# In[ ]:


df.isna().sum()


# In[ ]:


#Replacing the null values with the mean and median respectively

df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(),inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(),inplace=True)
df['Insulin'].fillna(df['Insulin'].median(),inplace=True)
df['BMI'].fillna(df['BMI'].median(),inplace=True)


# ### Histogram

# In[ ]:


df.hist(figsize=(12,10));


# Above plot shows the distribution of all the features in the datset.Most of them dont have a normal distribution.Later we may use feature scaling to overcome this problem.

# ### Pair Plot

# In[ ]:


import seaborn as sns
sns.pairplot(df,hue='Outcome')


# From the pair plot we can see that there is no clear seperation for the data.So it will not be easy to predict he Diabetes patients.

# ### Correlations

# In[ ]:


sns.heatmap(df.corr(),annot = True);


# So we can see that Age and Number of Pregancy has high correlation.This is obvious as older people have had more chance of being pregnant.
# 
# Glucose and Diabetics has high correlation.People with higher glucose level are generally diabetic.

# ### Exploring the Data

# In[ ]:


df.info()


# So we have 786 entries in the dataset.Features have all numerical values.

# In[ ]:


df.describe()


# So the features have high variance.We need to scake the data for better model prediction.

# ### Scaling the data

# In[ ]:


from sklearn.preprocessing import StandardScaler 
from keras.utils import to_categorical


# In[ ]:


sc = StandardScaler()
X = sc.fit_transform(df.drop('Outcome',axis=1))
y = df['Outcome'].values
y_cat = to_categorical(y)


# Standard scaler considerd the mean and standard deviation value to make the calculation.Mean value is subtracted from each value and then divided by the standard deviation.So the features will be scaled to mean of Zero and with a standard deviation of one.

# In[ ]:


#X


# In[ ]:


X.shape


# In[ ]:


#y_cat


# ### Building the Neural Network

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y_cat,random_state=22,test_size=0.2)


# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam


# In[ ]:


model = Sequential()
model.add(Dense(32,input_shape=(8,),activation ='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
#model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(Adam(lr=0.05),loss='categorical_crossentropy',metrics=['accuracy'])


# So our Neural network has 2 layers with 8 input features,32 nodes,relu as activation function,Output layer has two nodes to predict he categorical values,Softmax as the output function,Adam optimizer,Loss as categorical cross entropy and metrics of measurnment as accuracy.

# In[ ]:


model.fit(X_train,y_train,epochs=20,verbose=2,validation_split=0.1)


# In[ ]:


model.summary()


# ### Calulation of parameters for each layer
# 32*8 + 32 = 288
# 
# 32*32 + 32 = 1056
# 
# 32*2 + 2 = 66

# In[ ]:


y_pred = model.predict(X_test)


# y_pred is gives us the probability values.We have to convert them to binary class using argmax functionas shown below.

# In[ ]:


y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)


# ### Evaluating Model Performance

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# #### Accuracy

# In[ ]:


print('Accuracy of model is:',accuracy_score(y_test_class,y_pred_class))


# #### Classification report

# In[ ]:


print(classification_report(y_test_class,y_pred_class))


# #### Confusion matrix

# We can see that the Precision and Recall are no for 1 ie Diabetes prediction.We can improve this by further optimizing out model or by increasing the data.Here we have small set of data.So this kernel is just a demonstration of how to build a neural network to predict Diabetes.

# In[ ]:


confusion_matrix(y_test_class,y_pred_class)


# ### Making Sense of Our model

# In[ ]:


pd.Series(y_test_class).value_counts()


# In[ ]:


pd.Series(y_test_class).value_counts()/len(y_test_class)


# So in our data 64.9 % of the patients dont have diabetes.So without building a machine learning model if we were to categorise out patients 100 % as no diabetic we would be accurate 64.9 % of the times.Our model accuracy is 72% so we are better than the worst case.

# ### Benchmarking Deep Learning with Machine Learning Algorithms

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

for mod in [RandomForestClassifier(),SVC(),GaussianNB()]:
    mod.fit(X_train,y_train[:,1])
    y_pred = mod.predict(X_test)
    print("="*80)
    print(mod)
    print("_"*80)
    print("Accuracy score:{:0.3}".format(accuracy_score(y_test_class,y_pred)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_class,y_pred))


# So we can see the accuracy level with Random Forest,SVM and Gaussian Naive Bayes are not better than the result obtained by machine learning.

# ### Which Features are More Important to Predict Diabetes

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 
model= RandomForestClassifier(n_estimators=100,random_state=0)
X=df[df.columns[:8]]
Y=df['Outcome']
model.fit(X,Y)
pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)


# We can see that Glucose content in the blood is the major indicator of diabetes in a patient followed by Body Mass Index and Age.
