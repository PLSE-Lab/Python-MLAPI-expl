#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


iris_df=pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
iris_df.head()


# # Data Inspection

# In[ ]:


print("Shape of the data frame: ",iris_df.shape)
print("Total null values: ",iris_df.isna().sum().sum())
print("Duplicate values: ",iris_df.duplicated().sum() )


# > Removing the duplicated entries

# In[ ]:


iris_df.drop_duplicates(inplace=True)
print("Shape of the data frame: ",iris_df.shape)
print("\n")
print("Species categories with its count \n",iris_df["species"].value_counts())


# In[ ]:


iris_df.describe()


# > On comparng the mean and median values for these four paramater we can observe skewness

# In[ ]:


#iris_df.plot(kind='box')
#plt.show()
sns.boxplot(data=iris_df)


# > from the above box plot it's clear 
# 1. sepal_width has outliers and it's right skewed
# 2. petal_length and petal_with are left skewed
# 3. sepal_length is symmetrical

# # Measure of skewness
# #https://pythontic.com/pandas/dataframe-computations/skew

# In[ ]:


skewness_value=iris_df.skew(axis=0)
print("Measure of skewness column wise:\n",skewness_value)  


# 1. > If skewness value lies above +1 or below -1, data is highly skewed. If it lies between +0.5 to -0.5, it is moderately skewed. If the value is 0, then the data is symmetric
# 1. > For Symmetric distribution, mean=median
# 1. > Positively skewed data:
# If tail is on the right as that of the second image in the figure, it is right skewed data. It is also called positive skewed data.
# Common transformations of this data include square root, cube root, and log.
# 1. > Negatively skewed data:
# If the tail is to the left of data, then it is called left skewed data. It is also called negatively skewed data.
# Common transformations include square , cube root and logarithmic.
# * https://medium.com/@TheDataGyan/day-8-data-transformation-skewness-normalization-and-much-more-4c144d370e55* 

# # Measure of kurtosis
# #https://pythontic.com/pandas/dataframe-computations/kurtosis
# #https://pythontic.com/pandas/dataframe-computations/kurtosis

# In[ ]:



kurtosis_values=iris_df.kurt(axis=0)
print(kurtosis_values)


# # Handling Categorical values
# #https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621

# In[ ]:



from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
iris_df["species"]=label_encoder.fit_transform(iris_df["species"])
print(iris_df.head(10))
print("\n")
print(iris_df["species"].value_counts())


# # Feature Selection

# In[ ]:


X=iris_df.drop(["species"],axis=1)
Y=iris_df["species"]


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=3)


# # Selected model - Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logistic_reg=LogisticRegression()


# # Accuracy prior scaling/normalizing

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
logistic_reg.fit(X_train1,Y_train1)
predicted_result1=logistic_reg.predict(X_test1)
print("Accuracy Score: ",accuracy_score(Y_test1, predicted_result1))

print("Confusion Matrix:\n ",confusion_matrix(Y_test1, predicted_result1))

print("Classification Report:\n ",classification_report(Y_test1, predicted_result1))


# >  From the above result it is clear that the model is 96% accurate and f1score as 1 indicating very low false positive and false negative prediction

# 
# # Implementing Scaling,LDA followed by generating a model:-

# 1.Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train2,X_test2,Y_train2,Y_test2=train_test_split(X,Y,test_size=0.2,random_state=5,stratify=Y)


# 2. Standard Scaler

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train2=ss.fit_transform(X_train2)
X_test2=ss.transform(X_test2)


# 3.LDA

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
LDA=LinearDiscriminantAnalysis()
X_train2=LDA.fit_transform(X_train2,Y_train2)
X_test2=LDA.transform(X_test2)


# 4.LG Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train2,Y_train2)
y_prediction=LR.predict(X_test2)


# 5.Analysis

# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("\n Accuracy Score:",accuracy_score(Y_test2,y_prediction))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test2,y_prediction))
print("Classification Report:")
print(classification_report(Y_test2,y_prediction))

