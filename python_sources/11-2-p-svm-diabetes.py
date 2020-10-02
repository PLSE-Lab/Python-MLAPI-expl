#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machines - SVM

# # Context
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the 
# dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in 
# the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all 
# patients here are females at least 21 years old of Pima Indian heritage.

# # Content
# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the 
# number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

# # Can you build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

# # 1. Import Libraries and load the dataset

# In[ ]:


#Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


diabetes = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
print(diabetes.columns)


# In[ ]:


diabetes.head()


# # 2. Check dimension of dataset

# In[ ]:


print("dimension of diabetes data: {}".format(diabetes.shape))
#The diabetes dataset consists of 768 data points, with 9 features


# # 3. Check distribution of dependent variable, Outcome and plot it

# In[ ]:


print(diabetes.groupby('Outcome').size())


# # 4. Out of 768 data points, 500 are labeled as 0 and 268 as 1.
# Outcome 0 means No diabetes, outcome 1 means diabetes, Give a countplot

# In[ ]:


import seaborn as sns

sns.countplot(diabetes['Outcome'],label="Count")
#data has more No diabetic data as compared to diabetic data which would give a biased prediction towards no diabetic


# In[ ]:


diabetes.info()


# # 5. Check data distribution using summary statistics and provide your findings(Insights)

# In[ ]:


diabetes.describe().transpose()


# In[ ]:


# Few Insights
# Min blood pressure of 0 is invalid, so impute it with appropriate values. Same with few other variables like BMI
# Mean and Median values of Insuline is very different
# Insuline has very high Standard deviation
# We will ignore all these issues for now to concentrate more on Model


# # 6. Do correlation analysis and bivariate viualization with Insights

# In[ ]:


colormap = plt.cm.viridis # Color range to be used in heatmap
plt.figure(figsize=(15,15))
plt.title('Pearson Correlation of attributes', y=1.05, size=19)
sns.heatmap(diabetes.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
#There is no strong correlation between any two variables.
#There is no strong correlation between any independent variable and class variable.


# # 7. Plot a scatter Matrix

# In[ ]:


spd = pd.plotting.scatter_matrix(diabetes, figsize=(20,20), diagonal="kde")


# # 8. Do train and test split with stratify sampling on Outcome variable to maintain the distribution of dependent variable

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=11)


# In[ ]:


X_train.shape


# # 9. Train Support Vector Machine Model

# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))


# In[ ]:


#The model overfits substantially with a perfect score on the training set and only 65% accuracy on the test set.

#SVM requires all the features to be on a similar scale. We will need to rescale our data that all the features are approximately
#on the same scale and than see the performance


# # 10. Scale the data points using MinMaxScaler

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# # 11. Fit SVM Model on Scale data and give your observation

# In[ ]:


svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))


# Scaling the data made a huge difference.But now we are actually in an underfitting regime, where training and test set 
# performance are quite similar but less close to 100% accuracy.
# From here, we can try increasing either C or gamma to fit a more complex model.

# # 12. Try improving the model accuracy using C=1000

# In[ ]:


svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(
    svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))


# # Here, increasing C allows us to improve the model, resulting in 81.2% train set accuracy.
