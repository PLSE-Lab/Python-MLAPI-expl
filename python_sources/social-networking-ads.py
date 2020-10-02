#!/usr/bin/env python
# coding: utf-8

# ----------------**CLASSIFICATION PROBLEM**-----------------

# **Goal:-**
# To Predict the users on the social network who on interacting wiht the advertisements either purchased the product or not.

# In[ ]:


#importing important libraries
import pandas as pd  #for data analysis
import numpy as np   #for linear algebra
import matplotlib.pyplot as plt   #for visulaization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#importing dataset
data=pd.read_csv('../input/-social-networking-ads/Social_Network_Ads.csv')


# In[ ]:


# displaying all the features of the data
data.columns


# In[ ]:


data.head()


# In[ ]:


#Checking the shape of the dataset
data.shape 


# In[ ]:


#checking the information of the Data
data.info()


# In[ ]:


#Checking if there is any missing Value present in the dataset
data.isnull().sum()
#There is no missing value present in the dataset


# In[ ]:


#Checking the Datatype of the features
data.dtypes


# **We see that there are four continous type variables and one categorical type variable 'Gender'.**

# In[ ]:


#Information About the Data
data.describe()


# This gives us the describtion about the continous datatype variables. 

# **Univariate analysis**

# In[ ]:


data['Purchased'].value_counts()


# In[ ]:


data['Purchased'].value_counts().plot.bar()


# Out of 400,143 people have purchased the advertised product which is around 35% of the total. 

# In[ ]:


sns.distplot(data['Age'])


# We infer that maximum of the people lie in the age group between 25-45 and this variable is normally distributed.

# In[ ]:


sns.distplot(data['EstimatedSalary'])
#EstimatedSalary is normally distributed


# In[ ]:


plt.boxplot(data['EstimatedSalary'])
#Boxplot help us to check if there is any outlier present in the feature
#There is no outlier present in the EstimatedSalary Feature


# In[ ]:


plt.boxplot(data['Age'])
#There is no outlier present in the Age feature as well


# In[ ]:


plt.boxplot(data['User ID'])


# Bivariate Analysis

# In[ ]:


#Scatter Plot of estimatedsalary and Dependent variable purchased
sns.scatterplot(x='EstimatedSalary',y='Purchased',data=data)


# In[ ]:


sns.scatterplot(x='Age',y='Purchased',data=data)


# In[ ]:


sns.scatterplot(x='User ID',y='Purchased',data=data)


# In[ ]:


#We have one Categorical Variable as well
data.Gender.unique()


# In[ ]:


df1=pd.get_dummies(data=data)
#Here we change the categorical variable into continous variable


# In[ ]:


#Here we change the categorical variable Gender into continous variables Gender_Male and Gender_female
df1.head()


# In[ ]:


#Correlation between different features of our dataset
sns.heatmap(df1.corr(),annot=True)


# 1) We infer from this correlation matrix that the dependent variable Purchased is most correlated with the variable Age and then with variable EstimatedSalary.                                                                            
# 2) Dependent variable "Purchased" is least correlated with UserId ,so we can drop the feature UserID for model building.       
# 3) We also observed that there is no multicollinearity between the independent features which is good for model building.

# In[ ]:


#The dependent variable Purchased is very less correlated with User ID so we drop that feature
#We creates dummy variables so we drop one 
df2=df1.drop('User ID',axis=1)
df2=df2.drop('Gender_Male',axis=1)


# In[ ]:


df2.head()


# In[ ]:


#Separating Depenent and Independent Features
X=df2.iloc[:,[0,1,3]]
Y=df2.iloc[:,2]


# In[ ]:


X.head()


# In[ ]:


Y.head()


# In[ ]:


#Checking the shape of independent and dependent variables
print("Shape of Independent features:",X.shape)
print("Shape of dependent feature:",Y.shape)


# In[ ]:


plt.title("Correlation matrix")
sns.heatmap(df2.corr(),annot=True)


# Model Building

# In[ ]:


#Splitting the Data into training and test dataset
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[ ]:


print("Shape of X_train:",X_train.shape)
print("Shape pf X_test:",X_test.shape)
print("shape of Y_train:",Y_train.shape)
print("Shape of Y_test:",Y_test.shape)


# In[ ]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
XX_train=sc.fit_transform(X_train)
XX_test=sc.transform(X_test)


# Logistic Regression Model

# In[ ]:


from sklearn.linear_model import LogisticRegression
loreg=LogisticRegression()


# In[ ]:


#Fitting logistic Regression to the training set
loreg.fit(XX_train,Y_train)


# In[ ]:


#Score of the training set
loreg.score(XX_train,Y_train)


# The score of training Dataset is 0.84375 whic is a good score.

# In[ ]:


#Predicting the test Dataset
pred=loreg.predict(XX_test)


# In[ ]:


# Accuracy Score of test set
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,pred)


# Score of test datset is 0.8125 which is good enough.

# In[ ]:


inter=loreg.intercept_
print(inter)


# In[ ]:


#Coefficients of regression model
coeff=loreg.coef_
print(coeff)


# In[ ]:


#Making Confusion Matrix
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,pred)
print(cm)


# * Confusion Matrix is a performance measurement for machine learning Classification where output can be two or more classes.
# * It gives us insight not only into the errors being made by a classifier but also the type of erroe being made.

# In[ ]:


TP=48
FP=12
TN=20
FN=3
acc=(TP+TN)/(TP+TN+FP+FN)
rc=TP/(TP+FN)
pre=TP/(TP+FP)


# * TP- observation is positive and predicted to be positive.
# * FN- observation is positive and predicted to be negative.
# * TN- observation is negative and predicted to be negative.
# * FP- observation is negative and predicted to be positive.

# In[ ]:


#Printing Accuracy,Recall and Precision

print(acc)
print(rc)
print(pre)


# * High Recall indicates the class is correctly recognised.
# * High Precision indicates an example labeled as positive is indeed positive.

# In[ ]:


f_measure=(2*rc*pre)/(rc+pre)
print(f_measure)


# We have two measures(Recall and precision) f-measure helps to have a measurement that represents both of them.

# Support vector Machine

# In[ ]:


#Importing support vector classifier
from sklearn.svm import SVC
svc=SVC()


# In[ ]:


#fitting the training set 
svc.fit(XX_train,Y_train)


# In[ ]:


#predicting the test dataset
pred1=svc.predict(XX_test)


# In[ ]:


#accuracy of training dataset
svc.score(XX_train,Y_train)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,pred1)


# This is good accuracy score for the test Dataset

# In[ ]:


from sklearn import metrics


# In[ ]:


#Making Confusion matrix
cm1=metrics.confusion_matrix(Y_test,pred1)
print(cm1)


# In[ ]:


TP=45
FP=4
FN=3
TN=28
acc=(TP+TN)/(TP+TN+FP+FN)
rc=TP/(TP+FN)
pre=TP/(TP+FP)
print(acc)
print(rc)
print(pre)


# * Accuracy of the model is 0.9125
# * Recall of the model is 0.9375
# * Precision is 0.91836

# In[ ]:




