#!/usr/bin/env python
# coding: utf-8

# Hello everyone....
# 
# Hope you all are doing well.... Today i am working on diabetes dataset but before we start i just share some tips and tricks based on my Experience...
# 
# Try to be Data player before Data scientist because if you are habituated of handling the data,you can save your time when you are handling real world projects.
#  
#  Tricks and Tips:-
#  1. Try to spend your 80% of the time in data wrangling ,data cleaning or data preprocessing.
#  2. Daily try practice different kind of dataset everytime (Ex:-csv,tsv,image file,text file etc)
#  3. It's okay if you do mistakes ,just focus on your practice...However you will by doing mistakes.
#  4. Try to apply new shortcuts or new technique to solving the same issue.
#  5. Add a comment and write a short note if you are working on any new topic,such that if you visit the same code         once again then it would be clear for you.
#  6. Not required to work only on complex code, you can even work on simple code using the same technique.
#  7. Code should be polished such that if a new user try to understand the code,he/she should understand it easily.
#  
#  
#  
#  These are some tricks and tips,Hope this is helpful for you!

# # ||About the dataset||
# 
# The dataset contains.....
# 
# 1. Pregnancies: Number of times pregnant
# 2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. BloodPressure: Diastolic blood pressure (mm Hg)
# 4. SkinThickness: Triceps skin fold thickness (mm)
# 5. Insulin: 2-Hour serum insulin (mu U/ml)
# 6. BMI: Body mass index (weight in kg/(height in m)^2)
# 7. DiabetesPedigreeFunction: Diabetes pedigree function
# 8. Age: Age (years)
# 9. Outcome: Class variable (0 or 1)

# # ||Goal--||
# 
#   We will try to build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?
#   

# # ||Exercise||

# In[ ]:


#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#Reading the dataset
df=pd.read_csv('../input/diabetes2.csv')


# In[ ]:


#Check the first five values
df.head()


# In[ ]:


#Checking the null values..
df.isnull().sum()


# In[ ]:


#Checking the meta data
df.info()


# In[ ]:


#Doing some basic statistic
df.describe()


# In[ ]:


#Checking the unique values of dependent variable
df["Outcome"].unique()


# In[ ]:


#Check the dimensions
df.shape


# In[ ]:


#Ploting the headmap to check the correlation between all variables
import seaborn as sns
sns.heatmap(df)


# In[ ]:


#Ploting the histogram to check the distribution of each  columns 
p = df.hist(figsize = (20,20))


# In[ ]:


#Pairplot to visualize the correlation with one variable and all other variables respectively.
p=sns.pairplot(df, hue = 'Outcome')


# In[ ]:


#Checking the correlation using heatmap
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap


# In[ ]:


#Shortcut to scale all independent variable in one go.... 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(df.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[ ]:


X.head()


# In[ ]:



#Separating dataset into independent and dependent variables
X = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values



# In[ ]:


#Splitting into training and testing dataset....
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)


# In[ ]:


#Fitting Decisiontree into dataset
from sklearn.tree import DecisionTreeClassifier
#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
#Check the accuracy
from sklearn.metrics import accuracy_score


dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree_c.fit(X_train,y_train)
dtree_pred=dtree_c.predict(X_test)
dtree_cm=confusion_matrix(y_test,dtree_pred)
print("The accuracy of DecisionTreeClassifier is:",accuracy_score(dtree_pred,y_test))





# In[ ]:


print(dtree_cm)


# In[ ]:


#Fitting Randomforest into dataset
from sklearn.ensemble import RandomForestClassifier
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
print("The accuracy of RandomForestClassifier is:",accuracy_score(rdf_pred,y_test))


# In[ ]:


print(rdf_cm)


# In[ ]:


#Fitting Logistic regression into dataset
from sklearn.linear_model import LogisticRegression
lr_c=LogisticRegression(random_state=0)
lr_c.fit(X_train,y_train)
lr_pred=lr_c.predict(X_test)
lr_cm=confusion_matrix(y_test,lr_pred)
print("The accuracy of  LogisticRegression is:",accuracy_score(y_test, lr_pred))


# In[ ]:


print(lr_cm)


# In[ ]:


#Fitting KNN into dataset
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
knn_cm=confusion_matrix(y_test,knn_pred)
print("The accuracy of KNeighborsClassifier is:",accuracy_score(knn_pred,y_test))



# In[ ]:


print(knn_cm)


# In[ ]:


#Fitting Naive bayes into dataset
from sklearn.naive_bayes import GaussianNB



gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
bayes_pred=gaussian.predict(X_test)
bayes_cm=confusion_matrix(y_test,bayes_pred)
print("The accuracy of naives bayes is:",accuracy_score(bayes_pred,y_test))


# In[ ]:


print(bayes_cm)


# In[ ]:




#confusion matrix.....
plt.figure(figsize=(20,10))
plt.subplot(2,4,3)
plt.title("LogisticRegression_cm")
sns.heatmap(lr_cm,annot=True,cmap="prism",fmt="d",cbar=False)

plt.subplot(2,4,5)
plt.title("bayes_cm")
sns.heatmap(bayes_cm,annot=True,cmap="binary_r",fmt="d",cbar=False)
plt.subplot(2,4,2)
plt.title("RandomForest")
sns.heatmap(rdf_cm,annot=True,cmap="ocean_r",fmt="d",cbar=False)

plt.subplot(2,4,1)
plt.title("DecisionTree_cm")
sns.heatmap(dtree_cm,annot=True,cmap="twilight_shifted_r",fmt="d",cbar=False)
plt.subplot(2,4,4)
plt.title("kNN_cm")
sns.heatmap(knn_cm,annot=True,cmap="Wistia",fmt="d",cbar=False)
plt.show()


# # ||Conclusion||
# 
#     Looks like from the above model Logistic regression is the best fit model for diabetes dataset.
