#!/usr/bin/env python
# coding: utf-8

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


# ## In this project we will be working with diabetes dataset, indicating whether or not a particular  patient has a diabetes disease or not . We will try to create a model that will predict whether or not  this person has the disease  based off the features of that patient.
# 
# ## This data set contains the following features:
# *  Pregnancies: Number of times pregnant
# *  Glucose : Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# *  BloodPressure : Diastolic blood pressure (mm Hg)
# *  SkinThickness : Triceps skin fold thickness (mm)
# *  Insulin : 2-Hour serum insulin (mu U/ml)
# *  BMI : Body mass index (weight in kg/(height in m)^2)
# *  DiabetesPedigreeFunction : Diabetes pedigree function
# *  Age : Age (years)
# * Outcome : Class variable (0 or 1) 268 of 768 are 1, the others are 0
# 

# ##  Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Set the style to fivethirtyeight

# In[ ]:


style.use('fivethirtyeight')


# ## Read the diabetes data and print the first five rows of it

# In[ ]:


df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')


# In[ ]:


df.head()


# ## display the shape of data
# ### it show that the data has 768 rows and 9 columns

# In[ ]:


df.shape


# ## Show information about the data 
# ## it show that 7 features are int and 2 are float 

# In[ ]:


df.info()


# ## Create statistical summary of the data

# In[ ]:


df.describe().T


# ## check for NaN 
# #### it seems that there is no null values

# In[ ]:


df.isnull().sum()


# ## check for duplicated rows
# #### it seems that there is no duplicated rows

# In[ ]:


df.duplicated().sum()


# ## ** create a countplot showing number of Pregnancies for each type of patients **
# ### ** the plot show that most of woman had number of Pregnancies 1  **

# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(df.Pregnancies,hue=df['Outcome'])


# ##  ** create a countplot showing ages of patients **
# ### ** the plot showing that the most of patients have ages in range 21 to 60 years **

# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(df.Age,palette='rainbow')
plt.tight_layout()


# ## ** create a age range columns in df and group by it **
# 

# In[ ]:


df['age_range'] = [np.floor(i/10)*10 for i in df['Age'] ]
display(df.head())
df1 = df.groupby('age_range',as_index=False)


# ## ** calculate the percentage of patient  of each age range that have the disease **
# ## ** it showing that patient in range 20 to 30 years 21% of them have the disease **
# ## ** it showing that patient in range 30 to 40 years 46% of them have the disease **
# ## ** it showing that patient in range 40 to 50 years 55% of them have the disease **
# ## ** it showing that patient in range 50 to 60 years 59% of them have the disease **
# ## ** it showing that patient in range 60 to 70 years 27% of them have the disease **
# ## ** it showing that patient in range 70 to 80 years 50% of them have the disease **
# ## ** it showing that patient in range 80 to 90 years 0% of them have the disease **

# In[ ]:


df1.agg({'Outcome':'mean'})


# ## ** drop age_range columns ** 

# In[ ]:


df.drop('age_range',axis=1,inplace=True)


# In[ ]:


df.head()


# ## ** create a pie plot for Outcome **
# ## ** it showing that 65.1% of patien didn't have the disease and 34.9% of them have it **
# 

# In[ ]:


plt.figure(figsize=(10,10))
plt.pie(df.Outcome.value_counts(),autopct='%0.2f%%',labels=[0,1])
plt.show()


# ## ** create a histogram for Glucose in each type of patient **
# ## ** it showing that most of type 0 have Glucose = 100 and type 1 have Glucose = 125 **
# ## ** the plot showing that some patient have Glucose = 0 and that's impossible **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='Set1',height=6,aspect=2)
g.map(plt.hist,'Glucose',alpha=0.6,bins=20)
plt.legend()


# ## ** create a histogram for BloodPressure in each type of patient **
# ## ** it showing that most of type 0 have BloodPressure = 70 and type 1 have BloodPressure = 75 **
# ## ** the plot showing that some patient have BloodPressure = 0 and that's impossible **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='coolwarm',height=6,aspect=2)
g.map(plt.hist,'BloodPressure',alpha=0.6,bins=20)
plt.legend()


# ## ** create a histogram for SkinThickness in each type of patient **
# ## ** the plot showing that some patient have SkinThickness = 0 and that's impossible **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='Set2',height=6,aspect=2)
g.map(plt.hist,'SkinThickness',alpha=0.6,bins=20)
plt.legend()


# ## ** create a histogram for Insulin in each type of patient **
# ## ** the plot showing that some patient have Insulin = 0 and that's impossible **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='dark',height=6,aspect=2)
g.map(plt.hist,'Insulin',alpha=0.6,bins=20)
plt.legend()


# ## ** create a histogram for BMI for each type of patient **
# ## ** the plot showing that some patient have BMI = 0 and that's impossible **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='rainbow',height=6,aspect=2)
g.map(plt.hist,'BMI',alpha=0.6,bins=20)
plt.legend()


# ## ** create a histogram for DiabetesPedigreeFunction for each type of patient **
# ## ** it showing that most of type 0 have DiabetesPedigreeFunction = 0.25 and type 1 have DiabetesPedigreeFunction = 0.30 **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='cool',height=6,aspect=2)
g.map(plt.hist,'DiabetesPedigreeFunction',alpha=0.6,bins=20)
plt.legend()


# ## ** create a histogram for Age for each type of patient **
# ### ** the plot showing that the most of patients have ages in range 21 to 50 years **

# In[ ]:


g = sns.FacetGrid(df,hue='Outcome',palette='hot',height=6,aspect=2)
g.map(plt.hist,'Age',alpha=0.6,bins=20)
plt.legend()


# ### ** calculate the mean of 'Glucose','BloodPressure','SkinThickness','Insulin','BMI' in each type of patient **

# In[ ]:


x1 = np.round(df[df['Outcome'] == 1][['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].mean())
x2 = np.round(df[df['Outcome'] == 0][['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].mean())
display(x1)
display(x2)


# ## ** As it impossible for 'Glucose','BloodPressure','SkinThickness','Insulin','BMI' to be zero we will replace zero by the mean of each one**

# In[ ]:


df.loc[df['Outcome'] == 1,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df.loc[df['Outcome'] == 1,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,x1)
df.loc[df['Outcome'] == 0,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df.loc[df['Outcome'] == 0,['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,x2)


# In[ ]:


df.head()


# ## ** create a pairplot for df showing histograms after replace zeros and scaterplot for all feature**

# In[ ]:


sns.pairplot(df,hue='Outcome',diag_kind='hist')


# ## Create the new statistical summary of the data

# In[ ]:


df.describe()


# ## ** create a heatmap for corr between features **

# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True,cmap='BuGn')
plt.tight_layout()


# ## ** violin plot for all df **

# In[ ]:


plt.figure(figsize=(20,20))

ax1 = plt.subplot(4,2,1)
sns.violinplot(x=df['Outcome'],y=df['Pregnancies'],palette='cool')

ax2 = plt.subplot(4,2,2)
sns.violinplot(x=df['Outcome'],y=df['Glucose'],palette='coolwarm')

ax3 = plt.subplot(4,2,3)
sns.violinplot(x=df['Outcome'],y=df['BloodPressure'],palette='hot')

ax4 = plt.subplot(4,2,4)
sns.violinplot(x=df['Outcome'],y=df['SkinThickness'],palette='dark')

ax5 = plt.subplot(4,2,5)
sns.violinplot(x=df['Outcome'],y=df['Insulin'],palette='Set1')

ax6 = plt.subplot(4,2,6)
sns.violinplot(x=df['Outcome'],y=df['BMI'],palette='Set2')

ax7 = plt.subplot(4,2,7)
sns.violinplot(x=df['Outcome'],y=df['DiabetesPedigreeFunction'],palette='rainbow')

ax8 = plt.subplot(4,2,8)
sns.violinplot(x=df['Outcome'],y=df['Age'])

plt.tight_layout()
plt.show()


# ## ** tried to do feature scaling but it reduce the accuracy so i stop it **

# In[ ]:


#from sklearn.preprocessing import StandardScaler
#scalar = StandardScaler()
#scalar.fit(df.drop('Outcome',axis=1))
#scaled_feat = scalar.transform(df.drop('Outcome',axis=1))
#scaled_feat = pd.DataFrame(scaled_feat,columns=df.columns[0:-1])


# In[ ]:


#scaled_feat.head()


# ## ** Split the data into training set and testing set using train_test_split**
# ## **  80% training and 20% testing **

# In[ ]:


#X = scaled_feat
X  = df.drop('Outcome',axis=1)
y = df['Outcome']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# ## ** Train and fit a KNN model on the training set.**
# ## ** Start with K = 3 **

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[ ]:


knn.fit(X_train,y_train)


# ## ** Now predict values for the testing data.**

# In[ ]:


pred = knn.predict(X_test)


# ## ** Create a classification report and confusion matrix for the model.**
# ### ** The classification_report show that the f1-score for 0 == 0.87 and for 1 == 0.78  **
# ### ** The confusion_matrix show that the model predicted  99 persons didn't have the disease and 55 have it **
# ### ** The Accuracy of the model at k = 3 equal 0.84% **

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:


from sklearn.metrics import accuracy_score
print("The Accuracy of the model with K=3 equal {} %".format(accuracy_score(y_test, pred)*100))


# ## Choosing a K Value
# ### ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. **

# In[ ]:


error_rate = []
for i in range(1,60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    predict = knn.predict(X_test)
    error_rate.append(np.mean(predict != y_test))


# ## ** create the following plot using the information from error rate list.**
# ## it show that K at 12 had the minimum rate error so we will choose K = 12 

# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(range(len(error_rate)),error_rate,marker='o',markerfacecolor='white',linestyle='--')


# ## ** Train, fit a KNN model on the training set then predict values for testing data.**
# ## ** set  K = 12 **

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
knn_predict = knn.predict(X_test)


# ## ** Create a classification report and confusion matrix for the model.**
# ### ** The classification_report show that the f1-score for 0 == 0.90 and for 1==0.81  **
# ### ** The confusion_matrix show that the model predicted  99 persons didn't have the disease and 55 have it **
# ### ** The Accuracy of the model at k = 12 equal 87% **

# In[ ]:


print(confusion_matrix(y_test,knn_predict))
print('\n')
print(classification_report(y_test,knn_predict))


# In[ ]:


print("The Accuracy of the model with K=12 equal {} %".format(accuracy_score(y_test, knn_predict)*100))


# ## plot the confusion matrix
# ### ** the model predicted that 92 persons didn't have the disease and in real they hadn't  **
# ###  ** the model predicted that 7 persons didn't have the disease and  in real they had  **
# ###  ** the model predicted that 13 persons  have the disease and  in real they hadn't  **
# ###  ** the model predicted that 42 persons  have the disease and  in real they had  **

# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,knn_predict)),annot=True,cmap='Dark2',cbar=False,linewidths=1,fmt='.3g')


# ## Choosing a P Value
# ### ** Create a for loop that trains various KNN models with different p_values, then keep track of the error_rate for each of these models with a list. **

# In[ ]:


error_rate = []
for i in range(1,25):
    knn = KNeighborsClassifier(n_neighbors=12,p=i)
    knn.fit(X_train,y_train)
    predict = knn.predict(X_test)
    error_rate.append(np.mean(predict != y_test))


# ## ** create the following plot using the information from error rate list.**
# ## it show that p at 1 and 2 had the minimum rate error so we will choose p= 1 

# In[ ]:


plt.figure(figsize=(10,8))
plt.plot(range(len(error_rate)),error_rate,marker='o',markerfacecolor='white',linestyle='--',color='orange')


# ## ** Train, fit a KNN model on the training set then predict values for testing data.**
# ## ** set  p= 1 **

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=12,p=1)
knn.fit(X_train,y_train)
knn_predict = knn.predict(X_test)


# ## ** Create a classification report and confusion matrix for the model.**
# ### ** The classification_report show that the f1-score for 0==  0.90 and for 1 == 0.81  **
# ### ** The confusion_matrix show that the model predicted  99 persons didn't have the disease and 55 have it **
# ### ** The Accuracy of the model at k=12 and p=1 equal 87% **

# In[ ]:


print(confusion_matrix(y_test,knn_predict))
print('\n')
print(classification_report(y_test,knn_predict))


# In[ ]:


print("The Accuracy of the model with K=12 and p = 1 equal {} %".format(accuracy_score(y_test, knn_predict)*100))


# ## plot the confusion matrix
# ### ** the model predicted that 92 persons didn't have the disease and in real they hadn't  **
# ###  ** the model predicted that 7 persons didn't have the disease and  in real they had  **
# ###  ** the model predicted that 13 persons  have the disease and  in real they hadn't  **
# ###  ** the model predicted that 42 persons  have the disease and  in real they had  **

# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(pd.DataFrame(confusion_matrix(y_test,knn_predict)),annot=True,cmap='Set3',cbar=False,linewidths=1,fmt='.3g')

