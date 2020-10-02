#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# Build a model that predicts the survival of passengers on the titanic

# # Exploratory Data Analysis (EDA)

# __Data Description__
# 
# The file has the structure below:
# 
# | Column Name | Description | Type | Sample Values |
# | --- | --- | --- | --- |
# | PassengerId | ID of the passengers in the titanic ship | Int | Integer numbers i.e 1,2,3... |
# | Survived | Indicates whether a passenger survived or not | Int | Binary numbers, 1=Yes, 0=No |
# | Pclass | Ticket class of the passengers | Int | 1=1st, 2=2nd, 3=3rd |
# | Name | Names of the passengers | Str | Strings |
# | Sex | The gender of the passengers | Str | male, female |
# | Age | The age of the passengers | Int | Integer numbers |
# | SibSp | Number of siblings / spouses aboard the Titanic | Int | Integer numbers |
# | Parch | Number of parents / children aboard the Titanic | Int | Integer numbers |
# | Ticket | Ticket numbers | Int | ticket numbers |
# | Fare | Passenger fare | Int/floats | 21,24... |
# | Cabin | Cabin number | Str | A6, D56.. |
# | Embarked | Port of Embarkation | Str | C=Cherbourg, Q=Queenstown, S=Southampton |

# __Importing the required libraries__

# In[ ]:


import numpy as np
import pandas as pd
from numpy import log

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import math
from collections import Counter
import pandas_profiling as pp
import scipy.stats as stats

#configuration settings
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# __Loading the data into a dataframe__

# In[ ]:


titanic_survival_test_df = pd.read_csv("../input/titanic/test.csv")
titanic_survival_training_df = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


#view the top 5 records for the training set
titanic_survival_training_df.head(5)


# In[ ]:


#view the top 5 records for the test set
titanic_survival_test_df.head(5)


# In[ ]:


#profile report generation
pp.ProfileReport(titanic_survival_training_df)


# In[ ]:


pp.ProfileReport(titanic_survival_test_df)


# __Dropping Irrelevant Columns__

# In[ ]:


titanic_survival_training_df = titanic_survival_training_df.drop(["Name", "Cabin", "Embarked", "Ticket", "Age"], axis=1)
titanic_survival_test_df = titanic_survival_test_df.drop(["Name", "Cabin", "Embarked", "Ticket", "Age"], axis=1)


# In[ ]:


titanic_survival_training_df.head()


# In[ ]:


titanic_survival_test_df.head()


# In[ ]:


#check the shape of the records to know how many records are in the training dataset
titanic_survival_training_df.shape


# In[ ]:


# Check for rows containing duplicate data in the training set
duplicate_rows_df = titanic_survival_training_df[titanic_survival_training_df.duplicated()]
print("Number of duplicate rows: ", duplicate_rows_df.shape)


# As seen above there are no duplicate rows

# __Missing or null values__

# In[ ]:


# Finding the null values in the training set.
titanic_survival_training_df.isnull().sum()


# In[ ]:


# Finding the null values in the test set.
titanic_survival_test_df.isnull().sum()


# In[ ]:


#Filling the missing value in the training set
titanic_survival_test_df['Fare'].fillna((titanic_survival_test_df['Fare'].mean()), inplace=True)


# In[ ]:


# Finding the null values in the test set.
titanic_survival_test_df.isnull().sum()


# __Detecting Outliers__

# In[ ]:


#plotting a boxplot
sns.boxplot(x=titanic_survival_training_df["Fare"])


# In[ ]:


titanic_survival_training_df.boxplot(column=['Fare'], by=["Survived"])


# ### Exploring the target variable (Survived)

# In[ ]:


#proportion of the 'Survived' variable
survived_vc = titanic_survival_training_df['Survived'].value_counts()
survived_df = survived_vc.rename_axis('survived').reset_index(name='counts')
survived_df


# In[ ]:


#ploting a pie chart and a bar graph
# Define the labels
survived_label =  '0', '1'

#Choose which proportion to explode
survived_explode = (0,0.1)

# Create the container which will hold the subplots
survived_fig = plt.figure(figsize = (25,12))

# Create a frame using gridspec
gs = gridspec.GridSpec(6,7)

# Create subplots to visualize the pie chart
pie_ax01 = plt.subplot(gs[0:,:-3])
pie_ax01.set_title(label="Survival Rate",fontdict={"fontsize":25})
pie_ax01.pie(survived_df["counts"],
            explode = survived_explode,
            autopct = "%1.1f%%",
            shadow = True,
            startangle = 90,
            textprops ={"fontsize":22})
pie_ax01.legend(survived_label, loc = 0, fontsize = 18, ncol=2)

# Set subplot to visualize the bargraph
bar_ax01 = plt.subplot(gs[:6,4:])
survived_label_list = survived_df["survived"]
survived_freq = survived_df["counts"]
index = np.arange(len(survived_label_list))
width = 1/1.5

bar_ax01.set_title(label="Survival Rate",fontdict={"fontsize":25})
bar_ax01.set_xlabel(xlabel="Survived",fontdict={"fontsize":25})
bar_ax01.set_ylabel(ylabel="Count",fontdict={"fontsize":25})
bar_ax01.set_xticklabels(survived_label_list,rotation="vertical",fontdict={"fontsize":25})
bar_ax01.bar(survived_label_list,survived_freq,width,color="blue")

plt.tight_layout(pad=5)


# In[ ]:


# Checking for imbalance in the target variable
def balance_calc(data, unit='natural'):
    base = {
        'shannon' : 2.,
        'natural' : math.exp(1),
        'hartley' : 10.
    }
    if len(data) <= 1:
        return 0
    
    counts = Counter()
    
    for d in data:
        counts[d] += 1
    
    ent = 0
    
    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])
            
    return ent/math.log(len(data))


# In[ ]:


balance_calc(titanic_survival_training_df["Survived"],'shannon')


# The target variable is imbalanced because the value obtained of __0.1414398__ is below __0.5__. We can also see in the pie chart and bar graph above that there is an imbalance in the dataset

# In[ ]:


#plotting a histogram with a fitted normal distribution
sns.distplot(titanic_survival_training_df["Survived"], fit=stats.norm, color='red', kde = False)


# In[ ]:


#skewness
titanic_survival_training_df["Survived"].skew(axis = 0)


# In[ ]:


#kurtosis
titanic_survival_training_df["Survived"].kurt()


# The skewness of the target variable (Survived), as indicated above is greater than zero. Hence, the data is __positively skewed__ and given that the value of skewness (__0.478523__) lies between -0.5 and 0.5, the data is fairly symmetrical. 
# 
# The data has __platykurtic kurtosis__ since the kurtosis value __-1.7750__ is less than __3__. As seen in the histogram plot above, the distribution is short, the peak is low and the tail is thin. This means that the data lack outliers

# In[ ]:


#Finding the relations between the variables.
plt.figure(figsize = (20,10))
correlation = titanic_survival_training_df.corr()
sns.heatmap(correlation, cmap='BrBG', annot=True)
correlation


# From the heatmap above we can see that the survival of the passengers onboard was dependent on the fare that the passenger paid.

# ## Support Vector Machine

# In[ ]:


titanic_survival_training_df.head()


# __Prediction of survived based on fare__

# In[ ]:


#training set and test set
X = titanic_survival_training_df.iloc[:, [6]].values
y = titanic_survival_training_df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


# Fitting SVM to the training set
svm_fare_model = SVC(kernel='rbf', random_state=0)
svm_fare_model.fit(X_train, y_train)


# In[ ]:


# Predicting the test set results
y_pred = svm_fare_model.predict(X_test)


# In[ ]:


# Accuracy score
svm_fare_model.score(X_train, y_train)


# In[ ]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


print(classification_report(y_test, y_pred))


# The accuracy of the model is __65.45%__ and there are __125__ accurate predictions and __54__ wrong predictions. Also from the classification report we can see that the model is __95%__ confident in predicting the passengers that __did not survive__. However, it is __30%__ confident when it comes to the prediction of the passengers that __survived__

# __Prediction of survived based on the passenger class__

# In[ ]:


#training set and test set
X = titanic_survival_training_df.iloc[:, [2]].values
y = titanic_survival_training_df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


# Fitting SVM to the training set
svm_class_model = SVC(kernel='rbf', random_state=0)
svm_class_model.fit(X_train, y_train)


# In[ ]:


# Predicting the test set results
y_pred = svm_class_model.predict(X_test)


# In[ ]:


# Accuracy score
svm_class_model.score(X_train, y_train)


# In[ ]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


print(classification_report(y_test, y_pred))


# The accuracy of the model has improved from __65.45%__ to __66.99%__. 
# 
# There are __128__ accurate predictions and __51__ wrong predictions, which is an improvement from the previous model. 
# 
# From the classification report we can see that the model is __87%__ confident in predicting the passengers that __did not survive__. It is __46%__ confident when it comes to the prediction of the passengers that __survived__, which is an improvement

# __Prediction of survived based on sibsp and parch__

# In[ ]:


#training set and test set
X = titanic_survival_training_df.iloc[:, [4,5]].values
y = titanic_survival_training_df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


# Fitting SVM to the training set
svm_sp_model = SVC(kernel='rbf', random_state=0)
svm_sp_model.fit(X_train, y_train)


# In[ ]:


# Predicting the test set results
y_pred = svm_sp_model.predict(X_test)


# In[ ]:


# Accuracy score
svm_sp_model.score(X_train, y_train)


# In[ ]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


print(classification_report(y_test, y_pred))


# The accuracy of the model has reduced from __66.99%__ to __66.29%__. 
# 
# There are __124__ accurate predictions and __55__ wrong predictions, which shows that the performance of the model has reduced in comparison to the previous model. 
# 
# From the classification report we can see that the model is __93%__ confident in predicting the passengers that __did not survive__. It is __32%__ confident when it comes to the prediction of the passengers that __survived__.

# __Prediction of survived based on the sex__

# In[ ]:


#Encoding the 'Sex' variable
#Frequency encoding
fe = titanic_survival_training_df.groupby('Sex').size()/len(titanic_survival_training_df)
titanic_survival_training_df.loc[:,'Sex_Enc'] = titanic_survival_training_df['Sex'].map(fe)
titanic_survival_training_df.sample(5)


# In[ ]:


#training set and test set
X = titanic_survival_training_df.iloc[:, [7]].values
y = titanic_survival_training_df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


# Fitting SVM to the training set
svm_sex_model = SVC(kernel='rbf', random_state=0)
svm_sex_model.fit(X_train, y_train)


# In[ ]:


# Predicting the test set results
y_pred = svm_sex_model.predict(X_test)


# In[ ]:


# Accuracy score
svm_sex_model.score(X_train, y_train)


# In[ ]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


print(classification_report(y_test, y_pred))


# The accuracy of the model has improved from __66.29%__ to __78.65%__. 
# 
# There are __141__ accurate predictions and __38__ wrong predictions, which is a great improvement from the previous models. 
# 
# From the classification report we can see that the model is __84%__ confident in predicting the passengers that __did not survive__. It is __71%__ confident when it comes to the prediction of the passengers that __survived__, which is also a great improvement.
# 
# This shows that the sex of the passengers is a strong predictor on whether the passengers survived or not

# __Prediction of survived based on various features__

# In[ ]:


titanic_survival_training_df.sample(5)


# In[ ]:


#training set and test set
X = titanic_survival_training_df.iloc[:, [2,4,5,7]].values
y = titanic_survival_training_df.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


# Fitting SVM to the training set
svm_all_model = SVC(kernel='rbf', random_state=0)
svm_all_model.fit(X_train, y_train)


# In[ ]:


# Predicting the test set results
y_pred = svm_all_model.predict(X_test)


# In[ ]:


# Accuracy score
svm_all_model.score(X_train, y_train)


# In[ ]:


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


print(classification_report(y_test, y_pred))


# The accuracy of the model has improved from __78.65%__ to __80.76%__.
# 
# I noted that the accuracy of the model reduced to __65%__ when fare was added, indicating that it's not a good predictor of whether the passengers survived or not. Also the passengers' class highly correlates with the fare.
# 
# There are __143__ accurate predictions and __36__ wrong predictions, which is a great improvement from the previous models. 
# 
# From the classification report we can see that the model is __87%__ confident in predicting the passengers that __did not survive__. It is __68%__ confident when it comes to the prediction of the passengers that __survived__, which is also a great improvement.
# 
# Hence; passengers' class, sex, number of siblings, spouses, parents and children are strong predictors of the survived column.

# In[ ]:


titanic_survival_test_df.head()


# In[ ]:


#Encoding the 'Sex' variable in the test set
#Frequency encoding
fe = titanic_survival_test_df.groupby('Sex').size()/len(titanic_survival_test_df)
titanic_survival_test_df.loc[:,'Sex_Enc'] = titanic_survival_test_df['Sex'].map(fe)
titanic_survival_test_df.sample(5)


# In[ ]:


X_test2 = titanic_survival_test_df.iloc[:, [1,3,4,6]].values


# In[ ]:


# Predicting the test set results using the svm_all_model
y_pred = svm_all_model.predict(X_test2)


# In[ ]:


y_pred


# In[ ]:




