#!/usr/bin/env python
# coding: utf-8

# Starting off with hackathons for the first time but dont know how to approach a problem? Built a basic model but not able to improve scores? This notebook is for you! 
# 
# When I started my journey, I often used to feel lost. This gave me inspiration to upload a notebook for those who struggle during their initial days of solving a problem. This notebook demonstrates the pre-processing and model tuning steps for the same. I have used the ever famous Loan Prediction problem for this notebook. With this approach I was among the top 2% in the competition.
# 
# **UPVOTE IF YOU LIKE MY EFFORTS. IT KEEPS ME MOTIVATED TO PUBLISH QUALITY CONTENT.**

# The problem statement requires automating the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. 

# # IMPORTING ALL THE REQUIRED LIBRARIES

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette("rainbow")
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from sklearn import model_selection
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[ ]:


df = pd.read_csv("../input/train.csv")


# Taking a first look at our data.

# In[ ]:


df.head()


# In[ ]:


df.info()


# We notice some missing values in the data. Also, the Dependents column represents a numerical value(number of dependents on the applicant). But it is of object datatype. So let us check what unique values does that column hold.

# In[ ]:


df['Dependents'].unique()


# # CHECKING FOR MISSING DATA AND FILLING THE VALUES

# In[ ]:


msno.matrix(df)


# The plot gives us a general idea about the completeness of the data where the white horizontal lines represent missing values. Let us go ahead and find the exact number of missing values in our data

# In[ ]:


missing_values = df.isnull().sum().sort_values(ascending = False)
missing_values


# We need to change the datatype of dependents to an integer and fill the missing values. For Loan amount and its term, we fill the missing values with their respective mean.

# In[ ]:


df['Dependents'].replace(to_replace = '3+', value = 3, inplace = True)
df['Dependents'].fillna(df['Dependents'].median(), inplace = True)
df['Dependents'] = df['Dependents'].astype(int)


# In[ ]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace = True)


# For the categorical columns,we use the ffill method.  'ffill' stands for 'forward fill' and will propagate last valid observation forward.

# In[ ]:


for column in ['Credit_History', 'Self_Employed', 'Gender', 'Married']:
    df[column].fillna(df[column].ffill(), inplace=True)


# # VISUALIZATIONS

# The boxplots give us an idea about outliers in the numerical columns. Also, the bar plots estimates the mean values of different categories.

# In[ ]:


sns.boxplot('Loan_Status','LoanAmount', data = df)


# In[ ]:


sns.boxplot('Loan_Status','CoapplicantIncome', data = df)


# In[ ]:


sns.boxplot('Loan_Status','ApplicantIncome', data = df)


# In[ ]:


sns.catplot(x="Loan_Status", y="LoanAmount", hue="Gender", kind="bar", data=df);


# In[ ]:


sns.catplot(x="Loan_Status", y="LoanAmount", hue="Education", kind= "bar" , data=df)


# # Dropping the outliers

# As we saw in the boxplots, there are some outliers in our data. It is best to remove them for better results.

# In[ ]:


df.drop(df[df['CoapplicantIncome']>10000].index, inplace= True)
df.drop(df[df['ApplicantIncome']>20000].index, inplace= True)
df.drop(df[df['LoanAmount']>550].index, inplace= True)


# Creating two new features as per our intuition.

# In[ ]:


df['total_amount'] = df['ApplicantIncome'] +df['CoapplicantIncome']
df['ratio']= df['total_amount']/df['LoanAmount']


# # CHECKING THE SKEWNESS OF THE DATA

# In[ ]:


sns.distplot(df['ApplicantIncome'])


# In[ ]:


df.describe()


# The ApplicantIncome looks skewed so we look at the other numerical columns for skewness and transform the required columns to fit normal distribution.

# In[ ]:


df.skew(axis=0).sort_values(ascending= False)


# In[ ]:


cols = ['ratio','LoanAmount','total_amount']
df[cols] = np.log1p(df[cols])


# Dropping the columns that are not required.

# In[ ]:


df.drop('Loan_ID', axis = 1, inplace = True)
df.drop('ApplicantIncome', axis = 1, inplace = True)
df.drop('CoapplicantIncome', axis = 1, inplace = True)


# # CONVERTING CATEGORICAL COLUMNS TO NUMERICAL ONES

# Categorical columns will give an error if you try to plug these variables into most machine learning models in Python without "encoding" them first. One hot encoding creates new (binary) columns, indicating the presence of each possible value from the original data.

# In[ ]:


df1 = pd.get_dummies(df, columns=list(df.select_dtypes(exclude=np.number)))
df1.head()


# In[ ]:


col= ['Loan_Status_N','Loan_Status_Y']
df1.drop(columns = col,axis = 1,inplace = True)


# In[ ]:


df1.head()


# In[ ]:


df1['Loan_status'] = df["Loan_Status"]
X = df1.iloc[:,0:16]
y = df1.iloc[:,17]


# # Splitting the data and fitting a basic model first

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


rf = RandomForestClassifier()
m = rf.fit(X_train, y_train)
y_pred = m.predict(X_test)
accuracy_score(y_test, y_pred)


# Observe that we get a decent accuracy score by pre processing our data correctly. 
# Let us go ahead and tune our model to improve the accuracy.

# # Model Fitting with Hyperparameter Tuning
# Now, parameter tuning is driven by the knowledge of parameters and your model. Given the amount of computational resources we have, we need that knowledge to select the range for our parameter search better. Do play around before setting up the parameter search. It will give you better idea into how your model is working for the dataset. So DO NOT AVOID GOING IN-DEPTH with your theory!
# 
# That knowledge will work wonders for your leaderboard in any competition you take part in. DO NOT SKIP IT! 

# In[ ]:


import sklearn.model_selection as ms
parameters= {'n_estimators':[5,50,500],
            'max_depth':[5,10,12,15],
            'max_features':[5,10,15],
            'min_samples_split' : [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4]}


rf = RandomForestClassifier()
rf_model = ms.GridSearchCV(rf, param_grid=parameters, cv=5)
rf_model.fit(X_train,y_train)

print('The best value of Alpha is: ',rf_model.best_params_)


# Now, this gives us the best parameters but takes a lot of time as it is an iterative process and tries out all the possiblities. Quite tiresome.

# In[ ]:


rf1 = RandomForestClassifier(n_estimators = 500,max_depth = 10, max_features=15, min_samples_leaf = 4, min_samples_split= 10,bootstrap = True)
rf_random = rf1.fit(X_train,y_train)

scores_rf = cross_val_score(rf_random, X_train, y_train, cv=5)
scores_rf.mean()


# In[ ]:


y_pred = rf_random.predict(X_test)
accuracy_score(y_test, y_pred)


# Now randomized search is supposed to be faster and give better results (if you know about your model and its parameters).

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
random_grid ={'bootstrap': [True, False],
             'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
             'max_features': [1,3,5,7,9,11,13,15],
             'min_samples_leaf': [1, 2,3, 4],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [50,100,500,1000]}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[ ]:


rf_random.fit(X_train,y_train)
rf_random.best_params_


# In[ ]:


rf = RandomForestClassifier(n_estimators = 500,max_depth = 10, max_features=5, min_samples_leaf = 3, min_samples_split= 10,bootstrap = True)
rf_random = rf.fit(X_train,y_train)

scores_rf = cross_val_score(rf_random, X_train, y_train, cv=5)
scores_rf.mean()


# In[ ]:


y_pred = rf_random.predict(X_test)
accuracy_score(y_test, y_pred)


# ## That was my beginner code and I am proud to say I have come a long way and look forward to keep going. This was something which gave me the boost to keep working. I hope it helps you the same! 
# 
# ## THANK YOU! Keep learning. Upvote | Comment | Share.
