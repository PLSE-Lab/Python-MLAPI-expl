#!/usr/bin/env python
# coding: utf-8

# # Prediction of Probability of You Getting an Admit in the US for Masters

# 
# 
# The dataset contains several arguments(basically the inputs you need to provide) which are considered for the application for Masters Programs. The parameters included are :
# 
# 
# GRE Scores ( out of 340 )
# TOEFL Scores ( out of 120 )
# Under Grad University Rating ( out of 5 )
# Statement of Purpose (out of 5 )
# Letter of Recommendation Strength ( out of 5 )
# Undergraduate GPA ( out of 10 )
# Research Experience ( either 0 or 1 )
# Chance of Admit ( ranging from 0 to 1 )
# 
# Dataset: https://www.kaggle.com/mohansacharya/graduate-admissions/home

# Importing the necessary libraries for implementing Analysis.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Exploratory Data Analysis 

# As the Dataset is clean and has no null values we are directly going to implement the Exploration

# In[ ]:


Reading = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
Reading.head() #printing the first five rows 


# In[ ]:


Reading.describe()


# # Finding out if there are any null values

# In[ ]:


Null=Reading.isnull()
Null.sum()


# # Renaming columns

# In[ ]:


Reading = Reading.rename(columns={'GRE Score': 'GRE Score', 'TOEFL Score': 'TOEFL Score', 'LOR ': 'LOR', 'Chance of Admit ': 'Admit Possibilty'})
Reading.head()


# # Dropping unwanted columns

# In[ ]:


Reading.drop('Serial No.', axis='columns', inplace=True)
Reading.head()


# # Visualization 

# In[ ]:



gre_score = Reading[["GRE Score"]] #selecting only the required coloumn
toefl_score = Reading[["TOEFL Score"]] 
uni_rating=  Reading[["University Rating"]]


# In[ ]:


fig=sns.distplot(gre_score,color='black',kde=False)
plt.title("GRE SCORES")
plt.show()

fig=sns.distplot(toefl_score,color='r',kde=False)
plt.title("TOEFL SCORES")
plt.show()

fig=sns.distplot(uni_rating,color='r',kde=False)
plt.title("UNIVERSITY RATING")
plt.show()



# In[ ]:


fig=sns.lmplot(x='GRE Score',y='CGPA',data=Reading)
plt.title("CGPA VS GRE SCORE")
plt.show()


fig=sns.jointplot(x='CGPA',y='Admit Possibilty',data=Reading,kind='kde')
plt.show()


# From the above plot we can see that the person who did well in UG also did well in GRE.

# In[ ]:


fig=sns.lmplot(x='CGPA',y='TOEFL Score',data=Reading)
plt.title("CGPA VS TOEFL")
plt.show()


# In[ ]:


sns.pairplot(data=Reading,vars=["GRE Score","Admit Possibilty"])
plt.show()


# We can see that the GRE SCORE is a deal breaker for getting an admit

# # Predictions

# # We are using a linear regression model. Why?

#  # This is a supervised model data and also the independent variable X having the parameters GRE, TOEFL etc are in high relationship with the  dependent variable y being the chance of admit.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# Splitting the data as x and y where x contains the dependent varaiable data and y contains the independent variable data

# In[ ]:


x=Reading.drop('Admit Possibilty',axis='columns')
y=Reading['Admit Possibilty']
x_train,x_test,y_train,y_test=train_test_split(x, y)



                      
                                                                


# Further Splitting the data as test and train where the train set contains the 80% of data and the test set contains 20% of the data where we can see that 300 out of 400 rows are taken

# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


linear_regression = LinearRegression()
linear_regression = linear_regression.fit(x_train,y_train)


# A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called k-fold CV, the training set is split into k smaller sets

# In[ ]:


def get_cv_scores(linear_regression):
    scores = cross_val_score(linear_regression,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')

# get cross val scores
get_cv_scores(linear_regression)


# The CV score says that the model is neither an underfit nor an overfit. Any value between 0 and 1 is good.

# The model is predicting on the test set

# In[ ]:


model = LinearRegression(normalize=True)
model.fit(x_test, y_test)
model.score(x_test, y_test)


# # Accuracy on the test set is 81.2%

# # The model finally predicts the data based on user input

# In[ ]:


print('The chance of you getting an admit in the US is {}%'.format(round(model.predict([[305, 108, 4, 4.5, 4.5, 8.35, 0]])[0]*100, 1)))


# # Future Scopes:
# 1) Deploying the model using Flask and a hosted database.
# 2) Dockerising the the file
