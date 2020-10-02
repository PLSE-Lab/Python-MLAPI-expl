#!/usr/bin/env python
# coding: utf-8

# # Linear Regression For Predicting University Admission Using Sklearn

# ---

# ## 1. Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# ---

# ## 2. Loading Data 

# In[ ]:


#reading csv file 
df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")


# ###### About The Data:
# * GRE Score is out of 340
# * TOEFL Score is out of 120
# * University Rating is out of 5
# * Statement Of Purpose Rating is out of 5
# * Letter of Reccomendation Rating is out of 5
# * CGPA is out of 10
# * Research has only two values:- 1(done research) and 0(not done research)
# * Chance of Admit is ranges from 0.00 to 1.00 

# In[ ]:


df.head()


# In[ ]:


df.info()


# ##### Initial Observations:
# * dataset has 9 columns and 500 rows, in which none of them is null (we can also verify that by df.isnull().sum()
# * out of 9 columns, 7 columns are independent( 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research' ) and 1 column is dependent( 'Chance of Admit' )
# * column 'Serial No.' doesn't seems to have any value as rows are already indexed from 0 to 499, so we are going to drop it further steps
# 

# #### Dropping Column :

# As mentioned above, column 'Serial No.' has no use of keeping it, thus we are going to drop it

# In[ ]:


#dropping 'Serial No.' column
df.drop(columns=['Serial No.'],axis=1,inplace=True)


# In[ ]:


df.head()


# ---

# ## 3. Exploration & Visualisation Of Data 

# #### Checking Null Values 

# In[ ]:


#checking null values
df.isnull().sum()


# no null values in any of the columns

# Statistical Summary

# In[ ]:


#checking statistical summary
df.describe()


# ###### Observations from statistical summary :
# * GRE Score ranges from 290 to 340
# * TOEFL Score ranges from 92 to 120
# * CGPA ranges from 6.8 to 9.92
# * Chance of Admit ranges from 34% to 97%

# #### Plotting Histogram 

# In[ ]:


#histogram plot of df
df.hist(bins=20,figsize=(20,20))


# ###### Observations from Histogram Plot :
# * only CGPA, Chance of Admit, GRE Score & TOEFL Score are continuous rest are discrete
# * majority of candidates have CGPA ranging from 7.8 to 9.2 & very few of them have CGPA less than 7
# * majority of candidates has chance of admission of greater than 65%
# * few candidates has GRE score greater than 330
# * most of them has letter of recommendation of rating more than 2
# * most of the candidates applyying for admission has already done some reasearch
# * as in LOR, same goes with statement of purpose that majority of them has rating greater than 2
# * in TOEFL score, majority has score of ranging from 98 to 113
# * university with rating 1 has least number of appliant while the university with rating 3 has highest number of applicant

# #### Plotting Pairplot

# In[ ]:


#pairplot of dataframe df
sns.pairplot(df)


# ###### observations from pairplot :
# * with increase in GRE score, there is a high chance of getting admission
# * same goes with TOEFL score, that greater the TOEFL score, higher are the chances
# * university rating has a less impact
# * SOP, LOR & CGPA plays important role in getting admission and with increase in their values might give you a greater chance of admission

# #### Plotting HeatMap

# In[ ]:


#plotting annotated correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


#      value closer to 1 means that elements are positively correlated
#      value closer to 0 means that elements are negatively correlated    

# ###### order(descending) of features that plays important role in predicting chance of admission :
# 1. CGPA (0.88) 
# 2. GRE Score (0.81)
# 3. TOEFL Score (0.79)
# 4. University Rating (0.69)
# 5. SOP (0.68)
# 6. LOR (0.65)
# 7. Research (0.55) 

# ---

# ## 4. Data Preparation

# #### dividing dataset into dependent and independent variables

# as pointed out earlier in initial observations, that columns ('GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research') are independet variables i.e. these are the input parameters & the output or dependent variable is ('Chance of Admit')
# Thus we are going to divide initial dataframe df into two dataframes:
# 
# * x (independent variable)
# * y (dependent variable)

# In[ ]:


#creating dataframe with all feature except 'Chance of Admit'
#thus we are going to drop 'Chance of Admit' column from df and save it in x

x = df.drop(columns = ['Chance of Admit '])


# In[ ]:


x


# In[ ]:


#after that we are going to create a dataframe that has only 'Chance of Admit' column which will behave as output 

y = df['Chance of Admit ']


# In[ ]:


y


# #### Converting x and y to numpy arrays

# To perform transformation and scaling, we first need to convert newly created dataframe into numpy arrays

# In[ ]:


#converting into numpy arrays

x = np.array(x)
y = np.array(y)


# In[ ]:


x.shape


# In[ ]:


y.shape


# In[ ]:


#reshaping y

y = y.reshape(-1,1)


# In[ ]:


y.shape


# #### Splitting Data Into Testing & Training Dataset

# now that we have done all prepocessing to our data, we can finally split them by using train_test_split function of sklearn's model_selection .

# In[ ]:


#splitting data into 85% to training & 15% to testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)


# ---

# ## 5. Building Linear Regression Model

# First, we will build the model by the function LinearRegression() of sklearn's linear_model package & then fit the model based on the training dataset we just created

# In[ ]:


#Model building
model = LinearRegression()


# In[ ]:


#Fitting the model
model.fit(x_train, y_train)


# We have now successfully build & fit the linear regression model which will be going to predict the chances of admission to the university.
# since the model is linear thus it has intercept and coefficients(slope) of independent variables and the equation looks similar to
#             
#                        y = b0 + b1.x1 + b2.x2 + b3.x3 + b4.x4 + b5.x5 + b6.x6 + b7.x7
#    
# where, b0 is the intercept
#        and b1, b2, b3, b4, b5, b6, b7 are the coefficients 
#        
# as this is a multiple linear regression and our input has 7 features that's why we are getting 7 coefficients. No way above equation is a line but a hyperplane(in simple terms, plane in more than 3 dimensions).

# In[ ]:


#intercept of linear model
model.intercept_


# In[ ]:


#coefficients of linear model
model.coef_


# now that we have intercept and coefficients, we can simply plug in the values of x1,x2,...x7 and we have  our chances of admission.

# ---

# ## 6. Accuracy Of  Model

# we are going to check the accuracy of our model we just build by passing the testing data, as the testing data is new to the model, so it can be a good understanding as to how well the model is going to tackle real-world data

# In[ ]:


#accuracy of our model
accuracy = model.score(x_test, y_test)
accuracy


# It is more than 80% (it may vary), which is kind of a deasent score for such simple model.

# ---

# ## 7. Predicting

# #### Predicting output from x_test

# In[ ]:


#predicting output from x_test
y_pred = model.predict(x_test)
y_pred


# #### comparing predicted outcome from actual outcome

# In[ ]:


#comparing y_pred with y_test
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1


# ---
