#!/usr/bin/env python
# coding: utf-8

# <font size="6"> **Predicting Customer Churn at a Bank ** </font>
# 
# 
# Customer Churn is the number that indicate the proportion of customers that discontinue using a product. it is therefore crucial for brands like banks to detect customer that are likely to churn and retain them. 
# 
# In this Notebook, we are going to work with a dataset. The dataset holds information on a bank customers and we are to predict customer churn based on several attributes provided. 
# 

# <font size="5">**Exploratory Analyis** </font>
# 
# 
# In this section, we are going to do the following: 
# 1. Data Preparation ( but we won't do  much, the dataset is pretty clean)
# 2. Exploratory analysis to  understand our data before modeling

# In[ ]:


# Importing all the libraries that will be needed in this analysis
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as plty
import plotly.express as px


# In[ ]:


#importing the dataset in a dataframe that we name df
df = pd.read_csv("../input/predicting-churn-for-bank-customers/Churn_Modelling.csv")
df.head()


# In[ ]:


# a look at a sample from the dataset
df.head()


# there is no missing data, thats great
# Now lets check the dataset shape to understand how many columns and rows we have. 
# we have 10,000 rows representing the numbers of customers and 14 columns representing different values

# In[ ]:


# checking the shape of the dataset
df.shape


# In[ ]:


# Checking number of attributes, null values and types of each column
df.info()


# ****Explaining all the dataset's columns: ********
# 1. RowNumber          10000 non-null int64 --> numbering columns
# 2. CustomerId         10000 non-null int64 --> Customer ID
# 3. Surname            10000 non-null object --> Customer name
# 4. CreditScore        10000 non-null int64 --> Customer Credit Score (likehood to payback credit)
# 5. Geography          10000 non-null object --> location of the client 
# 6. Gender             10000 non-null object --> Customer Gender
# 7. Age                10000 non-null int64 --> Customer age
# 8. Tenure             10000 non-null int64 --> Number of month customer spent as a customer
# 9. Balance            10000 non-null float64 --> customer balance
# 10. NumOfProducts      10000 non-null int64 --> number of product the customer has subcribed to 
# 11. HasCrCard          10000 non-null int64 --> does the customer have a credit card or not?
# 12. IsActiveMember     10000 non-null int64 --> is the customer an active member or not?
# 13. EstimatedSalary    10000 non-null float64 --> customer estimated income
# 14. Exited             10000 non-null int64 --> did the customer left the brand or not? 

# I am now going to delete unecessary columns, regarding the analysis

# In[ ]:


# deleting Rownumbers, customerID and Surname columns
df_1 = df.drop (['RowNumber', 'CustomerId','Surname'], axis= 1)
df_1.info()


# In[ ]:


# Checking statisctics on all numerical values
df_1.describe()


# In[ ]:


# checking unique values
df_1.nunique()


# we are now going to explore the variables distributions, as well as correlation with customer attrition

# In[ ]:


# Plotting an histogram to visualize the relationship (if any between the credit score, customer attrition and gender)
fig = px.histogram(df, x="CreditScore", y="Exited", color="Gender", marginal="rug",
                   hover_data=df.columns)
fig.show()


# In[ ]:


fig = px.histogram(df, x="Geography", color="Exited",
                   hover_data=df.columns)
fig.show()


# Participant in this dataset all from  3 countries. more than half of the participants are from france. and its seems germans tend to leave the brand much more frequently than clients in spain or France. 

# In[ ]:


fig = px.histogram(df, x="Gender", color="Exited")
fig.show()


# Female customer seems more likely to  compared to male customer. 

# In[ ]:


# Plotting balance 
fig_1 = px.histogram(df, x="Balance", y="Exited", color="HasCrCard", 
                   hover_data=df.columns)
fig_2 = px.histogram(df, x="Balance", y="Exited", color="IsActiveMember",hover_data=df.columns)
fig_1.show()
fig_2.show()


# people with low empty bank account will tend to churn more, 

# In[ ]:


fig = px.histogram(df, x="Tenure", color="Exited", color_continuous_scale=px.colors.diverging.RdYlGn[::-1])
fig.show()


# The Amount of time previously spent with the brand doesnt seem to affect departing for customers. long term customers seems to leavea as easily as newer customers. 

# In[ ]:


df.info()


# In[ ]:


fig = px.scatter(df, x="EstimatedSalary", y="Balance", color = "Exited", size= "Age")
fig.show()


# there seem to be no pattern between Estimated Salary, Balance , age and exiting the bank

# <font size="5"> **Prediction** </font>
# 
# in this section we are going to do the following: 
# 
# 1. Feature engineering
# 2. Data modeling

# In[ ]:


# studying the relationship between the variables in the dataset
df_1[df_1.columns].corr()


# In[ ]:


fig = px.scatter_matrix(df_1, 
                       dimensions=["creditScore", "Age", "bala])
fig.show()

fig = px.scatter_matrix(df_1,
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species")
fig.show()

