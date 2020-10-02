#!/usr/bin/env python
# coding: utf-8

# 
# # Complete data analysis of the "BLACK FRIDAY DATASET"
# 
# The following exercise seeks to analyze in a general way the purchasing behavior of customers on a "black Friday", it should be noted that in this analysis I only emphasize the treatment and preliminary analysis of the data in order to perform in a next version the Machine Learning models to make prediction, below I indicate the step by step made in this notebook:
# 
# * Creation of the Dataframe's
# * Preliminary analysis of the amount of data
# * Cleaning and preparation of data
# * Area of treatment and extraction of the most relevant information of our data set.
# 
# # NOTE: 
# I'm attentive to your comments that help improve my skills as a data scientist

# # Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading the data frame

# In[ ]:


bf=pd.read_csv('../input/BlackFriday.csv')


# In[ ]:


bf.head()


# # Basic Information data frame

# In[ ]:


bf.info()  


# In[ ]:


bf.columns


# # Data Cleaning
# 
# In this area we will proceed to observe what missing data we have to find a way to clean them

# In[ ]:


bf.isnull().head()


# In[ ]:


sns.heatmap(bf.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # Analysis :
# 
# As we can see in our heat map,have a lot of missing data in 2 of our product categories, the next step is to find a way to fill in this missing data, a way to make it is , fill the missing data with zeros , due to we can assume that these products were not purchased

# In[ ]:


#  fill the missing data with zeros

bf=bf.fillna(value='0')


# In[ ]:


# we can to observe that our DataFrame is clean

bf.head() 


# In[ ]:


sns.heatmap(bf.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#  
# NOTE : Our heat map shows that we have already completed our empty boxes and are ready to make an in-depth analysis of the data

# # Analysis of the quantity of products offered and the 10 most sold

# In[ ]:


len(bf['Product_ID'].value_counts()) # products offered


# In[ ]:


bf['Product_ID'].value_counts().head(10) # 10 most sold


# In[ ]:


sns.set_style('darkgrid')
bf['Product_ID'].value_counts().head(10).plot(kind='bar',color='green')


# In[ ]:





# # Top 10 of the users who make more purchases

# In[ ]:


# Sum of purchases made by user

Buy_by_User=bf[['User_ID','Purchase']].groupby('User_ID').sum() 

# Top 10 of the users who make more purchases

Buy_by_User.sort_values(by=['Purchase'],ascending=False).head(10)


# In[ ]:





# # Number of users in our database 

# In[ ]:


bf['User_ID'].nunique() # We note that have 5891 users


# # Number of users by category (AGE)
#  It was necessary to format the indexes to create a dataframe and thus be able to access the data

# In[ ]:


Age_by_User=bf[['User_ID','Age']].groupby(['User_ID', 'Age']).count() 
test=pd.DataFrame(Age_by_User)
# New dataframe to evaluate the number of clients by age category
New_Age_by_User=test.reset_index(inplace=False) 
New_Age_by_User.groupby('Age').count()


# In[ ]:


sns.set_style('darkgrid')
New_Age_by_User.groupby('Age').count().plot(kind='bar',color='green')


# # NOTE :
# 
# Of the 5891 users that we have in our data based,it is observed that most of our users are in the following ranges:
# 
# *  26-35
# *  36-45
# *  18-25
# 

#  # Number of users by category (GENDER)

# In[ ]:


Gender_by_User=bf[['User_ID','Gender']].groupby(['User_ID', 'Gender']).count() 
test_Gender=pd.DataFrame(Gender_by_User)
# New dataframe to evaluate the number of clients by age category
New_Gender_by_User=test_Gender.reset_index(inplace=False) 
New_Gender_by_User.groupby('Gender').count()


# In[ ]:


bf_gender=pd.DataFrame(index=['F','M']
                       ,columns=['# Users'],data=[1666,4225]) # we create a dataframe
plt.pie(New_Gender_by_User.groupby('Gender').count()
        ,autopct='%1.1f%%',labels=bf_gender.index,shadow=True, startangle=90)
plt.legend()


# NOTE : Of the 5891 users that we have in our data based, around 72% of our customers in a "Black Frifay" are men.

# # Who spend more on a black friday

# In[ ]:


bf[['Gender','Purchase']].groupby([ 'Gender']).sum()


# In[ ]:


bf_gender_pur=pd.DataFrame(index=['F','M'],columns=['Purchase']
                           ,data=[1164624021,3853044357]) # we create a dataframe
plt.pie(bf[['Gender','Purchase']].groupby([ 'Gender']).sum()
        ,autopct='%1.1f%%',labels=bf_gender_pur.index,shadow=True, startangle=90)
plt.legend()


# NOTE: About 77% of all purchases made are made by men on the other hand we note the low participation of women in a "Black Friday"

# #  Who spend more according to their marital status

# In[ ]:


bf[['Marital_Status','Purchase']].groupby([ 'Marital_Status']).sum()


# In[ ]:


bf_marital_sta=pd.DataFrame(index=['singles','married'],columns=['Purchase']
                           ,data=[2966289500,2051378878]) # we create a dataframe
plt.pie(bf[['Marital_Status','Purchase']].groupby([ 'Marital_Status']).sum()
        ,autopct='%1.1f%%',labels=bf_marital_sta.index,shadow=True, startangle=90,radius=0.8)
plt.legend(loc=1)


# NOTE: 59% of people who make purchases on a "Black Friday" are single, which makes a lot of sense

# # Which customers buy more according to their time in the city

# In[ ]:


Stay_City=bf[['Stay_In_Current_City_Years','Purchase']].groupby([ 'Stay_In_Current_City_Years']).sum()
Stay_City.sort_values(by=['Purchase'],ascending=False)


# In[ ]:


sns.set_style('darkgrid')
Stay_City.plot(kind='bar',color='g')


#  
# NOTE: As we observed earlier, customers who buy on a "Black Friday" are those who have come to be a year in the city

# # Most purchased products by men and women
# 
# First of all we are going to reset the indices to give a better management to our data

# In[ ]:


Prod_By_Gender=bf[['Product_ID','Purchase','Gender']].groupby([ 'Product_ID','Gender']).sum()
test_prod=pd.DataFrame(Prod_By_Gender)
New_product_by_gender=test_prod.reset_index(inplace=False)


# # Top 10 products purchased by men

# In[ ]:


New_product_by_gender[New_product_by_gender['Gender']=='M'].sort_values(by=['Purchase']
                                                        ,ascending=False).head(10)


# In[ ]:


New_product_by_gender[New_product_by_gender['Gender']=='F'].sort_values(by=['Purchase']
                                                        ,ascending=False).head(10)


# # Common products within the top 10 between both sexes
# 
# To carry out this analysis it is necessary that we know which is the top 10 most consumed products between men and women and then make use of the "Merge" function as follows

# In[ ]:


left=New_product_by_gender[New_product_by_gender['Gender']=='M'].sort_values(by=['Purchase']
                                                        ,ascending=False).head(10)


# In[ ]:


right=New_product_by_gender[New_product_by_gender['Gender']=='F'].sort_values(by=['Purchase']
                                                    ,ascending=False).head(10)


# In[ ]:


pd.merge(left, right, how='inner',on='Product_ID')


# NOTE : From the list of best-selling products between both sexes we can see that
# there are 7 products that match in the top 10

# # Analysis of the least sold products by gender.

# In[ ]:


left_less=New_product_by_gender[New_product_by_gender['Gender']=='M'].sort_values(by=['Purchase'],ascending=False).tail(10)
left_less    


# In[ ]:


right_less=New_product_by_gender[New_product_by_gender['Gender']=='F'].sort_values(by=['Purchase'],ascending=False).tail(10)
right_less


# In[ ]:


pd.merge(left_less, right_less, how='inner',on='Product_ID')


#  NOTE : 
# From the list of products sold less between both sexes we can see that there are only 2 products in common

# # Purchases according to the coustomer's occupation

# In[ ]:


#  Number of occupations
bf['Occupation'].nunique() 


# In[ ]:


# Sum of purchases made by user

Buy_by_User=bf[['Occupation','Purchase']].groupby('Occupation').sum() 

# Top 10 of the users who make more purchases

Buy_by_User.sort_values(by=['Purchase'],ascending=False)


# In[ ]:


sns.set_style('darkgrid')
Buy_by_User.plot(kind='bar',color='g')


# NOTE:
# In the following graph we can see that the occupations that most buy in a "Black Friday" are the occupations 4,0,7 and 1 respectively

# In[ ]:




