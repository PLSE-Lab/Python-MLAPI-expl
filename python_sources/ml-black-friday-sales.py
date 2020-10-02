#!/usr/bin/env python
# coding: utf-8

# # Black Friday Analysis
# ### Based on a retail stores data set on the sales from 2017

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# #### Importing the data on a csv file. 

# In[ ]:


df1 = pd.read_csv(r"../input/black-friday/BlackFriday.csv")


# Columns of the data set include -: 
# ##### 1) User ID 
# ##### 2) Product ID
# ##### 3) Gender
# ##### 4) Age
# ##### 5) Occupation -> 20 different occupations have been categorised numerically between 1-20.
# ##### 6) Category of City -> Cities are classified as tier A,B and C. 
# ##### 7) Number of years in a city -> Classified as 1,2,3, and 4+ years in a city.
# ##### 8) Marital Status -> 0 is unmarried and 1 is married. 
# ##### 9) Product Category -> There are 3 different type of product categories in 3 different columns and the specific product in the product type picked up in each category is given. 
# ##### 10) Purchase -> The amount spent by the customer is given under purchase. 

# In[ ]:


df1.head()


# #### Replacing na/nan values in the data frame with 0.

# In[ ]:


df1 = df1.fillna(0)


# #### Replacing 4+ years in the Stay_In_Current_City_Years column with a 4 because it is a string and an ML model requires integers to draw conclusion. Simillar technique is used in case of other columns to replace string instances with numeric data. Categories are replaced with 0,1,2... 

# In[ ]:


new_df = df1.Stay_In_Current_City_Years.replace(['4+'],4)
new_df.head(10)


# In[ ]:


df1['new_stay'] = new_df
df1.head(10)


# In[ ]:


new_df = df1.Age.replace(['0-17','18-25','26-35','36-45','46-50','51-55','55+'],[0,1,2,3,4,5,6])
new_df.head(10)


# In[ ]:


df1['new_age'] = new_df
df1.head(10)


# In[ ]:


new_df = df1.City_Category.replace(['A','B','C'],[0,1,2])
new_df.head(10)


# In[ ]:


df1['new_city'] = new_df
df1.head(10)


# In[ ]:


new_df = df1.Gender.replace(['M','F'],[0,1])
new_df.head(10)


# In[ ]:


df1['new_gender'] = new_df
df1.head(10)


# ## Data cleaning has been performed and a new table has been generated. Moving to visualisation.

# ### Split up of age,gender,city based on population

# In[ ]:


AgeGroup = df1['Age'].map(lambda n: n.split("|")[0].split(":")[0]).value_counts().head(20)
AgeGroup.plot.bar()


# In[ ]:


sns.countplot(df1['Gender'])


# In[ ]:


sns.countplot(df1['City_Category'])


# ### Split up based on (Age,Occupation) and Gender of the customers.

# In[ ]:


sns.countplot(df1['Age'],hue=df1['Gender'])


# In[ ]:


sns.countplot(df1['Occupation'],hue=df1['Gender'])


# In[ ]:


sns.countplot(df1['Purchase'],hue=df1['City_Category'])


# In[ ]:


print(df1['Gender'].value_counts())


# In[ ]:


df1_Male = df1.loc[df1['Gender'] == 'M']
df1_Female = df1.loc[df1['Gender'] == 'F']


# In[ ]:


total_spending_male = df1_Male['Purchase'].sum()
total_spending_male


# In[ ]:


total_spending_female = df1_Female['Purchase'].sum()
total_spending_female


# ### Total Spent by Males and Females

# In[ ]:


spending_data = [['M',total_spending_male],['F',total_spending_female]]
df2 = pd.DataFrame(spending_data, columns=('Gender','Purchase'))
df2


# In[ ]:


df1_A = df1.loc[df1['City_Category'] == 'A']
df1_B = df1.loc[df1['City_Category'] == 'B']
df1_C = df1.loc[df1['City_Category'] == 'C']


# In[ ]:


total_spending_A = df1_A['Purchase'].sum()
total_spending_A


# In[ ]:


total_spending_B = df1_B['Purchase'].sum()
total_spending_B


# In[ ]:


total_spending_C = df1_C['Purchase'].sum()
total_spending_C


# In[ ]:


### Total Spent by each city category.


# In[ ]:


spending_data_City = [['A',total_spending_A],['B',total_spending_B],['C',total_spending_C]]
df3 = pd.DataFrame(spending_data_City, columns=('City_Category','Purchase'))
df3


# In[ ]:


df1['combined_G_M'] = df1.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
print(df1['combined_G_M'].unique())


# ### Number of males and females who are married and unmarried in each age group.

# In[ ]:


df5 = df1.groupby(['combined_G_M','Age']).size()
df5


# In[ ]:


sns.countplot(df1['Age'],hue=df1['combined_G_M'])


# ### Box plot comparing purchase of various age groups

# In[ ]:


sns.boxplot('Age','Purchase', data = df1)
plt.show()


# In[ ]:


Age_buy = df1.groupby(["Age"])["Purchase"].sum()
Age_buy.plot.bar()


# In[ ]:


Occu_buy = df1.groupby(["Occupation"])["Purchase"].sum()
Occu_buy.plot.bar()


# In[ ]:


City_buy = df1.groupby(["City_Category"])["Purchase"].sum()
City_buy.plot.bar()


# ### Bar chart plotting amount spent in each product category by various age groups

# In[ ]:


product_age = df1.groupby(["Age"])["Product_Category_1", "Product_Category_2", "Product_Category_3"].sum()
product_age.plot.bar()


# ### Determing the statistics of the purchase by the customers. Thus determinding what makes a good and a bad customer.

# In[ ]:


df1.Purchase.describe()


# ### Plotting a scatter plot and correlation matrix to see any pattern in the purchase to be able to draw a regression line. 

# In[ ]:


df10 = df1.head(1000)
df10.plot.scatter(x = "User_ID",y="Purchase")


# In[ ]:


df1.corr()


# ### We have taken any purchase over that of 12073 to be that of a good customer. 12073 is the 75 percentile mark based on our data description. 

# #### Getting the purchase in a binary form based on whether a customer is a good one or not a good one. 1 implies it is a good customer. 0 implies it is not a good customer. 

# In[ ]:


clean_data = df1.copy()
clean_data['good_customer'] = (clean_data['Purchase'] > 12073)*1
print(clean_data['good_customer'])


# In[ ]:


y=clean_data[['good_customer']].copy()


# In[ ]:


y.head()


# In[ ]:


df1.columns


# In[ ]:


customer_features = ['new_age', 'Occupation', 'new_city',
       'new_stay', 'Marital_Status','new_gender', 'Product_Category_1',
       'Product_Category_2', 'Product_Category_3']


# #### Building an X and Y dataframe to perform a decision tree analysis.

# In[ ]:


X = clean_data[customer_features].copy()


# In[ ]:


X.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)


# In[ ]:


good_customer_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
good_customer_classifier.fit(X_train, y_train)


# In[ ]:


predictions = good_customer_classifier.predict(X_test)


# #### Predictions for the first 10 customers.

# In[ ]:


predictions[:10]


# #### Comparison with actual results

# In[ ]:


y_test['good_customer'][:10]


# ### Accuracy % of our decision tree model

# In[ ]:


accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
print(accuracy * 100)


# In[ ]:




