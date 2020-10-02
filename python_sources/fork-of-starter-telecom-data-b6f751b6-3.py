#!/usr/bin/env python
# coding: utf-8

# ## Telecom churn - Exploratory data analysis

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/telecom_churn.csv')
df.head()


# In[ ]:


print(df.shape)


# In[ ]:


print(df.columns)


# In[ ]:


df.info()


# In[ ]:


### Let us change the 'Churn' feature to 'int64'
df['churn'] = df['churn'].astype('int64')
df.head()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include=['object','bool'])


# In[ ]:


df['churn'].value_counts()


# In[ ]:


df['churn'].value_counts(normalize = 'True')


# ### sorting

# In[ ]:


df.sort_values(by='total day charge', ascending=False).head()


# In[ ]:


df.sort_values(by=['churn','total day charge'], ascending=[True, False]).head()


# ### Indexing and retrieving data

# ##### proportion of churned users in our data

# In[ ]:


df['churn'].mean()


# #### Boolean Indexing

# #### Let us find the average values of numerical features for churned users

# In[ ]:


df[df['churn'] == 1].mean()


# #### How much time(on average) do churned users spend on phone during daytime ?

# In[ ]:


df[df['churn'] == 1]['total day minutes'].mean()


# #### What is the maximum length of international calls among loyal users who do not have an international plan ?

# In[ ]:


df[(df['churn'] == 0) & (df['international plan'] == 'no')]['total intl minutes'].max()


# #### loc method is used for indexing by name

# In[ ]:


df.loc[0:5, 'state':'area code']


# #### iloc() is used for indexing by number

# In[ ]:


df.iloc[0:5, 0:3]


# #### Let us first line of the data frame

# In[ ]:


df[:1]


# 

# #### printing the last line of data frame 

# In[ ]:


df[-1:]


# ### Applying Functions to cells, columns and rows

# #### find the maximum of each column

# In[ ]:


df.apply(np.max)


# #### The apply method can be used to apply a function to each row. To do this, specify axis=1. Lambda functions are very convenient in such scenarios.

# #### Select all states starting with W ( use a lambda function )

# In[ ]:


df[df['state'].apply(lambda x : x[0] == 'W')].head()


# #### How to replace values in a column ? 
# #### This can be done by using map method ( by passing a dictionary of the form {old_value : new_value} as its argument

# #### Let us replace 'yes' & 'no' status of 'international plan' with 'True' & 'False'

# In[ ]:


d = {'yes' : True, 'no' : False}
df['international plan'] = df['international plan'].map(d)
df.head()


# #### we can also use the 'replace' method. 
# #### replace the 'yes' & 'no' of 'voice mail plan' to 'true' & 'false'

# In[ ]:


df = df.replace({'voice mail plan' : d})
df.head()


# ### Grouping
# #### Syntax : df.groupby(by=grouping_columns)[columns_to_show].function()
# #### 1. First, the groupby method divides the grouping_columns by their values. They become a new index in the resulting dataframe.
# #### 2. Then, columns of interest are selected (columns_to_show). If columns_to_show is not included, all non groupby clauses will be included.
# #### 3 Finally, one or several functions are applied to the obtained groups per selected columns.
# #### Example : we group the data according to the values of the 'churn' variable and display statistics of three columns in each group:

# In[ ]:


columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])


# In[ ]:


df.head()


# ### Summary Tables
# #### By using the 'crosstab' method we can build a contingency table
# #### For example : we will see how the observations in our sample are distributed in the contex of two variables ( 'churn' & 'international plan')

# In[ ]:


pd.crosstab(df['churn'], df['international plan'])


# In[ ]:


pd.crosstab(df['churn'], df['international plan'], normalize = True)


# #### Let us see the relation between 'churn' & 'voice mail plan'

# In[ ]:





# In[ ]:


pd.crosstab(df['churn'],df['voice mail plan'], normalize = True)


# #### From above two tables, we can notice that most of the users are loyal and do not use additional services like 'international plan' or 'voice mail'

# ### Pivot Tables
# #### pivot_table method takes the following parameters:
# ##### values : a list of variables to calculate statistics for,
# ##### index : a list of variables to group data by,
# ##### aggfunc : what statisitcs we need to calculate for groups, ex. sum, mean, maximum, minimum or something else
# #### Find the averages for total day calls, total eve calls, total night calls for each area (area code)  

# In[ ]:


df.pivot_table(['total day calls', 'total eve calls', 'total night calls'],
              ['area code'],
              aggfunc = 'mean')


# ### DataFrame Transformations

# #### Create a new feature 'total calls' and insert as a last column

# In[ ]:


total_calls = df['total day calls'] + df['total eve calls'] +                 df['total night calls'] + df['total intl calls']
df.insert(loc=len(df.columns), column = 'total calls', value = total_calls)
df.head()


# 

# #### we can add a column without creating an intermediate series instance:

# In[ ]:


df['total charge'] = df['total day charge'] + df['total eve charge'] +                      df['total night charge'] + df['total intl charge']
df.head()


# #### drop method is used to delete the rows or columns
# #### we need to pass the required indexes and the axis parameter ( 1 to delete columns, and nothing or 0 to delete rows )
# #### The inplace argument tells whether to change the original DataFrame.
# #### with inplace = False, the drop method doesn't change the existing DataFrame and returns a new one with dropped rows or columns
# #### with inplace = True, it alteres the DataFrame

# #### Delete those two features we created earlier 

# In[ ]:


df.drop(['total charge', 'total calls'], axis=1, inplace=True)
df.head()


# #### Let us drop the some rows

# In[ ]:


df.drop([1,2]).head()


# # Let us predict the churn

# #### What is the relation between the churn and international plan ?

# In[ ]:


pd.crosstab(df['churn'], df['international plan'], margins = True, normalize = True)


# In[ ]:


sns.countplot(x='international plan', hue='churn', data=df);


# #### From the above plot, we can notice that churn rate is much higher with the international plan. This indicates that customers with international plan are not happy with this telecom operator

# ### Let us check relation between the customer service calls and churn 

# In[ ]:


pd.crosstab(df['churn'], df['customer service calls'], margins = True)


# In[ ]:


sns.countplot(x = df['customer service calls'], hue = 'churn', data = df);


# #### From the above  table & plot it is clear that churn rate increased a lot if the customer calls are more than 4.
# #### now we will add a new binary feature ( customer service calls > 3 ) and then we will check the churn

# In[ ]:


df['many service calls'] = (df['customer service calls'] > 3).astype('int')
df.head()


# In[ ]:


pd.crosstab(df['many service calls'], df['churn'], margins = True)


# In[ ]:


sns.countplot(x = df['many service calls'], hue = df['churn'], data = df);


# #### It's time to see the relation between the international plan and 'many intenational calls'

# In[ ]:


pd.crosstab(df['many service calls'] & df['international plan'], df['churn'])


# In[ ]:


pd.crosstab(df['many service calls'] & df['international plan'], df['churn'], normalize = True)


# ## Conclusion
# #### we found a relation i.e
# ### "international plan = True & customer service calls > 3 => churn = 1, else churn = 0"
# #### The accuracy with above relation is 85.80% ( i.e 85.23 + 00.57) and error is 14.19% ( 13.92+0.27)
# #### this is a good starting point for further studies
