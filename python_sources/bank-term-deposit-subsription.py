#!/usr/bin/env python
# coding: utf-8

# ### The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. 
#  
#  
# #### A term deposit is a fixed-term investment that includes the deposit of money into an account at a financial institution. Term deposit investments usually carry short-term maturities ranging from one month to a few years and will have varying levels of required minimum deposits

# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


df =pd.read_csv("/kaggle/input/portuguesebankidirectmarkcampaignsphonecalls/bank-full2.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ### Target Variable Value Counts

# In[ ]:


df['y'].value_counts()


# In[ ]:


df['y'].value_counts(normalize=True)


# 88 % of the data is of people who haven't subsribed to the term deposit and around 12 % is of those who subscribed.

# In[ ]:


df2=df.copy()


# In[ ]:


df['y']=df['y'].map({'no':0,'yes':1})


# In[ ]:


df.head()


# ### Missing/Unknown Values 

# In[ ]:


df.isnull().sum()


# There are no 'Nan' or Null values in the dataset, let's look into each variable to get better insights about the data.

# ### Analysing Categorical Variables

# In[ ]:


cat_col = df.select_dtypes('object').columns


# In[ ]:


cat_col


# In[ ]:


for i in cat_col:
    
    print("Feature : ",i)
    print(df[i].value_counts(normalize=True))
    print("\n")


# Unknowns present in job, education,contact and poutcome

# #### 1. Poutcome

# In[ ]:


df['poutcome'].value_counts(normalize=True)


# About 82% of the data is 'unknown'

# In[ ]:


df[df['poutcome']=='unknown'][['pdays','poutcome']]


# In[ ]:


df[(df['poutcome']=='unknown') & (df['pdays']!=-1)][['pdays','poutcome']]


# In[ ]:


df[df['pdays']==-1].shape


# In[ ]:


df[(df['poutcome']=='unknown') & (df['pdays']==-1)].shape


# 36954 people were not previously contacted before this campaign, so for these records, poutcome can be changed from unknown to 'not_contacted_prev'
# 
# 5 records present where they were contacted previously , but we do not have outcome of the same. (pdays != -1 and poutcome is unknown)

# In[ ]:


df.loc[(df['poutcome']=='unknown') & (df['pdays']==-1),'poutcome'] = 'not_contacted_prev'


# In[ ]:


df[(df['poutcome']=='not_contacted_prev') & (df['pdays']==-1)].shape


# In[ ]:


df[df['poutcome']=='other']


# In[ ]:


df[(df['poutcome']=='other') & (df['pdays']==-1)]


# As we have limited domain knowledge about what 'others' signify in poutcome, let us keep it as it is and assume that these were people who did not take a decision to either subscribe or not for the previous campaign.

# In[ ]:


print("Analysis of feature : 'Poutcome'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='poutcome',order=df['poutcome'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['poutcome']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='poutcome',x='Percentage',hue='y',data=prop_df,order=df['poutcome'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df


# Majority of the people have not been contacted in the previous campaign for term deposit.
# 
# The people who have subscribed for a term deposit in the previous campaign has 65% chances of subscribing it again in the current campaign. These customers can be targeted more to increase the rate of subscription.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(x='poutcome',data=df,hue='y',ax=ax1[0])
prop_df = (df['poutcome'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['poutcome'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed people, most of the people are not contacted in the previous campaign, followed by people who subscribed for term deposit inthe previous campaign.

# ### 2. Job

# In[ ]:


df['job'].value_counts(normalize=True)


# There are 12 categories present in the job feature, in which one is 'unknown'.
# 
# Blue collar jobs followed by management jobs and technician jobs are most common in the dataset.

# In[ ]:


df[df['job']=='unknown'].shape


# In[ ]:


df[(df['job']=='unknown') & (df['pdays']==-1)].shape


# There are 288 'unknown' values(0.637%) present in the field 'job'. Out of which the majority of the records (255) are of people who were not contacted in the previous campaign.

# In[ ]:


print("Analysis of feature : 'Job'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='job',order=df['job'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['job']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='job',x='Percentage',hue='y',data=prop_df,order=df['job'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)


# Students and retired customers are more likely to subscribe to the term deposit compared to other job categories.
# 
# These customers can be targeted more to increase the rate of subscription.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(y='job',data=df,hue='y',order=df['job'].value_counts().index,ax=ax1[0])
prop_df = (df['job'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['job'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed customers, most common jobs are 'management' followed by 'technician' and 'blue collar' jobs.

# ### 3. Education

# In[ ]:


df['education'].value_counts(normalize=True)


# There are 4 categories present in the education feature, in which one is 'unknown'.
# 
# 51% of the customers are having secondary education.

# In[ ]:


df[df['education']=='unknown'].shape


# In[ ]:


df[(df['education']=='unknown') & (df['pdays']==-1)].shape


# There are 1857 'unknown' values(4.1%) present in the field 'education'. Out of which the majority of the records (1534) are of people who were not contacted in the previous campaign.

# In[ ]:


print("Analysis of feature : 'Education'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='education',order=df['education'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['education']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='education',x='Percentage',hue='y',data=prop_df,order=df['education'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)


# The customers with teritiary level of education has slightly higher chances of subscription of the term deposit, compared to other education levels.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(y='education',data=df,hue='y',order=df['education'].value_counts().index,ax=ax1[0])
prop_df = (df['education'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['education'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed customers, most customers have secondary level of education followed by teritiary.

# ### 4. Marital

# In[ ]:


df['marital'].value_counts(normalize=True)


# There are 3 categories present in the marital feature.
# 
# 60% of the customers are married.

# In[ ]:


print("Analysis of feature : 'Marital'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='marital',order=df['marital'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['marital']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='marital',x='Percentage',hue='y',data=prop_df,order=df['marital'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)


# The customers who are single has a slightly higher chances of subscription of the term deposit compared to customers who are married or divorced.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(y='marital',data=df,hue='y',order=df['marital'].value_counts().index,ax=ax1[0])
prop_df = (df['marital'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['marital'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed customers, most customers are married followed by single customers.

# ### 5. Default

# In[ ]:


df['default'].value_counts(normalize=True)


# 98 % of the customers are do not have any credit in default.

# In[ ]:


print("Analysis of feature : 'Default'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='default',order=df['default'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['default']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='default',x='Percentage',hue='y',data=prop_df,order=df['default'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)


# The customers who do not have any credits in default, has a slightly higher chances of subscription of the term deposit compared to customers who have credits in default.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(y='default',data=df,hue='y',order=df['default'].value_counts().index,ax=ax1[0])
prop_df = (df['default'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['default'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed customers, most customers do not have any credits in default.

# ### 6. Housing

# In[ ]:


df['housing'].value_counts(normalize=True)


# 55.5% of the customers have taken housing loans while the other 45.5% do not have any housing loans in their name.

# In[ ]:


print("Analysis of feature : 'Housing'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='housing',order=df['housing'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['housing']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='housing',x='Percentage',hue='y',data=prop_df,order=df['housing'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)


# The customers who do not have a housing loan has a slightly higher chances of subscription of the term deposit compared to customers who already have a housing loan.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(y='housing',data=df,hue='y',order=df['housing'].value_counts().index,ax=ax1[0])
prop_df = (df['housing'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['housing'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed customers, most customers do not have housing loans.

# ### 7. Loan

# In[ ]:


df['housing'].value_counts(normalize=True)


# 55.5% of the customers have taken housing loans while the other 45.5% do not have any housing loans in their name.

# In[ ]:


print("Analysis of feature : 'Housing'")
fig, ax1 = plt.subplots(1,2,figsize=(15,5))
sns.countplot(y='housing',order=df['housing'].value_counts().index,data=df,ax=ax1[0])
prop_df = (df['y'].groupby(df['housing']).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
sns.barplot(y='housing',x='Percentage',hue='y',data=prop_df,order=df['housing'].value_counts().index,ax=ax1[1])
ax1[1].set(xticks = np.array(range(0,100,5)))
plt.show()


# In[ ]:


prop_df[prop_df['y']==1].sort_values(by='Percentage',ascending=False)


# The customers who do not have a housing loan has a slightly higher chances of subscription of the term deposit compared to customers who already have a housing loan.

# In[ ]:


fig, ax1 = plt.subplots(1,2,figsize=(15,7))
sns.countplot(y='housing',data=df,hue='y',order=df['housing'].value_counts().index,ax=ax1[0])
prop_df = (df['housing'].groupby(df['y']).value_counts().rename('Values').reset_index())
prop_df = prop_df[prop_df['y']==1]
plt.pie(prop_df['Values'],labels=prop_df['housing'],autopct='%1.1f%%')
ax1[1].set_title("Percentage of each category out of Subcribed people")
plt.show()


# Out of the subscribed customers, most customers do not have housing loans.

# In[ ]:


cat_col


# In[ ]:


df[df['education']=='unknown'].shape


# In[ ]:


df[(df['education']=='unknown') & (df['pdays']==-1)].shape


# In[ ]:


df[df['contact']=='unknown'].shape


# In[ ]:


df[(df['contact']=='unknown') & (df['pdays']==-1)].shape


# Most of the records with 'unkowns' as a category in the features - job, education,contact and poutcome are data of people who haven't been contacted in the previous campaign.

# In[ ]:


num_cols = df.select_dtypes('int64').columns


# ### Outliers

# In[ ]:


for i in num_cols:
    sns.boxplot(y=i,data = df)
    plt.show()


# All numerical variables have high outliers.

# In[ ]:





# In[ ]:


for i in num_cols:
    sns.boxplot(y='age',x='y',data = df)
    plt.show()


# In[ ]:





# In[ ]:


#df_log = df[num_cols]
#df_log.apply(np.log)


# ### Correlations

# In[ ]:


sns.pairplot(df2)


# In[ ]:


sns.pairplot(df2,hue='y')


# In[ ]:


df2.corr()


# No considerable correlations present between features

# ### Numerical Variable analysis

# In[ ]:


for i in num_cols:
    sns.barplot(x='y',y=i,data = df)
    plt.show()


# Balance, duration,campaign, pdays, previous are having considerable difference in both classes.

# ### Distributions

# In[ ]:


df0=df[df['y']=='no']
df1=df[df['y']=='yes']


# In[ ]:


for i in num_cols:
    sns.distplot(df0[i])
    sns.distplot(df1[i])
    plt.show()


# In[ ]:


cat_col1 = df2.select_dtypes('object').columns
cat_col1


# In[ ]:





# In[ ]:


df2['prev_contacted'] = list(map(lambda x : 'Yes' if x != -1 else 'No',df2['pdays']))


# In[ ]:


df2[df2['prev_contacted']=='Yes']


# In[ ]:


#fig, ax1 = plt.subplots(figsize=(15,5))
#plt.figure(figsize=(15,5))

for col in cat_col1:
    print("Analysis of feature : ",col)
    fig, ax1 = plt.subplots(1,2,figsize=(15,5))
    #sns.countplot(y=col,order=df[col].value_counts().index,hue='y',data=df,ax=ax1[0])
    sns.countplot(y=col,order=df2[col].value_counts().index,data=df2,ax=ax1[0])
    prop_df = (df2['y'].groupby(df2[col]).value_counts(normalize=True).mul(100).rename('Percentage').reset_index())
    sns.barplot(y=col,x='Percentage',hue='y',data=prop_df,order=df2[col].value_counts().index,ax=ax1[1])
    ax1[1].set(xticks = np.array(range(0,100,5)))
    plt.show()


# Job:- Most contacted - Blue collar, campaign successful mostly among students followed by retired people.
# 
# Marital- Most contacted - Married, campaign successful mostly among single people.
# 
# Education - Most contacted - Secondary, campaign successful mostly among people with teritiary education.
# 
# Default - Most contacted - people with no defaults, campaign successful mostly among people with no defaults
# 
# Housing - Most contacted - people who have taken housing loans, campaign successful mostly among people who have not taken housing loans
# 
# Loan - Most contacted - people who have taken housing loans, campaign successful mostly among people who have not taken housing loans
# 
# Contact - Most contacted - Cellular, campaign successful in similar rates among people contacted through telephone as well as cellular.
# 
# Month -  Most contacted - May, campaign successful mostly among people contacted during the months March, December, September and October.

# In[ ]:


for col in cat_col1:
    print("Analysis of feature : ",col)
    print(df2.groupby(col)['y'].value_counts(normalize=True))


# In[ ]:


df[df['campaign']==0]


# All are contacted during the current campaign. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




