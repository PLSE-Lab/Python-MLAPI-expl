#!/usr/bin/env python
# coding: utf-8

# # Don't Eat Hair Conditioner EDA
# As stated in the accompanying PDF, data in this EDA reflects information AS REPORTED and does not represent any conclusion about whether a product actually caused any adverse events. For any given report, there is no certainty that a suspected product caused a reaction. I'm assuming that each row in the dataset means that the patient ingested that product.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../input/CAERS_ASCII_2004_2017Q2.csv')


# firstly convert dates to more usable formats

# In[3]:


df['RA_CAERS Created Date']=pd.to_datetime(df['RA_CAERS Created Date'], format='%m/%d/%Y')
df['AEC_Event Start Date']=pd.to_datetime(df['AEC_Event Start Date'], format='%m/%d/%Y')


# Interestingly the dataset is approx. 2/3 female.

# In[4]:


fig, ax = plt.subplots(figsize=(12,8))
sns.countplot(df['CI_Gender'])


# README file states that there are duplicates in the dataset, let's investigate that.

# In[5]:


print(df[23:31].drop(['RA_CAERS Created Date','AEC_Event Start Date','PRI_Product Role'],axis=1))


# The rows printed are for the same patient. So rows are duplicated for patients who have ingested many products. In addition to this the README states that there may be multiple reports for a single patient when the reports have been submitted erroneously by multiple people. Perhaps we could build a system to identify these. Lets look at which types of products are potentially causing the most health events

# In[6]:


fig, ax = plt.subplots(figsize=(8,12))
sns.set_style("ticks")
sns.countplot(y=df['PRI_FDA Industry Name']).set_title('Health event counts by product type')


# So Vit/Min/Prot/Unconv Diet(Human/Animal) and Cosmetics are by very far related to the most health events. Let's see which specific products cause many health events. By inspection of the most common products there is a 'redacted' value with 6081 observations, we will remove this as well as low frequency products to get a better picture of the data

# In[8]:


fig, ax = plt.subplots(figsize=(8,14))
product_count=df.groupby('PRI_Reported Brand/Product Name').size()
product_count_large=product_count[(product_count>150) & (product_count.index!='REDACTED')]
#print(product_count_large)
product_count_df=pd.DataFrame({'product':product_count_large.index,'count':product_count_large}, index=None)
new=product_count_df.merge(df[['PRI_Reported Brand/Product Name','PRI_FDA Industry Name']],how='inner', left_on='product', right_on='PRI_Reported Brand/Product Name').drop_duplicates()[['count','product','PRI_FDA Industry Name']]
sns.barplot(x='count',y='product',hue='PRI_FDA Industry Name',data=new,dodge=False).set_title("Products with more than 150 health events")


# Most of these are vitamin supplements as we may have expected after limiting by product type. Also raw oysters and hair conditioner. Don't eat these. Peanut butter is presumably due to nut allergy sufferers. 
# 
# ## Symptoms
# Let's take a look at the symptoms of these events split by product type (adapted from Aleksey Bilogur, Food Event Outcomes kernel)

# In[ ]:


outcomes=[]
for _, reactions in df['SYM_One Row Coded Symptoms'].astype(object).str.split(",").iteritems():
    outcomes += [str(l).strip().title() for l in pd.Series(reactions).astype(object)]

outcome_df=pd.DataFrame({'Outcome':pd.Series(outcomes).value_counts().index, 'Count':pd.Series(outcomes).value_counts()})[:100]
fig, ax = plt.subplots(figsize=(10,23))
sns.barplot(x='Count',y='Outcome', data=outcome_df).set_title('Health event counts by product type')


# Lots of diarrhoea, vomiting and nausea. Lovely.

# ## Effect of age

# In[ ]:


fig, ax = plt.subplots(figsize=(14,10))
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Soft Drink/Water')]['CI_Age at Adverse Event'], label='Soft Drink/Water')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Fishery/Seafood Prod')]['CI_Age at Adverse Event'], label='Fishery/Seafood Prod')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Vit/Min/Prot/Unconv Diet(Human/Animal)')]['CI_Age at Adverse Event'], label='Vit/Min/Prot/Unconv Diet(Human/Animal)')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Nuts/Edible Seed')]['CI_Age at Adverse Event'], label='Nuts/Edible Seed')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Cosmetics')]['CI_Age at Adverse Event'], label='Cosmetics').set(xlim=(0, 100))


# So 50-60 year olds are particularly susceptible to cosmetics, children and mature adults are susceptible to nuts, and the elderly are susceptible to vitamins/minerals. Late 50s seems to generally be the dangerous period.

# In[ ]:


print(df['RA_CAERS Created Date'].max())
print(df['RA_CAERS Created Date'].min())

#fig, ax = plt.subplots(figsize=(22,30))
#sns.countplot(y=df['AEC_Event Start Date']).set_title('Health event counts by product type')


# reports range from 2004 to June this year

# In[ ]:


print(df['AEC_Event Start Date'].max())


# ## Most deadly
# Let's do the morbid task of finding the most deadly products

# In[ ]:


deadly=df[(df['SYM_One Row Coded Symptoms']!=np.NaN) & (df['SYM_One Row Coded Symptoms'].str.contains('DEATH'))]
fig, ax = plt.subplots(figsize=(10,20))
product_count=deadly.groupby('PRI_Reported Brand/Product Name').size()
product_count_large=product_count[(product_count>1) & (product_count.index!='REDACTED')]
product_count_df=pd.DataFrame({'product':product_count_large.index,'count':product_count_large}, index=None)
new=product_count_df.merge(deadly[['PRI_Reported Brand/Product Name','PRI_FDA Industry Name']],how='inner', left_on='product', right_on='PRI_Reported Brand/Product Name').drop_duplicates()[['count','product','PRI_FDA Industry Name']]
sns.barplot(x='count',y='product',hue='PRI_FDA Industry Name',data=new).set_title("Products which were consumed by more than one patient who died")


# Perhaps needless to say, don't eat raw oysters.
# 
# Additional data I'd like to have: 
#  * Pre-existing medical conditions that might help explain particular health events. 
#  * The quantity a patient consumed.
