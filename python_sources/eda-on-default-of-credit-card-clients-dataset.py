#!/usr/bin/env python
# coding: utf-8

# # EDA on Default of Credit Card Clients Dataset
# 
# This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
# 
# ### Data Source
# Kaggle: [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/home)
# 
# Original: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
# 

# # Content
# 
# Total 25 columns, with 1 target variable **default.payment.next.month**, and 23 explanatory variables (ID excluded).
# * **ID**: ID of each client
# * **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
# * **SEX**: Gender (1 = male, 2 = female)
# * **EDUCATION**: (1 = graduate school, 2 = university, 3 = high school, 0,4,5,6 = others)
# * **MARRIAGE**: Marital status (0 = others, 1 = married, 2 = single, 3 = divorce)
# * **AGE**: Age in years
# * **PAY_0**: Repayment status in September, 2005 
# 
#  (-2 = No consumption, -1 = paid in full, 0 = use of revolving credit (paid minimum only), 1 = payment delay for one month, 2 = payment delay for two months, ... 8 = payment delay for eight months, 9 = payment delay for nine months and above)
#  
# * **PAY_2**: Repayment status in August, 2005 (scale same as above)
# * **PAY_3**: Repayment status in July, 2005 (scale same as above)
# * **PAY_4**: Repayment status in June, 2005 (scale same as above)
# * **PAY_5**: Repayment status in May, 2005 (scale same as above)
# * **PAY_6**: Repayment status in April, 2005 (scale same as above)
# * **BILL_AMT1**: Amount of bill statement in September, 2005 (NT dollar)
# * **BILL_AMT2**: Amount of bill statement in August, 2005 (NT dollar)
# * **BILL_AMT3**: Amount of bill statement in July, 2005 (NT dollar)
# * **BILL_AMT4**: Amount of bill statement in June, 2005 (NT dollar)
# * **BILL_AMT5**: Amount of bill statement in May, 2005 (NT dollar)
# * **BILL_AMT6**: Amount of bill statement in April, 2005 (NT dollar)
# * **PAY_AMT1**: Amount of previous payment in September, 2005 (NT dollar)
# * **PAY_AMT2**: Amount of previous payment in August, 2005 (NT dollar)
# * **PAY_AMT3**: Amount of previous payment in July, 2005 (NT dollar)
# * **PAY_AMT4**: Amount of previous payment in June, 2005 (NT dollar)
# * **PAY_AMT5**: Amount of previous payment in May, 2005 (NT dollar)
# * **PAY_AMT6**: Amount of previous payment in April, 2005 (NT dollar)
# * **default.payment.next.month**: Default payment (1=yes, 0=no)
# 
# Description gather from Kaggle Dataset > Overview & Discussion. But some column need to be rename and value need to be edit later. 

# ## Explore Original Data (before data cleaning)
# 
# 1. Import library needed. 
# 2. Read CSV.
# 3. View data info.
#     * Columns = 25
#     * Rows = 30,000
#     * No null value
# 4. View data sample.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sns.set()


# In[ ]:


df = pd.read_csv('../input/UCI_Credit_Card.csv')
df.info()
df.sample(10)


# ### Check unique value of each categorical variable

# In[ ]:


print('SEX ' + str(sorted(df['SEX'].unique())))
print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))
print('MARRIAGE ' + str(sorted(df['MARRIAGE'].unique())))
print('PAY_0 ' + str(sorted(df['PAY_0'].unique())))
print('default.payment.next.month ' + str(sorted(df['default.payment.next.month'].unique())))


# # Data Cleaning
# 
# 1. Value in **EDUCATION** not define completely. Since 0, 4, 5, 6 all being define as other (or unknown), will be better to group them together and name it 4.
# 2. Column name '**default.payment.next.month**' is better to name it '**DEFAULT**'. Shorter and without dot that could probably cause error in code.
# 3. Column name '**PAY_0**' would be better to name it '**PAY_1**', to make it consistent with 'BILL_AMT1' and 'PAY_AMT1'.

# In[ ]:


fill = (df.EDUCATION == 0) | (df.EDUCATION == 5) | (df.EDUCATION == 6)
df.loc[fill, 'EDUCATION'] = 4

print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))


# In[ ]:


df = df.rename(columns={'default.payment.next.month': 'DEFAULT', 
                        'PAY_0': 'PAY_1'})
df.head()


# ## Data Cleaned
# * Value **0, 5, 6** in EDUCATION all been **replaced with 4** to represent category of **other**.
# * Column PAY_0 renamed to **PAY_1**.
# * Column default.payment.next.month renamed to **DEFAULT**
# We have clean our data and ready for further analysis.

# # Explore Data with Visualization
# 
# Before looking at charts, take a look at some common statistical output in table below.
# * From the mean of DEFAULT, it shows around 22% default.
# * Refer to PAY_1 ~ PAY_6, at least 75% of client are not in payment delay status.

# In[ ]:


df.describe().transpose()


# ## Visualize with Heatmap
# Look at DEFAULT correlation with other variables. 
# * Lowest is -0.15 correlate with LIMIT_BAL.
#     *     negative correlation indicates higher Credit Limit, lower Default.
# * Highest is 0.32 correlate with PAY_1. 
#     *    positive correlation indicates longer period of Delay Payment, higher Default.
# * In general PAY_1 ~ PAY_6 have higher correlation to DEFAULT compare to other variables.
#     *    clients payment behaviour give strong indication on Default.

# In[ ]:


sns.set(rc={'figure.figsize':(27,10)})
sns.set_context("talk", font_scale=0.7)
    
sns.heatmap(df.iloc[:,1:].corr(), cmap='Reds', annot=True);


# ## Visualize Categorical Data with Barchart
# We first study how Demographic relate to Default.
# ### Explore Education vs. Default.
# 

# In[ ]:


sns.set(rc={'figure.figsize':(9,7)})
sns.set_context("talk", font_scale=0.8)

edu = sns.countplot(x='EDUCATION', hue='DEFAULT', data=df)
edu.set_xticklabels(['Graduate School','University','High School','Other'])
plt.show()


# ### Show Education level and Default Count in table.

# In[ ]:


default0 = df.groupby(df['EDUCATION'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')
default1 = df.groupby(df['EDUCATION'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')
total = df.groupby('EDUCATION').size().reset_index(name='TOTAL')

eduTable = default0.join(default1['DEFAULT']).join(total['TOTAL'])
eduTable['EDUCATION'] = ['Graduate School','University','High School','Other']

eduTable


# ### Turn Default Count into Percentage.

# In[ ]:


eduTable['NOT_DEFAULT'] = round((default0['NOT_DEFAULT']/total['TOTAL'])*100,2)
eduTable['DEFAULT'] = round((default1['DEFAULT']/total['TOTAL'])*100,2)

eduPct = eduTable.iloc[:,0:3]
eduPct = eduPct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})

eduPct


# ### Visualize again using Stacked chart
# 

# In[ ]:


sns.set(rc={'figure.figsize':(9,4)})
sns.set_context("talk", font_scale=0.8)

ax = eduPct.plot(x='EDUCATION', kind='barh', stacked=True, title='Education Level vs. Default')
ax.set_xlabel('PERCENT')
ax.get_legend().set_bbox_to_anchor((1, 0.9))
plt.show()


# ### Summary (Education Level vs. Default)
# Although the **Default Count** for High School is lower than Graduate School & University. But in terms of **Default Percentage**, those with High School level have higher chance of Default.

# ## Explore Marriage vs. Default (with same method).

# In[ ]:


sns.set(rc={'figure.figsize':(9,7)})
sns.set_context("talk", font_scale=0.8)

marri = sns.countplot(x="MARRIAGE", hue='DEFAULT', data=df )
marri.set_xticklabels(['Others','Married','Single','Divorce'])
plt.show()


# In[ ]:


default0 = df.groupby(df['MARRIAGE'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')
default1 = df.groupby(df['MARRIAGE'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')
total = df.groupby('MARRIAGE').size().reset_index(name='TOTAL')

marriTable = default0.join(default1['DEFAULT']).join(total['TOTAL'])
marriTable['MARRIAGE'] = ['Others','Married','Single','Divorce']

marriTable


# In[ ]:


marriTable['NOT_DEFAULT'] = round((default0['NOT_DEFAULT']/total['TOTAL'])*100,2)
marriTable['DEFAULT'] = round((default1['DEFAULT']/total['TOTAL'])*100,2)

marriPct = marriTable.iloc[:,0:3]
marriPct = marriPct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})

marriPct


# In[ ]:


sns.set(rc={'figure.figsize':(9,4)})
sns.set_context("talk", font_scale=0.8)

ax = marriPct.plot(x='MARRIAGE', kind='barh', stacked=True, title='Marital Status vs. Default')
ax.set_xlabel('PERCENT')
ax.get_legend().set_bbox_to_anchor((1, 0.9))
plt.show()


# ### Summary (Marital Status vs. Default)
# Although the **Default Count** for Divorce is way lower than Single & Married. But in terms of **Default Percentage**, those who Divorce have higher chance of Default.

# ## Explore Credit Behaviour 
# Now we explore PAY_1. As information in the heatmap, PAY_1 is the higest positive correlated variable.
# 
# * **PAY_1**: Repayment status in September, 2005 (most recent month)
# 
#  (-2 = No consumption, -1 = paid in full, 0 = use of revolving credit (paid minimum only), 1 = payment delay for one month, 2 = payment delay for two months, ... 8 = payment delay for eight months, 9 = payment delay for nine months and above)

# In[ ]:


sns.set(rc={'figure.figsize':(15,7)})
sns.set_context("talk", font_scale=0.8)

pay1 = sns.countplot(y="PAY_1", hue='DEFAULT', data=df)
pay1.set_yticklabels(['No Consumption','Paid in Full','Use Revolving Credit','Delay 1 mth','Delay 2 mths'
                     ,'Delay 3 mths','Delay 4 mths','Delay 5 mths','Delay 6 mths','Delay 7 mths','Delay 8 mths'])
pay1.set_title('Credit Behaviour (most recent month)')

plt.show()


# In[ ]:


default0 = df.groupby(df['PAY_1'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')
default1 = df.groupby(df['PAY_1'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')
total = df.groupby('PAY_1').size().reset_index(name='TOTAL')

pay1Table = default0.join(default1['DEFAULT']).join(total['TOTAL'])
pay1Table['PAY_1'] = ['No Consumption','Paid in Full','Use Revolving Credit','Delay 1 mth','Delay 2 mths'
                     ,'Delay 3 mths','Delay 4 mths','Delay 5 mths','Delay 6 mths','Delay 7 mths','Delay 8 mths']

pay1Table


# In[ ]:


pay1Table['NOT_DEFAULT'] = round((default0['NOT_DEFAULT']/total['TOTAL'])*100,2)
pay1Table['DEFAULT'] = round((default1['DEFAULT']/total['TOTAL'])*100,2)

pay1Pct = pay1Table.iloc[:,0:3]
pay1Pct = pay1Pct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})

pay1Pct


# In[ ]:


sns.set(rc={'figure.figsize':(9,5)})
sns.set_context("talk", font_scale=0.8)

ax = pay1Pct.sort_index(ascending=False).plot(x='PAY_1', kind='barh', stacked=True, title='Credit Behaviour vs. Default')
ax.set_xlabel('PERCENT')
ax.get_legend().set_bbox_to_anchor((1, 0.9))
plt.show()


# ### Summary (Credit Behaviour vs. Default)
# * Those Using Revolving Credit (paid only minimum) and those delayed for 2 months have the highest **Default Count**.
# * When payment is delayed more than 2 months, the **chances of default** goes higher than 50%. 
# 

# ## Incomplete Description
# There are some contradicting data in the dataset. It could be error when building up the data, or incomplete [definition](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/home) and explanation. Althought there is additional information in [discussion](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset/discussion/34608), still couldn't explain all.
# * One of the contradicting information is those who Paid in Full in recent month, turn out to be default. Shown in chart above. 
# * Some client paid more than they should be, resulting a Negative Bill Amount. But still end up being default. Refer table below.
# * Some client Bill Amount is much higher than their Credit Limit provided, which seldom happen in real case. Refer 2nd table below.
# 
# More description is needed from the datasest originator, to explain some of these contradicting data.

# In[ ]:


error1 = df.query('BILL_AMT1 < 0 and DEFAULT == 1').loc[:,('ID','BILL_AMT1','DEFAULT')]
error1.sample(5)


# In[ ]:


error2 = df.query('BILL_AMT1 > LIMIT_BAL').loc[:,('ID','LIMIT_BAL','BILL_AMT1')]
error2.sample(5)


# ## Visualize Numerical Data with Histogram
# 
# ### Explore Age vs. Default

# In[ ]:


df['AGE'].describe()


# In[ ]:


sns.distplot(df['AGE'], norm_hist=False, kde=False);


# * Youngest client is 21 years old, and oldest is 79.
# * Most client is age range from 26 to 35. 
# * With some specific age group having extra high number of people.

# ### Compare All Client vs. Defaulted Client
# 

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))

ax1.set_title('All Client', fontsize=14)
ax2.set_title('Defaulted Client', fontsize=14)

sns.distplot(df['AGE'], norm_hist=False, kde=False, ax=ax1);
sns.distplot(df['AGE'][df['DEFAULT'] == 1], norm_hist=True, kde=False, ax=ax2);


# ### We may still use tables and histogram to analyse percentage of Not_Default and Default.

# In[ ]:



default0 = df.groupby(df['AGE'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')
default0 = default0.fillna(0)
default1 = df.groupby(df['AGE'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')
default1 = default1.fillna(0)
total = df.groupby('AGE').size().reset_index(name='TOTAL')

ageTable = total.join(default0.set_index('AGE'),on='AGE').join(default1.set_index('AGE'),on='AGE')
ageTable = ageTable[['AGE', 'NOT_DEFAULT', 'DEFAULT', 'TOTAL']]
ageTable = ageTable.fillna(0)
ageTable


# In[ ]:


ageTable['NOT_DEFAULT'] = round((ageTable['NOT_DEFAULT']/ageTable['TOTAL'])*100,2)
ageTable['DEFAULT'] = round((ageTable['DEFAULT']/ageTable['TOTAL'])*100,2)

agePct = ageTable.iloc[:,0:3]
agePct = agePct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})

agePct


# In[ ]:


sns.set(rc={'figure.figsize':(9,10)})
sns.set_context("talk", font_scale=0.5)

ax = agePct.sort_index(ascending=False).plot(x='AGE', kind='barh', stacked=True, title='Age vs. Default')
ax.set_xlabel('PERCENT')
ax.get_legend().set_bbox_to_anchor((1, 0.98))
plt.show()


# * For each age group > 60, we only have few data. So the percentage may not be reliable.
# * Take a close look at age range from 26 to 35, they have comparative lower default rate. This is also supported by large amount of data.

# # Modelling with Logistic Regression
# Build a Logistic Regression Classification model to predict Default probability, based on mixed variables.

# * Assign Dependent Variable (Target) into y.
# * Assign one or multiple Independent Variable (Predictor) into X.
# * In first example, we use all explanatory variable, excluding ID which give no meaning.

# In[ ]:


X = df[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE'
        ,'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'
        ,'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'
        ,'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
y = df['DEFAULT'] 


# * Import required modelling tools from Scikit-learn
# * Split data into Train and Test set. We are using 70% from dataset as train data, and 30% of as test data
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# * Calculate the accuracy score. This calculation depends on X variables we have selected. In these example we have included all independent variables. We will calculate again later, with different mixed of variables.

# In[ ]:


from sklearn.metrics import accuracy_score

logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

accuracy_score(y_test,predictions)


# * If we use all variables to predict, the Accuracy Score is 0.784

# ## Modelling with Demographic data
# * When we use all demographic data to predict, the accuracy score have no big difference.

# In[ ]:


X = df[['SEX','EDUCATION','MARRIAGE','AGE']]
y = df['DEFAULT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

accuracy_score(y_test,predictions)


# ## Modelling with Credit Behaviour data
# * When we use all credit behaviour data to predict, the accuracy score increase to 0.814.

# In[ ]:


X = df[['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]
y = df['DEFAULT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

accuracy_score(y_test,predictions)


# ## Modelling with single independent variable
# * When we use **most recent month credit behaviour** to predict, the accuracy score increase to 0.826.

# In[ ]:


X = df[['PAY_1']]
y = df['DEFAULT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

accuracy_score(y_test,predictions)


# # Final Words
# 
# ### This is what we found:
# 
# * Credit behaviour, which shows their delay status, is the most important indicator for Default.
#     * When payment is delayed more than 2 months, the chances of default goes higher than 50%.
# * Demographic data have lower correlation to Default. However we can still look for some indication.
#     * Those who Divorce have higher chance of Default.
#     * Those with High School level have higher chance of Default.
#     * Those age range from 26 to 35, have lower Default rate.
# * Some unexplain data to take note, more description and explanation is needed.
#     * Some client who Paid in Full in recent month, turn out to be default.
#     * Some client have a Negative Bill Amount. But still end up being default.
#     * Some client Bill Amount is much higher than their Credit Limit provided.
# 
# 
# ### From the insights that we gained here, we are able to propose to management a few things:
# 
# * To lower the risk of default, must be very cautious on clients payment behaviour.
# * More cautious on Divorce and High School level clients.
# * Marketing campaign should be aiming on clients' age from 26 to 35.
# * Communicate with data collection team. 
#     * Ensure proper documentation of data description.
#     * Reduce data error, check the data collection and tracking system.
# 

# In[ ]:




