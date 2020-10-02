#!/usr/bin/env python
# coding: utf-8

# ## Problem statement

# University admissions can be confusing and stressful. Most of the times, in order to know the status of admission it can take up a lot of time.
# 
# In this kernel, we will try to explore how we can maximize the Chance of Admit by focussing only on the most prominent factors.

# ## Importing necessary libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import scipy.stats as stats
import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
warnings.filterwarnings('ignore')


# ## Reading and understanding data
# 
# Let's choose the dataset Version1.1 with more number of columns

# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()


# In[ ]:


df.sample(5)


# In[ ]:


df.columns


# Dropping the column `Serial No.`, as it won't be necessary for analyzing the chance of admit.

# In[ ]:


serial_nums = df['Serial No.']
df.drop(columns=['Serial No.'],inplace=True)


# In[ ]:


df.shape


# In the columns `Chance of admit` and `LOR`, there are extra spaces in the end, so we will remove that from the columns name.

# In[ ]:


df.rename(columns={"Chance of Admit ":"Chance of Admit",
                  "LOR ":"LOR"},inplace=True)


# In[ ]:


df.describe()


# Separating numerical and categorical columns for analysis.

# In[ ]:


df.info()


# In[ ]:


df.nunique().sort_values()


# Even though few columns are integer or float types but they seem to have discrete unique values and hence they are most probably categorical in nature.

# In[ ]:


cat_columns = ['University Rating','SOP','LOR','Research']


# In[ ]:


num_columns = list(set(df.columns) - set(cat_columns))
num_columns 


# **Check for nulls**

# In[ ]:


df.isna().sum()


# There are no null values in the dataset so we can skip that (treating null values) part. However, we will still have to check for outliers in the dataset (exclusing the dependant variable).

# ## Observing the Dependant variable

# ### Chance of Admit

# In[ ]:


df['Chance of Admit'].head()


# In[ ]:


df['Chance of Admit'].describe()


# Creating a new columns high chance to identify profiles who are most likely to get an admit.

# In[ ]:


q3 = df['Chance of Admit'].quantile(q=0.75)
df['High Chance'] = df['Chance of Admit'].apply(lambda x: 1 if x>= q3 else 0)


# Chance of Admit is a `continuious numerical` variable so we can use box plot, violin plot,histogram or density plot. 
# 
# Let's observe a density plot below.

# In[ ]:


y = df['Chance of Admit']
y_skew = y.skew()
ylog_skew = np.log(y).skew()
y_kurt = y.kurtosis()
ylog_kurt = np.log(y).kurtosis()
print('Skew range, current:',y_skew)
print('Skew range, log:',ylog_skew)
print('Kurtosis, current:',y_kurt)
print('Kurtosis range, log:',ylog_kurt)

plt.figure(figsize=(10,8));
plt.title('Normal v/s Log normal plot')
sns.distplot(y,color='g',label="Normal")
sns.distplot(np.log(y),label="Log Normal")
plt.legend()
plt.show()


# Since the skew and kurtosis values are not improving with transformation, skipping the transformation as such.

# ## Observing the independant variables

# ### Univariate analysis

# #### Categorical columns

# In[ ]:


cat_columns


# Checking and removing outliers in the categorical data.

# In[ ]:


n_cat_cols = len(cat_columns)
# fig,ax = plt.subplots(1,n_cat_cols,figsize=(8*n_cat_cols,6))
# for i,col_name in enumerate(cat_columns):
#     plt.subplot(ax[i])
plt.rcParams['figure.figsize'] = 8,10
sns.boxplot(data=df.loc[:,cat_columns],orient='h')
plt.show()   


# An outlier was spotted only in the LOR column (in the left side of plot), we will treat that using **IQR method**.

# In[ ]:


df.iloc[347]


# In[ ]:


q1 = df['LOR'].quantile(q=0.25)
q3 = df['LOR'].quantile(q=0.75)

lower_limit = q1 - (q3-q1)*1.5
df['LOR'].loc[df['LOR'] < lower_limit] = np.NaN

# I checked the forward value and it is 2, so it will be safe to replace with forward fill
df['LOR'].fillna(method="ffill",inplace=True)


# To get an idea of data distribution, lets check categorical columns. 

# In[ ]:


fig,ax = plt.subplots(1,n_cat_cols,figsize=(6*n_cat_cols,6))

for i,col_name in enumerate(cat_columns):
    plt.subplot(ax[i])
    df[col_name].value_counts(normalize=True).plot.bar(title=col_name+"(% per category)")
plt.show()


# #### Numeric columns

# In[ ]:


num_columns


# In[ ]:


# Since Chance of Admit is our target/dependant varaiable, removing it from the num_columns
num_columns.remove('Chance of Admit')
num_columns


# *Check numerical columns for outliers*

# In[ ]:


# n_num = len(num_columns)
# fig,ax = plt.subplots(1,n_num,figsize=(6*n_num,6))
# for j,col_name in enumerate(num_columns):
#     plt.subplot(ax[j])
#     sns.distplot(df[col_name],bins=50)
#     plt.subplot(ax[i,j])
plt.rcParams['figure.figsize'] = 20,8
sns.boxplot(data=df.loc[:,num_columns],orient='h')
plt.show()


# Since the numerical columns are not having any outliers we can directly move on to the analysis.
# 
# We use **z-score method** to spot outliers in the numerical distributions.

# #### GRE Score

# In[ ]:


sns.FacetGrid(df,hue='High Chance',height=5)    .map(sns.distplot,'GRE Score')    .add_legend()
plt.show()


# #### TOEFL Score

# In[ ]:


sns.FacetGrid(df,hue='High Chance',height=5)    .map(sns.distplot,'TOEFL Score')    .add_legend()
plt.show()


# #### CGPA

# In[ ]:


sns.FacetGrid(df,hue='High Chance',height=5)    .map(sns.distplot,'CGPA')    .add_legend()
plt.show()


# The higher the PDF(probability distribution function) of two classes be separated from each other the more likely the feature will help in predicting the output (Chance of Admit).
# 
# And from the above diagrams, it can be seen that CGPA has the most separation. Which implies CGPA will be the most prominent factor determining a high chance of admit.

# ### Bivariate analysis

# In[ ]:


n_cat_cols,cat_columns


# #### Categorical v/s Conitnuous

# In[ ]:


fig,ax = plt.subplots(2,n_cat_cols,figsize=(6*n_cat_cols,12))
for i,col_name in enumerate(cat_columns):
    plt.subplot(ax[0,i])
    sns.violinplot(data=df,x=col_name,y='Chance of Admit',hue="High Chance",split=True)
    plt.subplot(ax[1,i])
#     sns.countplot(data=df,y=col_name,hue="High Chance")
    sns.stripplot(data=df,x=col_name,y='Chance of Admit',hue="High Chance",jitter=True)

plt.show()


# The following insights can be extracted from the above charts:
# - **University Rating**: It shows that if a student's University rating is good, the chance of admit increases. A student who studied in a university with rating 4 or 5 has a higher chance.
# - **SOP**: An SOP score more than 4 will be favourable.
# - **LOR**: Higher chance is seen for candidates with score more than 4.5.
# - **Research**: Many students who had high acceptance chance had a history of publishing paper. However, there are many students who published paper yet having low chance.

# ### Does having a good GRE score or TOEFL score or CGPA aids in higher chance of Admit?
# 
# Continuous v/s continuous variables

# In[ ]:


sns.jointplot(data=df,x="GRE Score",y='Chance of Admit',kind="kde")
sns.lmplot(data=df,y="GRE Score",x='Chance of Admit',hue="High Chance")
plt.show()


# In[ ]:


sns.jointplot(data=df,x="TOEFL Score",y='Chance of Admit',kind="kde")
sns.lmplot(data=df,x='Chance of Admit',y="TOEFL Score",hue="High Chance")
plt.show()


# In[ ]:


sns.jointplot(data=df,x="CGPA",y='Chance of Admit',kind="kde")
sns.lmplot(data=df,x='Chance of Admit',y="CGPA",hue="High Chance")
plt.show()


# Probably the answer for the above questionm is Yes!
# 
# All the 3 three features have a strong linear relationship with the `Chance of Admit`.

# Now let us compare the relationship between 
# - GRE Score and TOEFL Score
# - GRE Score and CGPA
# - CGPA and TOEFL Score

# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.scatterplot(data=df,x="GRE Score",y='TOEFL Score',size="Chance of Admit",hue="High Chance")


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.scatterplot(data=df,x="GRE Score",y='CGPA',size="Chance of Admit",hue="High Chance")


# In[ ]:


plt.figure(figsize=(8,8))
ax = sns.scatterplot(data=df,x="TOEFL Score",y='CGPA',size="Chance of Admit",hue="High Chance")


# ### Categorical v/s Categorical

# In[ ]:


cat_columns


# In[ ]:


fig,ax = plt.subplots(1,n_cat_cols,figsize=(6*n_cat_cols,6))
for i,col in enumerate(cat_columns):
    # Create a cross table for stacked graph
    pd.crosstab(df[col],df['High Chance'])

    ct = pd.crosstab(df[col],df['High Chance'],normalize="index")
    ct.plot.bar(stacked=True,ax=ax[i])
plt.show()


# ### Correlation between columns

# In[ ]:


# Removing serial no. and checking correlation.
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
fig,ax = plt.subplots(figsize=(8, 8))
with sns.axes_style("white"):
#     cmap="YlGnBu"
    sns.heatmap(corr, ax=ax, annot=True, mask=mask, square=True,fmt= '.2f',cmap="coolwarm")
plt.show()


# From the map, we can see that the target `Chance of Admit` has the most linear relation with the following :
# <br/>
# 
# Columns | Corelation 
# ---|---
# CGPA | 0.88
# GRE Score | 0.81
# TOEFL | 0.79
# 
# <br/>
# The least correlated featues is `Research` .

# ### Pair Plot

# In[ ]:


ax = sns.pairplot(df,hue="High Chance")


# ### Conclusion
# 
# > Focusing on LESS to get MORE
# 
# If a student wants to secure higher `Chance of Admit` with minimal amount of effort then the person should be focussing more on maximizing their CGPA, GRE Score and TOEFL Score.
