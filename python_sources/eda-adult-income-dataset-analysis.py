#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as mpatches


# # Import data

# In[ ]:


df = pd.read_csv('../input/adult-income-dataset/adult.csv')
df.head(10)


# Data description

# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). The prediction task is to determine whether a person makes over $50K a year.

# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df['income'].value_counts()


# In[ ]:


df['income']=df['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
df.head()


# In[ ]:


df.describe(include='all')


# # Data Cleaning

# In our datset, we can see that missing values are present in ("workclass","native-country","occupation") form of "?".

# In[ ]:


from numpy import nan
df = df.replace('?',nan)
df.head()


# In[ ]:


null_values = df.isnull().sum()
null_values = pd.DataFrame(null_values,columns=['null'])
j=1
sum_total=len(df)
null_values['percentage'] = null_values['null']/sum_total
round(null_values*100,3).sort_values('percentage',ascending=False)


# By observation of above, In  ("workclass","native-country","occupation") null values are present 

# In[ ]:


print('workclass',df.workclass.unique())
print('education',df.education.unique())
print('marital-status',df['marital-status'].unique())
print('occupation',df.occupation.unique())
print('relationship',df.relationship.unique())
print('race',df.race.unique())
print('gender',df.gender.unique())
print('native-country',df['native-country'].unique())


# In[ ]:


df['native-country'].fillna(df['native-country'].mode()[0],inplace = True)


# In[ ]:


df['workclass'].fillna(df['workclass'].mode()[0],inplace = True)


# In[ ]:


df['occupation'].fillna(df['occupation'].mode()[0],inplace = True)


# In[ ]:


null_values = df.isnull().sum()
null_values = pd.DataFrame(null_values,columns=['null'])
j=1
sum_total=len(df)
null_values['percentage'] = null_values['null']/sum_total
round(null_values*100,3).sort_values('percentage',ascending=False)


# lets look the data it again:  with the help of above oprations all null values the remove,he
# now our data is null free.

# # Exploratory Data Analysis

# In[ ]:


sns.pairplot(df)


# # Univariate analysis

# # Age Distribution:

# In[ ]:


df['age'].hist(figsize = (6,6))
plt.show


# By observation age attribute is right-skewed and not symetric.
# min and max age in btw 17 to 90.

# # finalwieght Distribution:

# In[ ]:


df['fnlwgt'].hist(figsize = (5,5))
plt.show()


# It seems like Rightly skewed.

# # Capital Gain Distribution:

# In[ ]:


df['capital-gain'].hist(figsize=(5,5))
plt.show()


# capital-gain shows that either a person has no gain or has gain of very large amount(10k or 99k).

# In[ ]:


df['capital-loss'].hist(figsize=(5,5))
plt.show()


# This histogram shows that most of the "capital-loss" values are centered on 0 and only few are non zero(2282).
# This attribute is similar to the capital-gain i.e. most of the values are centered on 0(nearly 43000 of them)
# 

# # Relation btw in capital-gain and capital-loss

# In[ ]:


sns.relplot('capital-gain','capital-loss',data= df)
plt.xlabel('capital-gain')
plt.ylabel('capital-loss')
plt.show()


# 1.both capital-gain and capital-loss can be zero(0)
# 2.if capital-gain is Zero then capital-loss being high or above zero.
# 3.if capital-loss is Zero then capital-gain being high or above zero.

# In[ ]:


df.head(1)


# # Hours-per-week Distribution:

# In[ ]:


df['hours-per-week'].hist(figsize=(5,5))
plt.show()


# In this data the hours per week atrribute varies within the range of 1 to 99.
# By observayion,30-40 hrs people work per week,around 27000 people.
# There are also few people who works 80-100 hours per week and some less than 20 which is unusual.

# # Workclass Distribution:

# In[ ]:


plt.figure(figsize=(12,5))

total = float(len(df['income']))

a = sns.countplot(x='workclass',data=df)

for f in a.patches:
    height = f.get_height()
    a.text(f.get_x() + f.get_width()/2., height+3, '{:1.2f}'.format((height/total)*100),ha="center")
plt.show()


# most of them belong to private workclass that is around 75%.
# without-play and never-play workclass has min count

# # Education Distribution:

# In[ ]:


plt.figure(figsize=(20,5))

a= float(len(['income']))

a= sns.countplot(x='education',data=df)
for s in a.patches:
    height = s.get_height()
    a.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
plt.show()
    


# Hs-grad has 32.32% of all the education attribute.
# pre-school has min.

# # marital-status Distribution:

# In[ ]:


plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sns.countplot(x="marital-status", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# Married-civ-spouse has maximum number of samples.
# Married-AF-spouse has minimum number of obs.
# 

# # Occupation Distribution:

# In[ ]:


plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sns.countplot(x="occupation", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# Prof-specialty has the maximum count.
# Armed-Forces has minimum samples in the occupation attribute.

# # Relationship Distribution:
#     

# In[ ]:


plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sns.countplot(x="relationship", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# Husband has maximum percentage among all.

# # Race Distribution:

# In[ ]:


plt.figure(figsize=(15,8))
total = float(len(df) )

ax = sns.countplot(x="race", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# white is maximun among all about 85.50%.
# black is second maximun.

# In[ ]:


# plt.figure(figsize=(5,5))
total = float(len(df) )

ax = sns.countplot(x="gender", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# there are 2 unique categories in gender.
# frequency of male is higher than female.
# 

# # Income(TArget variable) Distribution:

# In[ ]:


plt.figure(figsize=(5,5))
total = float(len(df) )

ax = sns.countplot(x="income", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In income there is 2 group,group1(who earns more than 50k) 23.93% belong to income and group2(who earns less than 50k) 76% belong to income

# # 2.Bivariate analysis
# 

# # Boxplot(Age relationship with income):

# In[ ]:


#Box plots
#--------------------------------------------------------------------------------
fig = plt.figure(figsize=(5,5))
sns.boxplot(x='income',y='age',data=df).set_title('Box plot of INCOME and AGE')
#blue_patch = mpatches.patch(color='blue',label='class_1')
#orange_patch = mpatches.patch(color='orange',label='class_2')
#plt.legend(handels=[blue_patch,orange_patch],loc=1)
plt.show


# Income group(<=50k) has lower median "age"(34 year) than the Income group(>50k) which has median "age"(42 year).

# # Boxplot(workclass relationship with income):
# 

# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.countplot(x='workclass',hue ='income',data=df).set_title("workclass vs count")


# The data seems to mainly consist private employees.
# 
# In All the workclasses number of people earning less then 50k are more then those earning 50k.

# # Boxplot (capital-gain Relationship with income):
# 

# In[ ]:


plt.figure(figsize=(5,5))
sns.boxplot(x="income", y="capital-gain", data=df)
plt.show()


# 
# Most of the capital gains value is accumulated at 0 for both the income group .

# # Boxplot (capital-loss Relationship with income):

# In[ ]:


plt.figure(figsize=(5,5))
sns.boxplot(x="income", y="capital-loss", data=df)
plt.show()


# This boxplot is similar to the capital gain boxplot where most of the values are concentrated on 0.

# # Boxplot (relationship Relationship with income):

# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(x="relationship", hue="income",
            data=df);


# Mostly a person with relation as husband in a family has most count of people with more then 50k income

# # Boxplot (race Relationship with income):
# 

# In[ ]:


plt.figure(figsize=(20,5))
sns.catplot(y="race", hue="income", kind="count",col="gender", data=df);


# It is clear people with Gender male and race as white has the most people with income more then 50k.

# In[ ]:





# In[ ]:


plt.figure(figsize=(12,5))
total = float(len(df["income"]) )

ax = sns.countplot(x="workclass", hue="income", data=df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format((height/total)*100),
            ha="center") 
plt.show()


# In[ ]:


#violin plot
#---------------------------------------------------------------------------------
sns.violinplot(x="income", y="age", data=df, size=8)
plt.title('Violin plt of AGE and Survival status')
blue_patch = mpatches.Patch(color='blue', label='class_1')
orange_patch = mpatches.Patch(color='orange', label='class_2')
plt.legend(handles=[blue_patch,orange_patch],loc=1)
plt.show()


# by observation of violin plot. This plot gives the combined information of PDF and box plot. The curve denotes the PDF and middle area denotes box plot.

# In[ ]:


sns.catplot(y="education", hue="income", kind="count",
            palette="pastel", edgecolor=".6",
            data=df);


# This data mostly consist of people who has education as hs-grad

# In[ ]:


ct = sns.catplot(y='marital-status',hue='gender',col='income',data=df,kind='count',
                height=4,aspect=.7)


# The people with marital status as Married-civ-spouce has highest people with income more then 50k.

# In[ ]:


sns.countplot(y="occupation", hue="income",
            data=df)


# In[ ]:


plt.figure(figsize=(10,7))
sns.countplot(x="relationship", hue="income",
            data=df);


# In[ ]:


plt.figure(figsize=(20,7))
sns.catplot(y="race", hue="income", kind="count",col="gender", data=df);


# In[ ]:


sns.heatmap(df.corr())

