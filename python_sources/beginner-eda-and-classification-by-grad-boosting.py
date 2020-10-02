#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing Packages

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
sns.set(style = 'darkgrid')
warnings.filterwarnings('ignore')


# ### Loading Censous Data

# In[ ]:


df = pd.read_csv('/kaggle/input/adult-census-income/adult.csv')


# # EDA

# In[ ]:


df.head()


# ##### Important: There are Missing vlaues in 'Workclaas' and 'Occupation" columns identified by '?'

# In[ ]:


df.info()


# In[ ]:


df.describe(include='all')


# ### About the data
#     1. There are 48842 instances and 15 features.
#     2. 9 out of 15 of our features are categorical.

# ## Univariate Analysis
#     Let's first take a look at Categorical features

# In[ ]:


workclass = df['workclass'].value_counts()
sns.countplot(df['workclass'],palette='icefire_r')
plt.xticks(rotation = 60)
print(workclass)


# 1. Almost 70% people work in private firms
# 2. No. of missing values is around 5%
# 3. There are very few people not working or not getting paid

# In[ ]:


print(df['education'].value_counts())
plt.xticks(rotation = 60)
sns.countplot(df['education'])
plt.show()


# 1. Most of the people are either High Scholl gradute or went to some college or have Bachelor degree

# In[ ]:


print(df['marital.status'].value_counts())
sns.countplot(df['marital.status'])
plt.xticks(rotation = 60)
plt.show()


# 1. Most people are either married(46%) or never married(33%) or divorced(13%).

# In[ ]:


print(df['occupation'].value_counts())
sns.countplot(df['occupation'])
plt.xticks(rotation = 90)
plt.show()


# 1. There are many occupations that people have and for every occupation there are many people in that occupation
# 2. There are also 2809 missing values
# 3. there are only 15 people working in armed forces. 

# In[ ]:


print(df['relationship'].value_counts())
sns.countplot(df['relationship'])
plt.show()


# 1. Almost 3/4 people live with their family

# In[ ]:


print(df['race'].value_counts())
sns.countplot(df['race'])
plt.show()


# 1. Attribute 'race' has almost all instances as 'white'.So, this feature will be of no use 
#    in predicting income hence we will drop it.

# In[ ]:


print(df['sex'].value_counts())
sns.countplot(df['sex'])
plt.show()


# 1. There are almost 67% men and 33% women in the data.

# In[ ]:


print(df['income'].value_counts())
sns.countplot(df['income'])
plt.show()


# 1. More than 75% of the people have income less than 50K
# 2. The target variable is imbalance

# In[ ]:


print(df['native.country'].value_counts())
sns.countplot(df['native.country'])
plt.xticks(rotation = 90)
plt.show()


# 1. The data is very skewed since almost 90% of the people have native-country as United States
# 2. It will be better to drop this feature too.

# ##### Filling the missing values of occupation and workclass fature by most frequent category

# In[ ]:


attrib, counts = np.unique(df['workclass'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
df['workclass'][df['workclass'] == '?'] = most_freq_attrib 

attrib, counts = np.unique(df['occupation'], return_counts = True)
most_freq_attrib = attrib[np.argmax(counts, axis = 0)]
df['occupation'][df['occupation'] == '?'] = most_freq_attrib 


# **Now let's take a look at numerical features**

# In[ ]:


sns.distplot(df['hours.per.week'])
plt.show()
sns.violinplot(df['hours.per.week'],)
plt.show()


# 1. Most people work 35 to 50 hours per week
# 2. Very less people work either 90+ or 0 hours per week

# In[ ]:


sns.distplot(df['age'])
plt.show()
sns.boxplot(df['age'])
plt.show()


# 1. The median age of people is around 37 years
# 2. There are also some people with age 80+ years

# **Converting the target variable into numeric datatype**

# In[ ]:


df['income'] = LabelEncoder().fit_transform(df['income'])


# In[ ]:


sns.distplot(df['income'],kde = False)
plt.show()


# In[ ]:


sns.distplot(df['fnlwgt'])
plt.xticks(rotation = 90)
plt.show()
sns.boxplot(df['fnlwgt'])
plt.xticks(rotation = 90)
plt.show()


# 1. I do not understand what is this attribute for
# 2. Median value for this attribute is around 170000
# 3. There are very large outlires in the attribute

# In[ ]:


sns.distplot(df['education.num'])
plt.show()
sns.boxplot(df['education.num'])
plt.show()


# **'Educational-Num' is also a categorical attribute**
# 1. I think it is somehow related with the 'Education' column
# 2. I will deal with this later while doing multivariate analysis

# In[ ]:


sns.distplot(df['capital.gain'],kde = False)
plt.show()
sns.boxplot(df['capital.gain'])
plt.show()


# 1. This attribute highly left sided skewed
# 2. There are also some large outliers in this attribute

# In[ ]:


sns.distplot(df['capital.loss'],kde = False)
plt.show()
sns.boxplot(df['capital.loss'])
plt.show()


# 1. This attribute also has same trend as capital loss.
# 2. It has many outliers also but not that large as of 'capital-gain'

# ### Multivariate Analysis
#   **Let's try to interpret the relation between target varibale and numeric independent variables**

# In[ ]:


sns.relplot('capital.gain','capital.loss',data = df,hue = 'income')
plt.show()


# In[ ]:


sns.catplot('income','capital.gain',data =df,kind = 'violin')
plt.show()


# In[ ]:


sns.catplot('income','capital.loss',data =df,kind = 'violin')
plt.show()


# 1. Those who have zero capital-loss have high chance having income >=50k
# 2. Those Who have less capital-gain have high chance of having income <50k
# 3. Perhaps we can drop these two attributes and use new feature 'capital-change' = 'capital-gain'-capital-loss'
#    by doing this it will be also somewhat easy dealing with the outliers of these attributes

# In[ ]:


sns.catplot(y = 'fnlwgt',x = 'income',data = df,kind = 'violin')
plt.show()


# 1. Looking at this plot,still i can't see any relation between 'fnlwgt' and income category,I will deal with this later

# In[ ]:


sns.catplot(y = 'hours.per.week',x = 'income',data = df,hue = 'income',kind = 'violin')
plt.show()


# 1. It can be seen that most people work for more than 35+ hours per week
# 2. People with income >=50k work more hours per week than people with income<50k
# 3. It can be seen that number of people working less than 35 hours per week is more in <50k income category.

# In[ ]:


sns.catplot(y = 'age',x= 'income',data = df,hue = 'income',kind = 'violin')
plt.show()


# 1. It can be seen that as age increases income increases as well.
# 2. Most people less than 30 years of age have income <50k.

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot = True,cmap = 'vlag',linewidths=0.25)
plt.show()


# In[ ]:


df.head(20)


# ###### Dropping unnecessery Ctegorical features

# In[ ]:


df.drop(['native.country'],axis = 1,inplace = True)
df.head()


# **Now with caegorical attributes**

# In[ ]:


sns.countplot('workclass',data = df,hue = 'income' )
plt.xticks(rotation = 60)
plt.show()
(df['workclass'].groupby(by = df['income']).value_counts())


# 1. People working in private companies have high probability of having income <50k
# 3. Generally every workclass have more no. of people in income <50k category

# In[ ]:


sns.countplot('education',data = df,hue = 'income')
plt.xticks(rotation = 60)
plt.show()
(df['education'].groupby(by = df['income']).value_counts())


# 1. People with higher eductional degree are more likely to fall into >50k income category.

# In[ ]:


sns.countplot('education.num',data = df,hue = 'income')
plt.show()
df['education.num'].groupby(by = df['income']).value_counts()


# 1. 'Educational-num' feature is just the Ordered LabelEncoded form of 'Education' feature 
# 2. I will drop 'education' column for the model training.

# In[ ]:


sns.countplot('marital.status',data = df,hue = 'income')
plt.xticks(rotation = 60)
plt.show()
df['marital.status'].groupby(by = df['income']).value_counts()


# In[ ]:


sns.countplot('occupation',data = df,hue = 'income')
plt.xticks(rotation = 60)
plt.show()
df['occupation'].groupby(by= df['income']).value_counts()


# In[ ]:


sns.countplot('relationship',data = df,hue = 'income')
plt.xticks(rotation = 60)
plt.show()
df['relationship'].groupby(by= df['income']).value_counts()


# In[ ]:


sns.countplot('sex',data = df,hue = 'income')
plt.xticks(rotation = 60)
plt.show()
df['sex'].groupby(by= df['income']).value_counts()


# In[ ]:


sns.countplot('race',data = df,hue = 'income')
plt.show()
df['race'].groupby(by = df['income']).value_counts()


# # Feature Preprocessing

# In[ ]:


df.drop('education',axis = 1,inplace = True) ##It's already Ordered encoded in the column 'eduction-num'


# **Encoding workclass,race,relationship,marital.status,occupation features by replacing labels with percentage of frequency they occured**

# In[ ]:


df.workclass = df.workclass.map((df.workclass.value_counts()/len(df.workclass)).to_dict())*100


# In[ ]:


df.rename(columns={'marital.status' : 'marital_status'},inplace = True)


# In[ ]:


df.marital_status = df.marital_status.map((df['marital_status'].value_counts()/len(df['marital_status'])).to_dict())*100
df.occupation = df.occupation.map((df.occupation.value_counts()/len(df.occupation)).to_dict())*100
df.relationship = df.relationship.map((df.relationship.value_counts()/len(df.relationship)).to_dict())*100
df.race = df.race.map((df.race.value_counts()/len(df.race)).to_dict())*100

df.head()


# In[ ]:


df = pd.get_dummies(df)


# In[ ]:


df.head()


# In[ ]:


df['capital-change'] = df['capital.gain'] - df['capital.loss']
df.drop(['capital.gain','capital.loss'],axis = 1,inplace = True)


# # Splitting ths data and training the Model

# In[ ]:


y = df.income
df.drop('income',axis = 1,inplace = bool(1))


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(df,y,random_state = 42)


# In[ ]:


xtrain.shape,xtest.shape,ytrain.shape,ytest.shape


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbclf = GradientBoostingClassifier(random_state=42,n_estimators=300,max_depth=5,learning_rate=0.01)


# In[ ]:


gbclf.fit(xtrain,ytrain)


# In[ ]:


train_predict = gbclf.predict(xtrain)


# In[ ]:


from sklearn.metrics import classification_report,accuracy_score,roc_auc_score
print(classification_report(ytrain,train_predict),accuracy_score(ytrain,train_predict),roc_auc_score(ytrain,train_predict))


# In[ ]:


test_predict = gbclf.predict(xtest)
print(classification_report(ytest,test_predict),accuracy_score(ytest,test_predict),roc_auc_score(ytest,test_predict))


# In[ ]:


(dict(sorted(zip(df.columns,gbclf.feature_importances_))))

