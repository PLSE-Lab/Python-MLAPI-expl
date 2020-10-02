#!/usr/bin/env python
# coding: utf-8

# ## Import data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler

columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']

train = pd.read_csv('../input/adult-training.csv', names=columns)
test = pd.read_csv('../input/adult-test.csv', names=columns, skiprows=1)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train.head()


# More information about dataset (including what fnlgwt is): [archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)

# In[ ]:


train.info()


# In[ ]:


test.info()


# ## Cleaning data
# Some cells contain ' ?', we convert them to NaN

# In[ ]:


train.replace(' ?', np.nan, inplace=True)
test.replace(' ?', np.nan, inplace=True)


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# As we see only Workclass, Occupation and Native country features have missing values.

# # Features engineering

# ## Income

# Simply change Income into 0's and 1's

# In[ ]:


train['Income'] = train['Income'].apply(lambda x: 1 if x==' >50K' else 0)
test['Income'] = test['Income'].apply(lambda x: 1 if x==' >50K.' else 0)


# ## Age

# In[ ]:


plt.hist(train['Age']);


# Age looks skewed, it needs to be normalized. It'll be done later with sklearn.preprocessing.StandardScaller().

# ## Workclass

# There are many empty rows, let's replace them with 0 and check how data plot looks like.

# In[ ]:


train['Workclass'].fillna(' 0', inplace=True)
test['Workclass'].fillna(' 0', inplace=True)


# In[ ]:


sns.factorplot(x="Workclass", y="Income", data=train, kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=45);


# In[ ]:


train['Workclass'].value_counts()


# As Never-worked and Without-pay look very similar, we merge them.

# In[ ]:


train['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)
test['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)


# ## fnlgwt

# In[ ]:


train['fnlgwt'].describe()


# Fnlgwt feature has high numers and big sandard deviation, let's take logarithm of that.

# In[ ]:


train['fnlgwt'] = train['fnlgwt'].apply(lambda x: np.log1p(x))
test['fnlgwt'] = test['fnlgwt'].apply(lambda x: np.log1p(x))


# In[ ]:


train['fnlgwt'].describe()


# ## Education

# In[ ]:


sns.factorplot(x="Education",y="Income",data=train,kind="bar", size = 7, 
palette = "muted")
plt.xticks(rotation=60);


# Primary education is devided into grades, they all give almost the same result. We can merge them into one feature - Primary.

# In[ ]:


def primary(x):
    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:
        return ' Primary'
    else:
        return x


# In[ ]:


train['Education'] = train['Education'].apply(primary)
test['Education'] = test['Education'].apply(primary)


# In[ ]:


sns.factorplot(x="Education",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);


# ## Education num

# In[ ]:


sns.factorplot(x="Education num",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);


# ## Marital Status

# In[ ]:


sns.factorplot(x="Marital Status",y="Income",data=train,kind="bar", size = 5, 
palette = "muted")
plt.xticks(rotation=60);


# In[ ]:


train['Marital Status'].value_counts()


# There are very few Married-AF-spouse features. They are similar to Married-civ-spouse, so we can merge them.

# In[ ]:


train['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)
test['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)


# In[ ]:


sns.factorplot(x="Marital Status",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);


# ## Occupation

# In[ ]:


train['Occupation'].fillna(' 0', inplace=True)
test['Occupation'].fillna(' 0', inplace=True)


# In[ ]:


sns.factorplot(x="Occupation",y="Income",data=train,kind="bar", size = 8, 
palette = "muted")
plt.xticks(rotation=60);


# In[ ]:


train['Occupation'].value_counts()


# Everything looks good, except Armed-Forces. They are similar to 0 and that's what we replace them with.

# In[ ]:


train['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)
test['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)


# In[ ]:


sns.factorplot(x="Occupation",y="Income",data=train,kind="bar", size = 8, 
palette = "muted")
plt.xticks(rotation=60);


# ## Relationship

# In[ ]:


sns.factorplot(x="Relationship",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=60);


# In[ ]:


train['Relationship'].value_counts()


# Looks good.

# ## Race

# In[ ]:


sns.factorplot(x="Race",y="Income",data=train,kind="bar", size = 6, 
palette = "muted")
plt.xticks(rotation=45);


# In[ ]:


train['Race'].value_counts()


# Nothing to change.

# ## Sex

# In[ ]:


sns.factorplot(x="Sex",y="Income",data=train,kind="bar", size = 4, 
palette = "muted");


# Here neither.

# ## Capital Gain , Capital Loss,  Hours/Week

# These features just need to be standarized.

# ## Native country 

# In[ ]:


train['Native country'].fillna(' 0', inplace=True)
test['Native country'].fillna(' 0', inplace=True)


# In[ ]:


sns.factorplot(x="Native country",y="Income",data=train,kind="bar", size = 10, 
palette = "muted")
plt.xticks(rotation=80);


# We need to segregate these countries into a few categories.

# In[ ]:


def native(country):
    if country in [' United-States', ' Cuba', ' 0']:
        return 'US'
    elif country in [' England', ' Germany', ' Canada', ' Italy', ' France', ' Greece', ' Philippines']:
        return 'Western'
    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', ' Columbia', ' Laos', ' Portugal', ' Haiti',
                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 
                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Vietnam', ' Holand-Netherlands' ]:
        return 'Poor' # no offence
    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', ' Japan', ' Yugoslavia', ' China', ' Hong']:
        return 'Eastern'
    elif country in [' South', ' Poland', ' Ireland', ' Hungary', ' Scotland', ' Thailand', ' Ecuador']:
        return 'Poland team'
    
    else: 
        return country    


# In[ ]:


train['Native country'] = train['Native country'].apply(native)
test['Native country'] = test['Native country'].apply(native)


# In[ ]:


train['Native country'].value_counts()


# In[ ]:


sns.factorplot(x="Native country",y="Income",data=train,kind="bar", size = 5, 
palette = "muted")
plt.xticks(rotation=60);


# # One-hot encoding

# Now we need to encode categorical features, we are going to do it with pd.get_dummies(). As this method may cause some problems, we merge datasets. It ensures that dimensions for both datasets are equal and also that given feature corresponds to the same dimension in both train and test datasets.

# In[ ]:


#merge datasets
joint = pd.concat([train, test], axis=0)


# We need to analize features, find categorical ones and one-hot encode them.

# In[ ]:


joint.dtypes


# In[ ]:


#list of columns with dtype: object
categorical_features = joint.select_dtypes(include=['object']).axes[1]

for col in categorical_features:
    print (col, joint[col].nunique())


# In[ ]:


#one-hot encode
for col in categorical_features:
    joint = pd.concat([joint, pd.get_dummies(joint[col], prefix=col, prefix_sep=':')], axis=1)
    joint.drop(col, axis=1, inplace=True)


# In[ ]:


joint.head()


# We separate train and test datasets.

# In[ ]:


train = joint.head(train.shape[0])
test = joint.tail(test.shape[0])


# We devide data frame into features and targets. Then standarize features.

# In[1]:


Xtrain = train.drop('Income', axis=1)
Ttrain = train['Income']

Xtest = test.drop('Income', axis=1)
Ttest = test['Income']

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)
Xtrain.head()
Xtest.head()


# ## Logistic regression data prediction

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[ ]:


model = LogisticRegression()
model.fit(Xtrain, Ttrain)

Ytrain = model.predict(Xtrain)
Ytest = model.predict(Xtest)


# In[ ]:


print(classification_report(Ttrain, Ytrain))


# In[ ]:


print(classification_report(Ttest, Ytest))

