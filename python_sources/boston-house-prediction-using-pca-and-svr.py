#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Importing the dataset
dataset = pd.read_csv('/kaggle/input/Boston.csv')
dataset.shape


# There are 398 observations and 9 attributes

# In[ ]:


dataset.dtypes


# In[ ]:


dataset.head()


# In[ ]:


dataset.isnull().sum()


# here, in our dataset no NA values are present

# # Univariate analysis for CRIM column:

# In[ ]:


dataset['crim'].describe()


# In[ ]:


sns.distplot(dataset['crim'])


# In[ ]:


print("Skewness: %f" % dataset['crim'].skew())
print("Kurtosis: %f" % dataset['crim'].kurt())


# In[ ]:


sns.boxplot(dataset['crim'])


# In[ ]:


dataset[dataset['crim']<dataset['crim'].quantile(0.02)]['crim']


# In[ ]:


dataset.loc[dataset['crim']<dataset['crim'].quantile(0.02),['crim']]=dataset['crim'].quantile(0.02)


# In[ ]:


dataset[dataset['crim']>dataset['crim'].quantile(0.95)]['crim']


# In[ ]:


dataset.loc[dataset['crim']>dataset['crim'].quantile(0.95),['crim']]=dataset['crim'].quantile(0.95)


# In[ ]:


sns.boxplot(dataset['crim'])


# In[ ]:


sns.distplot(dataset['crim'])


# # Univariate analysis for zn column

# In[ ]:


dataset['zn'].describe()


# In[ ]:


sns.boxplot(dataset['zn'])


# In[ ]:


dataset.loc[dataset['zn']<dataset['zn'].quantile(0.1),['zn']]=dataset['zn'].quantile(0.1)
dataset.loc[dataset['zn']>dataset['zn'].quantile(0.9),['zn']]=dataset['zn'].quantile(0.9)
sns.boxplot(dataset['zn'])


# In[ ]:


sns.distplot(dataset['zn'])


# In[ ]:


print("Skewness: %f" % dataset['zn'].skew())
print("Kurtosis: %f" % dataset['zn'].kurt())


# # univariate analysis of INDUS variable

# In[ ]:


dataset['indus'].describe()


# In[ ]:


sns.distplot(dataset['indus'])


# In[ ]:


sns.boxplot(dataset['indus'])


# In[ ]:


print("Skewness %f" % dataset['indus'].skew())
print("Kurtosis %f" % dataset['indus'].kurt())


# # Univariate analysis for CHAS column

# In[ ]:


dataset['chas'].describe()


# In[ ]:


dataset['chas'].value_counts()


# In[ ]:


sns.countplot(dataset['chas'])


# # univarte anlysis of RM variable

# In[ ]:


dataset['rm'].describe()


# In[ ]:


sns.boxplot(dataset['rm'])


# In[ ]:


sns.distplot(dataset['rm'])


# # univariate  analysis of LSTAT variable

# In[ ]:


dataset['lstat'].describe()


# In[ ]:


sns.boxplot(dataset['lstat'])


# In[ ]:


sns.distplot(dataset['lstat'])


# # univariate analysis of PTRATIO 

# In[ ]:


dataset['ptratio'].describe()


# In[ ]:


sns.boxplot(dataset['ptratio'])


# In[ ]:


dataset[dataset['ptratio']<dataset['ptratio'].quantile(0.02)]['ptratio']


# In[ ]:


dataset.loc[dataset['ptratio']<dataset['ptratio'].quantile(0.02),['ptratio']]=dataset['ptratio'].quantile(0.02)


# In[ ]:


dataset[dataset['ptratio']>dataset['ptratio'].quantile(0.98)]['ptratio']


# In[ ]:


dataset.loc[dataset['ptratio']>dataset['ptratio'].quantile(0.98),['ptratio']]=dataset['ptratio'].quantile(0.98)


# In[ ]:


sns.boxplot(dataset['ptratio'])


# In[ ]:


sns.distplot(dataset['ptratio'])


# # univariate analysis of MEDV variable

# In[ ]:


dataset['medv'].describe()


# In[ ]:


sns.boxplot(dataset['medv'])


# In[ ]:


sns.distplot(dataset['medv'])


# In[ ]:


feature_cols = ['crim', 'zn', 'indus','nox', 'rm','age', 'dis', 'rad', 'tax', 'ptratio','black', 'lstat']
target_col='medv'


# In[ ]:


X = dataset[feature_cols].values
y = dataset[target_col].values
print("X dimensions are",X.shape)
print("y dimensions are",y.shape)


# # Split the data into train and test sets

# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("X_train dimensions are",X_train.shape)
print("y_train dimensions are",y_train.shape)
print("X_test dimensions are",X_test.shape)
print("y_test dimensions are",y_test.shape)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# In[ ]:


print("dimensin of X_train",X_train.shape)


# In[ ]:


round(pd.Series(explained_variance),2)


# In[ ]:


#pca.components_.shape[0]
pd.DataFrame(pca.components_)


# In[ ]:



sns.heatmap(pd.DataFrame(pca.components_).corr(),annot=True)


# In[ ]:


# number of components
n_pcs= pca.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

most_important_names = [feature_cols[most_important[i]] for i in range(n_pcs)]

dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.Series(dic.items())
dic.items()


# In[ ]:


# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
fitted=regressor.fit(X_train, y_train)
y_predict_train=fitted.predict(X_train)
y_predict_test=fitted.predict(X_test)


# In[ ]:


fitted.score(X_train,y_train)


# In[ ]:



from sklearn.metrics import r2_score
print('R square for train:', r2_score(y_train,y_predict_train))
print('R square for test:', r2_score(y_test,y_predict_test))


# In[ ]:




