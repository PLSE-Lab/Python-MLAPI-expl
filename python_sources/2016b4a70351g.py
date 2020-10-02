#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.head()


# In[ ]:


num_rows = df.shape[0]
num_cols = df.shape[1]
(num_rows, num_cols)


# In[ ]:


df.info()


# In[ ]:


df.describe()         
# numerical variables


# In[ ]:


df.describe(include='object')
# categorical variables


# In[ ]:


df['rating'].value_counts()


# In[ ]:


df.isnull().any()


# In[ ]:


missing_values = df.isnull().sum()
missing_values[missing_values > 0]


# In[ ]:


# replacing missing values using mean of numerical features
df.fillna(value = df.mean(), inplace=True)


# In[ ]:


# no missing value present anymore
df.isnull().any().any()


# In[ ]:


sns.boxplot(x='rating', y='feature3', data = df)
# a lot of overlap => might not be a good feature


# In[ ]:


sns.boxplot(x='rating', y='feature5', data = df)


# In[ ]:


sns.boxplot(x='rating', y='feature6', data = df)


# In[ ]:


sns.boxplot(x='rating', y='feature7', data = df)


# In[ ]:


# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[ ]:


sns.boxplot(x='rating', y='feature8', data = df)


# In[ ]:


sns.boxplot(x='rating', y='feature11', data = df)


# In[ ]:


sns.distplot(df['feature6'],kde = False)


# In[ ]:


#df['feature6'] = np.log(df['feature6'])
#sns.distplot(df['feature6'],kde = False)


# In[ ]:


sns.boxplot(x='type', y='rating', data = df)


# In[ ]:


df.corr()


# In[ ]:


cols = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9',
        'feature10','feature11','type']
X = df[cols]
y = df['rating']


# In[ ]:


X = pd.get_dummies(data=X, columns=['type'])


# FIRST SUBMISSION

# In[ ]:


real_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


real_test.isnull().sum()


# In[ ]:


real_test.fillna(real_test.mean(), inplace=True)


# In[ ]:


X2 = real_test[cols]


# In[ ]:


X2 = pd.get_dummies(data=X2, columns=['type'])


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=4000,random_state=42, bootstrap=True).fit(X,y)


# In[ ]:


y_pred1 = clf.predict(X2)
y_pred1 = np.rint(y_pred1)


# In[ ]:


final = pd.DataFrame(real_test['id'])
final['rating'] = y_pred1


# In[ ]:


final.head()


# In[ ]:


final.to_csv('pred4.csv', index=False)


# SECOND SUBMISSON

# In[ ]:


real_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')


# In[ ]:


real_test.isnull().sum()


# In[ ]:


real_test.fillna(real_test.mean(), inplace=True)


# In[ ]:


X2 = real_test[cols]


# In[ ]:


X2 = pd.get_dummies(data=X2, columns=['type'])


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=2000,random_state=4, bootstrap=True).fit(X,y)


# In[ ]:


y_pred1 = clf.predict(X2)
y_pred1 = np.rint(y_pred1)


# In[ ]:


final = pd.DataFrame(real_test['id'])
final['rating'] = y_pred1


# In[ ]:


final.head()


# In[ ]:


final.to_csv('pred4.csv', index=False)

