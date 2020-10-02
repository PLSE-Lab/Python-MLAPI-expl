#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Let me know if you need anymore details or if you have suggestions
#Linkedin : @justsuyash


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data  = pd.read_csv('../input/fish-market/Fish.csv')


# # ****Trying to see an overview of Data that we have****

# In[ ]:


df = data.copy() #Copying it just in case
df.head(10)


# ## Data Columns
# * Species - species name of fish
# * Weight - weight of fish in Gram g
# * Length1 - vertical length in cm
# * Length2 - diagonal length in cm
# * Length3 - cross length in cm
# * Height - height in cm
# * Widthdiagonal - width in cm
# 
# We have been given the task to predict weight.

# ### Lets what are the species, maybe it will help us to include that in out features later on

# In[ ]:


df['Species'].unique()


# ### Lets Check if there are any Null Values in the data set:

# In[ ]:


df.info()


# In[ ]:


#Lets Check if there are any Null Values in the data set:
df.isnull().values.any()


# ### Hmm, so no null values in dataset,but how about wrong values?

# In[ ]:


df.describe()


# **We can see right away that minimum weight is 0 which is not possible also since the median weight is 398 and std is 357, the maximum weight 1650 seems like and outlier, this could throw off our predictions in the Linear Regression Model. But lets confrim out assumption**

# In[ ]:


first_quarlitle_weight = df['Weight'].quantile(0.25)
third_quarlitle_weight = df['Weight'].quantile(0.75)

inter_quartile_weight = third_quarlitle_weight - first_quarlitle_weight

lower_range_weight = first_quarlitle_weight - 1.5*inter_quartile_weight
upper_range_weight = third_quarlitle_weight + 1.5*inter_quartile_weight


# In[ ]:


#This way we get a resonable estimate of an outlier
df[ (df['Weight'] < lower_range_weight) | (df['Weight']>upper_range_weight)]


# **Since we have very less data to work on every set counts, Lets make is even more solid before we drop these,
# Lets estimate the outliers for Length**

# In[ ]:


first_quarlitle_length1 = df['Length1'].quantile(0.25)
third_quarlitle_length1 = df['Length1'].quantile(0.75)

inter_quartile_length1 = third_quarlitle_length1 - first_quarlitle_length1

lower_range_length1 = first_quarlitle_length1 - 1.5*inter_quartile_length1
upper_range_length1 = third_quarlitle_length1 + 1.5*inter_quartile_length1


# In[ ]:


df[ (df['Length1'] < lower_range_length1) | (df['Length1']>upper_range_length1)]


# **So thats two features telling us that these are outliers, we should from them to keep them throwing us off of out predictions**

# In[ ]:


excess_weight  = df[ (df['Weight'] < lower_range_weight) | (df['Weight']>upper_range_weight)]
df.drop(excess_weight.index,inplace=True)


# **So now our outliers are dropped, there is just one problem, weight cannot be 0 as shown in the minimum below, so we need to drop that too:** 

# In[ ]:


df.describe()


# In[ ]:


zero_weights  = df [ data['Weight'] == 0]
df.drop(zero_weights.index,inplace=True)


# In[ ]:


df.describe()


# **Now that we have removed outliers and errorenous values, lets check out what are features important for our prediction**

# In[ ]:


sns.heatmap(df.corr(),annot=True, cmap='YlGnBu')


# **Now we have come across a probelm that we could have had a hunch about,the variables realted to the dimensions of the fish are all correlated
# But is that really a problem? 
# I mean we have 3 features that are correlated to each other and also the target
# This makes the probablity of a the most useful feature being selected the highest if we give it all three, moreover we are not concerned
# with speed as the data set is quite small. 
# **
# 
# ### Verdict : We keep all three
# 
# 

# In[ ]:


#Lets do a pairplot to see if we can find something.
sns.pairplot(df, hue='Species')


# ### Looks like there is a realtion between species of fish and the weights, makes sense, lets check further how much of that is true

# In[ ]:


df['Species'] = df['Species'].astype('category')
df['species_cat'] = df['Species'].cat.codes


# In[ ]:


c = df['Species'].astype('category')

d = dict(enumerate(c.cat.categories))
print (d)


# In[ ]:


sns.pairplot(df,hue='species_cat')


# **Thats a huge correlation, now that we have discovered this, there are two approach towards this one is "One Hot Encoding"
# The other is as you can see above Categorical Numbering,
# I will prefer to create dummy variable as it assigns the value '1' to each category and hence it works better in algorithms as it doesnt add any extra weight to any particular category**
# 
# 
# 
# **So lets drop 'species_cat'**

# In[ ]:


data_with_dummies = df.drop(['species_cat'],axis=1)


# ### Crearting Data frame with dummies

# In[ ]:


data_with_dummies = pd.get_dummies(df, drop_first=True)


# In[ ]:


data_with_dummies


# ## Now lets train out model using Linear Regression

# In[ ]:


data_with_dummies.columns


# In[ ]:


x = data_with_dummies.drop(['Weight'],axis=1)
y = data_with_dummies['Weight']


# ### Classic Train Test Split

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)


# ### Linear Regression

# In[ ]:


lin = LinearRegression()


# In[ ]:


lin.fit(x_train,y_train)


# ### Determining Succes with Train DataSet:

# In[ ]:


y_train_predicted = lin.predict(x_train)
r2_score(y_train, y_train_predicted)


# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(lin, x_train, y_train, cv=10, scoring='r2')
print(cross_val_score_train)


# In[ ]:


cross_val_score_train.mean()


# In[ ]:


predict = lin.predict(x_test)


# In[ ]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))


# In[ ]:


print(r2_score(y_test, predict))


# ## Not at all bad

# In[ ]:


plt.scatter(y_test,predict)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ### I will repeat Not at all bad

# # EXTRAS :

# ## Now lets think about what can be done to improve the model?

# ### For Starters we can look at the pairplot again

# In[ ]:


sns.pairplot(df,hue='species_cat')


# In[ ]:


c = df['Species'].astype('category')

d = dict(enumerate(c.cat.categories))
print (d)


# **It is safe to assume that whats throwing us off is lesser data, as fish 5:'Smelt' is centered around a certain weight, 
# also there are only 6 white fishes with a wide distribution. But thats just my 2 cent, Let me know if you have any suggestions. Lets Learn together**

# In[ ]:




