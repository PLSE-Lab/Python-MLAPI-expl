#!/usr/bin/env python
# coding: utf-8

# **Importing Modules**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt #data visualization
import seaborn as sns #data visualization


# **Reading Dataset**

# In[ ]:


df = pd.read_csv("../input/MarathonData.csv")
df.head()


# From the explanation about the variable in dataset, i conclude that name, CATEGORY, id, and Marathon columns doesn't important for analyzing and predicting the dataset. Therefore i will drop those columns

# In[ ]:


df = df.drop(columns=['id','Marathon','CATEGORY'])
df.head()


# In[ ]:


df = df.drop(columns=['Name'])
df.head()


# **Handling Missing Values**

# Checking missing values in every columns

# In[ ]:


df.isna().sum()


# In[ ]:


df['CrossTraining'].unique()


# From the result above, nan values in CrossTraining column means that the runner doesn't have another Cross Training activity, therefore i will fill the missing values with 'nonct' that stands for non Cross Training

# In[ ]:


df['CrossTraining'].fillna('nonct',inplace=True)
df.isna().sum()


# i want to check whether Category column has an effect to marathon time or not, so i make another data frame that not containing missing values so it can be visualized

# In[ ]:


dfn = df.dropna(how='any')
dfn.isna().sum()


# In[ ]:


f,axes = plt.subplots(figsize=(15,5))
sns.swarmplot(x = 'Category', y='MarathonTime',data=df, ax=axes)
plt.title('Time distribution on different Category')
plt.xlabel('Category')
plt.ylabel('Marathon Time')
plt.show()


# From the swarmplot above, i conclude that Category column doesn't important for predicting Marathon Time. Therefore i will drop Category Column and drop the missing values

# In[ ]:


df = df.dropna(how='any')
df = df.drop(columns=['Category'])
df.head()


# In[ ]:


df.isna().sum()


# **Data Visualization**

# In[ ]:


df.info()


# Wall21 column type need to be changed so it can be visualized

# In[ ]:


df['Wall21'] = df['Wall21'].astype(float)
df.info()


# In[ ]:


plt.scatter(x = df['km4week'], y=df['MarathonTime'])
plt.title('km4week Vs Marathon Time')
plt.xlabel('km4week')
plt.ylabel('Marathon Time')
plt.show()


# In[ ]:


plt.scatter(x = df['sp4week'], y=df['MarathonTime'])
plt.title('sp4week Vs Marathon Time')
plt.xlabel('sp4week')
plt.ylabel('Marathon Time')
plt.show()


# turns out there is one outliers in sp4week data that need to be remove

# In[ ]:


df = df.query('sp4week<2000')


# In[ ]:


plt.scatter(x = df['sp4week'], y=df['MarathonTime'])
plt.title('sp4week Vs Marathon Time')
plt.xlabel('sp4week')
plt.ylabel('Marathon Time')
plt.show()


# In[ ]:


plt.scatter(x = df['Wall21'], y=df['MarathonTime'])
plt.title('Wall21 Vs Marathon Time')
plt.xlabel('Wall21')
plt.ylabel('Marathon Time')
plt.show()


# In[ ]:


f,axes = plt.subplots(figsize=(15,5))
sns.boxplot(x = df['CrossTraining'], y=df['MarathonTime'],ax=axes)
plt.title('CrossTraining Vs Marathon Time')
plt.xlabel('CrossTraining')
plt.ylabel('Marathon Time')
plt.show()


# from the graph above, i conclude that Cross Training doesn't have big impact for predicting MarathonTime

# In[ ]:


df = pd.get_dummies(df)


# **Checking Correlation**

# In[ ]:


correlated = df.corr().abs()['MarathonTime'].sort_values(ascending=False)
correlated


# In[ ]:


correlated = correlated[:5]
correlated


# make a new dataframe that only contain variable that have high correlation with Marathon Time

# In[ ]:


df = df.loc[:,correlated.index]
df.head()


# **Predicting**

# In this section, i will built a model to predict Marathon Time with selected features. I will use Linear Regression and Polynomial Regression to predict the dataset. i also will show the R squared, adjusted R squared , Root Mean Squared and Cross Validation Score for each method to compare.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score


# Split the data into train and test for helping the prediction

# In[ ]:


X = df.drop(columns=['MarathonTime'])
y = df['MarathonTime']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)


# Linear Regression

# In[ ]:


clf = LinearRegression()
clf.fit(X_train,y_train)
r2 = clf.score(X_train,y_train)
def adjustedR2(r2,n,k):
    return r2-(k-1)/(n-k)*(1-r2)
a = adjustedR2(r2, X_train.shape[0],4)
a


# 

# In[ ]:


prediction = clf.predict(X_test)
rmse = np.sqrt(np.mean((prediction-y_test)**2))
rmse


# In[ ]:


cv1 = cross_val_score(clf,X_train,y_train,cv=5).mean()
cv1


# Polynomial Regression

# In[ ]:


poly = PolynomialFeatures()
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)
clf.fit(X_train2,y_train)
PolyR2 = clf.score(X_train2,y_train)
PolyR2


# In[ ]:


b = adjustedR2(r2, X_train2.shape[0],4)
b


# In[ ]:


prediction2 = clf.predict(X_test2)
rmse2 = np.sqrt(np.mean((prediction2-y_test)**2))
rmse2


# In[ ]:


cv2 = cross_val_score(clf,X_train2,y_train,cv=5).mean()
cv2


# In[ ]:


results = pd.DataFrame({'Model': [],
                        'Root Mean Squared Error (RMSE)':[],
                        'R-squared (test)':[],
                        'Adjusted R-squared (test)':[],
                        '5-Fold Cross Validation':[]})
r = results.shape[0]
results.loc[r] = ['Linear Regression', rmse, r2, a, cv1]
results.loc[r+1] = ['Polynomial Regression', rmse2, PolyR2, b, cv2]


# In[ ]:


results

