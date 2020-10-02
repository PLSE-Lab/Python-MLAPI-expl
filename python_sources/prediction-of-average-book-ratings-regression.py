#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df=pd.read_csv('/kaggle/input/goodreadsbooks/books.csv', error_bad_lines=False)


# In[ ]:


df.head()


# In[ ]:


# Arrange by rating (high to low):

df.sort_values(by='average_rating',ascending=False)


# In[ ]:


# Count null values (if any):

df.isnull().sum()


# #### Barplot between Language and Average Rating:

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(data=df,x='language_code',y='average_rating')
plt.xlabel('Languages')
plt.ylabel('Average Rating')
plt.show()


# #### Observation: The books written in Welsch language have the highest average ratings while the ones written in Catalan language have the lowest average ratings.

# #### Scatter of Average Rating w.r.t. No. of Pages

# In[ ]:


plt.figure(figsize=(15,10))
sns.scatterplot(data=df,x='# num_pages',y='average_rating')
plt.xlabel('No. of Pages')
plt.ylabel('Average Rating')
plt.show()


# #### Plot for Observing Distribution of Average Ratings

# In[ ]:


plt.figure(figsize=(15,6))
sns.distplot(df['average_rating'],bins=20, color='red')
plt.title('Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.show()


# #### Observation: The majority of average ratings is distributed between 3.25 to 4.75.

# #### Boxplot for the No. of Pages by Language:

# In[ ]:


plt.figure(figsize=(15,10))
sns.boxplot(data=df,y='# num_pages',x='language_code')
plt.xlabel('Language')
plt.ylabel('No. of Pages')
plt.show()


# #### Observation: English language books have the maximum number of outliers in terms of the number of pages.

# #### Correlation between various columns in the Data Frame:

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),square=True,vmax=0.1,annot=True)
plt.xlabel('Language')
plt.ylabel('No. of Pages')
plt.show()


# #### Observation: The highest correlation exists between 'text_reviews_count' and 'ratings_count', while the lowest correlation is between 'average_rating' and 'bookID' (which stands to reason)

# #### Distribution Plot for the Number of Pages:

# In[ ]:


plt.figure(figsize=(15,7))
sns.distplot(df['# num_pages'],bins=20,color='gold')
plt.title('Distribution of Number of Pages')
plt.xlabel('No. of Pages')
plt.show()


# #### Statistical Analysis of Data Frame:

# In[ ]:


df.describe()


# ## Exploratory Analysis on the Dataset

# #### Range

# In[ ]:


cols = list(df.select_dtypes(exclude=['object']).columns)
df1 = df.loc[:,cols]


# In[ ]:


print("The Range of the dataset is : \n\n",df1.max()-df1.min())


# #### Interquartile Range:

# In[ ]:


IQR = df1.quantile(0.75)-df1.quantile(0.25)
IQR


# #### Crosstable between Language and Average Rating

# In[ ]:


ct= pd.crosstab(df['average_rating'],df['language_code'],normalize=False)
ct.head()


# #### To Count no. of books with Average Rating = 0:

# In[ ]:


df[df['average_rating']==0]


# In[ ]:


df[df['average_rating']==0].count().loc['bookID']


# #### Observation: There are 34 books with zero rating as Average.

# #### To find Average Rating of books by Language:

# In[ ]:


df.groupby('language_code')['average_rating'].agg(['mean'])


# #### Observation: 'wel' language books have the highest average rating while 'cat' language books have the lowest (as depicted in the distribution plot earlier)

# #### To count the number of books in each language:

# In[ ]:


df.groupby('language_code')['bookID'].agg(['count'])


# #### Observation: 'Eng' language has the highest number of books avaialbe, i.e. 10594, while 'wel' language (which has the highest average rating) has the lowest number of books avalable, i.e. only 1.

# #### To rank the Titles based on Average Rating:

# In[ ]:


df['Ranks']=df['average_rating'].rank(ascending=0,method='dense')


# In[ ]:


df


# In[ ]:


df.sort_values(by='average_rating',ascending=False)


# #### To rank books of each Language as per Average Rating:

# In[ ]:


df['Rank By Language']=df.groupby('language_code')['average_rating'].rank(ascending=0,method='dense')


# In[ ]:


df.sort_values(by='average_rating',ascending=False)


# #### Regression Plot between Number of Pages and Average Rating:

# In[ ]:


plt.figure(figsize=(10,10))
sns.regplot(data=df,y="average_rating",x="# num_pages",marker='*',color='k')
plt.xlabel('No. of Pages')
plt.ylabel('Average Rating')
plt.show()


# ### Feature Selection

# In[ ]:


df.head()


# In[ ]:


df1 = df.drop(['bookID','title','authors','isbn','isbn13','Ranks','Rank By Language'],axis=1)
df1.head()


# In[ ]:


lang = list(df1['language_code'].value_counts().head(6).index)


# In[ ]:


df1['language_code']=np.where(df1['language_code'].isin(lang),df1['language_code'],'others')


# In[ ]:


df1['language_code'].value_counts()


# In[ ]:


df1 = pd.get_dummies(df1, columns=['language_code'],drop_first=True)


# In[ ]:


df1.head()


# In[ ]:


import statsmodels.api as sm
X = df1.drop('average_rating',axis=1)
y =df1['average_rating']
xc = sm.add_constant(X)
lr = sm.OLS(y,xc).fit()
lr.summary()


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values,i) for i in range (X.shape[1])]
pd.DataFrame({'vif':vif}, index = X.columns)


# In[ ]:


X.head()


# In[ ]:


X = X.drop('text_reviews_count',axis=1)
X.head()


# In[ ]:


xc = sm.add_constant(X)
lr = sm.OLS(y,xc).fit()
lr.summary()


# In[ ]:


X = X.drop('language_code_en-US',axis=1)


# In[ ]:


xc = sm.add_constant(X)
lr = sm.OLS(y,X).fit()
lr.summary()


# In[ ]:


X = X.drop('ratings_count',axis=1)


# In[ ]:


xc = sm.add_constant(X)
lr = sm.OLS(y,X).fit()
lr.summary()


# ### Linear Regression Model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[ ]:


xc = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train).fit()
lr.summary()


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


xc = sm.add_constant(X_test)
lr = sm.OLS(y_test,X_test).fit()
lr.summary()


# In[ ]:


plt.figure(figsize=(7,5))
plt.scatter(y_test,y_pred, color='y')
plt.plot(y,y,color='b')
plt.show()


# #### The r-squared value of the model on the training data is 88.1%, while on the test data, the r-squared is 88.4%. Therefore, the model has a good accuracy.

# ### Random Forest Model

# #### Hyperparametric Tuning

# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


# In[ ]:


print('RMSE: ',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared', r2_score(y_test,y_pred))


# #### The value of r-squared is equal to -0.0698.

# In[ ]:


n_estimators = [int(x) for x in np.linspace(start = 10, stop=200, num=10)]
max_depth = [int(x) for x in np.linspace(10,100,num=10)]
min_samples_split = [2, 3, 4, 5, 10]
min_samples_leaf = [1, 2, 4, 10]

random_grid = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split}

print(random_grid)


# In[ ]:


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid, cv=3)

rf_random.fit(X_train, y_train)


# In[ ]:


rf_random.best_params_


# #### We performed hyperparametric tuning by running the cross-validation search and found the best values for random forest parameters (n_estimators, min_samples_split, min_samples_leaf, max_depth). Now we'll run the Random Forest model again and check if r-square value has improved or not.

# In[ ]:


rf = RandomForestRegressor(**rf_random.best_params_)
rf


# In[ ]:


rf.fit(X_train,y_train)
y_pred = rf.predict(X_test) 

# Predict values of y by applying the Random Forest model generated through train data to test data.


# In[ ]:


print('RMSE:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R-squared:',r2_score(y_test,y_pred))


# #### Before tuning, the r-squared value was negative (implying negative correlation) but now the r-squared value has improved to 0.0214. This is still less accurate than the Linear Regression Model done before. Also, as seen in the plot below, the model does not fit as good as the one plotted in Linear Regression above.

# In[ ]:


plt.figure(figsize=(7,5))
plt.scatter(y_test,y_pred, color='r')
plt.plot(y,y,color='b')
plt.show()


# In[ ]:




