#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
from warnings import filterwarnings
filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's explore the insurance data from kaggle and see if we can 
# 1. Find any useful information through Exploratory Data Analysis.
# 2. Use any machine learning algorithms to predict any useful parameters.

# First, we load the data into a dataframe and look at the sample/head of the data.

# In[ ]:


df = pd.read_csv('../input/insurance.csv')
df.head()


# > Looks pretty straight-forward. Let's check for other stats info and if the data has any nulls.

# In[ ]:


df.describe()


# In[ ]:


df.info()


# >Seems like we are going to miss out on all the fun of cleaning the dirty data today. May be wait for the next project. 
# >However, we can replace "yes" or "no" to binary (1s and 0s) to make it easier for the machine to understand.

# In[ ]:


df_clean = df.replace(to_replace={'yes':1, 'no':0})


# > Converted "smoker" column from "yes" or "no" to binary (1s, 0s)

# In[ ]:


df_clean.sex.unique()


# > Hmm. No other genders included. Not sure why. TODO: Research about how healthcare categorizes LGBTQ insured.

# In[ ]:


df_clean.region.unique()


# > We can convert these regions into numerical variables later on if needed. Keeping it as strings for easy interpretability.

# In[ ]:


#defining my own palette for smokers and non-smokers to appear as red and green respectively.
pal = ['#FF0000', #Red
       '#006400', #Green
      ]


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(df_clean.corr(), annot=True)


# >Charges are heavily correlated to being a smoker, followed by age, bmi, and children in that order.

# In[ ]:


sns.pairplot(df_clean, hue='smoker', palette=pal)


# > Visualizing the data definitely shows us some unique differences between the smokers and non-smokers. Let's dive a layer deeper.

# In[ ]:


sns.scatterplot(x='charges', y='age', data=df_clean, hue='smoker', palette=pal)


# In[ ]:


sns.boxplot(x='smoker', y='charges', data=df_clean, palette=pal, order=[1, 0])


# > So, smokers are charged significantly higher on average than non-smokers. Let's dive into what separates the two streams of smokers (one with higher charges and one with lesser).
# Since, BMI was another correlated factor to the charges let's use that as a hue to see the trend

# In[ ]:


sns.scatterplot(x='charges', y='age', data=df_clean[(df_clean['smoker']==1)], color="Red", hue='bmi', palette='Blues')
plt.title('Smoker\'s "Age vs Charges"')


# > Seems like BMI is the factor that splits the two categories. 

# In[ ]:


df_clean['BMI below limit'] = df_clean['bmi'].apply(lambda x: 1 if x<=30 else 0)
sns.scatterplot(x='charges', y='age', data=df_clean[(df_clean['smoker']==1)], color="Red", hue='BMI below limit', palette='Blues')
plt.title('Smoker\'s "Age vs Charges"')


# > After experimentation, I found BMI of 30 to be the "sweet-spot" that divides these streams.
# So, people who are smokers and have BMI over 30 pay significantly higher rates than other smokers.

# ## Time for some machine learning. Let's start with Linear Regression and see how it does

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


y= df_clean[df_clean['smoker']==1]['charges']
X= df_clean[df_clean['smoker']==1][['age', 'bmi', 'children']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[ ]:


coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
coefficients


# > These coeff how much a unit change in the age, bmi, and having children will affect the Charges. e.g. A Unit increase in "age" will increase "charges" by ~246.

# In[ ]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, classification_report

print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
print('R2 test_data', r2_score(y_test, predictions))


# > Lazy implementation gave around 69% accuracy which is not that great.

# In[ ]:


sns.distplot((y_test-predictions),bins=15)


# > We can normalize the data and recalculate the predictions but I think RandomForestRegressor should do better in this scenario as it uses bagging and the summation of results from different forests.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators =100, criterion = 'mse',random_state = 42,n_jobs = -1)
rfr.fit(X_train,y_train)
rfr_pred_train = rfr.predict(X_train)
rfr_pred_test = rfr.predict(X_test)


print('MSE train_data: ', round((mean_squared_error(y_train,rfr_pred_train)), 1))
print('MSE test_data: ', round(mean_squared_error(y_test,rfr_pred_test), 1))
print('R2 train_data: ',round(r2_score(y_train,rfr_pred_train), 2))
print('R2 test_data: ', round(r2_score(y_test,rfr_pred_test), 2))


# > 87% accuracy is pretty good given that we haven't normalized the data.

# In[ ]:


sns.distplot((y_test - rfr_pred_test),bins=30)


# In[ ]:




