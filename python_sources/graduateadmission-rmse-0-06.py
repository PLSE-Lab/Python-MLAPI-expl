#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Import libraries


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#### read in data


# In[ ]:


df = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv',index_col='Serial No.')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()####no missing values


# In[ ]:


df.info()


# In[ ]:


### pairplot
sns.pairplot(df)


# ### 1. Gre Score and TOEFL have high correlation
# 
# ### 2.CGPA and chance of admit have high and correlation

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


### Gre score is out of 340
### TOEFL is out of 120


# In[ ]:


df['Total_Perc_Score'] = ((df['GRE Score'] + df['TOEFL Score'])/(340+120))


# In[ ]:


df.head()


# In[ ]:


sns.heatmap(df.drop(['GRE Score','TOEFL Score'],axis=1).corr(),annot=True)


# In[ ]:


### Research
df['Research'].value_counts()


# In[ ]:


### Let us convert 0 to -1


# In[ ]:


# df['Research'] = df['Research'].replace(0,-1)


# In[ ]:


### Research
df['Research'].value_counts()


# In[ ]:


## SOP+LOR TOTAL
df['SOP_LOR_Total'] = ((df['SOP'] + df['LOR ']) /10)


# In[ ]:


df.head()


# In[ ]:


df['Total_Perc_Score']=df['Total_Perc_Score'].apply(lambda x : np.round(x,2))


# In[ ]:


sns.heatmap(df[['SOP_LOR_Total','Total_Perc_Score','Research','University Rating']].corr(),annot=True)


# In[ ]:


## As we can see, multicollinearity has been drasitcally reduced


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df_input = df[['University Rating','CGPA',
       'Research', 'Chance of Admit ', 'Total_Perc_Score', 'SOP_LOR_Total']]


# In[ ]:


df_input.head()


# In[ ]:


#### Arrays
X = df_input.drop('Chance of Admit ',axis=1).values
y = df['Chance of Admit '].values


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


## Train test split


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# In[ ]:


linear_regressor = LinearRegression()
rfc_regressor = RandomForestRegressor()
sgd_regressor = SGDRegressor()


# In[ ]:


np.mean(cross_val_score(linear_regressor,X_train,y_train,cv=5))


# In[ ]:


np.mean(cross_val_score(rfc_regressor,X_train,y_train,cv=5))


# In[ ]:


np.mean(cross_val_score(sgd_regressor,X_train,y_train,cv=5))


# In[ ]:


###
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


ridge = Ridge()


# In[ ]:


np.mean(cross_val_score(ridge,X_train,y_train,cv=5))


# In[ ]:


linear_regressor.fit(scaled_X_train,y_train)


# In[ ]:


predictions_linear = linear_regressor.predict(scaled_X_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


print(mean_absolute_error(y_test,predictions_linear))


# In[ ]:


print((mean_squared_error(y_test,predictions_linear)**(1/2)))


# In[ ]:


#Mean absolute error = 0.04
#rmse =0.06 

#linear models are the best!

