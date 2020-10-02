#!/usr/bin/env python
# coding: utf-8

# ### **Import packages**

# In[33]:


##Importing the packages
#Data processing packages
import numpy as np 
import pandas as pd 

#Visualization packages
import matplotlib.pyplot as plt 
import seaborn as sns 

#Machine Learning packages
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# ### **Import data**

# In[34]:


data = pd.read_csv('../input/insurance.csv')


# In[35]:


data.head()


# In[36]:


data.shape


# ### **Basic Stats**

# In[37]:


data.describe(include='all')


# ### **Check and remediate if there are any null values**

# In[38]:


data.info()


# **COMMENT:** Above output shows that there are No Null values.

# In[39]:


data.isnull().sum()


# **COMMENT:** Above output shows that there are No Null values.

# ### **Category Columns**

# In[40]:


cat_cols = data.columns[data.dtypes=='object']
data_cat = data[cat_cols]
cat_cols
data_cat.head()


# ### **Numerical Columns**

# In[41]:


num_cols = data.columns[data.dtypes!='object']
data_num = data[num_cols]
num_cols
data_num.head()


# ### **Correlation of Numerical Columns**

# In[42]:


#Inspecting the corrolation between the features
data.corr()


# In[43]:


#sns.heatmap(data_num.corr())
sns.heatmap(data_num.corr(), vmax=.8, linewidths=0.01,square=True,annot=True,cmap='magma',linecolor="black")


# **COMMENT:** There is some correlation betweeen BMI & Age and BMI & charges

# ### **Pairplot of Numerical Columns**

# In[44]:


sns.pairplot(data, hue='smoker')
#sns.pairplot(data_num, hue='smoker', palette="Set2", diag_kind="kde", size=2)


# In[45]:


data.hist(layout = (3, 3), figsize=(12, 9), color='blue', grid=False, bins=15)


# ### **Comparison of Category Columns**

# In[46]:


#pd.crosstab(data.sex, data.smoker)
pd.crosstab(data.sex, data.smoker, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENT:** Insurance charges are more for males as compared to females

# In[47]:


#pd.crosstab(data.sex, data.region)
pd.crosstab(data.sex, data.region, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENT:** Males are more in SouthEast as compared to other regions

# In[48]:


#pd.crosstab(data.smoker, data.region)
pd.crosstab(data.smoker, data.region, margins=True, normalize='index').round(2).style.background_gradient(cmap='autumn_r')


# **COMMENT:** Smokers are more in SouthEast as compared to other regions

# In[49]:


plt.figure(figsize=(24,6))
plt.subplot(131)  ; sns.boxplot(x='sex',y='charges',data=data)
plt.subplot(132)  ; sns.boxplot(x='smoker',y='charges',data=data)
plt.subplot(133)  ; sns.boxplot(x='region',y='charges',data=data)


# In[58]:


#plt.figure(figsize=(9,6))
sns.boxplot(x='sex',y='charges',data=data)


# **COMMENT:** 
# 1. Males pay more Insurance Charges (may due to more Smoker population in males)
# 2. Smokers pay more Insurance Charges (may due to high risk of diseases in smokers)
# 3. People in SouthEast region pay more Insurance charges (may due to more male poplulation, which also means more smoker population)

# In[32]:


plt.figure(figsize=(24,6))
plt.subplot(131)  ; sns.boxplot(x='sex',y='age',data=data)
plt.subplot(132)  ; sns.boxplot(x='sex',y='bmi',data=data)
plt.subplot(133)  ; sns.boxplot(x='sex',y='children',data=data)


# In[19]:


plt.figure(figsize=(24,6))
plt.subplot(131)  ; sns.boxplot(x='smoker',y='age',data=data)
plt.subplot(132)  ; sns.boxplot(x='smoker',y='bmi',data=data)
plt.subplot(133)  ; sns.boxplot(x='smoker',y='children',data=data)


# In[20]:


plt.figure(figsize=(24,6))
plt.subplot(131)  ; sns.boxplot(x='region',y='age',data=data)
plt.subplot(132)  ; sns.boxplot(x='region',y='bmi',data=data)
plt.subplot(133)  ; sns.boxplot(x='region',y='children',data=data)


# ### **Encoding Category Values into Numerical values**

# In[21]:


#sex
le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)


# In[22]:


data.head()


# ### **Splitting, Training & Testing using Linear Regression**

# In[23]:


x = data.drop(['charges'], axis = 1)
y = data.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test))


# ### **Splitting, Training & Testing using Polynomial Features**

# In[24]:


X = data.drop(['charges'], axis = 1)
Y = data.charges

quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test))


# In[ ]:





# In[ ]:





# In[ ]:




