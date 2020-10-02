#!/usr/bin/env python
# coding: utf-8

# #### importing libraries

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from scipy.stats import shapiro,levene
from sklearn.model_selection import train_test_split


# #### importing the dataset

# In[5]:


data=pd.read_csv("../input/USA_Housing.csv")
data.head()
# drop the Address column
data = data.drop('Address', axis=1)
#checking the head of the data
data.head()


# In[6]:


#describing the data
data.describe()


# In[7]:


data.info()


# In[8]:


#checking the shape of the data
data.shape


# #### checking for null values in the given dataset

# In[9]:


data.isnull().sum()


# #### hence there are no null values in the dataset EDA is not required

# #### plotting a heatmap to check the correlation between the variables 

# In[10]:


data.corr()


# In[11]:


sns.heatmap(data.corr(),annot=True)


# #### Pairplot to check the normalization and linearization of the  given dataset

# In[ ]:


sns.pairplot(data,diag_kind='kde')
plt.show()


# #### normalizing the data as there are different measuring units in the given dataset

# In[ ]:


data.columns


# In[ ]:


# renaming the columns
data.rename(columns={'Avg. Area Income':'Area_Income','Avg. Area House Age':'Area_House_Age','Avg. Area Number of Rooms':'Area_Number_of_Rooms','Avg. Area Number of Bedrooms':'Area_Number_of_Bedrooms','Area Population':'Area_Population'},inplace=True)


# In[ ]:


data.columns


# In[ ]:


X = data[data.columns[0:-1]]
Y = data["Price"]
X.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std =  sc.fit_transform(X)


# #### Fitting the model by using OLS method 
# #### backward elimination

# In[ ]:


model=smf.ols('Price~Area_Income+Area_House_Age+Area_Number_of_Rooms+Area_Number_of_Bedrooms+Area_Population',data).fit()
model.summary()


# #### as the p value is higher for Area_Number_of_Bedrooms column dropping off the Area_Number_of_Bedrooms column

# In[ ]:


#refitting  the model
model=smf.ols('Price~Area_Income+Area_House_Age+Area_Number_of_Rooms+Area_Population',data).fit()
model.summary()


# #### performing train_test_split and linear regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[ ]:


X=data.drop(['Area_Number_of_Bedrooms','Price'],axis=1)
Y=data['Price']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3,random_state=1)


# In[ ]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)
linreg=LinearRegression()
linreg.fit(X_train,Y_train)


# In[ ]:


ypred=linreg.predict(X_test)
ypred


# #### K-FOLD VALIDATION

# In[ ]:


from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve,auc


# In[ ]:


X.head()


# In[ ]:


Y.head()


# #### computing RMSE and R-squared values

# In[ ]:


kf=KFold(n_splits=5,shuffle=True,random_state=2)
root=[]
lst=[]
for train,test in kf.split(X,Y):
    linreg=LinearRegression()
    X_train,X_test=X.iloc[train,:],X.iloc[test,:]
    Y_train,Y_test=Y.iloc[train],Y.iloc[test]
    linreg.fit(X_train,Y_train)
    ypred=linreg.predict(X_test)
    root.append(np.sqrt(metrics.mean_squared_error(Y_test,ypred)))
    lst.append(linreg.score(X_train,Y_train))
    
print('Cross Validation Mean rmse is %1.2f'%np.mean(root))
print('Cross Validation Variance of rmse is %1.5f'%np.var(root,ddof=1))
print('Cross Validation Mean R square is %1.2f'%np.mean(lst))
print('Cross Validation Variance of R square is %1.5f'%np.var(lst,ddof=1))


# #### insights
Here i executed the model taking output variable as price and i observed that my model is showing an accuracy 0f 92% with mean rmse of 101187.14
# In[ ]:




