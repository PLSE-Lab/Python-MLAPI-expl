#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


base_data_train=pd.read_csv("../input/train.csv")


# In[ ]:


base_data_train


# In[ ]:


base_data_test=pd.read_csv("../input/test.csv")


# In[ ]:


base_data_test.shape


# In[ ]:


cont=base_data_train.describe().columns


# In[ ]:


cat=base_data_test.describe(include='object').columns


# In[ ]:


def nullvalue_function(base_dataset,percentage):
    
    # Checking the null value occurance
    
    print(base_dataset.isna().sum())

    # Printing the shape of the data 
    
    print(base_dataset.shape)
    
    # Converting  into percentage table
    
    null_value_table=pd.DataFrame((base_dataset.isna().sum()/base_dataset.shape[0])*100).sort_values(0,ascending=False )
    
    null_value_table.columns=['null percentage']
    
    # Defining the threashold values 
    
    null_value_table[null_value_table['null percentage']>percentage].index
    
    # Drop the columns that has null values more than threashold 
    base_dataset.drop(null_value_table[null_value_table['null percentage']>percentage].index,axis=1,inplace=True)
    
    # Replace the null values with median() # continous variables 
    for i in base_dataset.describe().columns:
        base_dataset[i].fillna(base_dataset[i].median(),inplace=True)
    # Replace the null values with mode() #categorical variables
    for i in base_dataset.describe(include='object').columns:
        base_dataset[i].fillna(base_dataset[i].value_counts().index[0],inplace=True)
  
    print(base_dataset.shape)
    
    return base_dataset


# In[ ]:


base_data_train_null=nullvalue_function(base_data_train,30)


# ## to create dummy variables  and label encodong variables 

# In[ ]:


from sklearn import preprocessing

def variables_creation(base_dataset,unique):
    
    cat=base_dataset.describe(include='object').columns
    
    cont=base_dataset.describe().columns
    
    x=[]
    
    for i in base_dataset[cat].columns:
        if len(base_dataset[i].value_counts().index)<unique:
            x.append(i)
    
    dummies_table=pd.get_dummies(base_dataset[x])
    encode_table=base_dataset[x]
    
    le = preprocessing.LabelEncoder()
    lable_encode=[]
    
    for i in encode_table.columns:
        le.fit(encode_table[i])
        le.classes_
        lable_encode.append(le.transform(encode_table[i]))
        
    lable_encode=np.array(lable_encode)
    lable=lable_encode.reshape(base_dataset.shape[0],len(x))
    lable=pd.DataFrame(lable)
    return (lable,dummies_table,cat,cont)


# In[ ]:


import numpy as np
lable,dummies_table,cat,cont=variables_creation(base_data_train_null,15)


# In[ ]:


lable


# In[ ]:


lable.shape


# In[ ]:


base_data_train_null.shape


# # append these lable data with the base_data_train_null data

# In[ ]:


for i in lable.columns:
    base_data_train_null[i]=lable[i]


# In[ ]:


base_data_train_null.drop(cat,axis=1,inplace=True)


# In[ ]:


base_data_train_null


# # now perform ouliers treatment

# In[ ]:


def outliers(df):
    import numpy as np
    import statistics as sts

    for i in df.describe().columns:
        x=np.array(df[i])
        p=[]
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        LTV= Q1 - (1.5 * IQR)
        UTV= Q3 + (1.5 * IQR)
        for j in x:
            if j <= LTV or j>=UTV:
                p.append(sts.median(x))
            else:
                p.append(j)
        df[i]=p
    return df


# In[ ]:


base_data_anly=outliers(base_data_train_null)


# In[ ]:


base_data_anly["Id"]


# In[ ]:


y=base_data_anly['SalePrice']
x=base_data_anly.drop('SalePrice',axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[ ]:


print(X_train.shape ,X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


y_train


# In[ ]:


y_test


# In[ ]:


# liinear regression 
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

lm.fit(X_train,y_train)

predicted=lm.predict(X_test)


# In[ ]:


actual=y_test.values


# In[ ]:


errors=pd.DataFrame(predicted,actual).reset_index()


# In[ ]:


errors


# In[ ]:


errors.columns=['acutal','predicted']


# In[ ]:


errors.plot()


# In[ ]:


errors[0:100].plot()


# In[ ]:


errors[100:200].plot()


# In[ ]:


errors[200:300].plot()


# In[ ]:


from sklearn.metrics import r2_score
r2_score(lm.predict(X_train),y_train)


# In[ ]:


# now apply variable reduction methods to improve the accuracy i.,e Rsquared  value


# # variable Reduction methods
# # 1.Low variace method
# # 2.correlation method
# # 3.Forward selection method
# # 4.Backward selection method
# # 5.Stepwise Method

# In[ ]:





# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as sfa 


# In[ ]:


variance_table=pd.DataFrame(X_train.var().sort_values(ascending=False))


# In[ ]:


variance_table


# In[ ]:





# In[ ]:


# low variance method
model=sm.OLS(y_train,X_train[variance_table.index[0:20]]) 
lm=model.fit()
lm.summary()
lm.rsquared


# In[ ]:


# forward selection
for i in range(1,40):
    model=sm.OLS(y_train,X_train[variance_table.index[0:i]]) 
    lm=model.fit()
    lm.summary()
    print(variance_table.index[0:i],lm.rsquared)


# In[ ]:


# backward selection
for i in range(40,1,-1):
    model=sm.OLS(y_train,X_train[variance_table.index[0:i]]) 
    lm=model.fit()
    lm.summary()
    print(lm.rsquared)


# In[ ]:


x=list(X_train.columns)


# In[ ]:


from random import shuffle
shuffle(x)


# In[ ]:


x


# In[ ]:


for i in range(1,40):
        shuffle(x)
        model=sm.OLS(y_train,X_train[x[0:i]]) 
        lm=model.fit()
        lm.summary()
        print(lm.rsquared)


# In[ ]:


X_train.corr()


# In[ ]:


x=[]
for i in X_train.describe().columns:
    for j in X_train.describe().columns:
        if i!=j:
            d=  {
                'X' :i,
                'y' :j ,
                'Corr' : X_train[[i,j]].corr().values[0][1]
                }
        
            x.append(d)


# In[ ]:


pd.DataFrame(x).sort_values('Corr',ascending=False).head()


# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as sfa 

model=sm.OLS(y_train,X_train) 
lm=model.fit()
lm.summary()


# In[ ]:


lm.params.sort_values(ascending=False).head()


# In[ ]:


lm.params.sort_values(ascending=True).head()


# In[ ]:


lm.params


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(lm.params.values)


# In[ ]:





# In[ ]:


pd.DataFrame(lm.params).reset_index()


# In[ ]:


coeff_tables=pd.DataFrame(lm.params).reset_index()


# In[ ]:


coeff_tables.head()


# In[ ]:


coeff_tables.tail()


# In[ ]:


coeff_tables.values


# In[ ]:





# # Printing MSE RMSE MAE

# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(actual, predicted)


# In[ ]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(actual, predicted))


# In[ ]:


sum(abs(actual-predicted))/len(actual)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




