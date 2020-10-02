#!/usr/bin/env python
# coding: utf-8

# * ### High Way mileage prediction of the Automobile vehicles in th Automobile Industry.

# In[ ]:


import pandas as pd
import numpy as np


# * ### Copy the Data of the Automobile showroom Details.

# In[ ]:


base_dataset = pd.read_csv("../input/Automobile_data.csv")


# In[ ]:


base_dataset


# In[ ]:


base_dataset.isna().sum()


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


(lable,dummies_table,cat,cont)=variables_creation(base_dataset,8)


# In[ ]:


cont


# In[ ]:


base_dataset = base_dataset[base_dataset.describe().columns]


# In[ ]:


base_dataset.shape


# In[ ]:


base_dataset.drop('highway-mpg',axis=1).columns


# In[ ]:


base_dataset.shape


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


outliers_treated=outliers(base_dataset[base_dataset.drop('highway-mpg',axis=1).columns])


# In[ ]:


outliers_treated.shape


# In[ ]:


import matplotlib.pyplot as plt
for i in outliers_treated:
    plt.hist(outliers_treated[i])
    plt.show()


# ### Model Building

# In[ ]:


outliers_treated=outliers_treated[outliers_treated.describe().columns]
outliers_treated['const']=1


# In[ ]:


outliers_treated


# In[ ]:


outliers_treated['target']=base_dataset['highway-mpg']
y=outliers_treated['target']
x=outliers_treated.drop('target',axis=1)


# In[ ]:


x


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[ ]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
predicted_values=lm.predict(X_test)


# In[ ]:


sum(abs(predicted_values-y_test.values))


# In[ ]:


Final = pd.DataFrame(predicted_values)
y_test=y_test.reset_index()
y_test = y_test.drop('index',axis = 1)
Final['y_test'] = y_test
print(Final)


# In[ ]:


from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(y_test.values,predicted_values)


# In[ ]:


from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y_test.values,predicted_values)


# In[ ]:


from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(y_test.values,predicted_values))


# In[ ]:


MAPE=sum(abs((y_test.values-predicted_values)/(y_test.values)))/X_test.shape[0]


# ### Graph( Actual vs Predicted)

# In[ ]:


error_table=pd.DataFrame(predicted_values,y_test.values)


# In[ ]:


error_table.reset_index(inplace=True)


# In[ ]:


error_table.columns=['pred','actual']


# In[ ]:


error_table.plot(figsize=(20,8))

