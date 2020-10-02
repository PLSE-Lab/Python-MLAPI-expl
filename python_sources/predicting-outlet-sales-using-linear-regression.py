import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
train.shape
train.isnull().sum()
test.shape
test.info()
data=pd.concat([train,test])
data.shape

data.info()
data.head(1)
data.dtypes
type(data.index)
data=data.set_index('Item_Identifier')
data.head(1)
data.info()
data.isnull().sum()
data=data.drop(["Outlet_Establishment_Year"],axis=1)
data['Item_Outlet_Sales']=data['Item_Outlet_Sales'].replace(0,np.NaN)
data.fillna(data.mean(),inplace=True)
data['Outlet_Size']=data['Outlet_Size'].replace(0,np.NaN)
data.fillna(data.median(),inplace=True)

data=data.drop(["Outlet_Size"],axis=1)
data.Item_Type.value_counts().plot(kind='bar',title='Item Type')

data.Item_Fat_Content.value_counts().plot(kind='bar',title='Item Fat Content')
data.groupby('Item_Type')[['Item_Type','Item_Outlet_Sales']].mean().sort_values(ascending=False,by='Item_Outlet_Sales').plot(kind='bar',color='r')

data.groupby('Outlet_Type')[['Outlet_Type','Item_Outlet_Sales']].mean().sort_values(ascending=False,by='Item_Outlet_Sales').plot(kind='bar',color='b')
data=pd.get_dummies(data,columns=['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Location_Type','Outlet_Type'])
data.dtypes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
y=data.Item_Outlet_Sales
x=data.drop(["Item_Outlet_Sales"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)
regr=LinearRegression()
reg=regr.fit(x_train,y_train)
pred=regr.predict(x_test)
pred[5]
y_test[[5]]
np.mean((pred-y_test)**2)#MSE for the model

