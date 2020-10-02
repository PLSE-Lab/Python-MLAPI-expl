import os

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler , StandardScaler ,OneHotEncoder

# Read Data
path = '../input/database.csv'
df = pd.read_csv(path)

# Feature Engineering
# fill deposits with zero
df['2010 Deposits'] = df['2010 Deposits'].fillna(0)
df['2011 Deposits'] = df['2011 Deposits'].fillna(0)
df['2012 Deposits'] = df['2012 Deposits'].fillna(0)
df['2013 Deposits'] = df['2013 Deposits'].fillna(0)
df['2014 Deposits'] = df['2014 Deposits'].fillna(0)
df['2015 Deposits'] = df['2015 Deposits'].fillna(0)
df['2016 Deposits'] = df['2016 Deposits'].fillna(0)

df['2010 Deposits Sum'] = df['2010 Deposits'].sum()
df['2011 Deposits Sum'] = df['2011 Deposits'].sum()
df['2012 Deposits Sum'] = df['2012 Deposits'].sum()
df['2013 Deposits Sum'] = df['2013 Deposits'].sum()
df['2014 Deposits Sum'] = df['2014 Deposits'].sum()
df['2015 Deposits Sum'] = df['2015 Deposits'].sum()

df['2010 Deposits Mean'] = df['2010 Deposits'].mean()
df['2011 Deposits Mean'] = df['2011 Deposits'].mean()
df['2012 Deposits Mean'] = df['2012 Deposits'].mean()
df['2013 Deposits Mean'] = df['2013 Deposits'].mean()
df['2014 Deposits Mean'] = df['2014 Deposits'].mean()
df['2015 Deposits Mean'] = df['2015 Deposits'].mean()

# decomposed Established Date
df['Established Date']  = df['Established Date'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
df['Established Year']  = df['Established Date'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')
df['Established Month'] = df['Established Date'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
df['Established Day']   = df['Established Date'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')

# drop Acquired Date
df_temp = df.drop(['Acquired Date'] , axis = 1)

# drop Established Date
df_temp = df_temp.drop(['Established Date'] , axis = 1)

# drop Institution Name
df_temp = df_temp.drop(['Institution Name'] , axis = 1)

# drop Main Office 
df_temp = df_temp.drop(['Main Office'] , axis = 1)

# LabelEncoder and MinMaxScalar
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()


for c in df_temp.columns:
    df_temp[c] = df_temp[c].fillna(-1)
    if df_temp[c].dtype == 'object':
        df_temp[c] = LEncoder.fit_transform(list(df_temp[c].values))
    
# if df_temp[c].dtype != 'int64':
#         df_temp[c] = MMEncoder.fit_transform(df_temp[c].values.reshape(-1, 1))

print(df_temp.head())
print(np.shape(df_temp))
print("-----------------------------------------")

# split traing data , testing data
from sklearn.model_selection import train_test_split

y_data = df_temp['2016 Deposits']
df_temp = df_temp.drop(['2016 Deposits'] , axis = 1)
x_data = df_temp

x_train , x_test , y_train , y_test = train_test_split(x_data , y_data ,test_size = 0.3 )

# use LinearRegression to predict 2016 Deposits 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Train model

lr = LinearRegression()

print("Training score = " , cross_val_score( lr, x_train, y_train, cv=5).mean())
print("----------------------------------------------")

lr.fit(x_train , y_train)

y_predict = lr.predict(x_test)

from sklearn.metrics import mean_squared_error , r2_score 

mse = mean_squared_error(y_test , y_predict)
r2 = r2_score(y_test , y_predict)

print("TestingData MSE =" , mse)
print("TestingData R2 =" , r2)