#!/usr/bin/env python
# coding: utf-8

# > ## Get that Avo Toast.  
# ***It's not like you can afford a home in the Bay anyway.***

# This is my third attempt at avocado price prediction. My first attempt was using fb prophet, second attempt was using regression, and both methods did not work. Fbprophet only worked for univariate dataset, but I wanted to use the region variable as well. And the regression models could not predict future dates.  
# It seems like for multivariate time series data, I would need a more complex model like neural networks. Hence I am trying LSTM this time. Let's see if third time will be a charm.  
# 
# *This is also my first Kaggle/data science project and any criticism/feedback is welcomed :) *
# 

# ### Goal
# Predict the region-specific avocado price based on total volume and historical average price.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import math, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/avocado.csv')
df.head()


# In[ ]:


df = df.drop(['Unnamed: 0'], 1)


# In[ ]:


scaler = StandardScaler()
df.loc[:,'Total Volume':'XLarge Bags']= scaler.fit_transform(df.loc[:,'Total Volume':'XLarge Bags'])


# Normalizing the volume columns.

# In[ ]:


df['type']=df['type'].replace('conventional', 0)
df['type']=df['type'].replace('organic', 1)


# Label encoding and one-hot encoding the categorical variables.

# In[ ]:


region_ohe = OneHotEncoder(categories = "auto", handle_unknown = "ignore")
X_encoded = region_ohe.fit_transform(df['region'].values.reshape(-1,1)).toarray()
X_encoded = pd.DataFrame(X_encoded, columns = [str(int(i)) for i in range(X_encoded.shape[1])])
X = df.drop(['year', 'region'], 1)
dff = pd.concat([X, X_encoded], axis = 1)
#moving AveragePrice to the last column
dff['Price']=dff.AveragePrice
dff.drop(['AveragePrice'], 1, inplace=True)
dff.head()


# In[ ]:


dff=dff.set_index('Date')
dff.tail()


# Using the date column as index.

# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# It seems like the quantities are highly correlated between each other, but not very much with the average price... This is not good since the variables should be independent. I should just keep one and drop the others, I will keep Total Volume.

# In[ ]:


df2 = dff.drop(dff.loc[:,'4046': 'XLarge Bags'], 1)


# In[ ]:


from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential


# In[ ]:


def load_data(df, seq_len):
    amount_of_features = len(df.columns) # 5
    data = df.as_matrix() 
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)
    print(result.shape)
    row = round(0.9 * result.shape[0]) # 90% split
    print("row: ", row)
    train = result[:int(row), :] # 90% date, all features 
    #print(train.shape)
    
    x_train = train[:, :-1] 
    y_train = train[:, -1][:,-1]
    
    x_test = result[int(row):, :-1] 
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


# In[ ]:


window = 10
X_train, y_train, X_test, y_test = load_data(df2, window)


# In[ ]:


def build_model(layers):
    d = 0.3
    model = Sequential()
    
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    
    # adam = keras.optimizers.Adam(decay=0.2)
        
    start = time.time()
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


# In[ ]:


model = build_model([57,window,1])


# In[ ]:


model.fit(X_train,y_train,batch_size=100,epochs=50,validation_split=0.1,verbose=1)


# In[ ]:


diff=[]
ratio=[]
p = model.predict(X_test)
print (p.shape)
# for each data index in test data
for u in range(len(y_test)):
    # pr = prediction day u
    pr = p[u][0]
    # (y_test day u / pr) - 1
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
    # Last day prediction
print(p[-1]) 


# In[ ]:


def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)


# In[ ]:


import matplotlib.pyplot as plt2
plt2.figure(figsize=(14,6))
plt2.plot(p,color='red', label='Prediction')
plt2.plot(y_test,color='blue', label='Actual')
plt2.legend(loc='best')
plt2.show()


# From the model score and the plot, it seems like the model is doing a pretty good job. But I'm not confident at all since I cannot justify all the parameters that I'm using (sequence length, batch size, units, number of layers, etc.)... I'm sure there is a lot of room for optimization, but I think (hope) that the overall structure of the model is right at least? 

# ### Now let's try predicting the price of a new/future date.

# I will go through a series of clunky steps to get the user's input into the right shape for the model.
# I'm aware that I might be taking a lot of detours and unecessary steps here... would love to know if there's a better alternative.

# In[ ]:


regiondict = {}
for key, value in enumerate(df.region.unique()):
    regiondict[key] = value


# In[ ]:


def OHE_region (region): #region is a string   
    #get the region number as mapped in the dict
    regionnum = list(regiondict.keys())[list(regiondict.values()).index(region)]   
    #create regionnum columns of 0, concat one column of 1 at index regionnum, then concat 53-regionnum columns of 0.
    before_df = pd.DataFrame(0.0, index=range(1), columns = list(range(regionnum)))
    after_df = pd.DataFrame(0.0, index=range(1), columns = list(range(regionnum+1, 54)))
    new_df = pd.DataFrame(1.0, index=range(1), columns = [str(regionnum)])
    OHE_df = pd.concat([before_df, new_df, after_df], axis=1) 
    return OHE_df


# In[ ]:


OHE_region('Tampa')


# In[ ]:


X_dict = {'Date': '2019-08-12', 'Total Volume': dff['Total Volume'].mean(), 'type': 0}
X_df = pd.DataFrame([X_dict])
new_X = pd.DataFrame(X_df, columns=X_dict.keys()) #hacky/dumb way of making sure that the columns in the df maintain the same order as in the dict.
encoded_region = OHE_region('Houston')
new_X = pd.concat([new_X, encoded_region], axis=1)
new_X=new_X.set_index('Date')
print(new_X)


# In[ ]:


#new_X=new_X.reshape(1,1,56)


# In[ ]:


#model.predict(new_X)


# ### stuck - TBC. 

# In[ ]:




