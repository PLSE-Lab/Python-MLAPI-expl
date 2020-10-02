#!/usr/bin/env python
# coding: utf-8

# **Call Center Optimization:** In this example, we take the Sanfrancisco 311 Call center data for last 10 years available at :https://data.world/sanfrancisco/mwjb-biik and use this to predict the Call volume for the next 3 months. We establish relation between Call Volume, Average Time to Answer in sec, Tranferred Calls percentage, and Service level achieved percentage (Call duration within 60 secs). Predicting these key parameters can improve Call Center operations, and help utilize resources efficiently.

# In[ ]:


import pandas
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from keras.layers import Dense   
from keras import optimizers

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas import DataFrame
from pandas import concat
from keras.models import load_model
from keras import optimizers
from matplotlib import pyplot
from math import sqrt
from keras import optimizers

# Any results you write to the current directory are saved as output.


# In[ ]:


numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('../input/311_call_metrics.csv', header=0, engine='python', skipfooter=3)
dataframe['Month'] = pd.to_datetime(dataframe.Month)
dataframe=dataframe.sort_values(by='Month',ascending=True)
regr_dataset = dataframe.copy()


# In[ ]:


dataframe.head()


# Next we sort the data based on the month and plot the data, to observe the trends. Since we are now going to predict the Call Volume of next 3 months based on historical data of Call Volume, we take the month and Call volume in our dataset

# In[ ]:


dataframe = dataframe.iloc[:,0:2]
dataset = dataframe.values[:,1:2]
dataset = dataset.astype('float32')

plt.plot(dataset)
plt.show()


# Next we Scale the dataset using MinMaxScaler in a range of 0 to 1

# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# We split the train and test data. We have 120 values, worth of 10 years of data, on a monthly basis. We take 20 values as test data, and remaining as train data. Since this is time series data, we take the first 106 values as train data and next 20 values as test data

# In[ ]:


train_size = int(len(dataset)) - 20
test_size = 20
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# **Approach 1: We use lookback period of 10, and create Supervised Training Data** for applying Deep Learning

# The next part is the most important step in creating the train data for a Supervised Learning problem. We shift the data with a lookback of 10 time steps and use this data to predict the value of the Call volume at the next time step.

# In[ ]:


#create dataframe series for t+1,t+2,t+3, to be used as y values, during Supervised Learning
#lookback = 10, means 10 values of TimeSeries (x) are used to predict the value at time t+1,t+2,t+3 (y)
def createSupervisedTrainingSet(dataset,lookback):

    df = DataFrame()
    x = dataset
    
    len_series = x.shape[0]

    df['t'] = [x[i] for i in range(x.shape[0])]
    #create x values at time t
    x=df['t'].values
    
    cols=list()
  
    df['t+1'] = df['t'].shift(-lookback)
    cols.append(df['t+1'])
    df['t+2'] = df['t'].shift(-(lookback+1))
    cols.append(df['t+2'])
    df['t+3'] = df['t'].shift(-(lookback+2))
    cols.append(df['t+3'])
    agg = concat(cols,axis=1)
    y=agg.values

    x = x.reshape(x.shape[0],1)

    len_X = len_series-lookback-2
    X=np.zeros((len_X,lookback,1))
    Y=np.zeros((len_X,3))
 
    for i in range(len_X):
        X[i] = x[i:i+10]
        Y[i] = y[i]

    return X,Y

look_back = 10
trainX, trainY = createSupervisedTrainingSet(train, look_back)
testX,testY = createSupervisedTrainingSet(test, look_back)


# In[ ]:


testY=testY.reshape(testY.shape[0],testY.shape[1])
trainY=trainY.reshape(trainY.shape[0],trainY.shape[1])
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[ ]:


#Check the sample train X and train Y, and match with original time series data
print1 = trainY[93,:].reshape(1,-1)
print("Train X at index 93")
print(np.around(scaler.inverse_transform(trainX[93,:,:])))
print("Train Y at index 93")
print(np.around(scaler.inverse_transform(print1)))
print("Actual Data")
print(np.around(scaler.inverse_transform(dataset[93:106])))        
#We used a lookback value of 10
#We inspect the X,Y values at a random index: 93
#As can be seen the 10 values of Time Series (Call Volume) from index 93 are being used as X to 
#predict the 3 values coming next (t+1,t+2,t+3)


# Use a Deep Learning technique with one hidden layer of 20 LSTM cells, outputting into 3 values, ie the predictions at time t+1,t+2,t+3. Input layer being 10 by 1 in size, for the 10 prior values of time series.

# In[ ]:


model = Sequential()
model.add(LSTM(20, input_shape=(look_back, 1)))
model.add(Dense(3))
myOptimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer)
history = model.fit(trainX, trainY, epochs=200,  validation_data=(testX,testY), batch_size=5, verbose=0)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], color=  'red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


#Once the model is trained, use it to make a prediction on the test data
testPredict = model.predict(testX)
predictUnscaled = np.around(scaler.inverse_transform(testPredict))
testYUnscaled = np.around(scaler.inverse_transform(testY))
#print the actual and predicted values at t+3
print("Actual values of Call Volume")
print(testYUnscaled[:,2])
print("Predicted values of Call Volume")
print(predictUnscaled[:,2])


# Plot the predicted and actual values at time t+1,t+2,t+3

# In[ ]:


pyplot.plot(testPredict[:,0])
pyplot.plot(testY[:,0],color='red')
pyplot.legend(['Actual','Predicted'])
pyplot.title('Actual vs Predicted at time t+1')
pyplot.show()

pyplot.plot(testPredict[:,1])
pyplot.plot(testY[:,1],color='red')
pyplot.legend(['Actual','Predicted'])
pyplot.title('Actual vs Predicted at time t+2')
pyplot.show()

pyplot.plot(testPredict[:,2])
pyplot.plot(testY[:,2],color='red')
pyplot.legend(['Actual','Predicted'])
pyplot.title('Actual vs Predicted at time t+3')
pyplot.show()


# In[ ]:


#Evaluate the RMSE values at t+1,t+2,t+3 to compare with other approaches, and select the best approach
def evaluate_forecasts(actuals, forecasts, n_seq):
    	for i in range(n_seq):
            actual = actuals[:,i]
            predicted = forecasts[:,i]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))
        
evaluate_forecasts(testYUnscaled, predictUnscaled,3)


# **Grid Search of parameters**
# We can improve the DeepLearning approach further by using Grid Search of Neural Network parameters using sklearn wrapper for Keras, KerasRegressor, and GridSearchCV. 

# **Regression analysis between Calls answered, transferred calls percentage, Average Speed of answer in secs, Service Level answered within 60 sec**

# In[ ]:


regr_dataset.head()


# In[ ]:


regr_dataset['Svc Level (% answered w/i 60 sec)'] = regr_dataset['Svc Level (% answered w/i 60 sec)'].apply(lambda x: np.nan if x in ['-'] else x[:-1]).astype(float)/100


# In[ ]:


regr_dataset = regr_dataset.fillna(regr_dataset.mean())


# In[ ]:


regr_dataset['Transferred Calls %'] = regr_dataset['Transferred Calls %'].str.rstrip('%').astype('float') / 100.0


# In[ ]:


regr_dataset.hist(bins=50,figsize=(20,15))
plt.show()


# In[ ]:


attributes = ["Calls Answered","Transferred Calls %","Svc Level (% answered w/i 60 sec)","Avg Speed Answer (sec)"]
scatter_matrix(regr_dataset[attributes],figsize=(12,8))
plt.show()


# The scatter matrix clearly shows relationship between avg_speed_answer_sec and calls_answered, ie as calls answered increases, the avg speed of answer reduces, and this is supported by correlation matrix
# **Correlation analysis**

# In[ ]:


corr_matrix = regr_dataset.corr()
corr_matrix["Calls Answered"].sort_values(ascending=False)


# The correlation matrix shows high negative correlation between the calls_answered and transferred_calls, as well 
# In order to minimize transferred calls, it is important to predict the call volume accurately.
# We decide to explore the correlation between Avg Speed of Answer in secs and Transferred Calls

# In[ ]:


from scipy.stats import spearmanr
from scipy.stats import pearsonr
regr_dataset['Avg Speed Answer (sec)'] =regr_dataset['Avg Speed Answer (sec)'].apply(lambda x: x if not pd.isnull(x) else regr_dataset['Avg Speed Answer (sec)'].mean())
regr_dataset['Transferred Calls %'] =regr_dataset['Transferred Calls %'].apply(lambda x: x if not pd.isnull(x) else regr_dataset['Transferred Calls %'].mean())
regr_dataset['Svc Level (% answered w/i 60 sec)'] =regr_dataset['Svc Level (% answered w/i 60 sec)'].apply(lambda x: x if not pd.isnull(x) else regr_dataset['Svc Level (% answered w/i 60 sec)'].mean())
corr_p, _ = pearsonr(regr_dataset['Avg Speed Answer (sec)'], regr_dataset['Transferred Calls %'])
print('Pearson correlation between Avg Speed of answer in secs and Transferred Calls: %.3f' % corr_p)

corr_p, _ = pearsonr(regr_dataset['Avg Speed Answer (sec)'], np.square(regr_dataset['Transferred Calls %']))
print('Pearson correlation between Avg Speed of answer in secs and square of Transferred Calls: %.3f' % corr_p)

corr_p, _ = pearsonr(regr_dataset['Svc Level (% answered w/i 60 sec)'], np.square(regr_dataset['Avg Speed Answer (sec)']))
print('Pearson correlation between Svc level and Avg speed of answer in secs: %.3f' % corr_p)


# We note there is high correlation between Avg speed of answer in secs and Transferred Calls. We develop Regression relation between Calls_Answered and transferred_calls, and use the output of this to predict Average speed of answer in secs for next 3 months. We also find high negative correlation between Svc level and Avg speed of answer in secs. As the Average Speed of Answer in secs increases, the Service level of answer within 60 secs decreases. We use this finding to predict these key parameters based on Calls Answered Volume for the next 3 months.

# In[ ]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(regr_dataset, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# In[ ]:


train_X = train_set.iloc[:,1] #take Calls answered as X variable
train_Y = train_set.iloc[:,4] #take Calls transferred as Y variable
test_X = test_set.iloc[:,1]
test_Y = test_set.iloc[:,4]


# In[ ]:


m=train_X.isnull().any()
print(m[m])
m=test_X.isnull().any()
print(m[m])


# In[ ]:


scaler1 = MinMaxScaler()
train_X = train_X.values.reshape(train_X.shape[0],1)
test_X = test_X.values.reshape(test_X.shape[0],1)
train_X = scaler1.fit_transform(train_X)
test_X = scaler1.transform(test_X)


# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(train_X, train_Y)


# In[ ]:


callTransfer_predictions = lin_reg.predict(test_X)
lin_mse = mean_squared_error(test_Y, callTransfer_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Calls transfered percentage RMSE with Lin Reg:"+str(lin_rmse))


# In[ ]:


sgd_reg = SGDRegressor(max_iter=500,penalty=None,eta0=0.1)
sgd_reg.fit(train_X, train_Y.values.ravel())


# In[ ]:


callTransfer_predictions = sgd_reg.predict(test_X)
sgd_mse = mean_squared_error(test_Y, callTransfer_predictions)
sgd_rmse = np.sqrt(sgd_mse)
print("Calls transfered percentage RMSE with SGD Reg:"+str(sgd_rmse))


# We decide to go with Linear Regression.

# In[ ]:


print(testX.shape)
currData = testX[7,:,:] #take the last set of 10 values to predict Calls Answered for next 3 months
currData = currData.reshape(1,currData.shape[0],currData.shape[1])
print(currData.shape)
currPredict = model.predict(currData)
currPredUnscaled = np.around(scaler.inverse_transform(currPredict))
print("prediction of Call Volume at time t+1:"+str(currPredUnscaled[:,0]))
print("prediction of Call Volume at time t+2:"+str(currPredUnscaled[:,1]))
print("prediction of Call Volume at time t+3:"+str(currPredUnscaled[:,2]))


# In[ ]:


CallVol_X = currPredict
print(CallVol_X)
CallVol_X = np.transpose(CallVol_X)
print(CallVol_X.shape)
callTransf_pred = lin_reg.predict(CallVol_X)
print("Predicted Calls transferred percentage")
print(callTransf_pred)


# We now fit Linear Regression between Transferred Calls and Average speed of Answer in secs

# In[ ]:


train_Y = train_set.iloc[:,3] # take Avg speed of answer in secs as Y variable
test_Y = test_set.iloc[:,3]
train_X = train_set.iloc[:,4] # take Calls transferred as X variable
test_X = test_set.iloc[:,4]
train_Y = train_Y.ravel().reshape(train_Y.shape[0],1)
test_Y = test_Y.ravel().reshape(test_Y.shape[0],1)
train_X = train_X.ravel().reshape(train_X.shape[0],1)
test_X = test_X.ravel().reshape(test_X.shape[0],1)


# In[ ]:


lin_reg1 = LinearRegression()
lin_reg1.fit(train_X, train_Y)


# In[ ]:


callTransf_pred = callTransf_pred.reshape(callTransf_pred.shape[0],1)
#scaler1.fit_transform(callTransf_pred)
print(callTransf_pred)
callAnsTime_predictions = lin_reg1.predict(callTransf_pred)
print("Avg speed of answer in secs predictions:")
print(callAnsTime_predictions)


# In[ ]:


train_Y = train_set.iloc[:,2] # take Svc level as Y variable
test_Y = test_set.iloc[:,2]
train_X = train_set.iloc[:,3] # take Avg speed of answer in secs as X variable
test_X = test_set.iloc[:,3]
train_Y = train_Y.ravel().reshape(train_Y.shape[0],1)
test_Y = test_Y.ravel().reshape(test_Y.shape[0],1)
train_X = train_X.ravel().reshape(train_X.shape[0],1)
test_X = test_X.ravel().reshape(test_X.shape[0],1)


# In[ ]:


lin_reg2 = LinearRegression()
lin_reg2.fit(train_X, train_Y)


# In[ ]:


callAnsTime_predictions =callAnsTime_predictions.reshape(callAnsTime_predictions.shape[0],1)
#scaler1.fit_transform(callTransf_pred)
print(callAnsTime_predictions)
SvcLevel_predictions = lin_reg2.predict(callAnsTime_predictions)
print("Svc level predictions:")
print(SvcLevel_predictions)


# **Final Predictions**

# In[ ]:


print("prediction of Call Volume at time t+1:"+str(currPredUnscaled[:,0]))
print("prediction of Call Volume at time t+2:"+str(currPredUnscaled[:,1]))
print("prediction of Call Volume at time t+3:"+str(currPredUnscaled[:,2]))


# In[ ]:


print("Predicted Calls transferred percentage, for next 3 months")
print(callTransf_pred)


# In[ ]:


print("Avg Call Ans Time predictions in secs, for next 3 months:")
print(callAnsTime_predictions)


# In[ ]:




