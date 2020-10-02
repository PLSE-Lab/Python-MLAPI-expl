"""
In last two succussful versions, you have learned basic of LSTM.
We have already done examples with 
1.-a) one stock and one feature(on close price column)
   b) one stock and two feature(on close price column)--it gives prediction two step ahead.
2. one stock with two features(open and close price)
"""

"""
Now, we are going to do
1.) two or more stocks with two or more features
"""

""" 
For best practice, you should already be familier with the following questions which were mentioned
in the previous versions of this kernel
1. Why is it called RNN/LSTM?
2. What is time_step(back_step), batch_size, feature, no of epoch, pipelines, dropout, learning rate, keep prob.,lstm_size(hidden_unit)?
3. Why should we scale the data? What type of scaling should be done on different type of data? Why data visualization is important?
4. How much should we split in test vs training data? Reshaping of training data according to input_shape to use in network?
5. Different aspect of network-->dropout, stateful/stateless,optimizer,error, return_sequence,dense?
6. What should we do in case of overfitting and underfit? Is shuffling neccesary?
7. how much error is good error?
"""
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input/nifty50intradayjan2018/"))#['AMBUJACEM.txt', 'CANBK.txt', 'CIPLA.txt']

# fix random seed for reproducibility.
np.random.seed(1)

# IMPORTANT STEPS:
# 1. import raw data
# 2. extract features
# 3. normalize/scale the data
# 4. split the date into train and test
# 5. find output and input. Define them in proper shape
# 6. apply DNN
# 7. make prediction and calculate loss
# 8. plot your data
# if you observe the steps, you will find that only step-6 is where we talk about neural network otherwise,
# everything else is simply data manipulation.

stock_symbol_list=['AMBUJACEM','CANBK','CIPLA']
stock_count=len(stock_symbol_list)

def get_stocks_data(stock_sym):
    stock_df = pd.read_csv('../input/nifty50intradayjan2018/' +stock_sym +'.txt')
    stock_df.columns =  ['sym','date','time','open','high','low','close','vol']
    return stock_df[0:8350]

def cleaned_stock_data(stock_df,feature_list):
    stock_df =stock_df[feature_list]
    return stock_df

def normalize_stock(normalize, stock_df):
    if normalize==1:
        # write down your customized scaling procedure
        print("you haven't written any customized scaling procedure")
    elif normalize==2:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_stock_df = scaler.fit_transform(stock_df)
        return scaled_stock_df,scaler
    elif normalize==3:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_stock_df = scaler.fit_transform(stock_df)
        return scaled_stock_df

# creating input,label/target
def create_input_output(stock_df,back_step):
    x= np.array([stock_df[i:i+back_step] for i in range(len(stock_df)-back_step)])
    y= np.array([stock_df[i+back_step] for i in range(len(stock_df)-back_step)])
    return x,y

# splitting (Train vs Test)
def split_test_train(x,y,split_ratio):
    train_size=int(len(x)*split_ratio)
    train_x,test_x=x[:train_size],x[train_size:]
    train_y,test_y=y[:train_size],y[train_size:]
    return train_x,train_y,test_x,test_y

def train(trainX, trainY,testX,lstm_size,n_batch,back_step,feature_num,epoch_no):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(lstm_size,return_sequences=True, stateful=False, batch_input_shape=(n_batch,back_step,feature_num)))# change the stateful to stateless
    model.add(Dropout(0.2))
    model.add(LSTM(10, stateful=False))
    model.add(Dropout(0.2))# do we need to add activation layer
    model.add(Dense(feature_num))
    start=time.time()
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(epoch_no):
        model.fit(trainX, trainY, batch_size= n_batch, shuffle=False,verbose=2)
        # model.reset_states()
    print ('compilation time : ', time.time() - start)
    
    # make predictions
    trainPredict = model.predict(trainX, batch_size=n_batch)
    # model.reset_states()
    testPredict = model.predict(testX,batch_size=n_batch)

    return trainPredict,testPredict

# invert predictions
def invert_predictions(trainPredict,testPredict,trainY,testY,scaler,normalize):
    if normalize==1:
        print("write here your own inverse transform function")
    elif normalize==2:
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform(trainY)
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform(testY)
        return trainPredict,testPredict,trainY,testY

# calculate root mean squared error
def calculate_error(trainPred,testPred,trainy,testy,i):
    trainScore = math.sqrt(mean_squared_error(trainy[:,0], trainPred[:,0]))# if we have more than one feature
    print('Train Score of'+stock_symbol_list[i] +' : %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testy[:,0], testPred[:,0]))
    print('Test Score of'+stock_symbol_list[i] +': %.2f RMSE' % (testScore))

def plot(trainPredict,testPredict,trainY,testY,stock_df,back_step,scaler):
    # f,ax = plt.subplots(figsize=(50, 50))
    for i in range(stock_count):
        f,ax = plt.subplots(figsize=(50, 50))
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(stock_df)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[back_step:len(trainPredict[i])+back_step, :] = trainPredict[i]
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(stock_df)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict[i])+(back_step):len(stock_df), :] = testPredict[i]
       # original stock plot
        stockPlot = np.empty_like(stock_df)
        stockPlot[:, :] = np.nan
        stockPlot[back_step:len(trainY[i])+back_step, :] = trainY[i]
        stockPlot[len(trainY[i])+(back_step):len(stock_df), :] = testY[i]
        # plot baseline and predictions
        plt.plot(stockPlot)
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.legend()
        plt.show()
    # plt.show()

def main():
    
    normalize = 2
    back_step=60
    feature_list=['close', 'vol']# could add more features
    feature_num = len(feature_list)
    split_ratio=0.3
    epoch_no=1 #no. of iteration
    n_batch=stock_count
    lstm_size=128
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    scaler_list=[]

    for i in range(stock_count):
        stock_df= get_stocks_data(stock_symbol_list[i])
        stock_df= cleaned_stock_data(stock_df,feature_list)
        stock_df,scaler= normalize_stock(normalize,stock_df)
        x,y = create_input_output(stock_df,back_step)
        train_x,train_y,test_x,test_y=split_test_train(x,y,split_ratio)
        trainX += list(train_x)
        trainY += list(train_y)
        testX += list(test_x)
        testY += list(test_y)
        scaler_list += [scaler]

    trainX= np.array(trainX)
    trainX.shape
    trainY= np.array(trainY)
    testX= np.array(testX)
    testY= np.array(testY)
    
    # our input shape=(7461, 60, 2)
    # train the model
    trainPredict,testPredict = train(trainX, trainY,testX,lstm_size,n_batch,back_step,feature_num,epoch_no)
    trainPred=[]
    testPred=[]
    trainy=[]
    testy=[]
    for i in range(stock_count):
        testPred += [testPredict[(i*(len(testPredict)//3)):(len(testPredict)//3)*(i+1)]]
        trainPred += [trainPredict[(i*(len(trainPredict)//3)):(len(trainPredict)//3)*(i+1)]]
        trainy +=[trainY[(i*(len(trainY)//3)):(len(trainY)//3)*(i+1)]]
        testy +=[testY[(i*(len(testY)//3)):(len(testY)//3)*(i+1)]]
        trainPred[i],testPred[i],trainy[i],testy[i] = invert_predictions(trainPred[i],testPred[i],trainy[i],testy[i],scaler_list[i],normalize)
        calculate_error(trainPred[i],testPred[i],trainy[i],testy[i],i)
        # 1. when stateful=True
        # Train Score ofAMBUJACEM : 2.93 RMSE Test Score ofAMBUJACEM: 6.59 RMSE 
        # Train Score ofCANBK : 7.12 RMSE Test Score ofCANBK: 12.92 RMSE 
        # Train Score ofCIPLA : 4.58 RMSE Test Score ofCIPLA: 9.38 RMSE
        # 2. when stateful = False
    
    # TODO: plot the graph corretly
    plot(trainPred,testPred,trainy,testy,stock_df,back_step,scaler_list)
    
    
    
if __name__=='__main__':
    main()


# I think, from these two versions, you have a basic understanding of LSTM/time-series.
# next we will add --
# 1. embedding vector, learing rate parameter
# 2. tensorflow
# 3. model which is constantly learning from oncoming data points
# 4. save the model in file for future use




