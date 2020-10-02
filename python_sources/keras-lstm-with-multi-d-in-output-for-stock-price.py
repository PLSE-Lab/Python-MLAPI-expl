#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# The purpose of this script is to show how we can apply the deep learning algorithm LSTM to predict stock price changes using Tensorflow/Keras framework. The script is derived from the excellent script hosted at https://www.kaggle.com/amarpreetsingh/stock-prediction-lstm-using-keras ("original script") with the following two major improvements:
#     
# * A more meaningful performance measure. The original script plots actual stock closing price with predicted closing price and shows they match extremely well, suggesting excellent prediction. IMHO, however, this is quite misleading. This is not the best way to reveal how good the LSTM model is at predicting stock price. The fact that the predicted stock price and the actual stock price curve very closely to each other on a serial plot does not mean the model has done a great job. A naive model that randomly generates a predictive stock price close to the previous day closing price might do just as well. This is because the stock price changes cover a wide range. If for the closing price today 20 you predict the closing price tomorrow to be 19 or 21, and for today's price 30 you predict tomorrow's price to be 29 or 31, ..., for 200 you predict to be 195 or 205, and so on and on, and you plot such data on a serial plot, you will see your predicted prices match the actual prices very well. However, you know this is not right, because stock price typically changes no more than 2~3%. A random predictor won't help you at all. Instead, we should focus the prediction of price changes. If we predict the stock price to rise tomorrow and it actually rises tomorrow, or if we predict the stock price to fall tomorrow and it actually falls tomorrow, then our prediction is a good prediction. If we have good prediction more often than we have bad predictions, then our prediction model is a good model.  To measure the performance of the predictive model, my personal preference is R-squared.
# 
# * Usage of multi-dimensional data as input or multiple features as predictors. The original script uses one feature -- "Close" price to predict one outcome -- "Close" price. For the purpose of showing how LSTM works, this might be OK. However, if you want to use LSTM to predict stock price changes to make money, this is certainly not enough. Besides the "Close" price, "Open", "High" and "Low" prices all offer valuable infomation. "Volume" is so important that it is well known for stock traders as an indicator to spot market bottom. There are also many other factors such as the moving averages, market panic index VIX, MACD, ... that are popular market status and change indicators. We want to be able to incorporate as many relevant predictors to our model as we like. So in this script, I would like to showcase how to use multiple predictors to predict stock price changes. Meanwhile, since we might want to predict multiple features (for instance, High & Low) altogether, I will show how this can be done using the python tensorflow/keras frameworks.

# **Methods and Results **
# 
# Now let's do the coding.

# In[ ]:


### import necessary packages
import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM,Dense
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import tensorflow.keras.callbacks as cb
from keras.models import load_model
from math import exp, log


# In[ ]:


### define some global parameters
# "look back" days. This is the number of historical days we use to predict tomorrow. Personally, I don't like the term "look back" because
# it is quite confusing. However, it seems that this term has become the convention in the field of CNN/RNN stock prediction, so I will use it
# here as well. In LSTM, this is the number of time steps.
LB = 10
# number of epochs. setting it too high might cause overfitting, setting it too low the model won't get enough training.
# so setting epochs to an adequate number is very important. How big this number should be is more an art. some people examine the
# test set performance metrics (such as loss, accuracy) during iterations of training and then decide how many epochs to go to. However,
# this approach peeks into the test set, and thus invalidates the purpose of the test set. Maybe RNN/CNN should internally divide its 
# training data into two parts, use one part for internal training, the other part for internal testing, or use something like Leave-one-out
# and find an appropriate number of epochs.
EPOCHS = 50

### load data file "DIA.csv"
data = np.array(pd.read_csv('../input/DIA.csv')).reshape(-1,7)
print("data shape = " + str(data.shape))
print(data[0:5]) # print the top 5 data lines
pv = data[:,(1,2,3,4,6)] # Prive & Volume. Only use Open, High, Low, Close, Volume columns for prediction
print(pv[0:5,]) # print the top 5 data lines


# Before we start training, we need to split our data into training set and testing set. A few things we need to be careful here:
# 
# * Do not use random sampling to split your data into training set and testing set. The daily stock prices are not independent from each other. If you know the stock information Monday and Wedesday last week (in your training set), you probably know something already about the stock price on Tuesday the same week (in your testing set). This is some kind of cheating. What we should do is to use a time point to separate the full dataset into training and testing datasets.
# 
# * Do not normalize or scale your full dataset before you split it into training set and testing set. I have watched several online tutorials using minmax scaling to the full dataset before splitting. This is some kind of cheating as well: when you use the min and max, you already know something about your testing set, because the min & max values are of all the data points. What we should do is to split the full dataset into training set (past) and testing test (future) with a time point. You can do whatever you want to do with the training set, but do not mess it with the testing set. For the testing test, though, you can use whatever information from the training set and do whatever you want (such as normalizing your prices by the last closing price). The past shall know nothing about the future, but the future can see the past.

# In[ ]:


### split the data into training set and testing (validating) set
splitIndex = int(pv.shape[0]*0.80)
print("total pv.length = " + str(pv.shape[0]) + ", splitIndex = " + str(splitIndex) + ", diff len = " + str(pv.shape[0]- splitIndex))
pv[0:5,]


# In[ ]:


#Create a function to process the data into LB day look back slices
def processData(data,lb):
    # X_orig: orignal X values
    # Y_orig: orignal Y values
    # X     : transformed X values as input
    # Y     : transformed Y values as predicted outcome (output)
    X_orig, Y_orig, X, Y = [], [], [], []
    
    for i in range(len(data) - lb - 1):
        xo1, x1 = [], []
        
        pmax = max(data[i:(i+lb),1]) # get the maximum price using the "High" column
        pmin = min(data[i:(i+lb),2]) # get the minimum price using the "Low" column
        pdiff = pmax - pmin
        vmax = max(data[i:(i+lb), 4]) # maximum volume
        vmin = min(data[i:(i+lb), 4]) # minimum volume
        vdiff = vmax - vmin

        for j in range(lb):
            # original, no scaling
            xo1.append(data[i + j,]) 
            # minmax scaling for prices (Open, High, Low, Close) and volume respectively
            x1.append(np.append((data[i + j, 0:4] - pmin) / pdiff, (data[i + j, 4] - vmin) / vdiff))

        # store the original data
        X_orig.append(xo1) 
        Y_orig.append(data[(i+lb),])
        # add a transformed model input entry, which is an array of "LB" elements of [Open, High, Low, Close, Volume] arrays.
        X.append(x1) 
        # add a transformed model output entry. The prices are normalized by the previous day close column (indexed 3)
        # and volume by the previous day volume
        y1 = []
        
        for j in range(4):
            y1.append(log(data[i+lb, j] / data[i+lb-1, 3])) # normalize by previous Close and log transform
        
        y1.append(log(data[i+lb, 4] / data[i+lb-1, 4])) # normalize by previous Volume and log transform
        Y.append(y1) 
        
    return np.array(X_orig), np.array(Y_orig), np.array(X), np.array(Y)

X_orig, Y_orig, X, Y = processData(pv,LB)
print("X rows: " + str(X.shape[0]))
print("Y rows: " + str(Y.shape[0]))
X_train_orig,X_test_orig = X_orig[:splitIndex],X_orig[splitIndex:]
Y_train_orig,Y_test_orig = Y_orig[:splitIndex],Y_orig[splitIndex:]
X_train,X_test = X[:splitIndex],X[splitIndex:]
Y_train,Y_test = Y[:splitIndex],Y[splitIndex:]

### do some sanity check
print("X_train len = " + str(len(X_train)) + ", Y_train len = " + str(len(Y_train)))
print("X_test len = " + str(len(X_test)) + ", Y_test len = " + str(len(Y_test)))

print("X_train_orig: " + str(X_train_orig[0:2]))
print("Y_train_orig: " + str(Y_train_orig[0:2]))
print("X_test_orig: " + str(X_test_orig[0:2]))
print("Y_test_orig: " + str(Y_test_orig[0:2]))

print("X_train:" + str(X_train[0:2]))
print("Y_train:" + str(Y_train[0:2]))
print("X_test:" + str(X_test[0:2]))
print("Y_test:" + str(Y_test[0:2]))


# In[ ]:


### Construct the model
model = Sequential()
model.add(LSTM(256,input_shape=(LB,5))) # note the values for "input_shape". 5 is the nubmer of features: Open, High, Low, Close, Volume
model.add(Dense(5))
model.compile(optimizer='adam',loss='mse', metrics=['mae', 'acc'])

### Reshape data for (Sample,Timestep,Features) 
X_train = X_train.reshape((X_train.shape[0],LB,5))
X_test = X_test.reshape((X_test.shape[0],LB,5))

#Fit model with history to check for overfitting
bestModelPath="DIA.LB" + str(LB) + "EPOCH{epoch:02d}.VAL_ACC{val_acc:.2f}.hdf5"
checkpoint = cb.ModelCheckpoint(bestModelPath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]
history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=64, validation_data=(X_test,Y_test) ,callbacks=callbacks_list, shuffle=False)


# In[ ]:


### plot the performance of the traing process
plt.plot(history.history['loss'], color='red')
plt.plot(history.history['val_loss'], color='green')


# Figure 1. Training loss (red) and validation loss (green) for each epoch.

# In[ ]:


### make predictions on the training and testing
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)


# In[ ]:


### define a function to reverse transform the predicted outcomes
def reverse(Xorig, Ypred):
    rst = [] # result

    for i in (range(len(Ypred))):
        pall, vall = [], []

        for j in range(LB):
            for k in range(4):
                pall.append(Xorig[i][j, k])    

            vall.append(Xorig[i][j, 4])

        pmax = max(pall)
        pmin = min(pall)
        pdiff = pmax - pmin
        vmax = max(vall)
        vmin = min(vall)
        vdiff = vmax - vmin

        orig1 = []

        for j in range(4):
            orig1.append(exp(Ypred[i][j]) * Xorig[i][LB-1,3])
            
        orig1.append(exp(Ypred[i][4]) * Xorig[i][LB-1, 4]) # volume
        rst.append(orig1)

    return np.array(rst)

### reverse the transforms
Y_train_pred_orig = reverse(X_train_orig, Y_train_pred)
Y_test_pred_orig = reverse(X_test_orig, Y_test_pred)


# In[ ]:


### plot out the predictions for Open, High, Low, Close, Volume respectively for the training & testing (validating) data sets
for i in range(5):
    plt.figure(i + 1)
    
    if i == 0:
        plt.title("Open")
    elif i == 1:
        plt.title("High")
        
    elif i == 2:
        plt.title("Low")
    elif i == 3:
        plt.title("Close")
    else: # i == 4
        plt.title("Volume")
        
    plt.plot(np.append(Y_train_orig[:,i], Y_test_orig[:,i]), color='red')
    plt.plot(np.append(Y_train_pred_orig[:,i], Y_test_pred_orig[:,i]), color='green')
    plt.axvline(x=splitIndex) # separate the training & testing time points


# Figure 2. The predicted and actual daily price/volume changes. Red is for training, green is for validating. The vertical line separates the training set from the validating set. 
# 
# As you can see above, at least for the Open, High, Low, Close price predictions, the actual prices and the predicted prices match almost perfectly well. Does this mean our predictive models have done very well? I would not think so. In following sections, I will show you why this is very misleading, and how we should better understand the performance of the predictive models.
# 
# Let's take a closer look at the first 100 predictions.

# In[ ]:


### plot just the predicted part for a closer look
NPOINTS = 100 # number of points to plot

### plot the comparision in price
print("Y_test: " + str(Y_test_orig[0:2]))
print("Y_test_pred: " + str(Y_test_pred_orig[0:2]))

for i in range(5):
    plt.figure(i+1)
    plt.figure(figsize=(20, 5))
    
    if i == 0:
        plt.title("Pred Open")
    elif i == 1:
        plt.title("Pred High")
        
    elif i == 2:
        plt.title("Pred Low")
    elif i == 3:
        plt.title("Pred Close")
    else: # i == 4
        plt.title("Pred Volume")
        
    #plt.figure(figsize=(15,10))
    plt.plot(Y_test_orig[0:NPOINTS, i], 'o', color='red')
    plt.plot(Y_test_pred_orig[0:NPOINTS, i], 'x', color='green')


# Figure 3.
# 
# Looking at plots above, we should check if a vertical pair of red and green dots go up and down together in relative to the previous red dot, and how much they deviate from the previous red dot. If you want, you can note down the number of pairs go in the same direction and the number of pairs in the opposite direction. If counting such numbers is too tedious to you, the code below plots the predictions in log transformed relative changes (positve meaning price increase, negative meaning price decrease), and count the numbers for you.

# In[ ]:


### compare the prediction in relative changes
# print("Y_test: " + str(Y_test[0:2]))
# print("Y_test_pred: " + str(Y_test_pred[0:2]))

for i in range(5):
    plt.figure(i+1)
    plt.figure(figsize=(16,4))
    
    if i == 0:
        msg = "Pred Open Change"
    elif i == 1:
        msg = "Pred High Change"
        
    elif i == 2:
        msg = "Pred Low Change"
    elif i == 3:
        msg = "Pred Close Change"
    else: # i == 4
        msg = "Pred Volume Change"
        
    plt.title(msg)
    plt.plot(Y_test[0:NPOINTS, i], 'o', color='red')
    plt.plot(Y_test_pred[0:NPOINTS, i], 'x', color='green')    
    plt.axhline(y=0) # separate the training & testing time points

    ### do some statistics
    topSame, topDiff, allSame, allDiff = 0, 0, 0, 0 # number of pairs in the same or different direction for the top NPOINTS points or all
    
    for j in range(NPOINTS):
        if Y_test[j, i] * Y_test_pred[j, i] >= 0:
            topSame += 1
        else:
            topDiff += 1
            
    for j in range(len(Y_test[:,i])):
        if Y_test[j, i] * Y_test_pred[j, i] >= 0:
            allSame += 1
        else:
            allDiff += 1
    
    print("\n" + msg)
    print("Number of points go in the same direction for the first " + str(NPOINTS) + " points: %d (%.2f%%)" % (topSame, 100 * topSame / (topSame + topDiff)))
    print("Number of points go in the diff direction for the first " + str(NPOINTS) + " points: %d (%.2f%%)" % (topDiff, 100 * topDiff / (topSame + topDiff)))
    print("Number of points go in the same direction for all " + str(len(Y_test[:,i])) + " points: %d (%.2f%%)" % (allSame, 100 * allSame / (allSame + allDiff)))
    print("Number of points go in the diff direction for all " + str(len(Y_test[:,i])) + " points: %d (%.2f%%)" % (allDiff, 100 * allDiff / (allSame + allDiff)))


# Figure 4.
# 
# If the pairs go in the same direction, it means the prediction is correct at least in terms of price change directions. If they go different directions, it means the prediction gets wrong in terms of price change directions. If a prediction model is good, it should make the green dots go in the same directions as the red dots more often than in different directions. As you can see above, based on the validation results from the test (validating) dataset, the predictions for Volume and High are the best -- the percentage of getting is bigger than 60% (it varies from script execution to script execution becase of the non-derministic nature of LSTM). This is awesome but far from perfect as some might have thought when looking at Figure 2 and seeing almost perfect matches of predicted and actual prices. As for closing price, our predictive power is almost zero because we get about 50% right and 50% wrong.
# 
# The above being said, I hope it makes sense to you that we should compare the predicted price/volume changes with the actual changes, but not the predicted price/volume with the actual price/volume.
# 
# 

# In[ ]:


### scatter plot prediction performance
from numpy.polynomial.polynomial import polyfit

for i in range(5):
    plt.figure(i+1)
    
    if i == 0:
        plt.title("Open")
    elif i == 1:
        plt.title("High")
        
    elif i == 2:
        plt.title("Low")
    elif i == 3:
        plt.title("Close")
    else: # i == 4
        plt.title("Volume")
        

    ### get the linear regression parameters
    b, m = polyfit(np.array(Y_test[:,i], dtype=float), np.array(Y_test_pred[:,i], dtype=float), 1)
    predReg = [0] * len(Y_test_pred[:,i]) # predicted based on regression

    predReg = b + m * Y_test[:,i]

    # draw the scatter plot
    plt.plot(Y_test[:,i], Y_test_pred[:,i], '.')
    plt.xlabel("actual change")
    plt.ylabel("predicted change")
    # draw the regression line
    plt.plot(np.array(Y_test[:,i], dtype=float), predReg, '-')


# Figure 5. Scatter plots of predicted price/volume changes versus actual price/volume changes. You can see that Volume has the highest predictability, High is predicted best among all prices, and Open/Low/Close are predicted pretty poorly.

# In[ ]:


# get the Rsq
from scipy import stats

### return the R-value (correlation coefficient) & p-value of the two input variables
def getRvalPval(var1, var2):
    slope, intercept, r_value, p_value, std_err = stats.linregress(var1, var2)

    return r_value, p_value
    
for i in range(5):
    if i == 0:
        title = "Open"
    elif i == 1:
        title = "High"
        
    elif i == 2:
        title = "Low"
    elif i == 3:
        title = "Close"
    else: # i == 4
        title = "Volume"
        
    print("\n" + title)
    r_value, p_value = getRvalPval(np.array(Y_test_orig[:,i], dtype=float), np.array(Y_test_pred_orig[:,i], dtype=float))
    print("Original Value R: %.3f\tp-value: %.2e" %(r_value, p_value))

    r_value, p_value = getRvalPval(np.array(Y_train[:,i], dtype=float), np.array(Y_train_pred[:,i], dtype=float))
    print("Training R: %.3f\tp-value: %.2e" %(r_value, p_value))

    r_value, p_value = getRvalPval(np.array(Y_test[:,i], dtype=float), np.array(Y_test_pred[:,i], dtype=float))
    print("Testing R: %.3f\tp-value: %.2e" %(r_value, p_value))


# Based on the R-value & p-value, we can see quantitatively that our models can predict reasonably well on Volume and High, but virtually nothing on Open, Low, and Close.

# **Conclusion and Discussion**
# 
# Now we have discussed in details how we can do multi-dimensional LSTM for stock price prediction using the Tensorflow/Keras framework. There is a lot you can do to improve the predictive power of the model, such as changing the look back days to include more historical information, add market panix index VIX as a predictor, focus on predicting High and Low and forget about Open and Close, and so on and so forth. This script is only intended to get you started with multi-dimensinal i/o LSTM, so feel free to change it to suit your own needs.
# 
# Should you have any suggestions or questions, please do not hesitate to contact me at moushengxu@gmail.com or simply reply here.
# 
