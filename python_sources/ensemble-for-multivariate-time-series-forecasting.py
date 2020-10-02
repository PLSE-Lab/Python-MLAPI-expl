#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# Welcome to this notebook on multiple Machine Learning and Statistical methods to tackle the task of predicting future oil prices, or more in general to forecast on multivariate time series.
# 
# The target variable for the models, however, will not be the oil prices, but the first order difference. This is done to remove the trend from the series, as it stabilizes the mean. A very useful analysis of this was done in another notebook which is linked below. At the end, though, we will undo the transformation and get the final results in the same scale.
# 
# Before the modelling aspect, we will use images, the covid data and stock and currency prices. Although NTL images may not be correlated with the current task, I think it's still valuable to show a couple of way to integrate them into an eventual model. Thus, the task of choosing which features are relevant will be left for another time. ;)
# 
# For the current task, 3 different ML approaches(XGBoost, Support Vector Machine (SVM) and Long Short Term Memory (LSTM)) will be shown and one statistical parametric approach (Autoregressive Integrated Moving Average (ARIMA) model).
# 
# I would also like to thank the work of other people that shared their notebooks, if you find any interesting content, **please remember to upvote the useful notebooks!**
# 
# [https://www.kaggle.com/snndemirhan/stock-prices-vs-oil-prices](https://www.kaggle.com/snndemirhan/stock-prices-vs-oil-prices)
# 
# [https://www.kaggle.com/sixxx6/notes-on-time-series-analysis](https://www.kaggle.com/sixxx6/notes-on-time-series-analysis)

# # **Index**
# 
# * [Loading packages](#packages)
# * [Data and Plots](#dataandplots)
# * [Getting Features](#gettingfeatures)
# * [XGBoost](#xgboost)
# * [SVM](#svm)
# * [LSTM](#lstm)
# * [ARIMA](#arima)
# * [Ensembling](#ensemblingresults)
# * [Final Comments](#finalcomments)

# ## **Loading packages**
# <div id="packages">
# </div>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image
from tqdm import tqdm 
import time
import cv2
import datetime
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
get_ipython().system('pip install yfinance')
import yfinance as yf
from sklearn.svm import SVR
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.applications.vgg16 import VGG16
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


# ## **Data and Plots**
# <div id="dataandplots">
# </div>

# In[ ]:


covid_data = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv")
ts = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-01-02_To2020-06-08.csv")
ts["Diff"] = ts["Price"].diff().dropna()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes[0].plot(ts["Price"])
axes[0].set_title("Oil Price")
axes[1].plot(ts["Diff"])
axes[1].set_title("Oil Price Differences")
fig.tight_layout()


# As expected the original time series has very strong external trends, which are very difficult for the models to pick up, while the differenced time series is centered around 0 and stationary. This time series is much more easy to model, and to check it, I encourage you to try in the later parts of the notebook, to change the target variable to the original price. The results will vary greatly!
# 
# Next let's start working on the other features our models will use to help predict the changes to oil prices.

# In[ ]:


f = plt.figure(figsize=(24, 16))

for i in range(0,6):
    im = image.load_img('/kaggle/input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/Italy-20200' + str(i+1) + '01.tif', color_mode='grayscale').convert("L")
    it = np.array(im)
    f.add_subplot(3, 2, i+1)
    plt.imshow(it)
    
plt.show()


# The images change greatly over time, in fact for the first day of each month, we can look at how the images of Italy get darker until March, and in April and May more lights are present. It should also be noted that sme images are whited out because of the presence of clouds.
# 
# 
# The next step is to feature engineer the images into usable features by the models. Two ways are shown below, the one commented out utilizes a CNN pretrained model (VGG16), and the flattened output of the last layer of the model is used to create the features. Overall it creates 25088 features for each image.
# 
# The other method involves getting the images' brightness (values greater than 220, this value is not set though), and contrast (difference between max pixel value and min pixel value in the image).
# 
# The images are also rescaled to 224x224, specifically for the VGG model input, but as they have different shape, it may still be useful to do it for the brightness and contrast features.
# The other functions allow us to assign the image to the correct data in the dataset.
# 
# 
# As the VGG method takes some time (and spoiler: it doesn't change much), we will use brightness and contrast as features.

# In[ ]:


# model = VGG16(weights='imagenet', include_top=False)

def get_datetime_img(imgstr):
    date = datetime.datetime(int(imgstr[-12:-8]),int(imgstr[-8:-6]),int(imgstr[-6:-4]))
    return date
                                 
def get_datetime_dtset(i):
    date = datetime.datetime(int(i.split("-")[0]),int(i.split("-")[1]),int(i.split("-")[2]))
    return date

def get_img_features(img_):
    img = image.load_img(img_, target_size=(224, 224)) 
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x) 
    features = model.predict(x) 
    # print(features.shape)
    features_reduce = features.flatten() 
    return features_reduce

def get_cont_bright(img_):
    img = image.load_img(img_, target_size=(224, 224),color_mode='grayscale') 
    img = np.array(img)
    bright = np.sum(img>220)
    cont = np.amax(img)-np.amin(img)
    return bright, cont
    
def get_img_dataset():
    img_features = covid_data["Date"]
    for dtset in ["Italy", "USA", "China"]: #, "France"]:
        feature_list = [dtset+str(x) for x in ["Brightness","Contrast"]] # range(0,25088)
        tmp = pd.DataFrame(0, index=pd.Series(range(0,110)), columns=feature_list)
        img_features = pd.concat([img_features, tmp], axis=1)
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in sorted(filenames):
                if dtset in filename and ".tif" in filename:
                    # feats = get_img_features(os.path.join(dirname, filename))
                    bright, cont = get_cont_bright(os.path.join(dirname, filename))
                    for j in img_features["Date"]:
                        if j == "2019-12-31" and "20200101" in filename:
                            # img_features.loc[img_features.Date==j,feature_list] = feats
                            img_features.loc[img_features.Date==j,dtset+"Brightness"] = bright
                            img_features.loc[img_features.Date==j,dtset+"Contrast"] = cont
                        if get_datetime_img(filename) == get_datetime_dtset(j):
                            # img_features.loc[img_features.Date==j,feature_list] = feats
                            img_features.loc[img_features.Date==j,dtset+"Brightness"] = bright
                            img_features.loc[img_features.Date==j,dtset+"Contrast"] = cont
        print("Finished:",  dtset)
    return img_features


# In[ ]:


train_ft = get_img_dataset()


# In[ ]:


# Plotting extracted features
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

train_ft.plot(kind='line',x='Date',y='ItalyBrightness', color='red', ax=ax[0])
train_ft.plot(kind='line',x='Date',y='USABrightness', color='blue', ax=ax[0])
train_ft.plot(kind='line',x='Date',y='ChinaBrightness', color='green', ax=ax[0])

train_ft.plot(kind='line',x='Date',y='ItalyContrast', color='red', ax=ax[1])
train_ft.plot(kind='line',x='Date',y='USAContrast', color='blue', ax=ax[1])
train_ft.plot(kind='line',x='Date',y='ChinaContrast', color='green', ax=ax[1])

plt.show()


# There are some peaks in brightness simultaneously all over the world every month. It would be interesting to try to understand why, but it falls out of current analysis. As for the contrast, it is almost always 255, except for one day in China. This feature is most likely irrelevant.

# ## **Getting Features**
# <div id="gettingfeatures">
# This is the same procedure as the notebook mentioned at the beginning does. Only a few other indeces where added like the VIX, a volatility index and a few other currencies' exchange rates.
# </div>

# In[ ]:


Yahoo_indeces=["^GSPC","^DJI","NQ=F","^FTSE","^GDAXI","^ISEQ","^N225","^XU100",
               "^FCHI","IMOEX.ME","^OMX","^OSEAX","XIU.TO","^BVSP","^MXX","^BSESN","^SSE50",
               "^KS11","^NZ50","^AXJO","^BFX","^ATX","^PSI20","BTCUSD=X","EURUSD=X",
              "MSFT","AAPL","AMZN","FB",
        "BLK","JNJ","V","PG","UNH",
        "JPM","INTC","HD","MA","VZ",
        "PFE","T","MRK","NVDA","NFLX","DIS",
        "CSCO","PEP","XOM","BAC","WMT","ADBE","CVX","BA","KO","CMCSA","ABT","WFC",
         "BMY","CRM","AMGN","TMO","LLY","COST","MCD","MDT","ORCL","ACN","NEE","NKE","UNP","AVGO","PM","IBM","LMT","QCOM","DAL",
              "ETHUSD=X", "JPY=X","GBPUSD=X","VXX","^VIX"]


Stock_indeces=["S&P500","DowJones","NASDAQ100","FTSE100","DAX","ISEQ","NIKKEI225","BIST100",
               "CAC40","MOEX","OMXS30","Oslo BorsAll-Share","TSX","IBOVESPA","IPCMexico","SHANGHAI50",
               "SENSEX","KOSPI","NZX50","ASX200","BEL20","ATX","PSI20","BTC/USD","EUR/USD",
              "Microsoft Corporation","Apple Inc.","Amazon","Facebook","BlackRock Inc.","Johnson & Johnson",
             "Visa Inc.","Procter & Gamble","UnitedHealth Group","JPMorgan",
            "Intel Corporation","Home Depot Inc.","Mastercard","Verizon Communications Inc.","Pfizer","AT&T Inc.","Merck & Co. Inc",
             "NVIDIA Corporation","Netflix Inc.","Walt Disney Company","Cisco Systems Inc.",
            "PepsiCo Inc.","Exxon Mobil Corporation","Bank of America Corp","Walmart Inc.","Adobe Inc.","Chevron Corporation",
             "Boeing Company","Coca-Cola Company","Comcast Corporation Class A","Abbott Laboratories","Wells Fargo & Company",
            "Bristol-Myers Squibb Company","Salesforce","Amgen Inc.","Thermo Fisher Scientific Inc.","Eli Lilly and Company",
             "Costco Wholesale Corporation","McDonald's Corporation","Medtronic Plc","Oracle Corporation","Accenture Plc Class A"
             ,"NextEra Energy Inc.",
            "NIKE Inc. Class B","Union Pacific Corporation","Broadcom Inc.","Philip Morris International","International Business Machines Corporation"
             ,"Lockheed Martin Corporation","QUALCOMM Incorporated","Delta Air","Ethereum-USD","Yen-USD","Pound-USD",
              "VIX-1Month","VIX"]

for i in range(len(Stock_indeces)):
    data = yf.download(Yahoo_indeces[i], start="2019-01-02", end="2020-08-31")
    stock_hist=pd.DataFrame(data[['Close','Volume']])
    stock_hist['Stock_indeces']=Stock_indeces[i]
    if i == 0:
        stock_history=stock_hist
    else:
        stock_history=pd.concat([stock_history,stock_hist])


# In[ ]:


stock_history['date']=stock_history.index
country_indeces=stock_history.pivot_table(index='date', columns='Stock_indeces', values='Close',fill_value=0)
country_indeces['Date']=country_indeces.index
country_indeces=country_indeces.reset_index(drop=True, inplace=False)
country_indeces['Date']=country_indeces['Date'].dt.strftime('%Y-%m-%d') #date column in downloaded data is not in wanted type.So I changed its type in here


# Now that we have all our features and datasets in order, we can merge them all through the date column, and then remove it for the modelling.

# In[ ]:


# Using covid world cases and deaths as features as well
data = covid_data[["Price", "Date", 'World_total_cases', 'World_total_deaths']]
data = pd.merge(data, country_indeces, on='Date')
data = pd.merge(data, train_ft, on="Date")

data["Price"] = data["Price"].diff()

data = data.drop("Date", axis=1) 

data.head()


# ## **XGBoost**
# <div id="xgboost">
# First we will scale the data to 0-1, to improve the[](http://) model performance and then we will divide into train and test set, with about a 75%-25% split. The series_timedep function also allows us to add temporal dependence into the data by adding columns at previous time lags. For now we will use the same time values only to predict.
#   <br>
#   XGBoost is one of the most used algorithms for ML, it can handle any type of numerical data, categorical data if encoded as number and  NA values, moreover it is very fast. Hyperparameter tuning for this model, such as grid search was not performed.
# </div>

# In[ ]:


def series_timedep(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
values = data
# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_timedep(scaled, 0, 1)
train_x, train_y, test_x, test_y = reframed.iloc[:81, 1:].values, reframed.iloc[:81, 0].values, reframed.iloc[81:, 1:].values, reframed.iloc[81:, 0].values


# In[ ]:


model_xg = xgb.XGBRegressor(colsample_bytree=0.5,
                 gamma=0,                 
                 learning_rate=0.005,
                 max_depth=5,
                 min_child_weight=0.25,
                 n_estimators=10000,                                                                    
                 reg_alpha=1,
                 reg_lambda=1,
                 subsample=1,
                 seed=777) 

model_xg.fit(train_x, train_y)


# In[ ]:


# make a prediction
yhat_xg = model_xg.predict(test_x)
yhat_xg = yhat_xg.reshape((len(yhat_xg), 1))
# invert scaling for forecast
inv_yhat_xg = np.concatenate((yhat_xg, test_x), axis=1)
inv_yhat_xg = scaler.inverse_transform(inv_yhat_xg)
inv_yhat_xg = inv_yhat_xg[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_x), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat_xg))
print('Test RMSE: %.3f' % rmse)


# ## **Support Vector Regression**
# <div id="svm">
# SVMs are mostly used for classification problems, however they can also be used for regression. Different kernel types were tried on the data, but the standard radial basis kernel had the best results. Other hyperparameters were not tested.
# </div>

# In[ ]:


model_svm = SVR(kernel='rbf', C=1e5, gamma="scale")

model_svm.fit(train_x, train_y)


# In[ ]:


# make a prediction
yhat_svm = model_svm.predict(test_x)
yhat_svm = yhat_svm.reshape((len(yhat_svm), 1))
# invert scaling for forecast
inv_yhat_svm = np.concatenate((yhat_svm, test_x), axis=1)
inv_yhat_svm = scaler.inverse_transform(inv_yhat_svm)
inv_yhat_svm = inv_yhat_svm[:, 0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat_svm))
print('Test RMSE: %.3f' % rmse)


# ## **LSTM**
# <div id="lstm">
# LSTMs are a kind of neural network, particularly RNNs, which can model long-term dependency. As such they are widely used for time series forecasting, but also in other settings (NLP, Speech Recognition, etc.).
# </div>

# In[ ]:


# reshape input to be 3D [samples, timesteps, features]
LSTMtrain_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
LSTMtest_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(LSTMtrain_x.shape, train_y.shape, LSTMtest_x.shape, test_y.shape)


# In[ ]:


# design network
dropout = 0.1
model_lstm = Sequential()
model_lstm.add(LSTM(100, return_sequences=True, input_shape=(LSTMtrain_x.shape[1], LSTMtrain_x.shape[2]), dropout=dropout, recurrent_dropout=dropout))
model_lstm.add(LSTM(50, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))
model_lstm.add(LSTM(25, dropout=dropout, recurrent_dropout=dropout))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer='adam')


# In[ ]:


# fit network
history = model_lstm.fit(LSTMtrain_x, train_y, epochs=1000, validation_data=(LSTMtest_x, test_y), batch_size=8, verbose=0, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# In[ ]:


# make a prediction
yhat_lstm = model_lstm.predict(LSTMtest_x)
# invert scaling for forecast
inv_yhat_lstm = np.concatenate((yhat_lstm, test_x), axis=1)
inv_yhat_lstm = scaler.inverse_transform(inv_yhat_lstm)
inv_yhat_lstm = inv_yhat_lstm[:, 0]
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat_lstm))
print('Test RMSE: %.3f' % rmse)


# ## **ARIMA**
# <div id="arima">
# ARIMA is one of the basic tools for time series analysis and has a long history of work behind it, but it is still very much used. ARIMA is a parametric modelling approach where the current value at time t is modelled as a function of past values and linear combination of error terms. The integration part means the differencing that we already performed.
#   <br>
#   The plots of the Autocorrelation function (acf) and Partial Autocorrelation function (pacf) help us in understanding the structure of the model based on the data.
# </div>

# In[ ]:


#plot acf and pacf
plot_acf(ts["Diff"][-1000:], lags=50)
plot_pacf(ts["Diff"][-1000:], lags=50)
plt.show()


# In this particular case, as the model needs more data to work properly, we took the last 1000 time values, and looked at the first lags values for acf and pacf. The first plot shows that there is a strong AR component in the model, as there is dependency on the previous values. the pacf plot shows also that a weak MA component is present.
# 
# The model results below show that our assumptions were correct and the presence of both components is statistically significant.

# In[ ]:


model_arima = ARIMA(ts["Diff"][-1000:].to_numpy(), order=(1,0,1))
model_arima = model_arima.fit()
print(model_arima.summary())


# In[ ]:


# plot residual errors
residuals = pd.DataFrame(model_arima.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# The plot of the residuals indicates that the model fits that data quite well, as it is Gaussian distributed and centered around 0, but the presence of some outlier is quite obvious. In very volatile time series such as oil prices, this is to be expected.
# 
# Below, we model the data to obtain one-step predictions, while basically using the previous predictions as a means to convey external information to the model. If that would not have been done, as time passes the fit of this particular model would be very poor.

# In[ ]:


ts_train = ts["Diff"][-1000:].to_numpy()
yhat_arima = []

for i in range(len(inv_y)):
    model_arima1step = ARIMA(ts_train, order=(1,0,1)).fit(disp=-1)
    yhat_arima1step = model_arima1step.forecast(1)[0]
    yhat_mix1step = (yhat_arima1step + inv_yhat_lstm[i] + inv_yhat_svm[i] + inv_yhat_xg[i])/4
    ts_train = np.append(ts_train, yhat_mix1step, axis=0)
    yhat_arima.append(yhat_mix1step)

yhat_arima = np.array(yhat_arima).reshape((len(yhat_svm), ))
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, yhat_arima))
print('Test RMSE: %.3f' % rmse)


# ## **Ensembling results**
# <div id="ensemblingresults">
# Simply by taking the mean of all our predictions. A different weighting scheme can be thought for this step.
# </div>

# In[ ]:


ensemble_preds = (yhat_arima + inv_yhat_lstm + inv_yhat_svm + inv_yhat_xg)/4
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, ensemble_preds))
print('Test RMSE: %.3f' % rmse)


# To get the actual RMSE on daily oil price, first we will check that our function to undo the differencing actually works, and the we will apply it to the predictions and check the final results.

# In[ ]:


base = covid_data["Price"][0]
orig = []

for i in range(1, len(covid_data["Price"])+1):
    orig.append(round(base + sum(data["Price"][1:i]),6))

# check there is no mistake in undoing the differencing
print((orig == round(covid_data["Price"], 6)).all())


# In[ ]:


def get_price(preds):
    base_test = covid_data["Price"][81]
    oil_preds = []
    for i in range(len(preds)):
        oil_preds.append(round(base_test + sum(preds[0:i]),6))
    return oil_preds

final_preds = get_price(ensemble_preds)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(orig[82:], final_preds))
print('Test RMSE: %.3f' % rmse)


# In[ ]:


# Our test data is the last month of available data

plt.plot(orig[82:], color="black", label="Oil Price")
plt.plot(final_preds, color="red", label="Ensemble Prediction")
plt.plot(get_price(yhat_arima), color="green", label="ARIMA Prediction")
plt.plot(get_price(inv_yhat_lstm), color="yellow", label="LSTM Prediction")
plt.plot(get_price(inv_yhat_xg), color="violet", label="XGBoost Prediction")
plt.plot(get_price(inv_yhat_svm), color="blue", label="SVM Prediction")
plt.legend()
plt.show()


# In[ ]:


sub = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/sampleSubmission.csv")
sub["Price"] = final_preds
sub.to_csv("sampleSubmission.csv",index=False)


# ## **Final Comments**
# <div id="finalcomments">
# The predictions of our models are very close to the actual ones, up until 10 days, after that they seem to predict a faster fall of the oil prices. A simple way to improve the results, would of course be a better choice of the features, like including stocks and commodities prices which relate very heavily to the WTI oil price (such as Brent, which basically follows the same distribution as WTI), and removing uncorrelated ones(image features, unrelated stock prices), but the aim of this notebook is to show some of the main tools for forecasting time series.  
#   <br>
#   As data is added, the performances of the models should improve, as right now the data is very few. If one decided not to use the covid and image data, a bigger dataset could be easily used as well.
# Also model finetuning for the ML models should be performed to improve their respective predictions.   
#   <br>
#   Some more updates may be coming in the next weeks, so stay tuned!
# </div>
