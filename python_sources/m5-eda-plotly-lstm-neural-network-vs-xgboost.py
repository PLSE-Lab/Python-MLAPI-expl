#!/usr/bin/env python
# coding: utf-8

# **Please upvote if you find interesting my Kernel :)**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_sales = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv", index_col="item_id")
df_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
df_calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", index_col = "date")
first_date = "d_1"
last_date = "d_1913"
#dates = pd.DataFrame([df_calendar.index, df_calendar.d, df_calendar.weekday, df_calendar.month, df_calendar.year]).transpose()
#dates.columns = ["Date", "d", "weekday", "month", "year"]
#dates_ = dates[["Date", "d"]]


# # FEATURE SELECTION

# In[ ]:


from sklearn.preprocessing import LabelEncoder

nonservingcols = ["wm_yr_wk", "wday"]
dates = df_calendar.drop(nonservingcols, axis = 1)
dates["Date"] = dates.index
dates.index = dates["d"]
dates = dates.fillna(0)

categorical_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
my_labeler = LabelEncoder()
for i in categorical_cols:
    dates[i] = my_labeler.fit_transform(dates[i].astype("str"))


# In[ ]:


def createFeatures(series):
    serie = series.transpose()
    df_products = pd.merge(serie, dates, left_index= True, right_index = True)
    df_products["Date"] = pd.to_datetime(df_products["Date"])
    df_products["quarter"] = df_products["Date"].dt.quarter.astype("uint8")
    df_products["Month"] = df_products["Date"].dt.month.astype("uint8")
    df_products["Year"] = df_products["Date"].dt.year.astype("uint8")
    df_products["dayofyear"] = df_products["Date"].dt.dayofyear.astype("uint8")
    df_products["dayofweek"] = df_products["Date"].dt.dayofweek.astype("uint8")
    df_products.index = df_products.Date
    df_products = df_products.drop(["Date", "weekday", "month", "d"], axis= 1)
    return df_products

def crearseries(data):
    a = data[0]
    df = df_sales.copy()
    first_date = "d_1"
    last_date = "d_1969"
    if a:
        final_df = df.groupby(data).sum()
        lnn = list()
        try:
            for i in final_df.index:
                nn = "_".join([i[0], i[1]])
                lnn.append(nn)
                final_df["final_name"] = lnn
                final_df.index = final_df["final_name"]
                final_df = final_df.drop("final_name", axis =1)
        except:
            pass
        return final_df
    else:
        df = df.loc[:,first_date:last_date]
        final_df = pd.Series(df.sum(axis = 0))
        return final_df


# In[ ]:


df_prices_stats = df_prices.loc[:,["item_id", "sell_price"]]
df_prices_stats = df_prices_stats.groupby("item_id").sell_price.agg([min, max, "mean"])
df_estados = df_sales.loc[:,"state_id":last_date]
df_estados = df_estados.groupby("state_id").sum()
df_estados_Q = pd.DataFrame(df_estados.sum(axis=1))


# In[ ]:


df_estados = df_estados.transpose()
df_estados= pd.merge(df_estados, dates, left_index= True, right_index = True)
df_estados.Date = pd.to_datetime(df_estados.Date)
df_estados.head()


# In[ ]:


df_sales_tot_ = pd.DataFrame(df_sales.loc[:, first_date:last_date].sum(axis=1))
df_sales_tot_a = df_sales_tot_.groupby(df_sales_tot_.index).sum()
df_sales_tot = pd.merge(df_sales_tot_a, df_prices_stats, right_index = True, left_index=True)
df_sales_tot["Total"] = df_sales_tot.iloc[:,0]*df_sales_tot.loc[:,"mean"]
df_sales_tot = df_sales_tot.Total.sort_values()
df_sales_tot_b = pd.DataFrame(df_sales_tot[-10:])
df_sales_tot_l = pd.DataFrame(df_sales_tot[:10])


# # EDA - Exploratory Data Analysis

# In[ ]:


import plotly.express as px
import plotly.graph_objects as go
fig_a = px.bar(df_sales_tot_b, x = df_sales_tot_b.index, y = df_sales_tot_b.iloc[:,0])
fig_a.show()

pms = df_sales_tot_b.index
pms_a = pd.DataFrame(df_sales.loc[pms, first_date:last_date])
pms_ = pms_a.groupby(pms_a.index).sum().transpose()
pms_d = pd.merge(pms_, dates, left_index = True, right_index = True)
#pms_d.head()

fig_c = go.Figure()
fig_c.add_trace(go.Scatter(x=pms_d.Date, y=pms_d.iloc[:,9], name=pms_d.columns[9],
                         line_color='deepskyblue'))
fig_c.add_trace(go.Scatter(x=pms_d.Date, y=pms_d.iloc[:,5], name=pms_d.columns[5],
                         line_color='dimgray'))
fig_c.add_trace(go.Scatter(x=pms_d.Date, y=pms_d.iloc[:,4], name=pms_d.columns[0],
                         line_color='red'))
fig_c.update_layout(title_text='Articles analysis timeseries',
                  xaxis_rangeslider_visible=True)
fig_c.show()


# In[ ]:


fig_l = px.bar(df_sales_tot_l, x = df_sales_tot_l.index, y = df_sales_tot_l.iloc[:,0])
fig_l.show()
pmsmin = pd.DataFrame(df_sales_tot[:10]).index
pms_a_min = pd.DataFrame(df_sales.loc[pmsmin, first_date:last_date])
pms_min = pms_a.groupby(pms_a_min.index).sum().transpose()
pms_d_min = pd.merge(pms_min, dates, left_index = True, right_index = True)
#pms_d.head()

fig = go.Figure()
fig.add_trace(go.Scatter(x=pms_d_min.Date, y=pms_d_min.iloc[:,0], name=pms_d_min.columns[0],
                         line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=pms_d_min.Date, y=pms_d_min.iloc[:,4], name=pms_d_min.columns[4],
                         line_color='dimgray'))
fig.add_trace(go.Scatter(x=pms_d_min.Date, y=pms_d_min.iloc[:,5], name=pms_d_min.columns[5],
                         line_color='red'))
fig.update_layout(title_text='Articles analysis timeseries',
                  xaxis_rangeslider_visible=True)
fig.show()


# In[ ]:


fig = px.choropleth(locations=df_estados_Q.index, locationmode="USA-states", color=df_estados_Q.iloc[:,0], scope="usa")
fig.show()

#fig = px.bar(df_estados_Q, x= df_estados_Q.index, y = df_estados_Q.iloc[:,0], color = df_estados_Q.index)
#fig.show()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=df_estados.Date, y=df_estados['CA'], name="California",
                         line_color='deepskyblue'))
fig.add_trace(go.Scatter(x=df_estados.Date, y=df_estados['TX'], name="Texas",
                         line_color='dimgray'))
fig.add_trace(go.Scatter(x=df_estados.Date, y=df_estados['WI'], name="Wisconsin",
                         line_color='red'))
fig.update_layout(title_text='State analysis timeseries',
                  xaxis_rangeslider_visible=True)
fig.show()


# We can observe some peaks along the time series, probably these are referring to weekends and months. So, let's plot it:

# In[ ]:


df_weekday = df_estados.loc[:,["CA", "TX", "WI", "weekday"]]
df_weekday = df_estados.groupby("weekday").sum()
df_weekday.sort_values("CA")


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(x=df_weekday.index,
                y=df_weekday.CA,
                name='California Stores',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=df_weekday.index,
                y=df_weekday.TX,
                name='Texas Stores',
                marker_color='rgb(55, 83, 220)'
                ))
fig.add_trace(go.Bar(x=df_weekday.index,
                y=df_weekday.WI,
                name='Wisconsin Stores',
                marker_color='rgb(55, 100, 30)'
                ))

fig.update_layout(
    title="Sells by State stores",
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Quantity (Products)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# In[ ]:


df_month = df_estados.loc[:,["CA", "TX", "WI", "month"]]
df_month = df_estados.groupby("month").sum()
df_month.sort_values("CA")
fig = go.Figure()
fig.add_trace(go.Bar(x=df_month.index,
                y=df_month.CA,
                name='California Stores',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=df_month.index,
                y=df_month.TX,
                name='Texas Stores',
                marker_color='rgb(55, 83, 220)'
                ))
fig.add_trace(go.Bar(x=df_month.index,
                y=df_month.WI,
                name='Wisconsin Stores',
                marker_color='rgb(55, 100, 30)'
                ))

fig.update_layout(
    title="Sells by State stores per month",
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Quantity (Products)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# In[ ]:


df_year = df_estados.loc[:,["CA", "TX", "WI", "year"]]
df_year = df_estados.groupby("year").sum()
df_year.sort_values("CA")
fig = go.Figure()
fig.add_trace(go.Bar(x=df_year.index,
                y=df_year.CA,
                name='California Stores',
                marker_color='rgb(55, 83, 109)'
                ))
fig.add_trace(go.Bar(x=df_year.index,
                y=df_year.TX,
                name='Texas Stores',
                marker_color='rgb(55, 83, 220)'
                ))
fig.add_trace(go.Bar(x=df_year.index,
                y=df_year.WI,
                name='Wisconsin Stores',
                marker_color='rgb(55, 100, 30)'
                ))

fig.update_layout(
    title="Sells by State stores per year",
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Quantity (Products)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    bargap=0.15, # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)
fig.show()


# Let's analyce some products across time with following techniques:
#  - Neural Networks LSTM
#  - Regression
#  
# Let's see which tools is providing us best results
# 
# Let's work with following products:
#  - Most sold:
#  - 10 most sold
#  - Random product
#  - Less sold
#  
# Let's divide the data in one month for testing and the rest of the data to train our model.

# In[ ]:


def rollingSeriesSuma(series, contador):
    db_roll = series.rolling(contador).sum()
    return db_roll[contador:]

def rollingSeriesMedia(series, contador=28):
    db_roll = series.rolling(contador).mean()
    db_roll = db_roll.fillna(db_roll.mean())
    return db_roll

def createDBprod(product, df_sales, first_date, last_date, contador = 28):
    df_products = df_sales.loc[product, first_date:last_date]
    df_products = df_products.groupby(df_products.index).sum().transpose()
    df_products = pd.merge(df_products, dates, left_index= True, right_index = True)
    df_products["Date"] = pd.to_datetime(df_products["Date"])
    df_products["quarter"] = df_products["Date"].dt.quarter.astype("uint8")
    df_products["Month"] = df_products["Date"].dt.month.astype("uint8")
    df_products["Year"] = df_products["Date"].dt.year.astype("uint8")
    df_products["dayofyear"] = df_products["Date"].dt.dayofyear.astype("uint8")
    df_products["dayofweek"] = df_products["Date"].dt.dayofweek.astype("uint8")
    df_products.index = df_products.Date
    df_products = df_products.drop([ "d", "Date", "weekday", "month"], axis= 1)
    df_products["Rolling"] = rollingSeriesMedia(df_products[product], contador)
    return df_products

df_products = createDBprod(pms[-1], df_sales, first_date, last_date)
df_products.tail(5)


# Let's implement two more features:
#  - Rolling over 7 days ago 
#  - Rolling over 25 days ago
#  - Expanding
#  
# And plot it.

# In[ ]:


products = [pms[-1], pms[0] , df_sales_tot.index[np.random.randint(1300,len(df_sales_tot))],df_sales_tot.index[np.random.randint(1300,len(df_sales_tot))] ,pmsmin[0]]
for i in products:
    db_roll = createDBprod(i, df_sales, first_date, last_date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=db_roll.index, y=db_roll[i], name = i,
                             line_color='deepskyblue'))
    fig.add_trace(go.Scatter(x=db_roll.index, y=db_roll["Rolling"], name = "Rolling 28 Days",
                             line_color='black'))
    fig.show()


# In[ ]:


products = [pms[-1], pms[0] , df_sales_tot.index[np.random.randint(1300,len(df_sales_tot))],df_sales_tot.index[np.random.randint(1300,len(df_sales_tot))] ,pmsmin[0]]
contador=28
for i in products:
    df_products = createDBprod(i, df_sales, first_date, last_date)
    db_exp = df_products[i].expanding(contador).mean()
    db_exp = pd.Series(db_exp[contador:])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=db_exp.index, y=db_exp, name="EXPANDING",
                             line_color='blue'))
    fig.add_trace(go.Scatter(x=df_products.index, y=df_products[i], name=i,
                             line_color='black'))
    fig.update_layout(title_text='Rolling analysis timeseries',
                      xaxis_rangeslider_visible=True)
    fig.show()


# In[ ]:


product_ = pms[-1]
df_products = createDBprod(product_, df_sales, first_date, last_date)
years = np.unique(dates["year"])
year = list()
fig = go.Figure()
colors = ["red", "black", "deepskyblue", "yellow", "orange", "green"]
color_ = dict(zip(years, colors))
for i in years:
    a = str(i)
    name_ = product_ + " - " + a
    year = df_products[df_products["year"]==i]
    fig.add_trace(go.Scatter(x=year.dayofyear, y=year[product_], name=name_,
                         line_color=color_.get(i)))
fig.show()


# # LSTM Neural Network

# In[ ]:


import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[ ]:


def LSTMnn(serie, split= 100, lb = 28, plot = False):
    #fix random seed for reproducibility
    np.random.seed(7)
    scaler = MinMaxScaler(feature_range = (0,1))
    df = scaler.fit_transform(serie)
    train_size = int(len(df)-split)
    test_size = len(df) - train_size
    train, test = df[0:train_size,:], df[train_size:len(df), :]
    def create_dataset(dataset, look_back = 28):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i+look_back,0])
        return np.array(dataX), np.array(dataY)
    trainX,trainY = create_dataset(train, lb)
    testX, testY = create_dataset(test, lb)

    # reshape input to be [samples, time steps, features]
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    # create and fit the lstm network
    
    model = Sequential()
    model.add(LSTM(4, input_shape = (1, lb)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", optimizer = "adam")
    model.fit(trainX, trainY, epochs = 10, batch_size = 1, verbose = False)
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    #invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lb:len(trainPredict)+lb, :] = trainPredict
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(df)-len(testPredict):, :] = testPredict
    testPredictPlot = pd.DataFrame(testPredictPlot)
    if plot == True:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie.index, y=serie.iloc[:,0], name="Original Serie",
                                 line_color='deepskyblue'))
        fig.add_trace(go.Scatter(x=df_estados.Date, y=Data_train.iloc[:,0], name="Dataset train",
                                 line_color='Green'))
        fig.add_trace(go.Scatter(x=df_estados.Date,y=Data_test.iloc[:,0], name="Dataset test",
                                 line_color='red'))
        fig.show()
    return trainPredictPlot, testPredictPlot


# # XGBoost
# 
# **TRAIN / TEST SPLIT**
# 
# Cut off last 28 days to use as our validation test

# In[ ]:


import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

def features_(df, train = True, label = None):
    cols_to_remove = ["year",label]
    X = df.drop(cols_to_remove, axis = 1)
    if label:
        y = df[label]
        return X, y
    else:
        return X

def funcionevaluacion(real_values, predict):
    mse = mean_absolute_error(real_values, predict)
    n = len(predict)
    suma =0
    for i in range(1, n):
        est = (real_values[i]-real_values[i-1])**2
        suma +=est
    a = 1/(n-1)*suma
    evaluacion = np.sqrt((1/n)*mse/a)
    return evaluacion


# In[ ]:


from sklearn.model_selection import GridSearchCV

def _AnalisisXGBoost(df_product, parametros, plot = False):
    _evals = list()
    #df_product = createDBprod(i, df_sales, first_date, last_date)
    y = df_product.iloc[:,0]
    split_date = df_product.index[-60]
    df_train = df_product.loc[df_product.index < split_date].copy()
    df_test = df_product.loc[df_product.index >= split_date].copy()
    X_train, y_train = features_(df_train, train = True, label = df_product.columns[0])
    X_test, y_test = features_(df_test, train = False, label = df_product.columns[0])

    model = xgb.XGBRegressor(n_estimators  =200)
    #model = GridSearchCV(model,
    #                    parametros,
    #                    cv = 2,
    #                    n_jobs=5,
    #                    verbose = False)
    model.fit(X_train, y_train,
           eval_set = [(X_train, y_train), (X_test, y_test)],
           early_stopping_rounds = 10,
           verbose = False)

    #_ = plot_importance(model, height = 1)
    #df_train["PREDICTION"] = df_train[i]
    df_test["PREDICTION"] = model.predict(X_test)
    _evals= funcionevaluacion(y_test, df_test["PREDICTION"])
    df_final = pd.concat([df_train, df_test], sort = False)
    if plot == True:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_final.index, y=df_final.PREDICTION, name="PREDICTION",
                         line_color='red'))
        fig.add_trace(go.Scatter(x=df_final.index, y=df_final.iloc[:,0], name="Dataset",
                         line_color='deepskyblue'))
        fig.show()
    return df_final, _evals


# In[ ]:


s1 = [[None],
      ["state_id"],
      ["store_id"],
      ["cat_id"],
      ["dept_id"],
      ["state_id", "cat_id"],
      ["state_id", "dept_id"],
      ["store_id", "cat_id"],
      ["store_id", "dept_id"],
      ["item_id"],
      ["item_id", "state_id"],
      ["item_id", "store_id"]]

prueba = crearseries(s1[1])


# In[ ]:


results_XGBoost = list()
results_LSTMNN = list()
for i in prueba.index:
    serie = prueba.loc[i,:]
    df_serie = createFeatures(serie)
    serie_pred_XGBoost, resultado_XGBoost = _AnalisisXGBoost(df_serie, "parametros")
    serie_pred_LSTMNN, resultado_LSTMNN = LSTMnn(pd.DataFrame(df_serie.iloc[:,0]))
    results_XGBoost.append([serie_pred_XGBoost, resultado_XGBoost])
    results_LSTMNN.append([serie_pred_LSTMNN, resultado_LSTMNN])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_serie.index, y=df_serie.iloc[:,0], name="REAL DATA",
                         line_color='blue'))
    fig.add_trace(go.Scatter(x=df_serie.index, y=serie_pred_XGBoost.PREDICTION, name="XGBoost Prediction",
                         line_color='red'))
    fig.add_trace(go.Scatter(x=df_serie.index, y=resultado_LSTMNN.iloc[:,0], name="LSTMNN Prediction",
                         line_color='black'))
    fig.show()


# # XGBOOST HIPERPARAMETRIZATION
# 
# PENDING TO FINALIZE

# In[ ]:


"""
parameters = {
    "learning_rate" : [.01, .03, .05],
    "max_depth" : [6,7, 8],
    "n_estimators": [500]
    }

products = [pms[-1], pms[0] , 
            df_sales_tot.index[np.random.randint(1300,len(df_sales_tot))],
            df_sales_tot.index[np.random.randint(1300,len(df_sales_tot))] ,
            df_sales_tot.index[np.random.randint(800,1300)] ,
            df_sales_tot.index[np.random.randint(200,800)] ,
            pmsmin[0],
            pmsmin[6],
            pmsmin[-1]]
totaleval = list()
for i in products:
    analisis_, evaluacion = _AnalisisXGBoost(i, parameters)
    totaleval.append([i, evaluacion])
    
totaleval = pd.DataFrame(totaleval)

import plotly.express as px
fig = px.bar(totaleval, x=0, y=1, color = 1)
fig.show()
"""

