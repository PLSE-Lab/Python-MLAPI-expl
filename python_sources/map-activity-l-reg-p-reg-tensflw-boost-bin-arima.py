#!/usr/bin/env python
# coding: utf-8

# # What you will get here?

#  
# * Uni-variate and Bi-variate Analysis on COVID Data
# * Correlation Matrix and Extraction program - ( Image + Excel ) of COVID Data
# * Modelling - Lin Reg. | Polynomial Reg. | XG Boost ( Bagging + Boosting ) | TensorFlow
# * ARIMA with Binning model on Oil Price Trends
# * Image Histogram on MAP Data and Activity Index generation
# * Seasonality reduction of Moonlight Effects
# * Activity Index vs Oil Price Analysis 

# In[ ]:





# # Authors

# This notebook results collective work of [Irfan Ahmad](https://www.linkedin.com/in/irfan-ahmad-31103055/) and [Sabarna Chatterjee](https://www.linkedin.com/in/sabarna-chatterjee-b942108b/)
# 

# # Introduction

# Agenda of this challenge is to investigate how coronavirus pandemic has changed oil markets and predict future oil prices. To solve this problem, we have below data files in hand.

# 1. COVID-19_and_Price_dataset.csv
# 2. Crude_oil_trend.csv
# 3. NTL-dataset.zip 

# In this notebook we explore all given data sets one by one and see how these data can help to predict oil price using different models.

# # Exploratory Data Analysis - EDA : First Data Set

# Lets start with first data set, given in COVID-19_and_Price_dataset.csv

# **Import Packages**

# In[ ]:


# Import the library

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pandas.io.json import json_normalize 
import ipywidgets as widgets
from ipywidgets import widgets, interact
from plotly.offline import iplot
import plotly.graph_objs as go

filePath = '../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv'


# **Load Data: Oil Trend along with Covid cases**

# In[ ]:


#Import Data
worldData = pd.read_csv(filePath)


# In[ ]:


print('Number of rows : ', worldData.shape[0])
print('Number of features : ', worldData.shape[1])


# In[ ]:


worldData.head(5)


# So this dataset have just 125 rows/records and quite high number of colum/features(i.e 834). We will now have a look on features availabe....

# 1. First column represent to Date and last column represnt to oil price on that perticular date.
# 

# 2. Along with these two column, we have 832(834 - 2) column in between.These column represents region wise COVID situation.

# 3. For each region there are 4 colums, which represnets : total_case, news_cases,total_deaths,new_deaths
# 

# 4. For example for region Aruba we have 4 coulms: Aruba_total_cases,Aruba_new_cases,Aruba_total_deaths,Aruba_new_deaths We can also see,records for few dates are missing.ex- Data records for 2020-01-04 and 2020-01-05 are not there,reason behind these days were either weekends or Holidays

# **Univariable study** : Here We'll just focus on the dependent variable (Oil Price) and try to know a little bit more about it

# In[ ]:


#descriptive statistics summary
worldData['Price'].describe()


# So, minimum value of oil price is 26.428933 and maximum value is 57.544133. But ,what? standard daviation is 12.193102 which is quite high. Lets check its distribution...

# In[ ]:


# Prob distribution
sns.distplot(worldData['Price']);


# ok ,so oil price does not look like completely normally distributed. Prices data are mostly close either 30 or 50.

# Oil price data is availabe here, from date 31-12-2019 to 29-06-2020(close to 7 Months). Let's look at trend of oil price during this period.

# In[ ]:


trace0 = go.Scatter(
    x = worldData['Date'],
    y = worldData['Price'],
    mode = 'lines',
    name = 'lines'
)

data = [trace0]  # assign traces to data
layout = go.Layout(
    title = 'Oil Price Trend during Covid',
    xaxis = dict(title = 'Date'), # x-axis label
    yaxis = dict(title = 'Oil Price'), # y-axis label
    hovermode ='closest' 
)
fig = go.Figure(data=data,layout=layout)
iplot(fig)


# **Multivariate study:** We'll try to understand how the dependent variable and independent variables are related to each other

# **Correlation check -Multicollinearity**

# As we know,we have 832 features in dataset.But many of them will be correlated to each other.This may results to multicollinearity in dataset.

# For any regression problem,input data should not have multicollinearity.

# Here we will look into correlation between independent variables(covide cases) and dependent variable(Oil Price) as well as
# correlation within independent variables(covide cases)

# Let's check relationship between total number of covid cases among different countries those could have possiable impact on oil price. As well as,lets look dependencies between oil price and total number cases in these countries.

# In[ ]:


colums = ['World_total_cases','World_total_deaths','UnitedStates_total_cases','Russia_total_cases',
          'UnitedKingdom_total_cases','Spain_total_cases','Germany_total_cases','France_total_cases',
          'Singapore_total_cases','Italy_total_cases','Israel_total_cases','Brazil_total_cases',
          'UnitedArabEmirates_total_cases','SaudiArabia_total_cases','Qatar_total_cases','Turkey_total_cases',
          'Iran_total_cases','Oman_total_cases','Kuwait_total_cases','Egypt_total_cases','China_total_cases',
          'India_total_cases','Price']
correlation = pd.DataFrame(worldData[colums].corr(method ='pearson'))


# In[ ]:


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(correlation, vmax=1, square=True);


# We could see most of the independent variables are highly co-related to each other.

# *Only total number of cases in China is not much co-related with total number of cases with other countries :)*

# Let's plot scatter plot between Oil price and few top correlated variables

# In[ ]:


#scatterplot
sns.set()
colums = ['World_total_cases','UnitedStates_total_cases',
          'Italy_total_cases','UnitedArabEmirates_total_cases','SaudiArabia_total_cases',
          'Iran_total_cases',
          'Price']
sns.pairplot(worldData[colums], height = 2.5)
plt.show();


# *Oil price is mostly co-related with total number of covid cases in these coutries except China and Brazil.So instead of taking all independent variables as input variable, we will choose only one.Lets consider only World_total_cases as input variable which we will use in model training*

# **We have just looked correlations between covid cases of few countries and oil prices. But actualy we have 832 coulmns available in dataset.**

# **Do you want to look into correlation between each and every coulmn? we have totla 832 column. So we can have 832 x 832 total comparisions,which is difficult to plot.**

# We have created a report below where we can choose multiple coulms and extract their correlations value into a csv file.

# In[ ]:


# Custom functions 
#Base Variables->
inputDetails = {'cb1': '', 'cb2': '', 'cb3': '', 'cb4': '', 'cb5': '','cb6': '', 
                'threshold_from' : '', 'threshold_to':'', 'cb7': ''}
selectedColumns = []
ColumnNameArrcorr = np.array([])
filteredWorlCovidData = []

def seperator():
    print('-------------------------------------------------------------------------------------')

def processData():
    #All Column Name
    colNameArr = np.array(list(worldData.columns))

    segColNames = { 
        'totalCasesColumnNameArr': [] , 
        'newCasesColumnNameArr': [], 
        'totalDeathCasesColumnNameArr': [],
        'newDeathCasesColumnNameArr': [],
        'price':[]
    }

    totalCasesColumnNameArr = []
    newCasesColumnNameArr = []
    totalDeathCasesColumnNameArr = []
    newDeathCasesColumnNameArr = []
    price = []
    world = []

    # Segregating the data into different Arrays
    for items in colNameArr:
        if 'total_cases' in items:
            totalCasesColumnNameArr.append(items)

    for items in colNameArr:
        if 'new_cases' in items:
            newCasesColumnNameArr.append(items)

    for items in colNameArr:
        if 'total_deaths' in items:
            totalDeathCasesColumnNameArr.append(items)

    for items in colNameArr:
        if 'new_deaths' in items:
            newDeathCasesColumnNameArr.append(items)

    for items in colNameArr:
        if 'Price' in items:
            price.append(items)

    for items in colNameArr:
        if 'World' in items:
            world.append(items)        
        
    segColNames['totalCasesColumnNameArr'] = totalCasesColumnNameArr
    segColNames['newCasesColumnNameArr'] = newCasesColumnNameArr
    segColNames['totalDeathCasesColumnNameArr'] = totalDeathCasesColumnNameArr
    segColNames['newDeathCasesColumnNameArr'] = newDeathCasesColumnNameArr
    segColNames['price'] = price
    segColNames['world'] = world

    if inputDetails['cb1'] == True:
        for item in segColNames['totalCasesColumnNameArr']:
            selectedColumns.append(item)
    if inputDetails['cb2'] == True:
        for item in segColNames['newCasesColumnNameArr']:
            selectedColumns.append(item)
    
    if inputDetails['cb3'] == True:
        for item in segColNames['totalDeathCasesColumnNameArr']:
            selectedColumns.append(item)
    
    if inputDetails['cb4'] == True:
        for item in segColNames['newDeathCasesColumnNameArr']:
            selectedColumns.append(item)
    
    if inputDetails['cb5'] == True:
        for item in segColNames['price']:
            selectedColumns.append(item)

    # selectedColumns
    for i in range(len(selectedColumns)):
        if 'World' in item:
            del selectedColumns[i]

    if inputDetails['cb6'] == True:
        for item in segColNames['world']:
            selectedColumns.append(item)        


    # Find the Relation Matrix
    filteredWorlCovidData = pd.read_csv(filePath, usecols=selectedColumns )
    label_encoder = LabelEncoder()
    filteredWorlCovidData.iloc[:,0] = label_encoder.fit_transform(filteredWorlCovidData.iloc[:,0]).astype('float64')
    corrWorlCovidData = filteredWorlCovidData.corr()
    ColumnNameArrcorr = np.array(corrWorlCovidData)
    rowNumber = ColumnNameArrcorr.shape[0]
    colNumber = ColumnNameArrcorr.shape[1]
    print(ColumnNameArrcorr)
    seperator()
    
    if inputDetails['cb7'] == True :
        # making a Relational List <= threshold
        not_related = []
        not_related_line = { 'COL1': '', 'COL2': '', 'CORRVALUE' : 0}
    
        for i in range(rowNumber):
            for j in range(colNumber):
                if ( float(inputDetails['threshold_from']) <= ColumnNameArrcorr[i, j] <= float(inputDetails['threshold_to'])):
                    not_related_line['COL1'] = filteredWorlCovidData.columns[i]
                    not_related_line['COL2'] = filteredWorlCovidData.columns[j]
                    not_related_line['CORRVALUE'] = ColumnNameArrcorr[i, j]
                    not_related.append(not_related_line)
                    not_related_line = { 'COL1': '', 'COL2': '', 'CORRVALUE' : 0}
            
        finalCorr = pd.DataFrame(json_normalize(not_related))

        #Export the Data to CSV
        exportToExcel = finalCorr.sort_values(by=['CORRVALUE'])
        exportToExcel.to_csv (r'.\exportCorr.csv', index = False, header=True)
        print('Data is exported to exportCorr.csv')
        seperator()
    if inputDetails['cb8'] == True :
        # fig = sns.heatmap(corrWorlCovidData,center=0, cmap='BrBG', annot=True,linecolor='white', linewidths=1)
        sfig = sns.heatmap(corrWorlCovidData, annot=True,cmap='coolwarm', linecolor='white', linewidths=1)
        # fig
        figure = sfig.get_figure()    
        figure.savefig('exportCorr.png', dpi=1080) 
        print('HeatMap is exported to exportCorr.png')
        seperator()
    
    


# *Let's say you want to compare correaltion within total number of cases of all countries as well as with price. Choose check box 'Total cases' and 'Price' and press sumbit.A CSV file will be generated with correation matrix.*

# In[ ]:


# Create text widget for output
input_threshold_from = widgets.Text( placeholder='Threshold from',value='-.9')
input_threshold_to = widgets.Text( placeholder='Threshold to',value='.9')

cb1 = widgets.Checkbox(description="Total Case")
cb2 = widgets.Checkbox(despcription="New Case")
cb3 = widgets.Checkbox(description="Total Death Case")
cb4 = widgets.Checkbox(despcription="New Death Case")
cb5 = widgets.Checkbox(despcription="Price",value=True)
cb6 = widgets.Checkbox(despcription="World",value=True)
cb7 = widgets.Checkbox(despcription="Take CSV Backup",value=True)
cb8 = widgets.Checkbox(despcription="Take IMG Backup")

pb = widgets.Button(
    description='Submit Choices',
    disabled=False,
    value='1',
    button_style='success',
    tooltip='Submit Input'
)


def on_button_clicked(b):
    inputDetails['cb1'] = cb1.value
    inputDetails['cb2'] = cb2.value
    inputDetails['cb3'] = cb3.value
    inputDetails['cb4'] = cb4.value
    inputDetails['cb5'] = cb5.value
    inputDetails['cb6'] = cb6.value
    inputDetails['cb7'] = cb7.value
    inputDetails['cb8'] = cb8.value
    inputDetails['threshold_from'] = input_threshold_from.value
    inputDetails['threshold_to'] = input_threshold_to.value
    print('Selected Value: ', inputDetails)
    processData()
    
pb.on_click(on_button_clicked)

ui = widgets.VBox([widgets.HBox([cb1, cb2, cb5]), 
                   widgets.HBox([cb3, cb4, cb6]), 
                   widgets.HBox([input_threshold_from, input_threshold_to]),widgets.HBox([pb,cb7,cb8])])


display(ui)
# for some reasons I need to reset descriptions
cb1.description="Total Case"
cb2.description="New Case"
cb3.description="Total Death Case"
cb4.description="New Death Case"
cb5.description="Price"
cb6.description="World"
cb7.description="Take CSV Backup"
cb8.description="Take IMG Backup"


# *Oil price is mostly co-related with total number of covid cases in these countries except China and Brazil.So instead of taking all independent variables as input variable we will choose only one.Lets consider only World_total_cases as input variable which we will use in model training*

# # Models

# **Linear Regression**

# Let's start with very simple linear regression model to predict future oil prices on the basis of covid cases and check how it goes...

# As we have decided to use only World_total_cases as independent variable because most of them are correlated to each other.So here we will be having only two variables one independent(World_total_cases) and one dependent variable(Price)

# In[ ]:


data = worldData[['World_total_cases','Price']]


# *Let's do some preprocessing!*

# In[ ]:


data['World_total_cases'].describe()


# Let's split the data set into training and test dataset so that we can validate performance of our models.

# In[ ]:


# Spliting data to training and test sets----------------->
train = data.iloc[:92,:].values
train = pd.DataFrame(train)
train.columns = ['World_total_cases','Price']

test = data.iloc[92:125,:].values
test = pd.DataFrame(test)
test.columns = ['World_total_cases','Price']


# Let's apply normalization to our data set. We are using here min-max normalization

# In[ ]:


##Min-Max Normalization
def norm_train(x):
  return (x - train_stats['mean']) / (train_stats['max']-train_stats['min'])
def norm_test(x):
  return (x - test_stats['mean']) /(test_stats['max']-test_stats['min'])

train_stats = train['World_total_cases'].describe()
train_stats = train_stats.transpose()

test_stats = test['World_total_cases'].describe()
test_stats = test_stats.transpose()

train['World_total_cases'] = norm_train(train['World_total_cases'])
test['World_total_cases'] = norm_train(test['World_total_cases'])


# In[ ]:


train['World_total_cases'].describe()


# In[ ]:


test['World_total_cases'].describe()


# In[ ]:


# import Linear Regression lib 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
X = np.reshape(train['World_total_cases'].values ,(92,1))
y = np.reshape(train['Price'].values ,(92,1))


# In[ ]:



lin_reg.fit(X, y)


# In[ ]:



# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Oil Price Prediction on training data (Linear Regression)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_linear()


# In[ ]:


X_test = np.reshape(test['World_total_cases'].values,(33,1))
y_test = np.reshape(test['Price'].values,(33,1))
#Let 's predict oil prices on test set for next 5 days
pred_linreg = lin_reg.predict(X_test)
pred_linreg = pd.DataFrame(pred_linreg.flatten())
pred_linreg.head(5)


# In[ ]:


#Let's check what were the actual prices in test set for next 5 days
test['Price'].head()


# Prediction price are quite low as compare to actula.Lets plot predictions on whole test set.

# In[ ]:


X_test = np.reshape(test['World_total_cases'].values,(33,1))
y_test = np.reshape(test['Price'].values,(33,1))
# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, lin_reg.predict(X_test), color='blue')
    plt.title('Oil Price Prediction on test data (Linear Regression)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_linear()


# *Clearly model is not predicting well as soon as it goes far*

# **Let's try with polynomial regression as actual curve of price does not look like linear!**

# **Polynomial Regression**

# 3 Degree Polynomial

# In[ ]:


# Fitting Polynomial Regression to the dataset: degree = 3
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Oil Price Prediction on train data (Polynomial Regression)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_polymonial()


# In[ ]:


#Let 's predict oil prices on test set for next 5 days
pred_poly  =  pol_reg.predict(poly_reg.fit_transform(X_test))
pred_poly = pd.DataFrame(pred_poly.flatten())
pred_poly.head(5)


# Predictions are not that great. Lets plot predictions on whole test set.

# In[ ]:


# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, pol_reg.predict(poly_reg.fit_transform(X_test)), color='blue')
    plt.title('Oil Price Prediction on test data (Polynomial Regression)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_polymonial()


# **Polynomial regression seems worst solution than linear regression! So What next?**

# **Let's try some boosting and bagging techninque for prediction**

# Gradient Boosting Decision Tree is one of the popular boosting technique. Simlarly Random forest in known for bagging. Both techniques are part of Ensemble learning. To learn more about bagging and boosting ,please follow link : [Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) 

# Instead of using either of these two separate algorithms , we will use [Xgboost](https://en.wikipedia.org/wiki/XGBoost). This combines both bagging and boosting technique.

# **Xgboost**

# In[ ]:


import xgboost as xgb


# In[ ]:


gbm = xgb.XGBRegressor(colsample_bytree = 1, learning_rate = 0.1,
max_depth = 10 , n_estimators = 50)


# In[ ]:


gbm.fit(X,y)


# In[ ]:


# Visualizing the results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, gbm.predict(X), color='blue')
    plt.title('Oil Price Prediction on training data (Xgboost)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_linear()


# In[ ]:


def viz_linear():
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, gbm.predict(X_test), color='blue')
    plt.title('Oil Price Prediction on test data (Xgboost)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_linear()


# **Note:** we can see model has overfitted to training data and prediction is worst on test data. We can control overfitting by doing hyperparmeter tuning that choosed in model(ex-learning_rate,n_estimators).

# In this note book, we are not going to tune hyper parameters.

# **Have you ever tried Tensorflow for Regression? Let's do it**

# **Tensorflow Regression**

# In[ ]:


import tensorflow as tf
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=64,activation='tanh', input_shape=[1]),
                             keras.layers.Dense(units=128,activation='tanh'),
                                 keras.layers.Dense(units=1)])
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss='mean_squared_error')
model.summary()


# In[ ]:


model.fit(X, y, epochs=500)


# In[ ]:


# Visualizing the results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, model.predict(X), color='blue')
    plt.title('Oil Price Prediction on training data (Tensorflow)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_linear()


# In[ ]:


def viz_linear():
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, model.predict(X_test), color='blue')
    plt.title('Oil Price Prediction on test data (Xgboost)')
    plt.xlabel('Total Number of Covid Cases')
    plt.ylabel('Oil Prices')
    plt.show()
    return
viz_linear()


# **Predictions on test data looks better than other models. We can even get better model by choosing different architecure(Number of hidden uints,number layers or types layers). This is just a sample model** 

# # Let's look into second data set...

# # EDA  : Second Data Set

# In[ ]:


# Import Lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import widgets, interact

filePath = '../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv'


# Let's download data from Crude_oil_trend_From1986-01-02_To2020-06-08 and check what info it has...

# In[ ]:


sns.set_style('whitegrid')
raw_df = pd.read_csv(filePath)
raw_df.head(5)


# In[ ]:


df = raw_df.groupby(['Date'])['Price'].sum().reset_index()
df['Price'].plot(kind='hist')
print(df['Price'].describe())


# **As we can see here we have only 2 colums 'Date'and 'Price'. So to predict future price we may consider this as a time series problem**

# **But here we will try time series with some variation.Instead to predicting actual price what if convert whole data set in number of bins and predict price falling into which bin?**

# Create quantile

# In[ ]:


bin_labels10 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
df['quantile_ex_1'] = pd.qcut(df['Price'],q=10,labels=bin_labels10)

bin_labels20 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
df['quantile_ex_2'] = pd.qcut(df['Price'],q=20,labels=bin_labels20)

df.head()
# df['quantile_ex_1'].value_counts()
# df['quantile_ex_2'].value_counts()


# Quantile Charts

# In[ ]:


new_df = pd.DataFrame(df)
df_melt = new_df.melt(id_vars='Date', value_vars=['quantile_ex_1', 'quantile_ex_2'])
fig = px.line(df_melt, x='Date' , y='value' , color='variable')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 month", step="month", stepmode="backward"),
            dict(count=6, label="6 month", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1 year", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
fig.show()


# In[ ]:


# Count the number of quantile_ex_2 Changes

# Count the Similar number of quantile_ex_2 Chnages in past records

# Find the Projection on the Data on Arima

# Try to find out the pattern


# In[ ]:


from pandas.io.json import json_normalize

binArr = np.array(df)

dataQuantileRanges1 = []
dataQuantileRanges2 = []

dataQuantileRangesLine = {'BINNO': '', 'LOW': '', 'HIGH': '', 'COUNT': 0, 'DELTA' : 0}

for binno in range(1,21):
    binVal = []
    for item in binArr:
        if item[3] == str(binno):
            binVal.append(item[1])
    dataQuantileRangesLine['BINNO'] = 'Bin no - ' + str(binno)
    dataQuantileRangesLine['LOW'] = round(np.array(binVal).min(),2)
    dataQuantileRangesLine['HIGH'] = round(np.array(binVal).max(),2)
    dataQuantileRangesLine['COUNT'] = len(np.array(binVal))
    dataQuantileRangesLine['DELTA'] = round(np.array(binVal).max(),2) - round(np.array(binVal).min(),2)
    dataQuantileRanges2.append(dataQuantileRangesLine)
    dataQuantileRangesLine = {'BINNO': '', 'LOW': '', 'HIGH': '', 'COUNT': 0, 'DELTA' : 0}

df_dataQuantileRanges2 = json_normalize(dataQuantileRanges2)

for binno in range(1,11):
    binVal = []
    for item in binArr:
        if item[2] == str(binno):
            binVal.append(item[1])
    dataQuantileRangesLine['BINNO'] = 'Bin no - ' + str(binno)
    dataQuantileRangesLine['LOW'] = round(np.array(binVal).min(),2)
    dataQuantileRangesLine['HIGH'] = round(np.array(binVal).max(),2)
    dataQuantileRangesLine['COUNT'] = len(np.array(binVal))
    dataQuantileRangesLine['DELTA'] = round(np.array(binVal).max(),2) - round(np.array(binVal).min(),2)
    dataQuantileRanges1.append(dataQuantileRangesLine)
    dataQuantileRangesLine = {'BINNO': '', 'LOW': '', 'HIGH': '', 'COUNT': 0, 'DELTA' : 0}

df_dataQuantileRanges1 = json_normalize(dataQuantileRanges1)

quantile_ex_1 = df['quantile_ex_1'].value_counts()
quantile_ex_2 = df['quantile_ex_2'].value_counts()


# **Check the dataset by clicking the below radio button to get the dataset,
# dataQuantileRanges1 ->10 Bin details dataQuantileRanges2 ->20 Bin details**

# In[ ]:


from IPython.display import clear_output
radbtn = widgets.RadioButtons( 
            options=['Display quantile_ex_1', 
                     'Display quantile_ex_2', 
                     'dataQuantileRanges1', 
                     'dataQuantileRanges2'],
    description='Your Choice',
    disabled=False )
def on_button_clicked(b):
    clear_output(wait=False)
    display(radbtn,pb)
    if radbtn.value == 'Display quantile_ex_1':
        print(quantile_ex_1)
    if radbtn.value == 'Display quantile_ex_2':
        print(quantile_ex_2)
    if radbtn.value == 'dataQuantileRanges1':
        print(df_dataQuantileRanges1)
    if radbtn.value == 'dataQuantileRanges2':
        print(df_dataQuantileRanges2)  
    
    
pb = widgets.Button(
    description='Submit Choices',
    disabled=False,
    value='1',
    button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Submit Input'
)

pb.on_click(on_button_clicked)

display(radbtn,pb)


# **Now ARIMA**

# Lets try to implement an ARIMA Model just to find out the trend of this bins predict the Bins. This arima model prediction data will give the response as the bin where the oil price could be

# In[ ]:


# Define the d and q parameters to take any value between 0 and 1
q = range(0, 3)
d = range(0, 2)
# Define the p parameters to take any value between 0 and 3
p = range(0, 4)


# In[ ]:


import warnings
import itertools
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
# pdq

new_df1 = df[['Date','quantile_ex_2']].copy()
train = new_df1
train.set_index(['Date'], inplace=True)
train['quantile_ex_2']=pd.to_numeric(train['quantile_ex_2'])
train
# train.set_index(['Date'], inplace=True)
# train = 
train.plot()
plt.ylabel('quantile_ex_2')
plt.xlabel('Date')
plt.show()


# In[ ]:


model = ARIMA(train,order= (2,2,0)) # 2,2,0
results = model.fit(disp=0)


# In[ ]:


forcastFromPast = results.forecast(28)


# In[ ]:


forcastFromPast


# In[ ]:


preds = forcastFromPast[0]
preds.astype(int)


# In[ ]:


# last 300 days array from Data frame
# Seperate line for frame

results.plot_predict()


# # Let' s come to third and last dataset. Most intresting and fun part!

# # Analysis on Map Data to extract information from it
# 
# What we are going to here is understand the pattern of the MAP images and get the extracted data out of it 
# 
# In brief we have catagorized the entire analysis into the below sections:
# 
# 1. Reading the Data
# 2. Understand the possible cluster
# 3. Traning the image data for KMEANS Algorithm
# 4. Populate a histogram and legend for the Image
# 5. Create a timeseries out of the each image 
# 6. Defination of Activity Magnitude
# 7. Understand the distorted seasonal effects (fullmoon effects / Cloud visibility) in the Map images
# 8. Refine the Map data by removing this distored data and extract
# 9. 
# 
# At Last we will conclude with a implementaion thoughts 
# 
# ## Defination Activity Ratio: Ratio of the White Color with the dark Section.

# ## 1. Reading the Data

# In[ ]:


# Import the necessary libs
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import ntpath

# Mention the file path to the Image which is to be considered 
filePath = '../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/USA-20200101.tif'
# filePath = './ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/Italy-S20200101-E20200110.tif'
fileName = ntpath.basename(filePath)
print(fileName)


# In[ ]:


# Now plot Map Display in Pyplot

image = cv2.imread(filePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.axis("off")
plt.imshow(image)


# ## 2. Understanding the possible cluster

# In[ ]:


# Get the Data again into the variable "img"
# This will give you the idea on how many cluster have to be choosen for KMEANS
# Clearly 5 sparks you are getting from the data

img = cv2.imread(filePath)
color = ('b','g','r')

for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.show()


# ## Details to Choose the Cluster no as 4
# 
# Now we have to discuss about the number clusters to be choosen.
# The activity data will be displayed when the colors are closer to "white".
# Based upon this Analysis we can take into consider that rightmost colors in the above graph 
# 
# ![image.png](attachment:image.png)
# 

# ## 3. Traning the image data for KMEANS Algorithm
# 
# Lets take the cluster as 4 and train the KMEANS

# In[ ]:


image = image.reshape((image.shape[0] * image.shape[1], 3))
#Set the Cluster
clt = KMeans(n_clusters = 4)
clt.fit(image)


# ## 4. Populate a histogram and legend for the Image

# The below methods works for ->
# 
# centroid_histogram -> This function taked the trained model input data and converts into a "hist[]" Array
# 
# plot_colors -> This function converts the percentages in "hist[]" Array and then plots the histogram with its speciifc width
# 
# plotAnalysis -> This function helps to create a legend for the histogram plot

# In[ ]:


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1) 
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors(hist, centroids):
    Legend = np.zeros((10, 90, 3), dtype = "uint8")
    bar = np.zeros((50, 200, 3), dtype = "uint8")
    startX = 0
    hist_array = []
    for (percent, color) in zip(hist, centroids):    
        endX = startX + (percent * 200)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),color.astype("uint8").tolist(), -1)
        startX = endX
        hist_array_line = { 
            'percent1': '', 
            'color1': ''
        }
        hist_array_line['percent1'] = percent
        hist_array_line['color1'] = color
        hist_array.append(hist_array_line)
    return bar , hist_array

############################
# printbar -> Plots the Bar Graph
# printvalue -> Plots the Value
# hist_array -> needed for Hist Value plot
# bar -> Plot the bar diagram
##############################

def plotAnalysis(printbar,printvalue,clt):

    hist = centroid_histogram(clt)
    bar, hist_array = plot_colors(hist, clt.cluster_centers_)
    
    blank_img = np.ones(shape=(200,800,3),dtype=np.int16)
    blank_img = blank_img * 255
    blank_img

    lineNumber = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in hist_array:
        dynamicYStart = 20 * lineNumber
        dynamicYEnd = dynamicYStart + 20
        runColor = line['color1'].astype("uint8").tolist()
        cv2.putText(blank_img,text=str(line['percent1']),org=(60,dynamicYStart*2 + 10 ),fontFace=font,fontScale=1,color=runColor,thickness=3,lineType=cv2.LINE_AA)
        lineNumber = lineNumber + 1
    
    if printbar == 'X':
        plt.figure()
        plt.axis("off")
        plt.imshow(bar)
    if printvalue == 'X':
        plt.figure()
        plt.axis("off")
        plt.imshow(blank_img)
    return hist_array


# In[ ]:


# Let's build a histogram of those clusters 
# Now Represent the number of pixels as color

plotAnalysis('X', # If X then print the bar
             'X', # If X then print the legend
             clt)


# -> These above values is done for USA-20200101.tif". Similary we can take all the imgaes of USA and create a nice Time series graph out of it to understand the Activity ratios in throughout Lockdown period. 
# 
# -> More Over this Data can be extrapolated to merge as a new Feature with the "World Covid Total Cases Excel". 

# ## What we can can understand from this?
# 
# As the image contains the 4 types of color cluster ( As choosen while the KMEANS). 
# We are going to seggregate these colors into different ranges.
# 
# catagory 1 - (Dark Black)
# catagory 2 - (Semi Black)
# catagory 3 - (Grey)
# catagory 4 - (Near White)

# ## Step for Mass Data Load

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import ntpath
import os

countrylist = []
filePath = '../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/'
filelist = os.listdir(filePath)

imageDataArr = []

for file in filelist:
    imageData = { 'country' : '', 'date':'', 'filename': '', 'fullfilepath':'', 'image' : [], 'clt': '' }
    country = file.split("-")
    imageData
    if country[0] not in countrylist:
        countrylist.append(country[0])
    imageData['country'] = country[0]
    imageData['date'] = country[1].split(".")[0]
    imageData['filename'] = file
    imageData['fullfilepath'] = filePath + file
    
    image = cv2.imread(imageData['fullfilepath'])
    imageData['image'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#   You can do the below section to execute all the KMEANS at a same time  
#     # Kmeans
#     image = image.reshape((image.shape[0] * image.shape[1], 3))
#     clt = KMeans(n_clusters = 4)
#     imageData['clt'] = clt.fit(image)
    
    imageDataArr.append(imageData)
        

# All file details in -> imageDataArr
# All Country Name -> countrylist


# In[ ]:


countrylist


# ## Load the below cell first - This converts the trained dataset to an Excel file
# 
# --> Preprocessing before Saving the trained model into excel 

# In[ ]:


import pandas as pd
from pandas.io.json import json_normalize 

def saveintofile():
    histDataArr = []
    
    for line in trainedData:
        imageData = line
        imageData['hist'] = plotAnalysis('', '', line['clt'])
        histDataArr.append(imageData)

    def arrangeHist(hist):
        finalReturn = []
        for line in hist:
            finalReturnLine = {}
            finalReturnLine['percent'] = line['percent1']
            finalReturnLine['color'] = line['color1'][0]
            finalReturn.append(finalReturnLine)
        pd_finalReturn = pd.DataFrame(json_normalize(finalReturn))
        pd_finalReturn = pd_finalReturn.sort_values(by=['color'])
        histDict = {}
        histDict['PERCENTCAT1'] = np.array(pd_finalReturn)[0][0]
        histDict['PERCENTCAT2'] = np.array(pd_finalReturn)[1][0]
        histDict['PERCENTCAT3'] = np.array(pd_finalReturn)[2][0]
        histDict['PERCENTCAT4'] = np.array(pd_finalReturn)[3][0]

        histDict['COLORCAT1'] = np.array(pd_finalReturn)[0][1]
        histDict['COLORCAT2'] = np.array(pd_finalReturn)[1][1]
        histDict['COLORCAT3'] = np.array(pd_finalReturn)[2][1]
        histDict['COLORCAT4'] = np.array(pd_finalReturn)[3][1]    
    
        return histDict



    refinedData = []
    for line in histDataArr:
        refinedDataline = {}
        refinedDataline['date'] = pd.to_datetime(str(line['date']),format='%Y%m%d')
        refinedDataline['filename'] = line['filename']
        refinedDataline['fullfilepath'] = line['fullfilepath']
        hist = arrangeHist(line['hist'])
        refinedDataline['PERCENTCAT1'] = hist['PERCENTCAT1']
        refinedDataline['PERCENTCAT2'] = hist['PERCENTCAT2']
        refinedDataline['PERCENTCAT3'] = hist['PERCENTCAT3']
        refinedDataline['PERCENTCAT4'] = hist['PERCENTCAT4']
    
        refinedDataline['COLORCAT1'] = hist['COLORCAT1']
        refinedDataline['COLORCAT2'] = hist['COLORCAT2']
        refinedDataline['COLORCAT3'] = hist['COLORCAT3']
        refinedDataline['COLORCAT4'] = hist['COLORCAT4']
        refinedData.append(refinedDataline)

    # Data Saved to excel
    df = pd.DataFrame(refinedData)
    transformedDataFilename = "./refinedData" + dropdown.value + ".xlsx"
    df.to_excel(excel_writer = transformedDataFilename)


# ## Follow these Steps
# 
# ## -> The below cell is to Train your dataset
# 
# 1. Execute the below section with each Country Name
# 2. Each succesfull execution will create a excel file in your system

# In[ ]:


import ipywidgets as widgets
from ipywidgets import widgets, interact

trainedData = []
selcountrydetails = []

pb = widgets.Button(
    description='Submit Choices',
    disabled=False,
    value='1',
    button_style='success',
    tooltip='Submit Input'
)

def trainData():
    run = 0
    len(selcountrydetails)
    for line in selcountrydetails:
        imageData = { 'country' : '', 'date':'', 'filename': '', 'fullfilepath':'', 'image' : [], 'clt': '' }
        imageData = line
        image = cv2.imread(line['fullfilepath'])    
        ###################################
        # Kmeans
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters = 4)
        imageData['clt'] = clt.fit(image)     
        ###################################
        trainedData.append(imageData)
        print(run)
        run = run+1
    print('--------Training Completed-------')
    
    #Convert the trained dataset to an Excel File
    saveintofile()

def on_button_clicked(b):
    print('Selected Country: ',dropdown.value)    
    for line in imageDataArr:
        if ( dropdown.value == line['country']):
            selcountrydetails.append(line)
    print('Image Dataset Length: ',len(selcountrydetails))    
    print('--------Training Started-------')
    trainData()

pb.on_click(on_button_clicked)

dropdown = widgets.Dropdown(
    options=countrylist,
    description='Number:',
    disabled=False
)

ui = widgets.VBox([widgets.HBox([dropdown, pb])])
display(ui)


# ## Load the Excel file
# 
# ## Execution of the below cell will create a Dropdown from which you can select any Country to do further analysis

# In[ ]:


import ipywidgets as widgets
from ipywidgets import widgets, interact
import pandas as pd
from pandas.io.json import json_normalize 

histImageCountry = ''
def fetchFromExcel(b):
    global histImageCountry
    fileName = "./refinedData" + dropdown.value + ".xlsx"
    histImageCountry = pd.read_excel(fileName)
    histImageCountry.sort_values("date",axis = 0, ascending = True, inplace = True)
    
    print('Data is loaded now to play with')

pb1 = widgets.Button(
    description='Submit Choices',
    disabled=False,
    value='1',
    button_style='success',
    tooltip='Submit Input'
)

pb1.on_click(fetchFromExcel)

dropdown = widgets.Dropdown(
    options=countrylist,
    description='Number:',
    disabled=False
)

ui = widgets.VBox([widgets.HBox([dropdown, pb1])])
display(ui)


# In[ ]:


# Lets Plot directly
import plotly.express as px
df_melt = histImageCountry.melt(id_vars='date', value_vars=['PERCENTCAT1','PERCENTCAT2','PERCENTCAT3','PERCENTCAT4'])
fig = px.line(df_melt, x='date' , y='value' , color='variable')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 month", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
fig.show()


# The above figure is directly ploted from the histogram values of all the images for the selected country. This might not make sense as the time series plot from this has no reference to check. So lets Change the dataset a bit and check what we can find out.

# Do analyze the standard Histogram dataset we have taken two types of approach:
# 
# 1. Taken the reference of the 1st day and plotted the values - Activity Ratio
# 2. Taken the sqroot of the summed value of (PERCENTCAT*)^2 - Activity Magnitude

# In[ ]:


createRatio = []
for line in np.array(histImageCountry):
    createRatioline = {}
    createRatioline['date'] = line[1]
    createRatioline['filename'] = line[2]
    createRatioline['fullfilepath'] = line[3]
    createRatioline['PERCENTCAT1'] = line[4] / np.array(histImageCountry)[0][4]
    createRatioline['PERCENTCAT2'] = line[5] / np.array(histImageCountry)[0][5]
    createRatioline['PERCENTCAT3'] = line[6] / np.array(histImageCountry)[0][6]
    createRatioline['PERCENTCAT4'] = line[7] / np.array(histImageCountry)[0][7]
    createRatioline['PRECENTVECTOR'] = np.sqrt( createRatioline['PERCENTCAT4']**2 + 
                                                createRatioline['PERCENTCAT3']**2 + 
                                                createRatioline['PERCENTCAT2']**2 + 
                                                createRatioline['PERCENTCAT1']**2)
    createRatio.append(createRatioline)

df = pd.DataFrame(createRatio)

print('All the dataset in Activity Ratio:::-->> ')

import plotly.express as px
df_melt = df.melt(id_vars='date', value_vars=['PERCENTCAT1','PERCENTCAT2','PERCENTCAT3','PERCENTCAT4','PRECENTVECTOR'])
fig = px.line(df_melt, x='date' , y='value' , color='variable')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 month", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
fig.show()

print('All the dataset in Activity Magnitude:::-->> ')

dfNew = df[['date','PRECENTVECTOR']]
df_melt = dfNew.melt(id_vars='date', value_vars=['PRECENTVECTOR'])
fig = px.line(df_melt, x='date' , y='value' , color='variable')
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1 month", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(step="all")
        ])
    )
)
fig.show()


# ## Now lets understand: why this spark is comming?
# ![image.png](attachment:image.png)

# In[ ]:


Lets plot dataset of USA From : 6th to 16th of Jan.


# In[ ]:


files = []
fig = plt.figure(figsize=(50, 50))
i=0
for days in ["%02d" % x for x in range(6,16)]:
    filePath = '../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/USA-202001' + days + '.tif'
    image = cv2.imread(filePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = i + 1
    sub = fig.add_subplot(5,2,i)
    sub.imshow(image)


# # From the above pictures we can extract below conclusion:
# 
# -In the above images, we can see sharp few white images on regular interval of times
# 
# **This due to the seasonal effects of the extra light due to fullmoon**
# 
# -> we need to refine this effects to get a smooth curve out of it:

# In[ ]:


# Our variable is dfNew
dfNew.head()


# In[ ]:


dfSeasonal = dfNew[['PRECENTVECTOR']]
dates = pd.date_range(start='1/1/2020', periods=160, freq='D')
dfseason = dfNew[['PRECENTVECTOR']]
dfseason.set_index


# In[ ]:


import statsmodels.api as sm
# Use seasonal Decompose
decomposition = sm.tsa.seasonal_decompose(dfSeasonal,model='multiplicative',period=30)
decomposition.plot()
dfpercent = decomposition.trend
dfpercent = pd.DataFrame(dfpercent)
dfpercent.plot()
dfpercent


# ## Now how to create a feature out of it -->

# In[ ]:


# Fetch the price from the COVID Cases ->
covidDatafilePath = '../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv'
worldData = pd.read_csv(covidDatafilePath)

worldDataPrice = worldData[ ['Date','Price' ] ]
worldDataPrice["Date"]= pd.to_datetime(worldDataPrice["Date"])   
worldDataPrice.head()
worldDataPriceArr = np.array(worldDataPrice)
len(worldDataPriceArr)


# In[ ]:


combinedData = []
i = 0
for line in np.array(dfpercent):
    try:
        combineLine = {}
        worldDataPriceArr[i]
        combineLine['Date'] = worldDataPriceArr[i][0]
        combineLine['Price'] = worldDataPriceArr[i][1]
        combineLine['ACTIVITYMAGNITUDE'] = line[0]
        combinedData.append(combineLine)
        i = i + 1
    except IndexError as ierr:
        print(i)
        break;

len(combinedData)


# In[ ]:


mergedData = pd.DataFrame(combinedData)
mergedData.head()
newData = mergedData[['Date','Price','ACTIVITYMAGNITUDE']]
newData.head()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
 
# create data
x = np.array(newData['Date'])
y = np.array(newData['Price'])
z = np.array(newData['ACTIVITYMAGNITUDE'])

# use the scatter function
plt.scatter(x, y, s=z, alpha=1)
plt.show()


# In[ ]:


plt.plot(np.array(newData['Price']),np.array(newData['ACTIVITYMAGNITUDE']))


# In[ ]:


newData.info()


# ## At Last we will concluding this Topic with an implementaion thoughts
# 
# This Dataset Contains error for each an every values which has a cloud cover or irregular images. 

# ## THE END-------------------------
