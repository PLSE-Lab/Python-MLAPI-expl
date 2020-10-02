#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


print(os.listdir('../input'))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()


# Now you're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: ../input/air_traffic_data.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# air_traffic_data.csv has 15007 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/air_traffic_data.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'air_traffic_data.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(10)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 12, 10)


# Let's check columns dtypes

# In[ ]:


df1.dtypes


# In[ ]:


numerical = [var for var in df1.columns if df1[var].dtype!='O'] #check numerical columns
categorical = [var for var in df1.columns if df1[var].dtype == 'O'] #check object columns

print('There are {} numerical variables'.format(len(numerical)))
print('There are {} categorical variables'.format(len(categorical)))


# > Check if any missing values

# In[ ]:


df1.isnull().sum()


# In[ ]:


#drop the only row which has NaN value
df1 = df1.dropna()

print(df1.shape)


# In[ ]:


df1.head()


# Grouping by Airline

# In[ ]:


grouped_by_airline = df1.groupby("Operating Airline").agg({ 
    "Operating Airline IATA Code" : "count",
    "Passenger Count" : lambda x : np.mean(x), #mean passengers count by airlines
    "Adjusted Passenger Count" : lambda x : np.mean(x) #mean adjusted passengerscount by airlines

})

grouped_by_airline.rename(columns = {"Operating Airline IATA Code" : "nb_flights", 
                                   "Passenger Count" : "mean_passenger_count", 
                                   "Adjusted Passenger Count" : "mean_adjusted_passenger_count"}, 
                          inplace = True)

grouped_by_airline = grouped_by_airline.sort_values(by = "nb_flights", ascending = False)

grouped_by_airline.head(10).round()


#  ## FEATURE ENGINEERING

# In[ ]:


def to_lower(data):
    """
    All columns in lower strings and underscored
    """    
    data.columns = map(lambda col: col.lower().replace(" ", "_"), data.columns)
    
    return data

def count_frequency(data, col, colname="frequency"):
    """
    Create a column from dataframe named frequency   
    """
    data[colname] = data.groupby(col)[col].transform('count')

    return data

# Dumify the object column
def dummify(data):
    """
    Dummify data columns object
    """
    dummify = data.loc[:, data.dtypes == object]
    for col in dummify.columns:
        df = pd.get_dummies(data[col], drop_first=True, prefix=col)
        data = pd.concat([data, df], axis=1)

    return data

def transform_df(data):
    "Wrapper of all functions"
    data = to_lower(data)
    data = count_frequency(data, 'operating_airline')
    data = dummify(data)
    
    return data


# In[ ]:


df1 = df1.drop(columns=['year', 'month'])
df1 = transform_df(df1)

print(df1.shape)


# In[ ]:


df1.head().T


# In[ ]:


prep_data = df1.copy()

prep_data = prep_data.drop(columns = prep_data.loc[:, prep_data.dtypes == object]) #drop all categorical columns


# In[ ]:


## Without a constant
import statsmodels.api as sm

X = prep_data.drop(['adjusted_passenger_count', 'passenger_count'], axis=1)
y = prep_data["adjusted_passenger_count"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[ ]:


df1.adjusted_passenger_count.mean()


# In[ ]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


# fit scaler
scaler = StandardScaler() # create an instance
scaler.fit(X_train) #  fit  the scaler to the train set


# In[ ]:


# feature scaling
from sklearn.preprocessing import StandardScaler

# for tree binarisation
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# to build the models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

# to evaluate the models
from sklearn.metrics import mean_squared_error

pd.pandas.set_option('display.max_columns', None)


# ### XGB REGRESSOR

# In[ ]:


xgb_model = xgb.XGBRegressor()

eval_set = [(X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

pred = xgb_model.predict(X_train)
print('xgb train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = xgb_model.predict(X_test)
print('xgb test mse: {}'.format(mean_squared_error(y_test, pred)))


# In[ ]:


importance = pd.Series(xgb_model.feature_importances_)
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.head(10).plot.barh(figsize=(18,6))


# ### RANDOM FOREST REGRESSOR

# In[ ]:


rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

pred = rf_model.predict(X_train)
print('rf train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = rf_model.predict(X_test)
print('rf test mse: {}'.format(mean_squared_error(y_test, pred)))


# In[ ]:


importance = pd.Series(rf_model.feature_importances_)
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.head(10).plot.barh(figsize=(18,6))


# ### LINEAR REGRESSION

# In[ ]:


lr = LinearRegression()
model = lr.fit(X_train,y_train)


# In[ ]:


predictions = lr.predict(X_test)


# In[ ]:


lr.score(X_test, y_test)


# In[ ]:


lr.intercept_


# In[ ]:


#to see the relationship between the training data values
#plt.scatter(X_train['activity_period'], y_train, c='red')
#plt.show()

#to see the relationship between the predicted 
#brain weight values using scattered graph
plt.plot(X_test, predictions)   
#plt.scatter(X_test['activity_period'], y_test,c='red')
plt.xlabel('act')
plt.ylabel('Passenger count')

#errorin each value
#for i in range(0,60):
#    print("Error in value number",i,(y_test[i]-predictions[i]))
#    time.sleep(1)

#combined rmse value
rss=((y_test-predictions)**2).sum()
mse=np.mean((y_test-predictions)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-predictions)**2)))


# ### LASSO

# In[ ]:


lasso = Lasso(random_state=2909)
lasso.fit(scaler.transform(X_train), y_train)

pred = lasso.predict(scaler.transform(X_train))
print('linear train rmse: {}'.format(np.sqrt(mean_squared_error(y_train, pred))))
pred = lasso.predict(scaler.transform(X_test))
print('linear test rmse: {}'.format(np.sqrt(mean_squared_error(y_test, pred))))


# In[ ]:


importance = pd.Series(np.abs(lasso.coef_.ravel()))
importance.index = X_train.columns
importance.sort_values(inplace=True, ascending=False)
importance.head(10).plot.barh(figsize=(18,6))


# In[ ]:


from sklearn.metrics import mean_squared_error

def regressionPlot(lasso,X_test,y_test, title):
    pred = lasso.predict(scaler.transform(X_test))
    plt.figure(figsize=(10,6))
    plt.scatter(pred,y_test,cmap='plasma')
    plt.title(title)
    plt.show()
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))


# In[ ]:


regressionPlot(lasso, X_test, y_test, "Lasso Model")


# ## NEW MODELS

# In[ ]:


cols = []

importance = pd.Series(np.abs(lasso.coef_.ravel()))
importance.index = X_train.columns
importance = importance.sort_values(ascending=False)
importance.head(10).plot.barh(figsize=(18,6))


# In[ ]:


pd.DataFrame(importance, ).head(10)


# In[ ]:


df1['airline'] = 'Other'

df1.loc[df1["operating_airline"] == 'United Airlines - Pre 07/01/2013', 'airline'] = 'United Airlines - Pre 07/01/2013'
df1.loc[df1["operating_airline"] == 'Alaska Airlines', 'airline'] = 'Alaska Airlines'
df1.loc[df1["operating_airline"] == 'SkyWest Airlines', 'airline'] = 'SkyWest Airlines'
df1.loc[df1["operating_airline"] == 'Northwest Airlines', 'airline'] = 'Northwest Airlines'
df1.loc[df1["operating_airline"] == 'Delta Air Lines', 'airline'] = 'Delta Air Lines'
df1.loc[df1["operating_airline"] == 'US Airways', 'airline'] = 'US Airways'
df1.loc[df1["operating_airline"] == 'ATA Airlines', 'airline'] = 'ATA Airlines'


df1['region'] = 'Other'

df1.loc[df1['geo_region'] == 'US', 'region'] = 'US'
df1.loc[df1['geo_region'] == 'Asia', 'region'] = 'Asia'
df1.loc[df1['geo_region'] == 'Europe', 'region'] = 'Europe'
df1.loc[df1['geo_region'] == 'Canada', 'region'] = 'Canada'


# In[ ]:


df1.columns


# In[ ]:


df1 = df1.drop(columns=['year', 'month', 'operating_airline', 'geo_region', 'adjusted_activity_type_code'])
df1 = to_lower(df1)
df1 = count_frequency(df1, 'airline')
df1 = dummify(df1)


print(df1.shape)


# In[ ]:


prep_data = df1.copy()
prep_data = prep_data.dropna()
prep_data = prep_data.drop(columns = prep_data.loc[:, prep_data.dtypes == object]) #drop all categorical columns

X = prep_data.drop(['adjusted_passenger_count', 'passenger_count'], axis=1)
y = prep_data["adjusted_passenger_count"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# In[ ]:


lasso = Lasso(random_state=2909)
lasso.fit(scaler.transform(X_train), y_train)

pred = lasso.predict(scaler.transform(X_train))
print('linear train rmse: {}'.format(np.sqrt(mean_squared_error(y_train, pred))))
pred = lasso.predict(scaler.transform(X_test))
print('linear test rmse: {}'.format(np.sqrt(mean_squared_error(y_test, pred))))


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Fork Notebook" button at the top of this kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
