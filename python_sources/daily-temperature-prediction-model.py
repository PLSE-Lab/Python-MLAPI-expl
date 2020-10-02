#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Hi, this is my first notebook post to kaggle. I'm looking for any and all advice, thanks :D

# ## Exploratory Analysis

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


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

# ### Let's check 1st file: /kaggle/input/city_temperature.csv

# In[ ]:


nRowsRead = None # specify 'None' if want to read whole file
# city_temperature.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/city_temperature.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'city_temperature.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# In[ ]:


df1.isnull().sum()


# We have unknown values for state. A possible solution to this would be to use the city, state, country, region to look up a pair of coordinates and then use that for our model. For simplicity sake, lets just drop the state column

# In[ ]:


df1.describe().round(1)


# Here we see that the minimum values for 'Year' and 'AvgTemperature' are very wrong, lets see in detail with a boxplot

# In[ ]:


df1.boxplot(column='Year')


# In[ ]:


df1.boxplot(column='AvgTemperature')


# -99 degrees and year 200 are obvious typos/placeholders, lets remove those columns.

# In[ ]:


print((df1['Year'] == 200).sum())
print((df1['AvgTemperature'] == -99).sum())


# imputing these missing values might give our model bias. Since we only have about 80,000 bad rows, and we have 3million total rows, we can drop these 

# In[ ]:


df1 = df1.drop(df1[(df1.Year == 200) | (df1.AvgTemperature == -99)].index)
df1 = df1.drop('State',1)
df1.head()


# In[ ]:


df1.groupby(['Year']).count()[['AvgTemperature']]


# We seem to have a good distribution of data points per year, nice.
# 
# Lets check if we have a good distribution of data points per city.

# In[ ]:


df1[(df1.Year != 2020)].groupby(['City','Year'])[['AvgTemperature']].count().describe()


# Most cities have around 363 readings per year, meaning most cities were polled daily for their temperatures.
# 
# We've excluded 2020 from this because 2020 is not yet over, so we obviously have less datapoints for that year.

# In[ ]:


df1.groupby('Region').describe()['AvgTemperature']


# We have many more data points for North America than other regions of the world, this may mean our model might have a bias towards north american temperatures, seasons, etc.

# In[ ]:


ax = sns.lineplot(x = 'Month', y = 'AvgTemperature', hue = 'Region', data = df1[(df1.Year != 2020)])
ax.set(xlabel = 'Month', ylabel='Average Temperature')
plt.show()


# Here we see the inverse relationship between seasons in different hemispheres

# In[ ]:


ax = sns.lineplot(x = 'Year', y = 'AvgTemperature', hue = 'Region', data = df1[(df1.Year != 2020)])
ax.set(xlabel = 'Year', ylabel='Average Temperature')
plt.show()


# In this graph, we see how the different regions compare in terms of average temperature over time

# In[ ]:


ax = sns.lineplot(x = 'Year', y = 'AvgTemperature', data = df1[(df1.Year != 2020)])
ax.set(xlabel = 'Year', ylabel='Average Temperature')
plt.show()


# We see a slight upwards trend towards higher temperature as the years increase. 2020 wasn't included in these graphs once again, because we don't have the whole year to compare to other years.

# # Preparing data for model
# 
# Since we want our model to predict the future, we'll train it on years 1995 - 2014 then validate on years 2015 to 2019 and use 2020 as our test

# In[ ]:


X = df1.copy()
y = X.AvgTemperature

X_train = X[(X.Year < 2015)].copy()
X_valid = X[(X.Year > 2014) & (X.Year != 2020)].copy()
X_test = X[(X.Year == 2020)].copy()

y_train = X_train.AvgTemperature
y_valid = X_valid.AvgTemperature
y_test = X_test.AvgTemperature

X_train.drop(['AvgTemperature'], axis=1, inplace=True)
X_valid.drop(['AvgTemperature'], axis=1, inplace=True)
X_test.drop(['AvgTemperature'], axis=1, inplace=True)


# In[ ]:


print(X_train.Year.unique())
print(X_valid.Year.unique())
print(X_test.Year.unique())


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

label_X_train = X_train.copy()
label_X_valid = X_valid.copy()
label_X_test = X_test.copy()

label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])
    label_X_test[col] = label_encoder.transform(X_test[col])


# In[ ]:


label_X_train.head()


# # Training and testing our model

# In[ ]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

model = XGBRegressor(n_estimators=1000,learning_rate=0.05,n_jobs=-1)
model.fit(label_X_train,y_train, early_stopping_rounds=5, eval_set=[(label_X_valid,y_valid)])
predictions = model.predict(label_X_test)
mae = mean_absolute_error(predictions,y_test)
print("Mean Absolute Error:" , mae)


# In[ ]:


pd.DataFrame(predictions).head()


# In[ ]:


pd.DataFrame(y_test).head()

