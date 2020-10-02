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
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


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

# ### Let's check 1st file: ../input/innercity.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# innercity.csv has 21613 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/innercity.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'innercity.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# In[ ]:


df1.head()


# In[ ]:


df1.info()


# In[ ]:


df1.describe()


# In[ ]:


df1.isna().sum()


# In[ ]:


df1.columns


# In[ ]:


df1.describe().T


# In[ ]:


df1.duplicated().sum()


# In[ ]:


df1.describe(include='all').T


# In[ ]:


#Missing values-

pd.isnull(df1).any()


# In[ ]:


df1["sold_year"] = df1["dayhours"].apply(lambda x:x.split('T')[0][:4])


# In[ ]:


df1.boxplot()


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plt.figure(figsize=(18, 8))
sns.boxplot(data=df1)


# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# In[ ]:


sns.heatmap(df1.corr(method="pearson"))


# In[ ]:


sns.heatmap(df1.corr(method="spearman"))


# In[ ]:


sns.heatmap(df1.corr(method="kendall"))


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 20, 10)


# 

# In[ ]:


df1.price.hist(bins=100)


# In[ ]:


df1.skew(axis = 0, skipna = True)


# In[ ]:


df1['price'].skew()


# In[ ]:


plt.figure(figsize=(10, 5))
sns.heatmap(df1.isnull())


# In[ ]:


#sns.pairplot(df1)


# In[ ]:


df1 = df1.drop(columns = 'cid')


# In[ ]:


df1 = df1.drop(columns = 'dayhours')


# In[ ]:


df1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


X = df1.drop("price", axis=1)
Y = df1["price"]
print(X.shape)
print(Y.shape)


# In[ ]:


#standard sclaer-
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler().fit(X)
scaled_X = scaler.transform(X)


# In[ ]:


df1


# In[ ]:


from sklearn.model_selection import train_test_split

seed = 42
test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = test_size, random_state = seed)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# Add silent=True to avoid printing out updates with each cycle
#my_model.fit(train_X, train_y, verbose=False)


# user variables to tune
folds   = 10
#metric  = "neg_mean_squared_error"
metric  = "neg_mean_absolute_error"

# hold different regression models in a single dictionary
models = {}
models["Linear"]        = LinearRegression()
models["Lasso"]         = Lasso()
models['Ridge']         = Ridge()
models["ElasticNet"]    = ElasticNet()
models["KNN"]           = KNeighborsRegressor()
models["DecisionTree"]  = DecisionTreeRegressor()
#models["SVR"]           = SVR()
models["AdaBoost"]      = AdaBoostRegressor()
models["GradientBoost"] = GradientBoostingRegressor()
models["RandomForest"]  = RandomForestRegressor()
models["XGBoost"]       = XGBRegressor()


# 10-fold cross validation for each model
model_results = []
model_names   = []
for model_name in models:
	model   = models[model_name]
	k_fold  = KFold(n_splits=folds, random_state=seed)
	results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)
	
	model_results.append(results)
	model_names.append(model_name)
	print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

# box-whisker plot to compare regression models
figure = plt.figure()
figure.suptitle('Regression models comparison')
axis = figure.add_subplot(111)
plt.boxplot(model_results)
axis.set_xticklabels(model_names, rotation = 45, ha="right")
axis.set_ylabel("Mean Squared Error (MSE)")
plt.margins(0.05, 0.1)


# In[ ]:


# create and fit the best regression model
best_model = GradientBoostingRegressor(random_state=seed)
best_model.fit(X_train, Y_train)

# make predictions using the model
predictions = best_model.predict(X_test)
print("[INFO] MSE : {}".format(round(mean_squared_error(Y_test, predictions), 3)))

#MAPE


# In[ ]:


model = GradientBoostingRegressor()
model.fit(X_train,Y_train)

##print("Intercept : ", model.intercept_)
##print("Slope : ", model.coef_)

#Predicting TEST & TRAIN DATA
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print("MAPE - Mean Absolute Percentage Error")
Y_train, train_predict = np.array(Y_train), np.array(train_predict)
print(np.mean(np.abs((Y_train - train_predict) / Y_train)) * 100)
print("Mape - Train:" , np.mean(np.abs((Y_train,train_predict))))
print("Mape - Test:" ,np.mean(np.abs((Y_test,test_predict))))


# In[ ]:


model = GradientBoostingRegressor()
model.fit(X_test,Y_test)

##print("Intercept : ", model.intercept_)
##print("Slope : ", model.coef_)

#Predicting TEST & TRAIN DATA
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print("MAPE - Mean Absolute Percentage Error")
Y_train, train_predict = np.array(Y_train), np.array(train_predict)
print(np.mean(np.abs((Y_train - train_predict) / Y_train)) * 100)
print("Mape - Train:" , np.mean(np.abs((Y_train,train_predict))))
print("Mape - Test:" ,np.mean(np.abs((Y_test,test_predict))))


# In[ ]:





# In[ ]:





# In[ ]:




