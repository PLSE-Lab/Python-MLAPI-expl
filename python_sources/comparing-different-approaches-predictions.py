#!/usr/bin/env python
# coding: utf-8

# # Problem :
# ## Jaipur Rainfall Data
# #### The problem is that Jaipur is one of the regions in India that has very limited amount of rainfall throught the year, So we will try to implement data analysis to predict what amount of rainfall will be recieved
# Some methods like padding dates were taken from other kernels. 
# Still you will see some new methods and analysis and the reason behind them in this kernels.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import MinMaxScaler


import keras
from keras.layers.core import Dense, Dropout
from keras.layers import Input , LSTM, Embedding
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l2

from os.path import join
from os import listdir


# In[ ]:


SEED = 1234
from numpy.random import seed
seed(SEED)
from tensorflow import set_random_seed
set_random_seed(SEED)


# In[ ]:


print(listdir('../input'))


# In[ ]:


# dataset = '/content/gdrive/My Drive/Colab Notebooks/DataSet-Jaipur'

# Location of Dataset
dataset = '../input'
filename = 'JaipurRawData3.csv'
filename = join(dataset, filename)


# In[ ]:


df = pd.read_csv(filename, index_col='date')
df.head()


# In[ ]:


print('Shape of Dataset: {}'.format(df.shape))
print('Shape of Each Row: {}'.format(df.iloc[0].shape))
print(df.dtypes)


# In[ ]:


plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig('correlationheatmap.png')
plt.show()


# ### Feature Engineering

# In[ ]:


def min_max_to_mean(df, input_columns, output_columnname):
    df[output_columnname] = (df[input_columns[0]] + df[input_columns[1]])/ 2
    df.drop(input_columns, axis=1, inplace=True)
    return df


# In[ ]:


df = min_max_to_mean(df, ['maxhumidity', 'minhumidity'], 'meanhumidity')
df.drop(['maxtempm', 'mintempm', 'maxdewptm', 'mindewptm', 'maxpressurem', 'minpressurem'], axis=1, inplace=True)


# In[ ]:


plt.figure(figsize=(10,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig('correlationheatmap.png')
plt.show()


# In[ ]:


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number])
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] 
    columnNames = list(df)
    if len(columnNames) > 10: 
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.savefig('scatterplot.png')

    plt.show()

    
plotScatterMatrix(df, 10, 10)


# ##### We see that Maximum Correlation is from Humidity and Dew Point, Looking at their distribution

# In[ ]:


def showplots(x):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                        gridspec_kw={"height_ratios": (.15, .85)})

    sns.boxplot(x, ax=ax_box)
    sns.distplot(x, ax=ax_hist)
    ax_box.set(yticks=[])
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)


# In[ ]:


showplots(df['meanhumidity'])


# In[ ]:


showplots(df['meandewptm'])


# ##### Thus There were no outliners as such, data comes from a distribution
# 
# ### Normalizing the Data we get

# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)


# In[ ]:


df = pd.DataFrame(df_scaled, columns=df.columns)


# ##### Pad Historic data for past 2 days with each row

# In[ ]:


def pad_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_meassurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_meassurements


# In[ ]:


df.columns


# In[ ]:


for column in df.columns:
#     if column != 'precipm':
    for n in range(1, 3):
        pad_nth_day_feature(df, column, n)


# In[ ]:


df.head()


# In[ ]:


# Changes in Shape
print('Shape of Dataset: {}'.format(df.shape))
print('Shape of Each Row: {}'.format(df.iloc[0].shape))


# In[ ]:


df.info()


# ##### Doing Some Data Cleaning Operations

# In[ ]:


# Check if there is only one value in the column remove that feature
def check_uniqueness(dataframe):
    for column in dataframe.columns:
        if len(pd.Series.unique(dataframe[column])) == 1:
            dataframe.drop(column, inplace=True, axis=1)

            
    return dataframe

df = check_uniqueness(df)


# In[ ]:


# Drop Na Columns
df.dropna(inplace=True)


# In[ ]:


df.describe().T


# ### Preparing Data

# For Precipitation Prediction

# In[ ]:


y_data = df['precipm']
x_data = df.drop(['precipm'], axis=1)


# In[ ]:


print('Shape of X: {}'.format(x_data.shape))
print('Shape of Y: {}'.format(y_data.shape))


# #### Split Train and Test Data
# 

# In[ ]:


# Split into Training and Test Set
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=SEED)


# In[ ]:


# Change them all to numpy array for faster computation
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[ ]:


# Print Final Shapes of Sets
print('Training Set : X -> {}, Y -> {}'.format(x_train.shape, y_train.shape))
print('Testing Set: X -> {}, Y -> {}'.format(x_test.shape, y_test.shape))


# #### Now we have Training Set and Testing Set

# ##### Applying RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rfg = RandomForestRegressor(random_state=SEED)
rfg.fit(x_train, y_train)


# In[ ]:


y_test_pred = rfg.predict(x_test)


# In[ ]:


# Root Mean Square Error
rmf_rmse = np.round(np.sqrt(mean_squared_error(y_test,y_test_pred)), 5)
print('Root Mean Square Error: {}'.format(rmf_rmse))


# ##### 1. Applying Linear Regression

# In[ ]:


lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)


# In[ ]:


y_test_predicted = lin_reg.predict(x_test)


# In[ ]:


# Root Mean Square Error
lin_rmse = np.round(np.sqrt(mean_squared_error(y_test,y_test_predicted)), 5)
print('Root Mean Square Error: {}'.format(lin_rmse))


# ##### 2. Fitting Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


polynomial_history = []


# In[ ]:


degree_array = [2,4,5]


# In[ ]:


for degree in degree_array:

    polynomial_features= PolynomialFeatures(degree=degree)
    x_train_poly = polynomial_features.fit_transform(x_train)
    x_test_poly = polynomial_features.fit_transform(x_test)
    lin_reg = LinearRegression()
    lin_reg.fit(x_train_poly, y_train)
    y_test_poly_predicted = lin_reg.predict(x_test_poly)
    rmse = np.round(np.sqrt(mean_squared_error(y_test, y_test_poly_predicted)), 5)
    print('Root Mean Square Error: {}'.format(rmse))
    
    polynomial_history.append((degree, rmse))
    


# #### Plotting Linear Regression with Polynomial Regression RMSE

# In[ ]:


def plot_RMSE(x_axis, y_axis, figsize=(12,8)):
    plt.figure(figsize=figsize)
    plt.title('Linear Regression and Polynomial Regression with their RMSE Value')
    plt.ylabel('Root Mean Square Error')
    bar_heights = plt.bar(x_axis, y_axis)
    for rect in bar_heights:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '{}'.format(height), ha='center', va='bottom')
    plt.show()





# In[ ]:


rmse_x = ['Random Forest', 'Linear Regression'] + ['Polynomial of Degree {}'.format(x[0]) for x in polynomial_history]
rmse_y = [rmf_rmse, lin_rmse] + [x[1] for x in polynomial_history]

plot_RMSE(rmse_x, rmse_y)


# #### Therefore, In case of Regression the Minimum Value of Root Mean Square Error is seen in ***Linear Regression***

# ##### 2. Recursive Feature Selection on Linear Regression
# We will use Wrapper Method of Recursive Feature Selection

# In[ ]:


lin_reg_fs = LinearRegression()
rfe = RFE(lin_reg_fs, 10)
fit = rfe.fit(x_data, y_data)


# In[ ]:


print("Num Features: {}".format(fit.n_features_))
print("Selected Features: {}".format(fit.support_))
print("Feature Ranking: {}".format(fit.ranking_))


# In[ ]:


x_feature_selected_data = rfe.transform(x_data)
x_feature_selected_data.shape


# Now Splitting the data to train and test set

# In[ ]:


x_train_fs, x_test_fs , y_train_fs, y_test_fs = train_test_split(x_feature_selected_data, y_data, test_size=0.2, random_state=SEED)
print(x_train_fs.shape, y_train_fs.shape, x_test_fs.shape, y_test_fs.shape)


# In[ ]:


lin_reg_fs.fit(x_train_fs, y_train_fs)


# In[ ]:


y_predicted = lin_reg_fs.predict(x_test_fs)


# In[ ]:


lin_rmse_fs = np.round(np.sqrt(mean_squared_error(y_test_fs,y_predicted)), 5)
print('Root Mean Square Error: {}'.format(lin_rmse_fs))


# In[ ]:


rmse_x = rmse_x + ['Feature Selected Regression']
rmse_y = rmse_y + [lin_rmse_fs]

plot_RMSE(rmse_x, rmse_y, (18,8))


# ## Feature Selection Reduced the Mean Square Error thus with Feature Selection we got a better fit

# #### Applying Ridge Regularization in Linear Regression

# In[ ]:


lin_reg_requ = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])


# In[ ]:


lin_reg_requ.fit(x_train_fs, y_train_fs)


# In[ ]:


y_pred_regu = lin_reg_requ.predict(x_test_fs)


# In[ ]:


lin_rmse_regu = np.round(np.sqrt(mean_squared_error(y_test_fs,y_pred_regu)), 5)
print('Root Mean Square Error: {}'.format(lin_rmse_regu))


# In[ ]:


rmse_x = rmse_x + ['Ridge Regularization']
rmse_y = rmse_y + [lin_rmse_regu]


# In[ ]:


plot_RMSE(rmse_x, rmse_y, (20,8))


# # Lets Analysis The response of Neural Netoworks in This Scenario

# In[ ]:


model = Sequential()

model.add(Dense(35, activation='relu', input_shape=x_train[0].shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[ ]:


history = model.fit(x_train, y_train, epochs=200, verbose=0)


# In[ ]:


y_predicted_nn = model.predict(x_test)


# In[ ]:


nn_rmse = np.round(np.sqrt(mean_squared_error(y_test,y_predicted_nn)), 5)
print('Root Mean Square Error: {}'.format(nn_rmse))


# In[ ]:


rmse_x = rmse_x + ['Deep Neural Network']
rmse_y = rmse_y + [nn_rmse]


# In[ ]:


plot_RMSE(rmse_x, rmse_y, (22,8))


# Applying Ridge Regularization in this neural network and retraining it with output we get

# In[ ]:


model_reg = Sequential()

model_reg.add(Dense(35, activation='relu', input_shape=x_train[0].shape, kernel_regularizer=l2(0.001)))
model_reg.add(Dropout(0.3))
model_reg.add(Dense(100, activation='relu', kernel_regularizer=l2(0.001)))
model_reg.add(Dropout(0.2))
model_reg.add(Dense(200, activation='relu', kernel_regularizer=l2(0.001)))
model_reg.add(Dropout(0.2))
model_reg.add(Dense(100, activation='relu', kernel_regularizer=l2(0.001)))
model_reg.add(Dropout(0.2))
model_reg.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
model_reg.add(Dropout(0.2))
model_reg.add(Dense(1))

model_reg.compile(loss='mean_squared_error', optimizer='adam')
model_reg.summary()


# In[ ]:


history_reg = model_reg.fit(x_train, y_train, epochs=200, verbose=0)


# In[ ]:


y_predicted_nn_reg = model_reg.predict(x_test)
nn_rmse_reg = np.round(np.sqrt(mean_squared_error(y_test,y_predicted_nn_reg)), 5)
print('Root Mean Square Error: {}'.format(nn_rmse_reg))


# In[ ]:


rmse_x = rmse_x + ['DNN Regularized']
rmse_y = rmse_y + [nn_rmse_reg]


# In[ ]:


plot_RMSE(rmse_x, rmse_y, (24,12))


# In[ ]:





# In[ ]:


model_ts = Sequential()

model_ts.add(Embedding(541, 100, input_length=len(x_train[0])))
model_ts.add(Dropout(0.3))
model_ts.add(LSTM(100, activation='relu', kernel_regularizer=l2(0.001), return_sequences=True))
model_ts.add(Dropout(0.2))
model_ts.add(LSTM(200, activation='relu', kernel_regularizer=l2(0.001)))
model_ts.add(Dropout(0.2))
model_ts.add(Dense(100, activation='relu', kernel_regularizer=l2(0.001)))
model_ts.add(Dropout(0.2))
model_ts.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
model_ts.add(Dropout(0.2))
model_ts.add(Dense(1))

model_ts.compile(loss='mean_squared_error', optimizer='adam')
model_ts.summary()


# In[ ]:


history_ts = model_ts.fit(x_train, y_train, epochs=200, verbose=0)


# In[ ]:


y_predicted_ts = model_ts.predict(x_test)
nn_rmse_ts = np.round(np.sqrt(mean_squared_error(y_test,y_predicted_ts)), 5)
print('Root Mean Square Error: {}'.format(nn_rmse_ts))


# In[ ]:


rmse_x = rmse_x + ['RNN Regularized']
rmse_y = rmse_y + [nn_rmse_ts]


# In[ ]:


plot_RMSE(rmse_x, rmse_y, (26,12))


# ## Conclusion : 
# For this dataset and This hyperparameters Models can be ranked liked this

# In[ ]:


result_frame = pd.DataFrame({'Model': rmse_x, 'RMSE' : rmse_y}, columns=['Model', 'RMSE']).sort_values('RMSE').reset_index(drop=True)
result_frame.index = np.arange(1, len(result_frame) + 1)
result_frame.index.names = ['Rank']


# In[ ]:


result_frame


# In[ ]:




