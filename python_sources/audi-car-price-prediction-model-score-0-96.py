#!/usr/bin/env python
# coding: utf-8

# ![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7f/Audi_logo_detail.svg/1280px-Audi_logo_detail.svg.png)
# 
# # We will use a simple linear regressor to model Audi prices  
# During the analysis, we will compare the performance of one-hot vs binary encoding.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn import metrics


# In[ ]:


path = r'/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv'
data = pd.read_csv(path)


# In[ ]:


data


# In[ ]:


# The data was cleaned prior to uploading on Kaggle but just checking for null values.
print(data.isnull().sum())


# In[ ]:


data.describe()


# # Feature Engineering  
# I will try out both one-hot and binary coding, and compare performance.

# In[ ]:


data_onehot = pd.get_dummies(data,columns=['model', 'transmission','fuelType'])


# In[ ]:


def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );


# In[ ]:


plotting_3_chart(data_onehot, 'price')
#skewness and kurtosis
print("Skewness: " + str(data_onehot['price'].skew()))
print("Kurtosis: " + str(data_onehot['price'].kurt()))


# In[ ]:


from sklearn.model_selection import train_test_split
X = data_onehot.drop(['price'],axis=1)
y = data_onehot['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X,y)


# In[ ]:


results = X_test.copy()
results["predicted"] = regressor.predict(X_test)
results["actual"]= y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
results


# # Conclusions   
# The model performs well, and better than the Ford prediction model. This is possibly due to the smaller range of values in the Audi dataset, whereas for Ford we see a much more diverse price range making it difficult to model linearly.  
# Next, we will see how performance changes with different encoding.

# In[ ]:


# binary encoding

import category_encoders as ce
data_bin = data.copy()
encoder = ce.BinaryEncoder(cols=['model','transmission','fuelType'])
data_bin = encoder.fit_transform(data_bin)
data_bin


# In[ ]:


plotting_3_chart(data_bin, 'price')
#skewness and kurtosis
print("Skewness: " + str(data_bin['price'].skew()))
print("Kurtosis: " + str(data_bin['price'].kurt()))


# In[ ]:


from sklearn.model_selection import train_test_split
X = data_bin.drop(['price'],axis=1)
y = data_bin['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X,y)


# In[ ]:


results = X_test.copy()
results["predicted"] = regressor.predict(X_test)
results["actual"]= y_test
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
results


# # Conclusions  
# Performance with binary encoding worse than using one-hot encoding.  
# 
# Next, I'll try one-hot encoding, and transform the data to reduce skew.

# In[ ]:


## trainsforming target variable using numpy.log1p,
log_data = data_onehot.copy()
log_data["price"] = np.log1p(log_data["price"])

## Plotting the newly transformed response variable
plotting_3_chart(log_data, 'price')
#skewness and kurtosis
print("Skewness: " + str(log_data['price'].skew()))
print("Kurtosis: " + str(log_data['price'].kurt()))


# In[ ]:


(log_data.corr()**2)["price"].sort_values(ascending = False)[1:]


# In[ ]:


from sklearn.model_selection import train_test_split
X = log_data.drop(['price'],axis=1)
y = log_data['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)


# In[ ]:


# Much improved score after adjusting distribution

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
print('Accuracy on Testing set: %.1f ' %(regressor.score(X_train,y_train)*100))


# In[ ]:


from sklearn.svm import SVR
pipeline_svr=Pipeline([('scalar1',StandardScaler()),
                     ('pca1',PCA(n_components=2)),
                     ('lr_classifier',SVR(kernel='linear'))])
pipeline_svr.fit(X_train, y_train)
pipeline_svr.score(X_test,y_test)


# In[ ]:


from sklearn.linear_model import Ridge,ElasticNet
ridge=Ridge(alpha=2,max_iter=1000,random_state=1)
ridge.fit(X_train,y_train)
print('Accuracy on Testing set: %.1f ' %(ridge.score(X_test,y_test)*100))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(random_state=1)
rf_reg.fit(X_train, y_train)
print('Accuracy on Testing set: %.1f ' %(rf_reg.score(X_test,y_test)*100))


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
GB=GradientBoostingRegressor(random_state=0)
GB.fit(X_test,y_test)
print('Performance Score(GB): %.1f ' %(GB.score(X_test,y_test)*100))


# In[ ]:


from xgboost import XGBRegressor
XGB=XGBRegressor(random_state=0)
XGB.fit(X_train,y_train)
print('Performance score(XGB): %.1f ' %(XGB.score(X_test,y_test)*100))


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


results = X_test.copy()
results["predicted"] = np.expm1(XGB.predict(X_test))
results["actual"]= np.expm1(y_test)
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(2)
results


# In[ ]:


custom = X_test.copy()
custom = custom[custom['model_ A4'] == 1]
custom = custom[custom['year'] == 2018]
custom = custom[custom['transmission_Automatic'] == 1]
custom = custom.iloc[1].copy()
# # custom['engineSize'] = 1.0
# # custom['mpg'] = 65
custom['mileage'] = 15000
# # custom['fuelType_Diesel'] = 0
# # custom['fuelType_Petrol'] = 1

custom


# In[ ]:


custom = custom.values.reshape(-1, 1)
flat_list = []
for sublist in custom:
    for item in sublist:
        flat_list.append(item)

print(np.expm1(GB.predict([flat_list])))


# In[ ]:




