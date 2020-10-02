#!/usr/bin/env python
# coding: utf-8

# > **EDA and Basic Model**
# 
# - For EDA I would be using the pandas profiling package (since that does a lot of the EDA for me) and will make changes to individual variable from there. If you have not looked at pandas profiling package then you should because it is a very powerful EDA package
# - The next step would be to create a simple GLM model and a RF model
# - I will then use the H2O package to run the model
# - Compare the results of manual tweaking vs H2O model

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pp
import random

#plotly packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tools


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


print('There are ',train.shape[0],'rows and ',train.shape[1],'columns in the train dataset')
print('There are ',test.shape[0],'rows and ',test.shape[1],'columns in the test dataset')


# In[ ]:


train.head(5)


# In[ ]:


print(list(train.columns))


# ## Looking at the Target Variable

# In[ ]:


data = go.Histogram(x=train.SalePrice)
fig = [data]
py.iplot(fig)


# Lets try the log of the target varirable

# In[ ]:


data = go.Histogram(x=np.log(train.SalePrice))
fig = [data]
py.iplot(fig)


# In[ ]:


#Saving the log_price variables
SalesPrice_log = np.log(train.SalePrice)
SalePrice = train.SalePrice
Id = train.Id


# In[ ]:


#Removing the ID and SalePrice Variable
train = train.drop(['Id','SalePrice'], 1)
test = test.drop(['Id'], 1)
print(train.shape)
print(test.shape)


# In[ ]:


#Combining the train and testing dataset
train_test = pd.concat([train,test], axis=0, sort=False)
train_test.shape


# # Removing Extreme Values from Train Dataset and Identifying Columns to Remove

# ## Splitting the data into numerical and categorical

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_num = train.select_dtypes(include=numerics)


# In[ ]:


columns_to_rem =  train_num.columns.tolist()
train_cat = train[train.columns.difference(columns_to_rem)]


# ## Numerical data EDA

# #### Removing outliers from numerical data using z-score and IQR (Interquartile Range) 

# In[ ]:


def missing_values(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    return missing_value_df

missing_values(train_num).sort_values(by=['percent_missing'], ascending = False).head()


# Removing missing values manually will be done in a later version

# In[ ]:


#Filling the missing values with mean
train_num = train_num.fillna(train_num.mean())


# In[ ]:


from scipy.stats import zscore

z = train_num.apply(zscore)
threshold = 3
np.where(z > 10)


# In[ ]:


Q1 = train_num.quantile(0.25)
Q3 = train_num.quantile(0.75)
IQR = Q3 - Q1


# In[ ]:


#Identifying rows to remove using the z-score and the interquartile range
print(train_num.shape)
train_num = train_num[(z < 10).all(axis=1)]
print(train_num.shape)


# In[ ]:


#Identifying rows to remove using the interquartile range
#ts = 20
#train_num = train_num[~((train_num < (Q1 - ts * IQR)) |(train_num > (Q3 + ts * IQR))).any(axis=1)]
#train_num.shape


# # Normalizing values based on Box-Cox Transformation

# Before normalizing, I would need to combine the test and train numeric dataset.

# In[ ]:


train_test_num = pd.concat([train_num,test[list(train_num.columns)]])
train_test_num.head()


# In[ ]:


#Filling the missing values with mean
train_test_num = train_test_num.fillna(train_test_num.mean())


# In[ ]:


profile = pp.ProfileReport(train_test_num)


# In[ ]:


profile


# ## Skew Features

# In[ ]:


from scipy.stats import skew

skew_features = train_test_num.apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews.head()


# In[ ]:


from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

#Setting threshold for skew
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

#creating a new df for skew
train_test_num_skew = pd.DataFrame()

for i in skew_index:
    train_test_num_skew[i]= boxcox1p(train_test_num[i], boxcox_normmax(train_test_num[i]+1))

#Changing col names of new df
skew_cols = [s + '_skew' for s in list(train_test_num_skew.columns)]

train_test_num_skew.columns = skew_cols
train_test_num_skew.head()


# Lets look at the transformations

# In[ ]:


print(train_test_num[skew_index].shape) 
print(train_test_num_skew.shape)

train_test_skew_compare = pd.concat([train_test_num[skew_index],train_test_num_skew], axis = 1)
train_test_skew_compare = train_test_skew_compare.reindex(sorted(train_test_skew_compare.columns), axis=1)
train_test_skew_compare.head()


# In[ ]:


def pick_color():
    colors = ["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink"]
    random.shuffle(colors)
    return colors[0]

def Hist_plot(data,i):
    trace0 = go.Histogram(
        x= data.iloc[:,i],
        name = str(data.columns[i]),
        nbinsx = 100,
        marker= dict(
            color=pick_color(),
            line = dict(
                color = 'black',
                width = 0.5
              ),
        ),
        opacity = 0.70,
  )
    fig_list = [trace0]
    title = str(data.columns[i])
    return fig_list, title
    
def Plot_grid(data, ii, ncols=2):
    plot_list = list()
    title_list = list()
    
    #Saving all the plots in a list
    for i in range(ii):
        p = Hist_plot(data,i)
        plot_list.append(p[0])
        title_list.append(p[1])
    
    #Creating the grid
    nrows = max(1,ii//ncols)
    i = 0
    fig = tools.make_subplots(rows=nrows, cols=ncols, subplot_titles = title_list)
    for rows in range(1,nrows+1):
        for cols in range(1,ncols+1):
            fig.append_trace(plot_list[i][0], rows, cols)
            i += 1
    fig['layout'].update(height=400*nrows, width=1000)
    return py.iplot(fig)


# In[ ]:


Plot_grid(train_test_skew_compare,len(train_test_skew_compare.columns),2)


# In[ ]:


#Combinig the skewed data with the original train test dataset
train_test_num = pd.concat([train_test_num, train_test_num_skew], axis = 1)
train_test_num.shape


# **Variable**: BsmtFinSF1  
# **Comments**: ~32.0% are zeros and there are also some extreme values (>2000)  
# **Findings**: Looks like there are zeros whererever the BsmtFinType1 and BsmtFinType2 is Unf (Unfinished). These basement sqft is not actually zero. I would have to convert it to -1.
# I will also remove rows greater than 2000 (4 rows)

# In[ ]:


#Lets check how the data looks like when BsmtFinSF1 is zero for columns that contain Bsmt
bsmt_cols = [col for col in train_test if 'Bsmt' in col]
train_bsmt = train[bsmt_cols][train['BsmtFinSF1']== 0]
train_bsmt.head(10)


# In[ ]:


def compare_df(data1, data2, column_name):
    data1 = data1.filter(regex = column_name).iloc[:,0]
    df1 = pd.DataFrame(data1.value_counts(normalize=True) * 100)
    
    data2 = data2.filter(regex = column_name).iloc[:,0]
    df2 = pd.DataFrame(data2.value_counts(normalize=True) * 100)
    
    df3 = pd.merge(df1,df2, how='outer', left_index=True, right_index=True)
    
    return df3

train_bsmt_cat = train_bsmt.select_dtypes(exclude=["number","bool_"])

for col in train_bsmt_cat.columns:
    print(compare_df(train_bsmt,train,col))


# In[ ]:


train_test_num['BsmtFinSF1'] = np.where(train_test_num['BsmtFinSF1'] <= 0, -1,train_test_num['BsmtFinSF1'])


# **Variable**: BsmtFinSF2  
# **Comments**: ~88.0% are zeros  
# **Findings**: This tells us that BsmtFinSF2 is zero when the Basement2 is unfinished. I will change it to -1

# In[ ]:


train_bsmt = train[bsmt_cols][train['BsmtFinSF2']== 0]
for col in train_bsmt_cat.columns:
    print(compare_df(train_bsmt,train,col))


# In[ ]:


train_test_num['BsmtFinSF2'] = np.where(train_test_num['BsmtFinSF2'] <= 0, -1,train_test_num['BsmtFinSF2'])


# In[ ]:


missing_values(train_test_num).sort_values(by=['percent_missing'], ascending = False).head()


# ## Categorical Variables EDA

# In[ ]:


train_test_cat = pd.concat([train_cat,test[list(train_cat.columns)]])
train_test_cat.head()


# In[ ]:


missing_values(train_test_cat).sort_values(by=['percent_missing'], ascending = False)


# In[ ]:


profile = pp.ProfileReport(train_test_cat)


# In[ ]:


profile


# ## Model Development

# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

