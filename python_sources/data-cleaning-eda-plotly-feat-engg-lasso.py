#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
from scipy import stats
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


#Plotting
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.figure_factory as ff
from plotly import subplots
# Display plot in the same cell for plotly
init_notebook_mode(connected=True)


# In[ ]:


import sklearn
from sklearn import linear_model,metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, make_scorer


# In[ ]:


print('Numpy : Version ',np.__version__)
print('Pandas : Version ',pd.__version__)
print('Plotly : Version ',plotly.__version__)
print('Plotly Express : Version ',plotly.express.__version__)
print('Scikit-Learn : Version ',sklearn.__version__)


# In[ ]:


# Colors from material design to make visualizations look awesome!
material_blue = '#81d4fa'
dark_blue = '#1e88e5'
material_green = '#a5d6a7'
dark_green = '#43a047'
material_indigo = '#3f51b5'
material_red = '#f44336'
bg_color = '#212121'


# In[ ]:


# Importing the train dataset
df_train = pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


# Importing the test dataset
df_test = pd.read_csv('../input/test.csv')
df_test.head()


# In[ ]:


# Column names
df_train.columns


# In[ ]:


# Shape => Tuple of no. of rows and columns
df_train.shape


# ### Observations:
# **a) Dataset contains null values in some rows.**
# 
# **b) The categorical variables are in object type. Hence, they should be transformed to 'category' type in Pandas.**

# In[ ]:


df_train.describe()


# ## 1) Data Cleaning
# - Visualizing data using Plotly.
# - Dealing with missing values.

# In[ ]:


total = df_train.isnull().sum().sort_values(ascending=False)
missing_cols = list(total.index)
total_values = list(total[:])
df_missing = pd.DataFrame(dict({'columns':missing_cols,'total':total_values}))
df_missing = df_missing[df_missing['total']>0]
df_missing


# In[ ]:


fig = px.bar(df_missing, x='columns',y='total')
fig.update_traces(marker_color=dark_blue)
iplot(fig)


# In[ ]:


missing_data_cols = list(df_missing['columns'])
df_train[missing_data_cols].head()


# In[ ]:


df_train[df_train['PoolArea']>0].shape[0]


# In[ ]:


df_train[df_train['Fireplaces']>0].shape[0]


# ### **Observations: -**
# 
# 1) PoolQC is a variable which represents Pool Quality. There are not many houses with pools. Hence, most of the values in the PoolArea column (Pool area in sq. ft.) are equal to zero.
# 
# 2) MiscFeature is used to represent Miscellaneous features. There are not many miscellaneous features.
# 
# 3) Alley is used to represent the type of alley access to the property. There is alley acces to very less number of properties.
# 
# 4) Fence is used to represent Fence Quality. Most of the houses do not have a fence.
# 
# 5) FireplaceQu is used to represent the Quality of the fireplace. Nearly half of the houses in the dataset do not have a fireplace. Hence, some values in the Fireplaces column (No. of fireplaces) are equal to zero.
# 
# 6) LotFrontage is a column used to denote the linear feet of street connected to the property. Since it contains more than 200 missing values, it is better to remove the column than replacing the missing values with the measures of central tendancy.
# 
# ### Inferences
# - The columns PoolQC, MiscFeature, Alley, Fence and FireplaceQu contain null values not because of errors during data entry, but because of them actually being null.
# 
# - The numerical columns PoolArea and Fireplace contain zero values at the null values of PoolQC and FireplaceQu respectively, since it makes sense that the qualitative variables contain null values if the PoolArea is zero
# 
# - The same can be said about Fireplaces and FireplaceQu.

# In[ ]:


cols_to_be_del = ['PoolQC','PoolArea','MiscFeature','Alley','Fence','Fireplaces','FireplaceQu','LotFrontage']
df_train.drop(cols_to_be_del, inplace=True, axis=1)
df_train.shape


# In[ ]:


df_test.head()


# In[ ]:


compare_df = df_train[df_train['GarageCars']>0]['Id'] == df_train[df_train['GarageArea']>0]['Id']
compare_df.shape[0]


# ### Inferences:
# - There are no abnormal values for Garage variables, i.e., there aren't cases where GarageCars > 0 and GarageArea = 0
# - The converse holds true as well, i.e., there aren't cases where GarageArea > 0 and GarageCars = 0
# - This means that whoever owns a house with a garage is maintaining atleast one car. Let's keep this in mind.

# In[ ]:


df_missing.tail(13)


# ### **Observations:-**
# 7) Although GarageCond, GarageQual are numerical in nature, they are actually categorical.
# 
# 8) GarageYrBlt, GarageFinish, GarageType are categorical variables which contain null values. These null values cannot be imputed since the null in this context means that there is no garage.
# 
# 9) There is no record where the GarageCars is zero for non null values of GarageType. This means that every house containing a garage also has a car associated with it.
# 
# 10) The features BsmtQual, BsmtCond, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 also follow the same trend.
# 
# ### Inferences
# - The garage variables i.e., the GarageCond, GarageQual, GarageYrBlt, GarageFinish, GarageType can be removed.
# - The variables GarageArea, GarageCars however can be useful, since there are only 81 zero values.
# - In the same way, BsmtArea is useful to represent the missing values from the corresponding Bsmt variables.
# 
# **Note : All of these inferences are made keeping simplicity in mind. Instead of just removing them, one could argue that a careful study of the data can be done to fill in the missing values accordingly. The assumption here is that most of the categorical variables do not affect the target variable heavily since they contain a lot of missing values and they are already being represented in this dataset by their respective categorical values.**

# In[ ]:


cols_to_be_del_2 = ['GarageCond', 'GarageQual', 'GarageYrBlt', 'GarageFinish', 'GarageType','MasVnrType']
df_train.drop(cols_to_be_del_2, inplace=True, axis=1)
df_train.shape


# In[ ]:


cols_to_be_del_3 = ['BsmtQual','BsmtCond','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical']
df_train.drop(cols_to_be_del_3, axis=1, inplace=True)
df_train.shape


# In[ ]:


df_missing.tail(3)


# In[ ]:


df_train['MasVnrArea'].describe()


# ### Inferences
# - MasVnrArea has very less number of missing values. Hence, it can be imputed with mean, median or mode.
# - To keep things even, let's replace with median since 0 is the most repeated value.

# In[ ]:


df_train.fillna(df_train.median(), inplace=True)
df_train.isnull().values.sum()


# ## 2) Univariate Analysis of SalePrice
# - The ultimtate goal of our analysis is to predict the value of SalePrice with the help of our features.
# - Data transformation for skewed variables needs to be done.
# - Let's explore the target variable first.

# In[ ]:


print("Skewness: {}".format(str(df_train['SalePrice'].skew())))
print("Kurtosis: {}".format(str(df_train['SalePrice'].kurt())))


# In[ ]:


fig = px.histogram(df_train, x='SalePrice', nbins=100)
iplot(fig)


# In[ ]:


fig = ff.create_distplot([df_train['SalePrice']],['SalePrice'],bin_size=20000, colors=[dark_blue],show_hist=False)
iplot(fig, filename='Basic Distplot')


# In[ ]:


def box_plot(dataframe, columns):
    data = []
    for column in columns:
        data.append(go.Box(y=df_train[column], name=column, boxmean='sd',fillcolor=material_blue,marker=dict(color=dark_blue)))
    return data


# In[ ]:


target_box_data = box_plot(df_train,['SalePrice'])
iplot(target_box_data)


# In[ ]:


def violin_plot(dataframe, columns):
    data = []
    for column in columns:
        data.append(go.Violin(y=dataframe[column], box_visible=True, line_color=bg_color,
                               meanline_visible=True, fillcolor=material_green, opacity=0.8,
                               x0=column))
    return data


# In[ ]:


violin_fig = violin_plot(df_train, ['SalePrice'])
iplot(violin_fig, filename = 'SalePriceViolin')


# ### Observations: -
# 1) SalePrice is fairly positively skewed, i.e., it is more inclined towards left in the distplot and violin plot.
# 
# 2) The values above 340k are treated as outliers.
# 
# ### Inferences: - 
# - SalePrice needs to be transformed to a known distribution (preferrably the one which brings the value of skewnwss to zero)

# In[ ]:


def qqplots(df, col_name, distribution):
    qq = stats.probplot(df, dist=distribution, sparams=(0.5))
    x = np.array([qq[0][0][0],qq[0][0][-1]])
    pts = go.Scatter(x=qq[0][0],
                     y=qq[0][1], 
                     mode = 'markers',
                     showlegend=False,
                     name=col_name,
                     marker = dict(
                            size = 5,
                            color = material_indigo,
                        )
                    )
    line = go.Scatter(x=x,
                      y=qq[1][1] + qq[1][0]*x,
                      showlegend=False,
                      mode='lines',
                      name=distribution,
                      marker = dict(
                            size = 5,
                            color = material_red,
                        )
                     )
    
    data = [pts, line]
    return data


# In[ ]:


#Plot data for 4 different distributions
norm_data = qqplots(df_train['SalePrice'], 'SalePrice','norm')
power_law_data = qqplots(df_train['SalePrice'], 'SalePrice','powerlaw')
poisson_data = qqplots(df_train['SalePrice'], 'SalePrice','poisson')
lognorm_data = qqplots(df_train['SalePrice'], 'SalePrice','lognorm')


# In[ ]:


fig = subplots.make_subplots(rows=2, cols=2, subplot_titles=('Normal Distribution', 'Power Law Distribution',
                                                          'Poisson Distribution', 'Log Normal Distribution'))
fig.append_trace(norm_data[0], 1, 1)
fig.append_trace(power_law_data[0], 1, 2)
fig.append_trace(poisson_data[0], 2, 1)
fig.append_trace(lognorm_data[0], 2, 2)
fig['layout'].update(height=600, width=900, title='Comparision of QQ-plots')

iplot(fig, filename='make-subplots-multiple-with-titles')


# ### Observations: -
# 3) The Straight line is seen in QQ-plot of SalePrice vs Log-Normal i.e., assuming that the points at the end are outliers.
# 
# Let's draw a concrete straight line just to be sure.

# In[ ]:


layout = dict(xaxis = dict(zeroline = False,
                           linewidth = 1,
                           mirror = True),
              yaxis = dict(zeroline = False, 
                           linewidth = 1,
                           mirror = True),
             )

fig = dict(data=lognorm_data, layout=layout)
iplot(fig, show_link=False)


# In[ ]:


# Creating a pipeline
df_pipe = df_train.copy()


# In[ ]:


df_pipe['SalePrice'] = np.log(df_train['SalePrice'])
print("Skewness: {}".format(str(df_pipe['SalePrice'].skew())))
print("Kurtosis: {}".format(str(df_pipe['SalePrice'].kurt())))


# In[ ]:


fig = px.histogram(df_pipe,'SalePrice')
iplot(fig)


# In[ ]:


fig = ff.create_distplot([df_pipe['SalePrice']],['SalePrice Log Normal'],bin_size=0.08, colors=[dark_blue], show_hist=False)
iplot(fig, filename='Distribution plot for Sale Price (Log transform)')


# In[ ]:


log_transformed_qqplot_data = qqplots(df_pipe['SalePrice'], 'SalePrice Log transform','norm')
layout = dict(xaxis = dict(zeroline = False,
                       linewidth = 1,
                       mirror = True),
          yaxis = dict(zeroline = False, 
                       linewidth = 1,
                       mirror = True),
         )
qqplot_fig = dict(data=log_transformed_qqplot_data, layout=layout)
iplot(qqplot_fig, show_link=False)


# In[ ]:


target_transformed_box_data = [go.Box(y=df_pipe['SalePrice'], name='SalePrice Log transform', boxmean='sd',fillcolor=material_green,marker=dict(color=dark_green))]
iplot(target_transformed_box_data)


# In[ ]:


target_transformed_violin_data = violin_plot(df_pipe,['SalePrice'])
iplot(target_transformed_violin_data, filename = 'SalePriceLogViolin', validate = False)


# In[ ]:


fig = subplots.make_subplots(rows=1, cols=2)
fig.append_trace(target_box_data[0], 1, 1)
fig.append_trace(target_transformed_box_data[0], 1, 2)
fig['layout'].update(height=600, width=950, title='SalePrice Unchanged vs Log transformed')
iplot(fig, filename='SalePrice-unch-vs-log-box')


# In[ ]:


print("Skewness: {}".format(str(df_pipe['SalePrice'].skew())))
print("Kurtosis: {}".format(str(df_pipe['SalePrice'].kurt())))


# #### Transformation Success!
# - Skewness is down significantly,
# - SalePrice is now closer to a normal distribution.

# ## 3) EDA and Feature Engineering
# - Combination of certain features.
# - Creating age based on year given.
# - Visualization of numerical features.

# In[ ]:


#Feature Engineering

df_pipe['TotalSF']=df_pipe['TotalBsmtSF'] + df_pipe['1stFlrSF'] + df_pipe['2ndFlrSF']
df_pipe['TotalSQR_Footage'] = (df_pipe['BsmtFinSF1'] + df_pipe['BsmtFinSF2'] +
                                df_pipe['1stFlrSF'] + df_pipe['2ndFlrSF'])

df_pipe['Total_Bathrooms'] = (df_pipe['FullBath'] + (0.5 * df_pipe['HalfBath']) +
                              df_pipe['BsmtFullBath'] + (0.5 * df_pipe['BsmtHalfBath']))

df_pipe['AgeSinceRemodel'] = 2010 - df_train['YearRemodAdd']
df_pipe['AgeSinceBuilt'] = 2010 - df_train['YearBuilt']


# In[ ]:


corr_matrix = df_pipe.corr()
corr_matrix = corr_matrix.abs()
fig = go.Figure(data=go.Heatmap(z=corr_matrix))
iplot(fig)


# In[ ]:


k = 15 #number of variables for heatmap
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cols_to_be_del_pipe = ['FullBath','1stFlrSF']
cols = list(cols)
for i in cols_to_be_del_pipe:
    cols.remove(i)

correlations = list(df_pipe[cols].corr().round(2).values)
correlation_matrix = [list(correlations[i]) for i in range(len(correlations))]
fig = ff.create_annotated_heatmap(z=correlation_matrix,x=list(cols),y=list(cols))
iplot(fig, filename='annotated_heatmap')


# In[ ]:


#Categorical variables
ordinal_cols = ['BldgType','HeatingQC','Functional']
binary_cols = ['PavedDrive','CentralAir']
df_pipe[ordinal_cols] = df_pipe[ordinal_cols].astype('category')
df_pipe[ordinal_cols].head()


# In[ ]:


def count_plot(df,col_name):
    value_counts_series = df[col_name].value_counts()
    categories = list(value_counts_series.index)
    values = list(value_counts_series)
    fig = go.Figure(data=[go.Bar(
            x=categories, 
            y=values,
            textposition='auto',
        )])
    iplot(fig)


# In[ ]:


count_plot(df_pipe, 'HeatingQC')


# In[ ]:


fig = px.box(df_train, x='HeatingQC', y='SalePrice')
iplot(fig)


# In[ ]:


# Label Encoding for HeatingQC
categories = list(df_pipe['HeatingQC'].unique())
encoding_dict = {col:x for col,x in zip(categories,range(5,0,-1))}
replace_dict_heatingqc = {'HeatingQC':encoding_dict}
df_pipe.replace(replace_dict_heatingqc, inplace = True)
df_pipe[ordinal_cols].head()


# In[ ]:


skews = []
kurts = []
for col in cols:
    skews.append(df_pipe[col].skew())
    kurts.append(df_pipe[col].kurt())
dict_skew_data = {'Feature':cols, 'Skew':skews, 'Kurt':kurts}
df_skews = pd.DataFrame(dict_skew_data, columns=['Feature','Skew','Kurt'])
df_skews


# In[ ]:


df_pipe['TotalSF'] = np.log(df_pipe['TotalSF'])


# In[ ]:


iplot(qqplots(df_pipe['TotalSF'],'TotalSF Log transform','norm'))


# In[ ]:


fig = px.scatter_matrix(df_pipe, dimensions=['TotalSF','GrLivArea','TotalSQR_Footage','SalePrice'],color='OverallQual')
iplot(fig)


# In[ ]:


fig = px.scatter_matrix(df_pipe, dimensions=['Total_Bathrooms','GarageArea','GarageCars','SalePrice'],color='OverallQual')
iplot(fig)


# In[ ]:


fig = px.scatter_matrix(df_pipe, dimensions=['AgeSinceBuilt','AgeSinceRemodel','TotRmsAbvGrd','TotalBsmtSF','SalePrice'],color='OverallQual')
iplot(fig)


# In[ ]:


def draw_scatter_plot(col_name_x, col_name_y):
    trace = go.Scatter(
        x = df_pipe[col_name_x],
        y = df_pipe[col_name_y],
        mode = 'markers'
    )
    data = [trace]
    iplot(data, filename='basic-scatter')


# ## 4) LASSO (Linear regression)
# - The model which can deal with collinearity due to L1 regualizer (Since it takes the absolute value, it can penalize unnecessary variables and make them equal to zero).
# - One of the most basic models.

# In[ ]:


y_tr = df_pipe['SalePrice']
x_tr = df_pipe[cols]
lasso = linear_model.Lasso()
parameters = {'alpha': [1]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring=make_scorer(metrics.mean_squared_error), cv=10)
lasso_regressor.fit(x_tr, y_tr)
y_pred_lasso = lasso_regressor.predict(x_tr)


# In[ ]:


mse = metrics.mean_squared_error(y_tr, y_pred_lasso, sample_weight=None)
rmse = np.sqrt(mse)
print(rmse)


# ## Conclusions:
# 1) So far, we have explored the numerical variables to a certain extent. However, most of the categorical variables remain to be explored.
# 
# 2) The feature selection of the numerical variables was based on correlations and scatterplot matrices. This should suffice for a basic approach.
# 
# 3) The Kernel will be continued . If you are readng this, thank you for checking out this kernel. It must've been tough, since I've included a lot of visualizations and I could not document some of my decisions.
# 
# 4) Please leave a message in the discussions if there's more to be included.
# 
# ### Thank you!

# ### References:
# 
# **1) Plotly**
# - https://plot.ly/python/
# - https://www.kaggle.com/thebrownviking20/intermediate-visualization-tutorial-using-plotly
# - https://www.kaggle.com/artemseleznev/plotly-tutorial-for-beginners
# - https://stackoverflow.com/questions/51170553/qq-plot-using-plotly-in-python
# 
# **2) Kernels**
# - https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# - https://www.kaggle.com/hamzaben/eda-feature-eng-and-model-blending-top-20
# - https://www.kaggle.com/erikbruin/house-prices-lasso-xgboost-and-a-detailed-eda
# - https://www.kaggle.com/masumrumi/a-stats-analysis-and-ml-workflow-of-house-pricing
# - https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1
