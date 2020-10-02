#!/usr/bin/env python
# coding: utf-8

# ![eee](http://googletoday.net/wp-content/uploads/2015/10/GoogleMerchandiseStore-Haul-1024x488.jpg)
# 
# Data science has a lot of applications in e-commerce and marketing. This competition introduces yet another challenge in these domains that can be tackled using Machine Learning techniques.
# 
# **The goal** is to predict the natural log of the sum of all transactions per user. In layman's terms, we want to design an algorithm that will identify clients who spend a lot of money on Google Merchandise Store and those who don't.
# 
# **This Kernel is dedicated to Exploratory Data Analysis**. I will try to gain as many insights as possible. I will do another kernel to benchmark different models on this dataset, from the most interpretable to the most complex one.
# 
# Here is what I will do :
# 1. Check missing values and data processing
# 2. Compute statistics on the target variable / Hypothesis Testing
# 3. Exploratory Data Analysis
# 4. Features Selection
# 
# Enjoy and feel free to give any constructive critics !

# In[ ]:



import numpy as np 
import pandas as pd 
from pandas.io.json import json_normalize
import json
import os

#Libraries for plots
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
from IPython.display import HTML
plt.style.use('fivethirtyeight')
sns.set_context(rc = {"lines.linewidth": 2})

import random
import datetime as dt

# For feature Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score



# Stats from Scipy for Hypothesis testing
from scipy.stats import norm
from scipy.stats import kurtosis, skew
from scipy.stats import shapiro
from scipy.stats import normaltest



def load_df(csv_path = '../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters = {column: json.loads for column in JSON_COLUMNS}, 
                     dtype = {'fullVisitorId': 'str'}, 
                     nrows = nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

print(os.listdir("../input"))

# For random color in my Plotly plots.
def randomc():
    r = random.randint(1,255)
    g = random.randint(1,255)
    b = random.randint(1,255)
    return('rgb({},{},{})'.format(r,g,b))


# I only load the training set to avoid memory issues. If you fork this kernel, the data processing steps on both sets.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = load_df()\n#test_df = load_df("../input/test.csv")')


# **A quick summary of the training set.**

# In[ ]:


train_df.info()


# **Data Pre-Processing**

# In[ ]:


# to numeric values
train_df['totals.transactionRevenue'] = pd.to_numeric(train_df['totals.transactionRevenue'])
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0)
train_df['totals.hits'] = pd.to_numeric(train_df['totals.hits']).fillna(0)
train_df['totals.pageviews'] = pd.to_numeric(train_df['totals.pageviews']).fillna(0)

#dates from int to Timestamp
train_df['date'] = pd.to_datetime(train_df.date, format='%Y%m%d')

# drop useless colums that have only 1 value
train_const_cols = [ col for col in train_df.columns if len(train_df[col].unique()) == 1]
train_df.drop(train_const_cols, axis = 1, inplace = True)


# Compute the ratio of Missing values

# In[ ]:


misvalue_dic = {}
for column in train_df :
    v = 100 * train_df[column].isna().sum() / len(train_df)
    column
    if v > 0 :
        misvalue_dic[column]=v


# In[ ]:


misvalue_dic


# First plot : 
# **Ratio of missing values**

# In[ ]:


trace1 = go.Bar(
                x = list(misvalue_dic.keys()),
                y = list(misvalue_dic.values()),
                name = "Missing Values",
                marker = dict(color=randomc()))
data = [trace1]
layout = go.Layout(
    xaxis = dict(tickangle = -25),
    title='Percentage of missing value for uncomplete columns',
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)

    


# Some columns have almost 100% of missing values ! 
# <br/> Might be tempting to drop them thinking they are useless. We don't know anything yet about this dataset, so it would be risky.

# In[ ]:


#Some insight on the target variable : Total Transaction Revenue
gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

#take non zero total transaction revenues
nz_df = gdf.loc[gdf["totals.transactionRevenue"] > 0]
log_nz = np.log1p(nz_df["totals.transactionRevenue"])
nz = nz_df["totals.transactionRevenue"]



# In[ ]:


print("Among all the visitors, only {0:.3f} percent have bought something from August 2016 to August 2017".format( round(100*len(nz)/len(gdf),4)))


# **Distribution of the Target Variable
# **
# 
# Let's plot the distribution of the Target Variable, non null values only<br/>
# Without taking the log of totals, here is what it looks like.

# In[ ]:


plt.subplots(figsize = (14, 7))
ax = sns.distplot(a = nz).set_title('Distribution of non null Totals Transaction Revenues')


# Taking the log here is what we get. Much nicer !

# In[ ]:


plt.subplots(figsize = (16, 7))
ax = sns.distplot(a = log_nz,axlabel = "ln(1+ totals.transactionRevenue)" ).set_title('Distribution of Ln(1+non null Totals Transaction Revenues)')


#  So the distribution has a nice bell shape but the tails are too light to be a normal distribution. We can verify this assumption with **normality tests**.  <br><br>
# Here is a quick description of the distribution of log values. Let's compute the [skewness](https://en.wikipedia.org/wiki/Skewness) and [kurtosis](https://fr.wikipedia.org/wiki/Kurtosis) before the normality test

# In[ ]:


log_nz.describe()


# In[ ]:


print( "The skewness of the distribution is {}.".format(log_nz.skew()))
print( "The kurtosis of the distribution is {}.".format(log_nz.kurt()))


# |skewness|< 0.5 ==> The distribution is symmetric around the mean <br>
# No conclusion for the kurtosis though as it is between -2 and 2.
# <br> <br>
# 
# **Normality Test:**
# We will perform [Shapiro-Wilk test](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test), based on expected values, and [d'Agostino test](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test), based on kurtosis and skewness. Noteworthy : the Shapiro test is said to be less acurate when n is large (>1000) .
# 

# In[ ]:


# Shapiro Normality Test
stat, p = shapiro(log_nz)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[ ]:


#D'Agostino Test
stat, p = normaltest(log_nz)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')
    


# Conclusion : WIth such a low p-value, we reject the hypothesis of normality <br>
# Link to remind [Misunderstanding of p-values](https://en.wikipedia.org/wiki/Misunderstandings_of_p-values)

# ![](https://juicebubble.co.za/wp-content/uploads/2018/03/normal-paranormal-distribution-white-400x400.png)

# Are there specific periods where people are show more interes for Google products ? For example when a new version of Android is released ? Let's see for ourself <br> <br>
# 
# **Number of visits per day : **

# In[ ]:


nz2 = train_df.loc[train_df['totals.transactionRevenue'] > 0]
z2 = train_df.loc[train_df['totals.transactionRevenue'] == 0]
fig, ax1 = plt.subplots(figsize = (18, 10))
plt.title("Revenue and Non Revenue visits");
z2.groupby(['date'])['totals.transactionRevenue'].count().plot()
ax1.set_ylabel('Visits count')
plt.legend(['Non-Revenue and Revenue users'], loc =(0.70,0.9) )
ax2 = ax1.twinx()
nz2.groupby(['date'])['totals.transactionRevenue'].count().plot(color='brown')
ax2.set_ylabel('Visits count')
plt.legend(['Revenue users'], loc = (0.7, 0.95))
plt.grid(False)


# **Worth noticing :**
# <br>
# Revenue transaction increase a lot in december. Good Christmas gift !

# In[ ]:


def barplot_visit(feat):
    feat_data = 100*train_df[feat].value_counts()/len(train_df)
    feat_data = feat_data.to_frame().reset_index()
    
    nz = 100 * train_df.loc[train_df['totals.transactionRevenue'] > 0][feat].value_counts() / len(train_df.loc[train_df['totals.transactionRevenue'] > 0])
    nz = nz.to_frame().reset_index()
    
    trace1 = go.Bar(
        x=feat_data['index'],
        y=feat_data[feat],
        name='Zero Revenue',
        marker=dict(color=randomc())
    )
    
    trace2 = go.Bar(
        x=nz['index'],
        y=nz[feat],
        name='Non Zero Revenue',
        marker=dict(color=randomc())
    )
    
    layout = go.Layout(
        title=feat,
        height=100, width=100,
        xaxis=dict(
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title='Percentage of visits for each {}'.format(feat),
            titlefont=dict(size=16),
            tickfont=dict(size=14)
        ),
        legend=dict(
            x=1.0,
            y=1.0,
            bgcolor='rgb(255, 255, 255)',
            bordercolor='rgb(255, 255, 255)'
        ),
       
    )
    

    fig = tools.make_subplots(rows=1, cols=2)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig['layout'].update(autosize=False, height=300, width=1000, title='Percentage of visits for each {}'.format(feat))

    iplot(fig)


# **Let's see how the continent, the device and other  features are related to the number of visits**

# In[ ]:


barplot_visit('channelGrouping')


# Visits from Referal Channel group tend to buy more whereas they represent only 11% of the visits

# In[ ]:


barplot_visit('device.operatingSystem')


# How surprising ! People using Macintosh seem buy more often than Windows and Linux users

# In[ ]:


barplot_visit('geoNetwork.continent')


# In[ ]:


barplot_visit('geoNetwork.subContinent')


# In[ ]:


barplot_visit('device.deviceCategory')


# As expected, most people visit the website from their desktop/laptop

# In[ ]:


def barplot_revenue(feat):
    feat_data = 100*train_df.groupby(feat)['totals.transactionRevenue'].sum()/train_df['totals.transactionRevenue'].sum()
    feat_data = feat_data.to_frame().reset_index()

    trace0 = go.Bar(
        x=feat_data[feat],
        y=feat_data['totals.transactionRevenue'],
        name=feat,
        marker=dict(color=randomc())
    )
    
    layout = go.Layout(
        title=feat,
        autosize=False,
        width=800,
        height=300,
        xaxis=dict(
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title='Percentage of revenue for each {}'.format(feat),
            titlefont=dict(size=16),
            tickfont=dict(size=14)
        ),
        legend=dict(
            x=1.0,
            y=1.0),
       
    )
    
    fig = go.Figure(data=[trace0], layout=layout)
    iplot(fig)


# In[ ]:


barplot_revenue('channelGrouping')


# In[ ]:


barplot_revenue('device.operatingSystem')


# In[ ]:


barplot_revenue('geoNetwork.continent')


# In[ ]:


barplot_revenue('device.deviceCategory')


# Revenue comes mostly from desktop users

# In[ ]:


date_revenue = 100* train_df.groupby('date')['totals.transactionRevenue'].sum()/train_df['totals.transactionRevenue'].sum()

plt.subplots(figsize = (14, 6))
plt.title("Percenntage of revenue per day");

date_revenue.plot(linewidth=1.25)


# **What we get from that :**
# <br>
# <br>
# For every month we see approximately 4 drop & rise patterns. And the peaks are sharper. We can easily infer than people are more likely to buy during the week end. So if we were doind manual feature engineering we could add a column of binary variable indicating if the corresponding day is in the week end or not.

# **Conclusion of EDA :** 
# <br>
# <br>
# We don't do EDA just for the sake of EDA, here is  what we can conclude :
# 
# 1. The distribution of the sums of non zero transaction revenues does not fit a popular distribution. Especially not the normal distribution. 
# 2. Appart from this last feature creation ( binary variable to indicate week end days), this EDA does not give many hint for potential manual feature engineering
# <br>
# <br>
# The dataset has many categorical features and only two numerical features. So I think ensembles will be of a good help for feature engineering and predictions too. If anyone has a non-blackbox method or a more interpretable way to do feature engineering, please tell me ! I would like to know. 
# 

# **Feature Selection**
# <br>
# <br>
# As manual feature engineering is not obvious here, let's use **Random Forest Algorithm** for Feature Selection. We will first one-hot encode the whole training set.

# In[ ]:


from sklearn.preprocessing import LabelEncoder

#categorical features
cat_feat = list(train_df.columns.values)
cat_feat.remove('totals.transactionRevenue')
cat_feat.remove("totals.pageviews")
cat_feat.remove("totals.hits")

#numerical features
num_feat = ["totals.hits", "totals.pageviews"]


for feat in cat_feat:
    lbl = LabelEncoder()
    lbl.fit(list(train_df[feat].values.astype('str')))
    train_df[feat] = lbl.transform(list(train_df[feat].values.astype('str')))
for feat in num_feat:
    train_df[feat] = train_df[feat].astype(float)

y= train_df['totals.transactionRevenue']

feats= cat_feat + num_feat

X = train_df[feats]

# Train test split without shuffle to keep the date order

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)



# In[ ]:


# Create a random forest classifier
from sklearn.ensemble import RandomForestRegressor

rgr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
           oob_score=False, random_state=0, verbose=1, warm_start=False)
# Train the classifier. You can try with other parameters, 
# Especially try to use more estimators if your machine is more powerful.
rgr.fit(X_train, y_train)

# Print the name and gini importance of each feature
for feature in zip(feats, rgr.feature_importances_):
    print(feature)


# In[ ]:


feat_score =sorted(zip(feats, rgr.feature_importances_), key=lambda tup: tup[1], reverse=True)
score_list=[x[1] for x in feat_score]
feat_list=[x[0] for x in feat_score]


# In[ ]:


trace0 = go.Bar(
                x = feat_list,
                y = score_list,
                name = "Score of features",
                marker = dict(color=randomc()))
data = [trace0]
layout = go.Layout(
    xaxis = dict(tickangle = -25),
    title='Score of features',
)
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **This is the end of my kernel**
# <br>
# <br>
# EDA did not give many hints so manual feature engineering did not seem useful. We have performed feature engineering with a randomforest algorithm. The most important feature seems to be the total page viewed, by far.
# <br>
# Quite interesting analysis. Though, I wonder how this problem would be tackled by a professional data science team. What happens when you have a dataset where the only methods you can use for both feature engineering/selection, are ensemble or black-box model. How do you explain the result to the client ? " I did some magic tricks and my algorithm has good performance " ?? Any insight please tell me I want to know !
# 
# Thank you for reading !
# 
