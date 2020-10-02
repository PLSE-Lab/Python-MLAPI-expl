#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# **"Ultimately, the greatest lesson that COVID-19 can teach humanity is that we are all in this together."-**[Kiran Mazumdar-Shaw](https://en.wikipedia.org/wiki/Kiran_Mazumdar-Shaw)
# 
# Welcome to the Global AI Challenge 2020!. In this year's challenge we will be estimating the economic impact of COVID-19 across the globe. Participants are asked to predict the crude oil price given the COVID-19 cases. The challenge here is to utilize various kinds of datasets given in order to predict the Price. 
# 
# Lockdown measures put in place to contain the spread of COVID-19 represent an unprecedented shock to global oil demand. The International Energy Agency (IEA) forecasts that the drop in global demand in April will be as much as 29 million barrels/day year-on-year (around 30% of demand), followed by another significant year-on-year fall of 26 million barrels/day in May. The world has returned to oil demand levels last seen in the 1990s.
# **Source: [Bruegel.org](https://www.bruegel.org/2020/04/covid-19-is-causing-the-collapse-of-oil-markets-when-will-they-recover/)**
# 
# 
# 
# 
# In this notebook we will use the Crude Oil Trend and Covid-19 train dataset to understand and visualize the impact of Covid-19. Additionally, we will train a simple Machine Learning model to predict the price such that people who are new to Machine learning can also participate and get encouraged.
# 
# 

# # Authors
# 
# This notebook is a collaborated work of **[Aravind](http://linkedin.com/in/imaravindr/)** and **[Shebin](https://www.linkedin.com/in/shebin-xp/)**. 
# 
# *Our objective here is to show the power of visualizations and motivate fellow employees to participate in this challenge.* 
# 
# "In vain have you acquired knowledge if you have not imparted it to others." - [Deuteronomy Rabbah](https://en.wikipedia.org/wiki/Deuteronomy_Rabbah)
# 
# **Please vote if you feel this kernel to be useful.!**

# # Exploratory Data Analysis - EDA
# 
# Let's explore the data given.

# ### Import Packages

# In[ ]:


import pandas as pd 
import numpy as np
import os

path = '/kaggle/input/ntt-data-global-ai-challenge-06-2020/'
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import animation, rc
from IPython.display import HTML, Image
rc('animation', html='html5')

get_ipython().system('pip install bar_chart_race')
import bar_chart_race as bcr


# ## Load Data - Crude Oil Trend

# In[ ]:


# Load Crude Oil data
data_1 = pd.read_csv(path+"Crude_oil_trend_From1986-10-16_To2020-03-31.csv")
print('Number of data points : ', data_1.shape[0])
print('Number of features : ', data_1.shape[1])
print('Features : ', data_1.columns.values)
data_1.head() # to print first 5 rows


# We have a time series data. We will understand how the Crude Oil Price fluctuates historically. The data given is from 1986-10-16 to 2020-03-31.  
# 
# The data given is on day to day basis. We will first see how the price fluctuates on yearly basis.

# In[ ]:


# Converting Date format
data_1['Date'] = pd.to_datetime(data_1['Date'])
data_1['Date'].dtype
# Year wise data
# mean price 
data_1_year = data_1.groupby(data_1.Date.dt.year)['Price'].agg('mean').reset_index()
data_1_year.head()


# ## Crude Oil Price over the years

# In[ ]:


# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim((1986, 2020))
ax.set_ylim(np.min(data_1_year.Price), np.max(data_1_year.Price)+1)
ax.set_xlabel('Year',fontsize = 14)
ax.set_ylabel('Price',fontsize = 14)
ax.set_title('Crude Oil Price Over the Years',fontsize = 18)
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_facecolor('#000000') 
line, = ax.plot([], [], lw=4,color='green')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)


# animation function. This is called sequentially
def animate(i):
    d = data_1_year.iloc[:int(i+1)] #select data range
    x = d.Date
    y = d.Price
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=40, repeat=True)


# In[ ]:


anim


# The above animated graph shows how the Crude oil prices had gone up over the year. But for our interest (Covid-19 situation), we are more interested in what's happening in 2020. 
# 
# *Note: #!conda install -c conda-forge ffmpeg # run this if the above plot shows RuntimeError: Requested MovieWriter (ffmpeg) not available*

# ## Price drop in 2020
# Weekly analysis

# In[ ]:


# Week wise data 2020 Jan to April
mask = (data_1['Date'] > '2019-12-31') & (data_1['Date'] <= '2020-03-31')
data_2020 = data_1[mask]
# mean price 
data_2020_weekly = data_2020.set_index('Date').resample('W').mean().reset_index()
data_2020_weekly.head()


# ## Crude Oil Price Per Week: (Jan - Mar, 2020)

# In[ ]:


# First set up the figure, the axis, and the plot element we want to animate
import datetime
fig, ax = plt.subplots(figsize=(8,6))

ax.set_xlim([datetime.date(2020, 1, 2), datetime.date(2020, 3, 31)])
ax.set_ylim(np.min(data_2020_weekly.Price), np.max(data_2020_weekly.Price)+1)
ax.set_xlabel('Date',fontsize = 14)
ax.set_ylabel('Price',fontsize = 14)
ax.set_title('Crude Oil Price Per Week 2020 Jan - Mar',fontsize = 18)
ax.xaxis.grid()
ax.yaxis.grid()
ax.set_facecolor('#000000') 
line, = ax.plot([], [], lw=4,color='green')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)


# animation function. This is called sequentially
def animate(i):
    d = data_2020_weekly.iloc[:int(i+1)] #select data range
    x = d.Date
    y = d.Price
    line.set_data(x, y)
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=14, repeat=True)


# In[ ]:


anim


# The gif shows how the price is impacted from the first week of Jan, 2020 till the last week of Mar, 2020. We will explore deep regarding the negative correlation of price and Covid-19.

# ## Load data - Covid-19 Train

# In[ ]:


# Load dataset
data_2 = pd.read_csv(path+"COVID-19_train.csv")
print('Number of data points : ', data_2.shape[0])
print('Number of features : ', data_2.shape[1])
data_2.head()


# This data contains the information about Covid-19 across the world.

# ## Loss of Life
# 
# Total deaths across the world.

# In[ ]:


# Lets take only few countries
cols = ['Date','China_total_deaths','Germany_total_deaths','Spain_total_deaths',
        'France_total_deaths','UnitedKingdom_total_deaths','India_total_deaths',
       'Italy_total_deaths','SouthKorea_total_deaths','UnitedStates_total_deaths','Russia_total_deaths']
data_deaths = data_2[cols]
data_deaths.set_index("Date", inplace = True) 
data_deaths.head()


# In[ ]:


bcr.bar_chart_race(df=data_deaths, filename=None, figsize = (3.5,3),title='COVID-19 Deaths by Country')


# The graph shows us how fast and when exactly the death counts in Italy and Spain took over its precedor China. 

# ## Transmission of Covid-19

# In[ ]:


# Modifying data
data_total_cases = data_2.filter(regex="total_cases|Date|Price")
# Drop countries with 0 cases
data_total_cases = data_total_cases.loc[:, (data_total_cases != data_total_cases.iloc[0]).any()] 
# countries = data_total_cases.columns.values[1:-1]
# countries = list(set([i.split('_')[0] for i in countries]))
data_total_cases.head()


# In[ ]:


# data transformation
dates = []
countries_ls = []
total_cases = []
prices = []
for index, row in data_total_cases.iterrows():
    df = pd.DataFrame(row).T
    c_ls = (df.iloc[:,1:-2].apply(lambda x: x.index[x.astype(bool)].tolist(), 1)[index])
    dates.extend([[df['Date'][index]]*len(c_ls)][0])
    prices.extend([[df['Price'][index]]*len(c_ls)][0])
    countries_ls.extend([col.split('_')[0] for col in c_ls])
    total_cases.extend([df[col][index] for col in c_ls])
    
data_2_mod = pd.DataFrame({'Date':dates,'Country':countries_ls,'Total_Cases':total_cases,'Price':prices})
data_2_mod.head()


# The column wise information is changed into rows.

# In[ ]:


from IPython.display import Image
sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.set_context("paper")
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "seaborn"
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot,plot
init_notebook_mode(connected=True)

fig = px.choropleth(
    data_2_mod, #Data
    locations= 'Country', #To get Lat and Lon of each country
    locationmode= 'country names', 
    color= 'Total_Cases', #color scales
    hover_name= 'Country', #Label while hovering
    hover_data= ['Country','Price'], #Data while hovering
    animation_frame= 'Date', #animate for each day
    color_continuous_scale=px.colors.sequential.Reds
)

fig.update_layout(
    title_text = "<b>COVID-19 Spread in the World up to Mar 31, 2020</b>",
    title_x = 0.5,
    geo= dict(
        bgcolor = 'black',
        showframe= False,
        showcoastlines= False,
        projection_type = 'equirectangular'
        
        
    )
)
iplot(fig)


# The plot shows how Covid-19 was spreading across the world for the first 3 months of 2020. 
# 
# We can observe that initial spread was in the neighbourhood of China and then suddenly into Europe and other parts of the world.
# 
# The crucial dates were from Jan 24 - Feb 03. In a span of 10 days many countries started reporting its positive cases.
# 
# Also, come March 31 Italy, Spain, Germany, Iran, China, France were the countries with most number of Covid-19 cases.

# ## Price Vs Covid-19
# 
# Let's get back to our target variable price. Let's visualize the price drop and total cases together

# In[ ]:



# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=data_2.Date, y=data_2.Price, name="Price"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=data_2.Date, y=data_2.World_total_cases, name="World Total Cases",line = dict(color = 'orangered')),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
#     title_text="Total cases vs Price"
    title='<b>Total cases vs Price</b>',
    plot_bgcolor='linen',
#     paper_bgcolor = 'grey',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=2,
                     label='2m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

# Set x-axis title
fig.update_xaxes(title_text="<b>Date</b>")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Price</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>World Total Cases</b>", secondary_y=True)

iplot(fig)


# We can see the direct relationship between World total cases and price. There is a steep decrease in price from Mid-Jan 2020 to Mar 2020. The steepness of the price curve indicates us that it will continue to fall down further in the following months. 

# ## Traditional Corr Plot?

# In[ ]:


# Top countries impacted as of Mar 31, 2020.
cols = ['World_total_cases','World_total_deaths','China_total_cases','Italy_total_cases','Germany_total_cases',
        'Spain_total_cases','Iran_total_cases','France_total_cases','Price']

cordata = pd.DataFrame(data_2[cols].corr(method ='pearson'))

fig = go.Figure(data=go.Heatmap(z=cordata,x=cols,y=cols,colorscale='burgyl'))

iplot(fig)


# In[ ]:


cordata


# The correlation data shows which country affects the target (price) variable the most. 
# 
# World total cases is highly correlated with Price column (-0.89)
# 
# Among countries, Iran contributes the most with a negative correlation of -0.897 followed by Italy (-0.858) and China (-0.841) and then the others.
# 
# Note - The countries themselves are multicorrelated therefore, while training a ML model remove independent features that are correlated among themselves. Learn more about multicolinearity in data science [here](https://towardsdatascience.com/multicollinearity-in-data-science-c5f6c0fe6edf).
# 

# # Simple Machine Learning for Beginners
# 
# 
# 
# The core idea behind any Machine Learning Model is to identify a function that can take independent features (x's) as input and return the target/dependent variable. i.e, f(x) = y.
# 
# Lets train a simple yet powerful Linear regressor to predict price.
# 
# Learn more about Linear regression [here](https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f).
# 
# Note: We are not including the process of feature engineering in this kernel. 

# ## Train & Cross Validation Split
# 
# We will split the available data into train and cv based on date.
# 
# CV data range: 20 Mar - 31 Mar

# In[ ]:


data_2 = data_2.set_index(data_2['Date'])
train = data_2[:'2020-03-19']
cv = data_2['2020-03-20':]
train.tail()


# In[ ]:


features = ['World_total_cases','World_total_deaths','China_total_cases','Italy_total_cases','Germany_total_cases',
        'Spain_total_cases','Iran_total_cases','France_total_cases'] # columns to be used for training
y_train = train['Price'].values # target column
X_train = train[features] # Let's only consider few columns.
print('Number of X_train data points : ', X_train.shape[0])
print('Number of features train: ', X_train.shape[1])
print('Features : ', X_train.columns.values)
X_train = X_train.values


# In[ ]:


y_cv = cv['Price'].values
X_cv = cv[features]
print('Number of X_cv data points : ', X_cv.shape[0])
print('Number of features cv: ', X_cv.shape[1])
print('Features : ', X_cv.columns.values)
X_cv = X_cv.values


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from matplotlib.legend_handler import HandlerLine2D
from tqdm import tqdm


lr = LinearRegression()
lr.fit(X_train,y_train)

y_train_hat = lr.predict(X_train)
y_train_rmse = sqrt(mean_squared_error(y_train,y_train_hat))

y_cv_hat = lr.predict(X_cv)
y_cv_rmse = sqrt(mean_squared_error(y_cv,y_cv_hat))
print('Linear Regression Model trained!')


# ### Train and CV predictions

# In[ ]:


# train predictions
line1, = plt.plot(y_train, color="r", label="Actual Price")
line2, = plt.plot(y_train_hat, color="g", label="Predicted Price")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('Train: Actual Vs Predicted Price')
plt.xlabel('Number of rows')
plt.ylabel('Price')


# In[ ]:


# cv predictions
line1, = plt.plot(y_cv, color="r", label="Actual Price")
line2, = plt.plot(y_cv_hat, color="g", label="Predicted Price")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.title('CV: Actual Vs Predicted Price')
plt.xlabel('Number of rows')
plt.ylabel('Price')


# Since the model is very basic [without any feature engineering done](https://en.wikipedia.org/wiki/Feature_engineering#:~:text=Feature%20engineering%20is%20the%20process,as%20applied%20machine%20learning%20itself.) the performance of the model is quiet low (Cases of Overfitting). 
# 
# 
# However, for understanding the submission process, we will use this model to predict the test data given.

# ## Test data - Submission

# In[ ]:


test = pd.read_csv(path+'COVID-19_test.csv')
X_test = test[features]
print('Number of X_train data points : ', X_test.shape[0])
print('Number of features : ', X_test.shape[1])
test.head()


# In[ ]:


y_test_hat = lr.predict(X_test)
#print("Predicted Price from April 01 - May 22, 2020: \n")
# Predicted Price - First week of april
print("Predicted - First week of april \n")
for i in range(0,7):
    print("Date: {}, Predicted Price: {}".format(test['Date'][i],y_test_hat[i]))


# ## Submission

# In[ ]:


submission_df = pd.DataFrame({'Date':test.Date,'Price':y_test_hat})
#submission_df.to_csv(path+'Submission.csv',index = False)


# Note: The model and prediction is only for basic undertanding. Not appropriate!

# ## Reality Though!

# In[ ]:


from IPython.display import Image
Image(filename='/kaggle/input/realitycovid19/Reality-covid.PNG') 


# **Stay inside! Stay safe!**
