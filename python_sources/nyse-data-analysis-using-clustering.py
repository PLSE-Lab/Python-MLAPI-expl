#!/usr/bin/env python
# coding: utf-8

# **New York Stock Exchange (NYSE) Data Analysis:
# 
# **Introduction:
# 
# In this project, we are analysing the NYSE Data. What is NYSE?
# 
# **NYSE:
# 
# The New York Stock Exchange (NYSE) is a stock exchange located in New York City that is 
# the largest equities-based exchange in the world, based on the total market capitalization 
# of its listed securities. Formerly run as a private organization, the NYSE became a public 
# entity in 2005 following the acquisition of electronic trading exchange Archipelago. 
# In 2007 a merger with Euronext, the largest stock exchange in Europe, led to the creation 
# of NYSE Euronext, which was later acquired by Intercontinental Exchange, the current parent
# of the New York Stock Exchange.
# 
# **KEY TAKEAWAYS:
# 
# 1)The NYSE, which dates back to 1792, is the largest stock exchange in the world based on 
# the total market capitalization of its listed securities.
# 
# 2)Many of the oldest publicly traded U.S. companies are listed on the Big Board, the nickname for the NYSE.
# 
# 3)The Intercontinental Exchange now owns the NYSE, having purchased the exchange in 2013.
# 

# In[ ]:


get_ipython().run_line_magic('reset', '-f')

import warnings
warnings.filterwarnings("ignore")

# 1.1 Data manipulation library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')
# 1.2 OS related package
import os
# 1.3 Modeling librray
# 1.3.1 Scale data
from sklearn.preprocessing import StandardScaler
# 1.3.2 Split dataset
from sklearn.model_selection import train_test_split
# 1.3.3 Class to develop kmeans model
from sklearn.cluster import KMeans
# 1.4 Plotting library
import seaborn as sns
# 1.5 How good is clustering?
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import re
# 1.6 Set numpy options to display wide array
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )


# In[ ]:


# Display multiple outputs from a jupyter cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# DateFrame object is created while reading file available at particular location given below

df=pd.read_csv("../input/nyse/fundamentals.csv",parse_dates = ['Period Ending'])


# In[ ]:


# Remove NaN or Null values from DataFrame

df.dropna(inplace=True)

# Displaying first 10 rows of DataFrame

df.head()


# In[ ]:


# Remove special characters from DataFrame

cols = {col : re.sub('[^A-Za-z0-9]+','_',col) for col in df.columns.values}

df.rename(columns = cols,inplace=True)

df.columns=df.columns.str.replace(r"[^a-zA-Z\d\_]+",'_')

df.info()


# In[ ]:


# Group data by Ticker Symbols and take a mean of all numeric variables.

gr1=df.groupby('Ticker_Symbol')

gr1.agg([np.mean]).head()


# In[ ]:


# This graph showing Ticker Symbol wise gross profit

px.histogram(data_frame =df,
                    x='Ticker_Symbol',
                    y='Gross_Profit',
                    histfunc="sum",
                    template="plotly_dark"
            )


# In[ ]:


# Graph showing the relationship of Net Income and Estimated shares outstanding

px.histogram(data_frame =df,
                    y='Net_Income',
                    x='Estimated_Shares_Outstanding',
                    histfunc="sum",
                    template="plotly_dark"
            )


# In[ ]:


# Relationship between Capital expenditures and capital surplus

px.density_contour(
                   data_frame =df,
                   x = 'Capital_Expenditures',
                   y = 'Capital_Surplus',
                   template="plotly_dark"
                   )


# In[ ]:


# Relationship between net income and estimated shares outstanding using heatmap

px.density_heatmap(
                   data_frame =df,
                   x = 'Net_Income',
                   y = 'Estimated_Shares_Outstanding',
                   template="plotly_dark",
                   nbinsx = 10,             
                   nbinsy = 20
                   )


# In[ ]:


# New column is created by extracting only day from date

df['Year']= df['Period_Ending'].dt.year

df.Year.unique()

fig=px.scatter(df,
         x = "Gross_Margin",
         y = "Profit_Margin",
         size = "Goodwill",
         range_x=[0,85],
         range_y=[0,120] ,
         animation_frame = "Year",   
         animation_group = "Ticker_Symbol",   
         color = "Ticker_Symbol"              
         )

# 5.3 The following code slows down animation
#  Ref: https://community.plotly.com/t/how-to-slow-down-animation-in-plotly-express/31309/6
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig.show()


# In[ ]:


sns.distplot(df.Gross_Profit,color='b')


# In[ ]:


sns.jointplot(df.Gross_Margin, df.Profit_Margin, kind = 'reg',color='b')


# In[ ]:


# Take the selected data from dataframe 

dfselecteddata = df[['Accounts_Payable','Accounts_Receivable','Gross_Profit',
               'Cost_of_Revenue','Gross_Margin','Gross_Profit','Net_Income','Profit_Margin','Total_Assets',
               'Total_Liabilities','Estimated_Shares_Outstanding']]

# New field named net_profit_loss where value is 1 if there is profit and value is 0 if there is loss 

dfselecteddata.loc[(dfselecteddata['Net_Income'] > 0),'Net_Profit_Loss'] = 1,
dfselecteddata.loc[(dfselecteddata['Net_Income'] <= 0),'Net_Profit_Loss'] = 0 

# Copy 'net_profit_loss' column to another variable and then drop it

y = dfselecteddata['Net_Profit_Loss'].values

dfselecteddata.drop(columns = ['Net_Profit_Loss'], inplace = True)

# Scale data using StandardScaler
    
ss = StandardScaler()                 # Create an instance of class
ss.fit(dfselecteddata)                # Train object on the data
X = ss.transform(dfselecteddata)      # Transform data
X[:5, :]                              # See first 5 rows
X_train, X_test, _, y_test = train_test_split( X,               # np array without target
                                               y,               # Target
                                               test_size = 0.25 # test_size proportion
                                               )
# 4.1 Examine the results

X_train.shape    

X_test.shape  


# In[ ]:


# Create an instance of modeling class
# We will have two clusters

clf = KMeans(n_clusters = 2)

# Train the object over data

clf.fit(X_train)

# So what are our clusters?

clf.cluster_centers_
clf.cluster_centers_.shape         
clf.labels_                        # Cluster labels for every observation
clf.labels_.size                   
clf.inertia_                       # Sum of squared distance to respective centriods, SSE


# In[ ]:


# For importance and interpretaion of silhoutte score, see:

silhouette_score(X_train, clf.labels_)    


# In[ ]:


# Make prediction over our test data and check accuracy

y_pred = clf.predict(X_test)
y_pred                 
np.sum(y_pred == y_test)/y_test.size
dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)


# In[ ]:


# Scree plot:
sse = []
for i,j in enumerate(range(10)):
    
    # How many clusters?
    n_clusters = i+1
    
    # Create an instance of class
    clf1 = KMeans(n_clusters = n_clusters)
    
    # Train the kmeans object over data
    clf1.fit(X_train)
    
    # Store the value of inertia in sse
    sse.append(clf1.inertia_ )

# Plot the line now
sns.lineplot(range(1, 11), sse)


# In[ ]:


# Silhoutte plot
visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        # Fit the data to the visualizer
visualizer.show()   

