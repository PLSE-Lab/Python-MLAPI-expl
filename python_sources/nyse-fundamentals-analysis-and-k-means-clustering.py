#!/usr/bin/env python
# coding: utf-8

# # New York Stock Exchange Data Analysis
# 
# The data used here is S&P 500 companies historical prices with fundamental data which consists of four csv files namely,
# 
# 
# - **prices.csv:** raw, as-is daily prices. Most of data spans from 2010 to the end 2016, for companies new on stock market date range is shorter. There have been approx. 140 stock splits in that time, this set doesn't account for that.
# - **prices-split-adjusted.csv:** same as prices, but there have been added adjustments for splits.
# - **securities.csv:** general description of each company with division on sectors
# - **fundamentals.csv:** metrics extracted from annual SEC 10K fillings (2012-2016), should be enough to derive most of popular fundamental indicators.
# 
# This notebook deals with preliminary data data analysis that includes preprocessing of data, DatViz and clustering. The basic goal is to create an unsupervised model ofr data analysis using **fundamentals and securities vectors.** 
# 
# ## Fundamentals
# The fundamentals vectors have companies' fundamental financial statement indicators including **Balance Sheet, Cash-Flow Statement** and **Income-Expense Statement**. In most of the cases all the ratios are present, but in some cases the liquidity indicators like current liability or some other may not be present thus ratios are not valid.
# 
# ## Securities
# This vector has stationery data related to *'Ticker Symbol'* like security name, sector, segment and address etc. It can be used to identify status of particular industry segment.

# In[ ]:


#STEP 1: Get right arrows in quiver

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance

import seaborn as sns
from sklearn.metrics import silhouette_score,silhouette_samples

import warnings #To hide warnings
import re  #RegEx


#1.1: Set the stage
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import display, HTML

InteractiveShell.ast_node_interactivity = "all"
pd.options.display.max_rows = 200
pd.options.display.max_columns = 300
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# Now we can read the data and check its basic structure like data types,Null values and shape etc

# In[ ]:


#STEP 2: Load Data
df_sec = pd.read_csv('/kaggle/input/nyse/securities.csv')
df_fn = pd.read_csv('/kaggle/input/nyse/fundamentals.csv')

#STEP 2.1:Analyse data structure
display(HTML('The fundamentals dataset has <b>' +str(df_fn.shape[1])+' features </b>and <b>'+str(df_fn.shape[0])
             +' observations</b>'))

df_fn.info()


# In[ ]:


# Check for NaN values
display(HTML(' <h2>Features having NaN values</h2>'))
df_fn[df_fn.columns[df_fn.isna().any()]].isna().sum().to_frame().T
#View Data sample(Random sample rather than head for better understanding)
display(HTML(' <h2>Random Sample of 5 observations</h2>'))
df_fn.sample(n = 5).T


# The preliminary data analysis shows that there are **1781 observations and 79 features in dataset in which *SIX* columns are holding some NaN values.** 
# - The null values are atmost 299 while in case of 'For Year' such occurance is 173.
# - 'Period Ending' feature has date values and none of them is nan thus we can directly parse the column to date.
# - 'For Year' feature is having year value of 'Period Ending', thus for nulls it can be calculated. It also suggests that we can use either of 'Period Ending' and 'For Year' but not both required.
# ### Jobs to do
# - We can can verify the occurance of nan is in same observations or is in random.
# - __ANOTHER JOB TO BE DONE IS RENAMING THE FEATURES TO REMOVE all PERIODS AND DOTS__
# - Reduce the shape of dataset

# In[ ]:


#2.2 Renaming Features
f_dct = {n : re.sub('[^A-Za-z0-9]+','',n) for n in df_fn.columns.values}
df_fn.rename(columns = f_dct,inplace=True)
df_fn['PeriodEnding'] = pd.to_datetime(df_fn['PeriodEnding'])
df_fn['ForYear'] = df_fn['PeriodEnding'].dt.year.astype('category')
#2.3 Clean it!
fwm = ["TickerSymbol","ForYear","AccountsPayable","AccountsReceivable","GrossProfit","Liabilities","NetCashFlow","OperatingIncome","TotalAssets","TotalEquity","TotalLiabilities","TotalLiabilities&Equity","TotalRevenue","EarningsPerShare"]
to_drop= [x for x in df_fn.columns.values if x not in fwm]
df_fn.drop(columns = to_drop,inplace = True)

display(HTML('<h2>Remaining NaN Columns after reduction</h2>'))
df_fn.isna().sum().sort_values(ascending=False).to_frame().head().T

display(HTML('<b> The reduced shape of dataset is '+str(df_fn.shape)))


# In this phase all features have been renamed and we have also reduced it to minimal shape for preliminary analysis. Now EarningsPershare has 219 null values
# 
# Now let us check a data sample again.

# In[ ]:


df_fn.head()


# ## Analysis of tickers
# Now to get basic idea we would try to measure the dataset by tickers available. To so we will have to check the tickers and correcponding summary for its each value.

# In[ ]:


print("The dataset has ",len(df_fn['TickerSymbol'].unique())," unique tickers")
grp_tick = df_fn.groupby('TickerSymbol')
t_agg = grp_tick.agg(np.nanmean)
t_agg.head()


# *After summarization we can now check for the performance by plotting some basic features like Earning/share, operating Income, asset-liablity etc.*

# In[ ]:


t_agg['Ticker'] = t_agg.index
bottom30 = t_agg.sort_values(by = 'EarningsPerShare').head(30)
top30 = t_agg.dropna().sort_values(by = 'EarningsPerShare',ascending=False).head(30)

fig = plt.figure(figsize = (20,15))
plt.subplot(1,2,1)
plt.title('Top 30 Tickers as per Earning/share')
sns.barplot(y = 'Ticker', x = 'EarningsPerShare', data = top30)
plt.subplot(1,2,2)
plt.title('Bottom 30 Tickers as per Earning/share')
sns.barplot(y = 'Ticker', x = 'EarningsPerShare', data = bottom30)


# Here the plot suggests that who had invested in PCLN or AZO must have got huge profits while on other side, investor of APA and DYN have faced huge losses.
# The plot ***actually*** does not suggests the health of company itself, for which we can check its asset-liability and Payable-Receivable etc

# In[ ]:


#top30.loc[:,['TotalAssets','TotalLiabilities','TotalEquity']]
top30.sort_values(by = 'TotalAssets',
                  ascending=False).plot(x='Ticker',y=['TotalAssets','TotalLiabilities','TotalEquity']
                                        ,kind='bar',figsize = (20,8),title='Asset-Liabilty composure'
                                        +' of Top30 companies')


# The asset-liablity chart shows that the second best returning ticker __AZO__ is having negative equity which is not a good sign while __4th best earning ticker AAPL__ has a good composure of asset-liability.
# We can alternatively check the possible pair plots for finding the structure. 

# In[ ]:


sns.pairplot(t_agg)


# # Prepare data for model

# In[ ]:


#tick = df_fn[:,['TickerSymbol','ForYear']] #for future use
df_fn.drop(columns = ['TickerSymbol','ForYear'],inplace=True)
y_col = df_fn.columns[df_fn.isna().any(axis =0)]
fn_pred = df_fn[df_fn.isna().any(axis =1)]
fn_prem = df_fn[~df_fn.isna().any(axis =1)]
y = fn_prem[y_col]
fn_prem.drop(columns = y_col)
ss = StandardScaler()
X = ss.fit_transform(fn_prem)
X_train, X_test, y_train, y_test = train_test_split( X,y, test_size = 0.2)


# ## Identification of number of clusters for the data
# We are using here k-means clustering for which the number of clusters is to be predetermined. In order to achieve that we can draw a Scree Chart or check silhouette Plot.

# In[ ]:


distortions = []
sil_sc = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(X_train)
    distortions.append(km.inertia_)
    if(i>1) : 
        sil_sc.append(silhouette_score(X_train, km.labels_))
        
# plot
plt.title('Scree Plot')
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

for i in range(0,9): print('\nFor ',(i+2),' clusters avg silhoutte score is #',sil_sc[i])
plt.plot(range(0, 10), distortions, marker='o')
plt.title('Silhouette Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[ ]:


for i in range(2,11):
    model = SilhouetteVisualizer(KMeans(i),title = 'Silhouette for n_cluster = '+str(i))
    model.fit(X_train)
    model.show()


# ### All these plots strongly recommend the number of clusters should be set at 2.

# In[ ]:


colors = ['orange', 'blue']
km = KMeans(n_clusters=2)
clusters = km.fit(X_train)
labels = clusters.labels_
ctr = clusters.cluster_centers_

fig = plt.figure(figsize = (10,10))
fig.suptitle('K-mean clusters formed with n_cluster = 2',fontsize = 16)
for k, col in zip(range(X_train.shape[0]), colors):
    my_members = (labels == k)
    cluster_center = ctr[k]
    plt.plot(X_train[my_members, 0], X_train[my_members, 1], 'w', markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)


# In[ ]:


from sklearn.manifold import TSNE
tsne = TSNE()
X_embedded = tsne.fit_transform(X_train)
X_embedded
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full',hue = np.sign(y_train['EarningsPerShare']))


# At last we check for the inter cluster distance map for our model.

# In[ ]:


visualizer = InterclusterDistance(km, embedding= 'tsne')
visualizer.fit(X_train)        
visualizer.show()              


# # **Please upvote if you like my work and give comments/suggestions for improvement**
