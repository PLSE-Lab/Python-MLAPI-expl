#!/usr/bin/env python
# coding: utf-8

# #### Please upvote, in case you find this notebook helpful
# > <font size=4 color='Blue'>(The Visualizations are updated every 24 Hours)</font>
# 
# <img align="left" src="https://www.vmcdn.ca/f/files/okotokstoday/images/okotoks-today/covid-19.jpg;w=635"></img>

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


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

# ### Let's check 1st file: /kaggle/input/Critical_Cases_April_14.csv

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# Critical_Cases_April_14.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/Critical_Cases_April_14.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Critical_Cases_April_14.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df1, 8)


# Scatter and density plots:

# In[ ]:


plotScatterMatrix(df1, 20, 10)


# <hr>
# ### COVID-19 is wreaking havoc across the globe!!!
# 1. How the countries are dealing with it
# 2.  Or one of those European countries that are literally on fire at the moment because of the virus? 
# 3.  How good is the US doing?
# 4.  While i am writing this kernal data for different countries in world is getting doubled each 2nd day, 4th day and so on. We all have been a complete failure for stopping the virus growth daily. As you can see in the trend below, how the data for counties is getting doubled each 2nd day or 4th day. Japan is the only country which is having doubling rate near about to week. Week later USA was doubling it count in 4 Days. This week i.e. as of 8th April ,the number of confirmed coronavirus cases in the U.S. surpassed 426,000 on Wednesday, according to figures provided by NBC, with 12,864 fatalities nationwide.Just one week back they were having around 213,000 cases. India is also doubling its figures every 4th-5th consecutive day now. India was having 3082 cases on 4th April and on 8th April it have been doubled to 5916. You can refer to image below as per doubling rate for different counties. As per the graph we can easily see the growth for Spain, Italy and Germany have started to steep down. We will discuss the reason for this further
# 5.  How Lockdown wordk? 
# 6.  Lessons to learn from countries who have done good so far?
# 
# 

# In[ ]:



get_ipython().system('pip install folium')
get_ipython().system('pip install plotly')


# In[ ]:


df_worldinfor1 = pd.read_csv('/kaggle/input/Critical_Cases_April_14.csv', delimiter=',', nrows = nRowsRead)
df_worldinfor1.dataframeName = 'Critical_Cases_April_14.csv'
nRow, nCol = df_worldinfor1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df_worlddata = df_worldinfor1[df_worldinfor1.Country != 'World']
df_worlddata.index = df_worlddata["Country"]
df_worlddata = df_worlddata.drop(['Country'],axis=1)
df_worlddata.head()


# <hr>
# 1. Validating Testing Data around the world
# 2. How different counties are performing test wise around the world
# 3. Let us visulize as per total number of test and test among Million people

# In[ ]:


df_test=df_worlddata.drop(['Total Cases','Cases','Total Deaths','Deaths','Total Recovers','Active','Serious Critical',
                           'Total Cases/1M pop','Deaths/1M pop'],axis=1)


# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_test.sort_values('Tests/1M pop')["Tests/1M pop"].index[-50:],df_test.sort_values('Tests/1M pop')["Tests/1M pop"].values[-50:],color="red")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Tests/1M pop ",fontsize=18)
plt.title("Top Countries (Tests/1M pop )",fontsize=20)
plt.grid(alpha=0.3)


# <hr>
# **As per figure above you can notice following points**
# 1. USA is doing maximum number of test nowdays and that is the reason they are having so much count nowdays
# 2. India have also increase number of test at daily basis now
# 3. South Korea, despite being less number of cases have done more number of test, and that is the reason they are able to make the curve flat after their count of 9 k cases

# **
# **Let us now analyze top 20 countries with number of Total cases**
# **Will analyze Top counties as per confirmed, Most number of deaths, recovered and Critical case(Requiring ICU)**

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_worlddata.sort_values('Total Cases')["Total Cases"].index[-20:],df_worlddata.sort_values('Total Cases')["Total Cases"].values[-20:],color="red")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Total Cases",fontsize=18)
plt.title("Top Countries (Total #)",fontsize=20)
plt.grid(alpha=0.3)


# **Top Contries as per Active Cases**

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_worlddata.sort_values('Active')["Active"].index[-20:],df_worlddata.sort_values('Active')["Active"].values[-20:],color="darkcyan")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Active",fontsize=18)
plt.title("Top Countries (Active #)",fontsize=20)
plt.grid(alpha=0.3)


# <hr>
# ### Let us now identify critical cases
# 1. This will be most important as this will identify the number of deaths in coming days
# 2. USA will have more number of death in coming day as they are having more number of serious cases
# 3. France and Spain followed by USA

# In[ ]:


f = plt.figure(figsize=(20,15))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_worlddata.sort_values('Serious Critical')["Serious Critical"].index[-20:],df_worlddata.sort_values('Serious Critical')["Serious Critical"].values[-20:],color="crimson")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Serious Critical",fontsize=18)
plt.title("Top Countries (Critical #)",fontsize=20)
plt.grid(alpha=0.3)


# In[ ]:




