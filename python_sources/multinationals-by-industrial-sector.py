#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
df = pd.read_csv('../input/cusersmarildownloadsmultinationalscsv/multinationals.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)
df.dataframeName = 'multinationals.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df.head(5)


# In[ ]:


df.describe()


# The code below is from @rpowers9. Thanks for sharing (kernel 54163d971a) learn-with-other-users competition 2019.

# In[ ]:


# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in df.columns if
                    df[cname].nunique() < 10 and 
                    df[cname].dtype == "object"]


# Select numerical columns
numerical_cols = [cname for cname in df.columns if 
                df[cname].dtype in ['int64', 'float64']]


# In[ ]:


print(categorical_cols)


# In[ ]:


print(numerical_cols)


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


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


# In[ ]:


plotPerColumnDistribution(df, 10, 5)


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
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='g', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# In[ ]:


plotCorrelationMatrix(df, 8)


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


# In[ ]:


plotScatterMatrix(df, 8, 6)


# In[ ]:


print ("Skew is:", df.year.skew())
plt.hist(df.year, color='palegreen')
plt.show()


# In[ ]:


target = np.log(df.year)
print ("Skew is:", target.skew())
plt.hist(target, color='purple')
plt.show()


# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px


# In[ ]:


ax = sns.swarmplot(x="year", y="year.1", hue="power_code code",
                   data=df, palette="Set2", dodge=True)


# In[ ]:


ax = sns.swarmplot(x="year", y="year.1", data=df)


# Pie chart from Melih Kanbay @melihkanbay. 

# In[ ]:


labels1=df.year.value_counts().index
sizes1=df.year.value_counts().values
plt.figure(figsize=(11,11))
plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")
plt.title("year",size=25)
plt.show()


# The codes below are from Fatih Bilgin. Thank you @fatihbilgin, again.

# In[ ]:


df.plot(kind='scatter', x='year', y='year.1', alpha=0.5, color='darkgreen', figsize = (12,9))
plt.title('year And year.1')
plt.xlabel("year")
plt.ylabel("year.1")
plt.show()


# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px


# In[ ]:


trace1 = go.Box(
    y=df["year"],
    name = 'year',
    marker = dict(color = 'rgb(0,145,119)')
)
trace2 = go.Box(
    y=df["year.1"],
    name = 'year.1',
    marker = dict(color = 'rgb(255, 111, 145)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='year', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))
fig = dict(data=data, layout=layout)
iplot(fig)


# In[ ]:


# It is a continous variable and hence lets look at the relationship of year with year.1 using a Regression plot

_ = sns.regplot(df['year'], df['year.1'])


# In[ ]:


f,ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.iloc[:,12:].corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()


# In[ ]:


MultinationalsIndSec=df.groupby(['reference period','year.1','power_code code'])['year'].mean().reset_index()

fig = px.scatter_mapbox(MultinationalsIndSec[MultinationalsIndSec["reference period"]=='Nan'], 
                        lat="year.1", lon="power_code code",size="year",size_max=12,
                        color="year", color_continuous_scale=px.colors.sequential.Inferno, zoom=11)
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# And that's what happened when you try to adapt other person codes for your dataset without knowing to work. Almost an Inferno as it's written in this last script. Sorry Fatih Bilgin, I coulnd't honor you.  