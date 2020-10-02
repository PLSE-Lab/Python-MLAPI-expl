#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from ipywidgets import interact
# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import *
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot,widgetbox

from bokeh.layouts import layout

from bokeh.embed import file_html

from bokeh.models import Text
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import LinearAxis
from bokeh.models import SingleIntervalTicker

from bokeh.palettes import Spectral6
output_notebook()

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[ ]:


import cufflinks as cf
cf.set_config_file(world_readable=True,offline=True)


# In[ ]:


# load data to pandas dataframe
df = pd.read_csv('../input/scmp2k19.csv')


# In[ ]:


# explore the top five rows
df.head()


# In[ ]:


# get some info about data 
df.info()


# In[ ]:


# make a copy of our data
data = df.copy()


# In[ ]:


# explore columns related to the addrress
data.loc[:,['district','mandal','location',]].sample(8,random_state=1)


# In[ ]:


# drop unnecessary columns
column_to_drop = ['district','mandal' ,'location', 'odate']
data.drop(columns=column_to_drop, axis=1,inplace=True)


# In[ ]:


# check the columns now
data.columns


# In[ ]:


# check for duplicate values
data.duplicated().sum()


# In[ ]:


# drop the duplicates
data.drop_duplicates(inplace=True)


# In[ ]:


# check for null values
((data.isna().sum()/data.shape[0])*100).round(2)


# In[ ]:


# check for unique values in the humidity_min  column
data.humidity_min.unique()


# In[ ]:


def get_humidity_min(x):
    '''
    extract the humidity_min value out of a string inside tuple
    '''
    # ensure that x is not Null and there is more than one humidity_min
    if not x or len(x) <= 1:
        return None
    humidity_min = [float(i[0].replace('min','').strip())  for i in x if type(i[0])== str]
    return round((sum(humidity_min)/len(humidity_min)),1)


# In[ ]:


# if we check for each value type
type(data.humidity_max[0])


# In[ ]:


import ast 
# get the before number of null values
data.humidity_min.isna().sum()


# In[ ]:


# check now
((data.isna().sum()/data.shape[0])*100).round(2)


# In[ ]:


# first let's drop the humidity_max column now
data.drop(columns='humidity_min',axis=1,inplace=True)
# test for data size
data.shape


# In[ ]:


# check for percentage of null values 
((data.isna().sum()/data.shape[0])*100).round(2)


# In[ ]:


import cufflinks as cf
cf.set_config_file(world_readable=True,offline=True)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 10,6
plt.xkcd() # let's have some funny plot
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # we will use this later, so import it now

from bokeh.io import output_notebook, show
from bokeh.plotting import figure


# In[ ]:


sns.heatmap(data.corr());


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df['district'].value_counts()


# In[ ]:


#sangareddy dist
rain_Sr =df[df.district == 'Sangareddy']
rain_Sr.head(4)


# In[ ]:


rain_Sr.iplot()


# In[ ]:


#sorting by datewise sangareddy dist
sort1 = rain_Sr.sort_values('odate')
sort1.head()


# In[ ]:


#maximum humidity sangareddy
sort1.head(10).plot(x = 'district',y='humidity_max' )


# In[ ]:


#datewise area with highest rainfall
rain_v = df.sort_values('odate')
rain_v.max()


# In[ ]:


dist=df.groupby('district')
dist.describe()


# In[ ]:


#district wise all max rainfall,humidity
dist.max().iplot()


# In[ ]:


#district wise maximum humidity graph
dist['humidity_max'].count().iplot('bar')


# In[ ]:


#district wise minimum humidity graph
dist['humidity_min'].count().iplot('bar')


# In[ ]:


#district wise cumm_rainfall graph
dist['cumm_rainfall'].count().iplot('bar')


# In[ ]:


df[['humidity_max']].iplot(
    kind='scatter',
    histnorm='percent',
    barmode='overlay',
    xTitle='cumm_rainfall',
    yTitle='odate',
    title='datewise rainfall')


# In[ ]:


first1=df[1:1000]


# In[ ]:


first1['district'].max()


# In[ ]:


#a plotting between cumm_rainfall,humidity_max 
first1.plot.scatter(x='cumm_rainfall', y='humidity_max', figsize=(10,8))


# In[ ]:


dstrct=df.groupby('district')


# In[ ]:


dstrct.count()


# In[ ]:


#Yadadri-Bhongir district
yddrb=df[df['district']=='Yadadri-Bhongir 	']
#cummulative rainfall in Yadadri-Bhongir 	district
yddrb['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Warangal (U) district
wrgl=df[df['district']=='Warangal (U)']
#cummulative rainfall in Warangal (U)district
wrgl['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Warangal (R) district
wrgl=df[df['district']=='Warangal (R)']
#cummulative rainfall in Warangal (R) district
wrgl['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Wanaparthy district
wrgl=df[df['district']=='Wanaparthy']
#cummulative rainfall in Wanaparthy district
wrgl['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Karimnagar district
krngr=df[df['district']=='Karimnagar']
#cummulative rainfall in Karimnagar district
krngr['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Karimnagar district
krngr=df[df['district']=='Karimnagar']
#cummulative rainfall in Karimnagar district
krngr['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


district=df.groupby('district')


# In[ ]:


district['humidity_min'].max().iplot()


# In[ ]:


district['humidity_max'].max().iplot()


# In[ ]:


district['cumm_rainfall'].max().iplot()


# In[ ]:


hyd=df[df['district']=='Hyderabad']


# In[ ]:


hyd


# In[ ]:


hyd.describe()


# In[ ]:


hyd['mandal'].unique()


# In[ ]:


district['mandal'].count().iplot('bar')


# In[ ]:


dstrct.count()


# In[ ]:


#adilabad
adb=df[df['district']=='Adilabad']


# In[ ]:


#cummulative rainfall in adilabad
adb['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#bhadradri-kothagudem
bdrd=df[df['district']=='Bhadradri-Kothagudem']


# In[ ]:


#cummulative rainfall in bhadradri-kothagudem
bdrd['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#hyderabad
hyd=df[df['district']=='Hyderabad']


# In[ ]:


#cummulative rainfall in hyderabad
hyd['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#jagtial
jgtl=df[df['district']=='Jagtial']


# In[ ]:


#cummulative rainfall in jagtial
jgtl['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#jangaon district
jngn=df[df['district']=='Jangaon']


# In[ ]:


#cummulative rainfall in jangaon district
jngn['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Jayashankar-Bhupalpally district
asnkr=df[df['district']=='ayashankar-Bhupalpally']


# In[ ]:


#cummulative rainfall in ayashankar-Bhupalpally  district
asnkr['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Jogulamba-Gadwal district
jglmba=df[df['district']=='Jogulamba-Gadwal']
#cummulative rainfall in Jogulamba-Gadwal district
jglmba['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Kamareddy district
kmrdy=df[df['district']=='Kamareddy']
#cummulative rainfall in Kamareddy district
kmrdy['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


#Karimnagar district
krmngr=df[df['district']=='Karimnagar']
#cummulative rainfall in Karimnagar district
krmngr['cumm_rainfall'].iplot(kind='histogram')


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


df.dropna(how='any', inplace=True)
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


subdivs = df['Aws ID'].unique()
num_of_subdivs = subdivs.size
print('Total # of Subdivs: ' + str(num_of_subdivs))
subdivs


# In[ ]:


subdivs = df['district'].unique()
num_of_subdivs = subdivs.size
print('Total # of Subdivs: ' + str(num_of_subdivs))
subdivs


# In[ ]:


fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
df.groupby('district').mean().sort_values(by='cumm_rainfall', ascending=False)['cumm_rainfall'].plot('bar', color='r',width=0.3,title='Subdivision wise Average Annual Rainfall', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print(df.groupby('district').mean().sort_values(by='cumm_rainfall', ascending=False)['cumm_rainfall'][[0,1,2]])
print(df.groupby('district').mean().sort_values(by='cumm_rainfall', ascending=False)['cumm_rainfall'][[28,29,30]])


# In[ ]:


fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
dfg = df.groupby('district').sum()['cumm_rainfall']
dfg.plot('line', title='Overall Rainfall in Each District', fontsize=20)

plt.ylabel('Overall Rainfall (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))


# In[ ]:


months = df.columns[2:14]
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
xlbls = df['district'].unique()
xlbls.sort()
dfg = df.groupby('district').mean()
dfg.plot.line(title='Overall Rainfall in Each Month of Year', ax=ax,fontsize=20)
plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'xx-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)

dfg = dfg.mean(axis=0)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))


# In[ ]:



fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
df.groupby('district').mean().plot.line(title='Overall Rainfall in Each Month', ax=ax,fontsize=20)
#plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'x-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


# In[ ]:


fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
xlbls = df['district'].unique()
xlbls.sort()
dfg = df.groupby('district').mean()
dfg.plot.line(title='Overall Rainfall in Each Month', ax=ax,fontsize=20)
plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'xx-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)

dfg = dfg.mean(axis=0)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Max: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))


# In[ ]:


fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
df.groupby('district').mean().plot.line(title='Overall Rainfall in Each Month', ax=ax,fontsize=20)
#plt.xticks(np.linspace(0,35,36,endpoint=True),xlbls)
plt.xticks(  rotation = 90)
plt.ylabel('Rainfall (mm)')
plt.legend(loc='upper right', fontsize = 'x-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


# In[ ]:


import sklearn.linear_model as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df2 = df[['Aws ID','cumm_rainfall','humidity_min','humidity_max']]
df2.columns = np.array(['Aws ID', 'x1','x2','x3'])

for k in range(1,9):
    df3 = df[['Aws ID','cumm_rainfall','humidity_min','humidity_max']]
    df3.columns = np.array(['Aws ID', 'x1','x2','x3'])
    df2 = df2.append(df3)
df2.index = range(df2.shape[0])
    
#df2 = pd.concat([df2, pd.get_dummies(df2['SUBDIVISION'])], axis=1)

df2.drop('Aws ID', axis=1,inplace=True)
#print(df2.info())
msk = np.random.rand(len(df2)) < 0.8

df_train = df2[msk]
df_test = df2[~msk]
df_train.index = range(df_train.shape[0])
df_test.index = range(df_test.shape[0])

reg =sk.LinearRegression()
reg.fit(df_train.drop('x1',axis=1),df_train['x1'])
predicted_values = reg.predict(df_train.drop('x1',axis=1))
residuals = predicted_values-df_train['x1'].values
print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))
df_res = pd.DataFrame(residuals)
df_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
df_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


predicted_values = reg.predict(df_test.drop('x1',axis=1))
residuals = predicted_values-df_test['x1'].values
print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))
df_res = pd.DataFrame(residuals)
df_res.columns = ['Residuals']

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
df_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)
ax.xaxis.set_ticklabels([])
plt.ylabel('Residual')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


# In[ ]:


df.hist(figsize=(18,18))


# In[ ]:


# Graph has date on the x-axis
p = figure(title="Graph 1:Rainfall per day in 2018", x_axis_type='datetime')

p.line(x='odate', y='Count') #build a line chart
p.xaxis.axis_label = 'odate'
p.yaxis.axis_label = 'humidity_min'

p.xgrid.grid_line_color = None

# add a hover tool and show the date in date time format
hover = HoverTool()
hover.tooltips=[
    ('odate', '@Date{%F}'),
    ('Count', '@Count')
]
hover.formatters = {'odate': 'datetime'}
p.add_tools(hover)
output_notebook() # show the output in jupyter notebook
show(p)


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


# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# scmp2k19.csv has 237273 rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('../input/scmp2k19.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'scmp2k19.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# In[ ]:


plotCorrelationMatrix(df1, 8)


# In[ ]:


plotScatterMatrix(df1, 12, 10)


# In[ ]:




