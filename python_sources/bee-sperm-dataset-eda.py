#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import os # accessing directory structure
import pandas as pd 
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"

import warnings  
warnings.filterwarnings('ignore')


# There is 1 csv file in the current version of the dataset:
# 

# In[ ]:


nRowsRead = 1000
dataframe = pd.read_csv('/kaggle/input/bee_sperm.csv', delimiter=',', nrows = nRowsRead,skiprows=3, error_bad_lines=False)
dataframe.dataframeName = 'bee_sperm.csv'
nRow, nCol = dataframe.shape


# **Column renaming**

# In[ ]:


dataframe.columns = ['Specimen', 'Treatment', 'Environment', 'TreatmentNCSS', 'Sample ID','Colony', 'Cage', 'Sample', 'SpermVolumePer500ul', 'Quantity',
       'ViabilityRawPercentage', 'Quality', 'Age', 'Infertil', 'AliveSperm','Quantity_Millions', 'Alive_Sperm_Millions', 'Dead_Sperm_Millions']


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


# Now we're ready to read in the data and use the plotting functions to visualize the data.

# ### Let's start investigation

# In[ ]:


nRowsRead = 1000 # specify 'None' if want to read whole file
# bee_sperm.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
dataframe = pd.read_csv('/kaggle/input/bee_sperm.csv', delimiter=',', nrows = nRowsRead,skiprows=3, error_bad_lines=False)
dataframe.dataframeName = 'bee_sperm.csv'
nRow, nCol = dataframe.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


dataframe.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# **Correlation Plots:**

# In[ ]:


def corrplot(df):
    plt.figure(figsize=(21,21))
    plt.title("Spearman Correlation Heatmap")
    corr = df.corr(method='spearman')
    mask = np.tril(df.corr())
    sns.heatmap(corr, 
               xticklabels=corr.columns.values,
               yticklabels=corr.columns.values,
               annot = True, # to show the correlation degree on cell
               vmin=-1,
               vmax=1,
               center= 0,
               fmt='0.2g', #
               cmap= 'coolwarm',
               linewidths=3, # cells partioning line width
               linecolor='white', # for spacing line color between cells
               square=False,#to make cells square 
               cbar_kws= {'orientation': 'vertical'}
               )

    b, t = plt.ylim() 
    b += 0.5  
    t -= 0.5  
    plt.ylim(b,t) 
    plt.show()
    
corrplot(dataframe)


# In[ ]:


#Alligned Correlation plot for better understanding correlations
def allignedCorrelationPlot(dataframe):
    df = dataframe.copy(deep=True)
    corr = df.corr(method='spearman')
    plt.figure(figsize=(18,10))
    mask = np.tril(df.corr())
    sns.heatmap(corr, 
               xticklabels=corr.columns.values,
               yticklabels=corr.columns.values,
               annot = True,
               vmin=-1,
               vmax=1,
               center= 1,
               cmap= 'Set2',
               linewidths=0.3,
               linecolor='green',
               square=False,
               cbar_kws= {'orientation': 'horizontal'},
               mask= mask)

    #To adjust the trim at bottom and top
    b, t = plt.ylim() 
    b += 1  
    t -= 2  
    plt.ylim(b, t) 

    plt.show()


# In[ ]:


allignedCorrelationPlot(dataframe)


# In[ ]:


processedFrame = dataframe.copy(deep=True)


# In[ ]:


Cage = processedFrame["Cage"]
Colony = processedFrame["Colony"]
fig = go.Figure()
fig.add_trace(go.Box(x=Cage,name="Cage"))
fig.add_trace(go.Box(x=Colony,name= "Colony"))
fig.show()


# In[ ]:


fig = px.violin(dataframe, y="Cage", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show()


# In[ ]:


fig = px.violin(dataframe, y="Colony", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show()


# In[ ]:


fig = go.Figure()
TreatmentNCSS = processedFrame["TreatmentNCSS"]
fig.add_trace(go.Box(x=TreatmentNCSS,name= "TreatmentNCSS"))
fig.show()


# In[ ]:


fig = px.violin(dataframe, y="TreatmentNCSS", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show()


# In[ ]:


# Specimen
Specimen = processedFrame["Specimen"]
fig = go.Figure()
fig.add_trace(go.Box(x=Specimen,name="Specimen"))
fig.show()


# In[ ]:


fig = px.violin(dataframe, y="Specimen", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show(size=(21,12))


# In[ ]:


#Sperm_Vol_per500 vs Quantity
Sperm_Vol_per500 = processedFrame["Sperm Volume per 500 ul"]
Quantity = processedFrame["Quantity"]
fig = go.Figure()
fig.add_trace(go.Box(x=Sperm_Vol_per500,name="Sperm_Vol_per500"))
fig.add_trace(go.Box(x=Quantity,name= "Quantity"))
fig.show()


# In[ ]:


Colony = processedFrame["Colony"]
Cage = processedFrame["Cage"]
Sample = processedFrame["Sample"]
ViabilityRaw = processedFrame["ViabilityRaw (%)"]
Quality = processedFrame["Quality"]

fig = go.Figure()
# Use x instead of y argument for horizontal plot.
fig.add_trace(go.Box(x=Quality,name= "Quality"))
fig.add_trace(go.Box(x=ViabilityRaw,name = "ViabilityRaw"))
fig.add_trace(go.Box(x=Colony,name= "Colony"))
fig.add_trace(go.Box(x=Cage,name="Cage"))
fig.add_trace(go.Box(x=Sample,name = "Sample" ))

fig.show()


# In[ ]:


fig = px.violin(processedFrame, y="ViabilityRaw (%)", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show()


# In[ ]:


fig = px.violin(dataframe, y="Quality", box=True, # draw box plot inside the violin
                points='all', # can be 'outliers', or False
               )
fig.show()

