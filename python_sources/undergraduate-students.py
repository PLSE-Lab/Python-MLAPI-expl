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
df1 = pd.read_csv('../input/cusersmarildownloadsetudiantscsv/etudiants.csv', delimiter=';', nrows = nRowsRead)
df1.dataframeName = 'etudiants.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[ ]:


df1.head(5)


# In[ ]:


df1.shape


# In[ ]:


df1.columns


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# The SWARMPLOTS!

# In[ ]:


ax = sns.swarmplot(x="sexe", y="effectif", hue="rentree",
                   data=df1, palette="Set2", dodge=True)


# In[ ]:


ax = sns.swarmplot(x="rentree", y="sexe", data=df1)


# In[ ]:


ax = sns.swarmplot(x="rentree", y="effectif", data=df1)


# In[ ]:


ax = sns.swarmplot(x="sexe", y="effectif", hue="rentree", data=df1)


# In[ ]:


ax = sns.swarmplot(x="sexe", y="effectif", data=df1, size=6)


# In[ ]:


ax = sns.boxplot(x="rentree", y="effectif", data=df1, whis=np.inf)
ax = sns.swarmplot(x="rentree", y="effectif", data=df1, color=".2")


# In[ ]:


g = sns.catplot(x="sexe", y="effectif",
                hue="rentree", col="secteur",
                data=df1, kind="swarm",
                height=4, aspect=.7);


# In[ ]:


df1.info


# In[ ]:


df1.describe()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


plotPerColumnDistribution(df1, 10, 5)


# In[ ]:


plotCorrelationMatrix(df1, 8)


# In[ ]:


plotScatterMatrix(df1, 15, 10)


# In[ ]:


df1.rentree.describe()


# In[ ]:


print ("Skew is:", df1.rentree.skew())
plt.hist(df1.rentree, color='purple')
plt.show()


# In[ ]:


target = np.log(df1.rentree)
print ("Skew is:", target.skew())
plt.hist(target, color='orange')
plt.show()


# In[ ]:


numeric_features = df1.select_dtypes(include=[np.number])
numeric_features.dtypes


# In[ ]:


corr = numeric_features.corr()

print (corr['rentree'].sort_values(ascending=False)[1:11], '\n')
print (corr['rentree'].sort_values(ascending=False)[-10:])


# In[ ]:


df1.rentree.unique()


# In[ ]:


df1.effectif.unique


# In[ ]:


#Define a function which can pivot and plot the intended aggregate function 
def pivotandplot(data,variable,onVariable,aggfunc):
    pivot_var = data.pivot_table(index=variable,
                                  values=onVariable, aggfunc=aggfunc)
    pivot_var.plot(kind='bar', color='purple')
    plt.xlabel(variable)
    plt.ylabel(onVariable)
    plt.xticks(rotation=0)
    plt.show()


# In[ ]:


pivotandplot(df1,'rentree','sexe',np.median)


# In[ ]:


# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# It is a continous variable and hence lets look at the relationship of rentree (start of the university year) with sexe (gender) using a Regression plot

_ = sns.regplot(df1['rentree'], df1['sexe'])


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)


# In[ ]:


df1.plot(kind='scatter', x='rentree', y='effectif', alpha=0.5, color='mediumorchid', figsize = (12,9))
plt.title('rentree And effectif')
plt.xlabel("rentree")
plt.ylabel("effectif")
plt.show()


# In[ ]:


ax = sns.scatterplot(x="sexe", y="effectif",                      hue="rentree", legend="full", palette='RdPu', data=df1)


# In[ ]:


# libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Dataset
df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })
 
# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df1['rentree'], df1['sexe'], df1['effectif'], c='darkolivegreen', s=60)
ax.view_init(30, 185)
plt.show()


# In[ ]:


ax = sns.violinplot(x="rentree", y="effectif", data=df1, 
                    inner=None, color=".8")
ax = sns.stripplot(x="rentree", y="effectif", data=df1, 
                   jitter=True)
ax.set_title('effectif vs rentree')
ax.set_ylabel('effectif rentree')


# In[ ]:


fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,8))

sns.boxplot(x='rentree', y='effectif', data=df1, ax=axis1);
axis1.set_title('effectif vs rentree')
axis1.set_ylabel('effectif index')
sns.boxplot(x='rentree', y='sexe', data=df1, ax=axis2);
axis2.set_title('sexe vs rentree')
axis2.set_ylabel('sexe index')
sns.boxplot(x='rentree', y='effectif_ing', data=df1, ax=axis3);
axis3.set_title('effectif_ing vs rentree')
axis3.set_ylabel('effectif index')

