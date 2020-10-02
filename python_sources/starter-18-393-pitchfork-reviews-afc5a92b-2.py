#!/usr/bin/env python
# coding: utf-8

# Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of representation learning. Specifically, we'll design a neural network architecture such that we impose a bottleneck in the network which forces a compressed knowledge representation of the original input. If the input features were each independent of one another, this compression and subsequent reconstruction would be a very difficult task. However, if some sort of structure exists in the data (ie. correlations between input features), this structure can be learned and consequently leveraged when forcing the input through the network's bottleneck.

# ## Introduction
# Greetings from the Kaggle maXbox bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing.
# What I want to show in the end is a correlation between genres and score with an autoencoder (target-context pairings based on genre and mean review score):
# 
# 
# [[scores over genre---------------genre - score - reviews count
#     
#     genre - score - #reviews
#     electronic [6.8] 3874
#     metal [6.2,] 862
#     rock [6.7] 9438
#     None [6.5] 2371
#     rap [6.5,] 1559
#     experimental [8.1,] 1815
#     pop/r&b [7.1] 1432
#     folk/country [7.3] 685
#     jazz [7.1,] 435
#     global [7.0] 219
# 
# 
# For example we tend to give experimental or folk/country more bonus than metal or rap, lets have a look at brian eno formal member of roxy music or pink floyd:
# 
# * > brian eno {'experimental'} review id - score - title
# * > [22714, 22061, 21743, 20032, 17303, 14828, 2808, 11732, 11731]
# * > [7.7, 10.0, 8.0, 7.7, 8.0, 7.4, 6.1, 7.8, 8.8]
# * > ['reflection', 'another green world', 'the ship', 'nerve net', 'lux', 'small craft on a milk sea', 'another day on earth', 'music for films','discreet music']
# * according to score: another green world
# 
# * > pink floyd {'rock'}
# * > [22663, 20006, 10968, 10643, 6307]
# * > [8.8, 5.7, 4.0, 9.4, 10.0]
# * > ['the early years 1965-1972', 'the endless river', 'oh, by the way', 'the piper at the gates of dawn  [40th anniversary edition]', 'animals']
# * according to score: animals
# 

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 as sql
#import numpy as np


# There is 0 csv file in the current version of the dataset:
# 

# In[ ]:


print(os.listdir('../input'))
db = sql.connect('../input/database.sqlite')
scores = pd.read_sql('SELECT reviewid, score FROM reviews', db)
artists = pd.read_sql('SELECT * FROM artists', db)
years = pd.read_sql('SELECT * FROM years', db)
genres2 = pd.read_sql('SELECT distinct genre FROM genres', db)
scores2 = pd.read_sql('SELECT reviewid, score, title FROM reviews', db)
print(genres2.genre)

#con.close()
#sqlpath = os.listdir('../input')
print(set(years))
#print(scores2.info())

cursor = db.cursor()
genres = {}
genre_lookup = {}
scores = {}
cursor.execute('select distinct genre from genres')
for row in cursor:
    genre_lookup[len(genre_lookup)] = row[0] 
    genres[row[0]] = []
    
print(genres, len(genres)) 
scores2['years'] = years.year
print(scores2.info())

# using List comprehension + isdigit() +split() 
# getting numbers from string to check artist name with numbers in it  
res = [int(i) for i in str(artists.artist.str.split()) if i.isdigit()] 


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
    
plotPerColumnDistribution(scores2, 2, 2)


# In[ ]:


# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = 'df.name'
    #df = df.dropna('columns') # drop columns with NaN
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
    
#scores2.drop('reviewid', axis=1, inplace=True)
#scores2.drop('title', axis=1, inplace=True)
#scores2.name= scores2
plotCorrelationMatrix(scores2, 6)
print(scores2.info())
print(scores2.corr())


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
    
plotScatterMatrix(scores2, 10, 12)


# Oh, no! There are no automatic insights available for the file types used in this dataset. As your Kaggle kerneler bot, I'll keep working to fine-tune my hyper-parameters. In the meantime, please feel free to try a different dataset.

# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
