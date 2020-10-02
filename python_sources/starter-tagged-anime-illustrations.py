#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings! This is an automatically generated kernel with starter code demonstrating how to read in the data and begin exploring. If you're inspired to dig deeper, click the blue "Fork Notebook" button at the top of this kernel to begin editing your own copy.

# ## Exploratory Analysis
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made.Remember, I'm only a kerneling bot, not Jeff Dean or a Kaggle Competitions Grandmaster!

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There are 12 csv files in the current version of the dataset:
# 

# In[ ]:


print(os.listdir('../input'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/007_nagato_yuki'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/046_alice_margatroid'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/065_sanzenin_nagi'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/080_koizumi_itsuki'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/096_golden_darkness'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/116_pastel_ink'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/140_seto_san'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/144_kotegawa_yui'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/164_shindou_chihiro'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/165_rollo_lamperouge'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/199_kusugawa_sasara'))
print(os.listdir('../input/moeimouto-faces/moeimouto-faces/997_ana_coppola'))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Plot the PCA with either 2 or 3 reduced components
def plotPCA(df, nComponents):
	df = df.select_dtypes(include =[np.number]) # keep only numerical columns
	df = df.dropna('columns') # drop columns with NaN
	if df.shape[1] < nComponents:
		print(f'No PCA graph shown: The number of numeric columns ({df.shape[1]}) is less than the number of PCA components ({nComponents})')
		return
	df = df.astype('float64') # Cast to float for sklearn functions
	df = StandardScaler().fit_transform(df) # Standardize features by removing the mean and scaling to unit variance
	pca = PCA(n_components = nComponents)
	principalComponents = pca.fit_transform(df)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component ' + str(i) for i in range(1, nComponents + 1)])
	fig = plt.figure(figsize = (8, 8))
	if (nComponents == 3):
		ax = fig.add_subplot(111, projection = '3d')
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_zlabel('Principal Component 3', fontsize = 15)
		ax.set_title('3 component PCA', fontsize = 20)
		ax.scatter(xs = principalDf.iloc[:, 0], ys = principalDf.iloc[:, 1], zs = principalDf.iloc[:, 2])
	else:
		ax = fig.add_subplot(111)
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_title('2 component PCA', fontsize = 20)
		ax.scatter(x = principalDf.iloc[:, 0], y = principalDf.iloc[:, 1])


# In[ ]:


# Histogram of column data
def plotHistogram(df, nHistogramShown, nHistogramPerRow):
	nunique = df.nunique()
	df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
	nRow, nCol = df.shape
	columnNames = list(df)
	nHistRow = (nCol + nHistogramPerRow - 1) / nHistogramPerRow
	plt.figure(num=None, figsize=(6*nHistogramPerRow, 8*nHistRow), dpi=80, facecolor='w', edgecolor='k')
	for i in range(min(nCol, nHistogramShown)):
		plt.subplot(nHistRow, nHistogramPerRow, i+1)
		df.iloc[:,i].hist()
		plt.ylabel('counts')
		plt.xticks(rotation=90)
		plt.title(f'{columnNames[i]} (column {i})')
	plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
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


# Now, read in the data and use the plotting functions to visualize the data.

# ### Let's check 1st file: ../input/moeimouto-faces/moeimouto-faces/080_koizumi_itsuki/color.csv

# In[ ]:


nRowsRead = 100 # specify 'None' if want to read whole file
# color.csv may have more rows in reality, but we are only loading/previewing the first 100 rows
df1 = pd.read_csv('../input/moeimouto-faces/moeimouto-faces/080_koizumi_itsuki/color.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'color.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Histogram of sampled columns:

# In[ ]:


plotHistogram(df1, 10, 5)


# 2D and 3D PCA Plots

# In[ ]:


plotPCA(df1, 2) # 2D PCA
plotPCA(df1, 3) # 3D PCA


# ### Let's check 2nd file: ../input/moeimouto-faces/moeimouto-faces/199_kusugawa_sasara/color.csv

# In[ ]:


nRowsRead = 100 # specify 'None' if want to read whole file
# color.csv may have more rows in reality, but we are only loading/previewing the first 100 rows
df2 = pd.read_csv('../input/moeimouto-faces/moeimouto-faces/199_kusugawa_sasara/color.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'color.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df2.head(5)


# Histogram of sampled columns:

# In[ ]:


plotHistogram(df2, 10, 5)


# 2D and 3D PCA Plots

# In[ ]:


plotPCA(df2, 2) # 2D PCA
plotPCA(df2, 3) # 3D PCA


# ### Let's check 3rd file: ../input/moeimouto-faces/moeimouto-faces/007_nagato_yuki/color.csv

# In[ ]:


nRowsRead = 100 # specify 'None' if want to read whole file
# color.csv may have more rows in reality, but we are only loading/previewing the first 100 rows
df3 = pd.read_csv('../input/moeimouto-faces/moeimouto-faces/007_nagato_yuki/color.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'color.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df3.head(5)


# Histogram of sampled columns:

# In[ ]:


plotHistogram(df3, 10, 5)


# 2D and 3D PCA Plots

# In[ ]:


plotPCA(df3, 2) # 2D PCA
plotPCA(df3, 3) # 3D PCA


# ## Conclusion
# This concludes this starter analysis! To continue from here, click the blue "Fork Notebook" button at the top. This will create a copy of the code and environment for you to edit, delete, modify, and add code as you please. Happy Kaggling!
