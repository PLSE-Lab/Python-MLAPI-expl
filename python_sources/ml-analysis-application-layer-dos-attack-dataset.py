#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is the data taken from the application layer of the system. Through the network analysis the lables are assigned having three classes. 1) Benign: Legit ,2) DoS slowloris:DoS attack and 3). DoS Hulk: DDoS attack. 

# ## Exploratory Analysis : Data Visualization
# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D #For Basic ploting
from sklearn.preprocessing import StandardScaler #Preprocessing
from sklearn import preprocessing    # Preprocessing
from sklearn.naive_bayes import GaussianNB #import gaussian naive bayes model
from sklearn.tree import DecisionTreeClassifier #import Decision tree classifier
from sklearn import metrics  #Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# There are 2 csv files in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data.

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

# ### Let's check 1st file: /kaggle/input/test_mosaic.csv

# In[ ]:


nRowsRead = 1000 # specify No. of row. 'None' for whole data
# test_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/test_mosaic.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'test_mosaic.csv'
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


plotCorrelationMatrix(df1, 19)


# 
# 
# Scatter and density plots:
# 

# In[ ]:




plotScatterMatrix(df1, 20, 10)


# ### Let's check 2nd file: /kaggle/input/train_mosaic.csv

# In[ ]:


nRowsRead = 1000 # specify No. of rows. 'None' for whole file
# train_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/train_mosaic.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'train_mosaic.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df2.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


plotPerColumnDistribution(df2, 10, 5)


# Correlation matrix:

# In[ ]:


plotCorrelationMatrix(df2, 19)


# 
# 
# Scatter and density plots:
# 

# In[ ]:


plotScatterMatrix(df2, 20, 10)


# Get the whole data to perform ML techniques. 

# > Training Data:

# In[ ]:


nRowsRead = None # specify No. of row. 'None' for whole data
# test_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
train_data = pd.read_csv('/kaggle/input/train_mosaic.csv', delimiter=',', nrows = nRowsRead)
train_data.dataframeName = 'train_mosaic.csv'
nRow, nCol = train_data.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a look what data looks like:

# In[ ]:


train_data.head()


# > Testing Data:

# In[ ]:


nRowsRead = None # specify No. of row. 'None' for whole data
# test_mosaic.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
test_data = pd.read_csv('/kaggle/input/test_mosaic.csv', delimiter=',', nrows = nRowsRead)
test_data.dataframeName = 'test_mosaic.csv'
nRow, nCol = test_data.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a look what data looks like:

# In[ ]:


test_data.head()


# Now, Identify the classes.

# In[ ]:


train_data['Label'].unique()
test_data['Label'].unique()


# Encode these string classes to numeric to perform further processes. 

# In[ ]:


# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
test_data['Label'] = label_encoder.fit_transform(test_data['Label'])


# > Lets check the updated data.

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# Next, Split the dataset into train and test. As convention in machine learning, X_train,X_test are used for features and y_train,y_test are used for classes. As our data is already in the form of two files. train_data and test data. We will split it in this way:

# In[ ]:


X_train = train_data.drop('Label',axis=1)
X_test = test_data.drop('Label',axis=1)
y_train = train_data['Label']
y_test = test_data['Label']


# Let's check the train and test data now. 

# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


X_test.head()


# In[ ]:


y_test.head()


# Now, we ll implement four algorithms Naive Bayes and Decision Tree. Then at the end we ll comapre the result's from all of these algorithms.

# **> Naive Bayes**

# In[ ]:


# create gaussian naive bayes classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(X_train,y_train)
#Predict the response for test dataset
gnb_pred = gnb.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy : ",metrics.accuracy_score(y_test,gnb_pred))


# > **Decision Tree**

# In[ ]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
dt_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, dt_pred))


# > **Decion Tree with Max Depth 3**

# In[ ]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
dt_pred1 = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, dt_pred1))


# **Comparison**

# In[ ]:


print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, gnb_pred))
print("Decision Tree Accuracy:",metrics.accuracy_score(y_test, dt_pred))


# **Above results shows that the Decision Tree gave high accuracy as compared to the Naive bayes.**
