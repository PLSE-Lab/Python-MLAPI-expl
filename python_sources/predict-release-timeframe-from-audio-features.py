#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Introduction
# 
# The Million Song Dataset (MSD) is a freely-available collection of audio features and metadata for a million contemporary popular music tracks. This is a subset of the MSD and contains audio features of songs with the year of the song. The purpose here is to predict the decade a song was released in, based on its audio features.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.utils import resample
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load and inspect data
# 
# The original dataset has release year as the label for each song. Lets convert this to release decade, since we are trying to predict the decade a song was released in, and not the exact year.

# In[ ]:


df = pd.read_csv('../input/year_prediction.csv')
df.sample(5)


# In[ ]:


# Group release years into decades
df['label'] = df.label.apply(lambda year : year-(year%10))


# The number of data samples are not uniform across release decades. There are too few samples of songs released before 1950.

# In[ ]:


sns.countplot(y="label", data=df)
plt.xlabel("Audio samples")
plt.ylabel("Release Decade")
plt.title("Samples in the dataset/release decade")


# Each sample has 90 features. Each feature takes a wide range of values.

# In[ ]:


print("(Samples, Features) {}".format(df.iloc[:,1:].shape))
df.iloc[:,1:].describe()


# ### Scale Features
# 
# After scaling these features using min-max scaling, each feature is reduced to a range of 0 to 1 

# In[ ]:


df.iloc[:,1:] = (df.iloc[:,1:]-df.iloc[:,1:].min())/(df.iloc[:,1:].max() - df.iloc[:,1:].min())
df.iloc[:,1:].describe()


# ### Downsample
# 
# We have over 500k samples, but there are too few samples for some categories and too many for others. Lets pick equal number of random samples for each category (release decade). Also we have too few samples of songs older than 1950. We will exclude these for now and revisit this later.

# In[ ]:


df_t = df[df.label>1940]
min_samples = df_t.label.value_counts().min()
decades = df_t.label.unique()
df_sampled = pd.DataFrame(columns=df_t.columns)
for decade in decades:
    df_sampled = df_sampled.append(df_t[df_t.label==decade].sample(min_samples))
df_sampled.label = df_sampled.label.astype(int)


# After downsampling our dataset has equal number of samples for each release decade.

# In[ ]:


sns.countplot(x="label", data=df_sampled)
plt.ylabel("Audio samples")
plt.xlabel("Release Decade")
plt.title("Downsampled dataset")


# ### Analyze features
# 
# Our dataset has 90 features. Heres a look at correlation between features and the output class.(Only first few features are included)

# In[ ]:


# Correlation between the release decade and features
corr = df_sampled.iloc[:,:20].corr()
fig, ax = plt.subplots(figsize=(10,10)) 
plt.title("Correlation")
sns.heatmap(corr, square=True)
plt.show()


# ### How do features differ by release decade?
# This heatmap visualizes how mean value of each feature differs based on the decade a song was released in. (Only first few features are included)

# In[ ]:


# How do features differ by release decade?
columns = df_sampled.groupby(['label']).mean().columns
labels = ["{:02d}'s".format(l%100) for l in sorted(df_sampled.label.unique())]
fig, ax = plt.subplots(figsize=(20,5)) 
sns.heatmap(df_sampled.groupby(['label']).mean().iloc[:,0:20], yticklabels=labels)
plt.ylabel("Release Decade")
plt.xlabel("Features (Mean)")
plt.show()


# Visualize how each feature differs by output class (decade a song was released in). Only the first few shown here.

# In[ ]:


for component in df_sampled.columns[1:11]:
    sns.FacetGrid(df_sampled, hue="label", size=3)        .map(sns.kdeplot, component)        .add_legend()
    plt.show()


# ### Dimensionality Reduction for Visualization
# 
# It is hard to visualize this high dimensional data (90 features). Lets explore couple of techniques for translating high-dimensional data into lower dimensional data. Purpose of dimensionality reduction here is visualization alone.
# 
# Use PCA to reduce to 20 principal components.

# In[ ]:


X = df_sampled.iloc[:,1:].values
y = df_sampled.iloc[:,0].values
print("X ", X.shape, ", y ", y.shape)


# In[ ]:


pca = PCA(n_components=20).fit(X)
X_pca = pca.transform(X)


# In[ ]:


principal_components = []
samples, features = X_pca.shape
for m in range(1, features+1):
    principal_components.append("Principal Component {}".format(m))
cols = principal_components+["Release Decade"]    
df_pca = pd.DataFrame(np.append(X_pca, y.reshape(samples,1), axis=1), columns=cols)
df_pca["Release Decade"] = df_pca["Release Decade"].astype(int)
print("df_pca.shape = ",df_pca.shape)


# Visualize principal components (only first two shown here)

# In[ ]:


sns.pairplot(df_pca, hue="Release Decade",x_vars="Principal Component 1",y_vars="Principal Component 2", size=10)


# Reducing this further to 2 components using t-SNE.

# In[ ]:


tsne_samples = df_pca.shape[0]
tsne = TSNE(n_components=2, verbose=2, perplexity=50, n_iter=1000)
tsne_results = tsne.fit_transform(df_pca.iloc[:tsne_samples,:-1])


# In[ ]:


df_tsne = pd.DataFrame(np.append(tsne_results, 
                                 df_pca.iloc[:tsne_samples,-1].values.reshape(tsne_results.shape[0],1), 
                                 axis=1), 
                       columns=["t-SNE Component 1","t-SNE Component 2","Release Decade"])
df_tsne["Release Decade"] = df_tsne["Release Decade"].astype(int)


# Visualize t-SNE components.

# In[ ]:


sns.pairplot(df_tsne, hue="Release Decade",x_vars="t-SNE Component 1",y_vars="t-SNE Component 2", size=10)


# t-SNE components of songs by release decade. There appears to be separation between output classes that are far apart, but not so much for songs released in adjacent decades. 

# In[ ]:


for component in df_tsne.columns[:-1]:
    sns.FacetGrid(df_tsne, hue="Release Decade", size=6)        .map(sns.kdeplot, component)        .add_legend()
    plt.show()


# In[ ]:


sns.pairplot(data=df_tsne.sample(1000), hue="Release Decade", vars=df_tsne.columns[:-1], size=4)


# In[ ]:


col = ["Greens", "Oranges", "Oranges","Purples", "Purples", "Blues", "Blues"]
for idx, year in enumerate([1950,1960,1970,1980,1990,2000,2010]):
    df_tsne_year = df_tsne[df_tsne['Release Decade']==year]
    sns.kdeplot(df_tsne_year['t-SNE Component 1'].values, df_tsne_year['t-SNE Component 2'].values,cmap=col[idx], shade=True, shade_lowest=False, n_levels=20)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("Songs Released in the {}s".format(year))
    plt.show()


# ### Classification 
# 
# Based on the analysis so far, there doesn't appear to be clear separation between output classes. Lets attempt classification using SVC. We will use complete set of features here and not the principal components visualized earlier.
# 
# Split the dataset into training and test set. Use grid search to find the best parameters for SVC.

# In[ ]:


df_sampled = shuffle(df_sampled)
df_train, df_test = train_test_split(df_sampled, test_size=0.3)


# In[ ]:


X_train = df_train.iloc[:,1:].values 
y_train = df_train.iloc[:,0].values
print("X_train ", X_train.shape, ", y_train ", y_train.shape)


# In[ ]:


#grid_search = GridSearchCV(svm.SVC(),
#                           {'kernel':['linear', 'rbf','poly'], 
#                            'C': [1, 5, 10,15,20,25], 
#                            'gamma' : [1, 5, 10,15,20]
#                           },
#                           cv=None)
#grid_search.fit(X_train, y_train)
#clf = grid_search.best_estimator_
#print(clf)
clf = svm.SVC(kernel='rbf',C=10,gamma=5);
clf.fit(X_train, y_train)


# Predict the release decade for songs in the test set using the trained SVC model. Print classification metrics.

# In[ ]:


tst = df_test
X_test = tst.iloc[:,1:].values 
y_test = tst.iloc[:,0].values
expected = y_test
predicted = clf.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
cnf_matrix = metrics.confusion_matrix(expected, predicted)


# Plot the confusion matrix.

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
labels = sorted(df_test.label.unique())
plot_confusion_matrix(cnf_matrix, classes=["{:02d}'s".format(label%100) for label in labels],
                      title='Confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["{:02d}'s".format(label%100) for label in labels], normalize=True,
                      title='Normalized')

plt.show()


# In[ ]:




