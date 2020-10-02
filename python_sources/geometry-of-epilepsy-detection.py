#!/usr/bin/env python
# coding: utf-8

# # The Geometry of Epilepsy Detection
# 
# This kernel is my first try on the problem of Epilesy Detection.
# . 
# I recently learned a new algorithm, that I appreciate very much. It maps the feature space onto 2 dimensions and we can therefore get a glimpse of the geometry of the classification problem:
# - Are the classes distinguishable by the features given?
# - What Machine learning approach might work best for this problem?
# 
# ## Background
# The kernel uses the **t-SNE Algorithm**.
# It is a dimensionality reduction algorithm, that has its focus on preserving the distances of the datapoints as much as possible, whille mapping them on a plain, that can be easily visualized by us humans.
# 
# If you want to know more about that algorithm: here is an article for you: It is free, but to read it, you need to login at Oreilly's website:
# [An illustrated introduction to the t-SNE algorithm](https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)
# 
# # Content
# 1. Loading the Libraries
# 2. Loadinging and preparing the data
# 3. Applying the algorithm
# 4. Visualization of the result
# 5. Interpretation of the Visualization
# 6. Trying K Nearest Neighbour
# 7. What do you think?

# ## 1. Loading Libraries
# We need only the standard libraries for this:
# - numpy
# - pandas
# - matplotlib
# - seaborn

# In[ ]:


# use numpy and pandas
import numpy as np
import pandas as pd

# We need sklearn for preprocessing and for the TSNE Algorithm.
import sklearn
from sklearn.preprocessing import Imputer, scale
from sklearn.manifold import TSNE

# WE employ a random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')

# We import seaborn to make nice plots.
import seaborn as sns


# ## 2. Loading and preparing the data
# We load and prepare the data for the t-SNE Algorithm
# 
# ### The t-SNE Algorithm expects
# - numeric input
# - no missing values are allowed
# - the data must be sorted by its target
# 
# ### Checks
# - we check the data for missing values
# - we check the data types for being numeric
# 
# ### Transformations:
# - there is a field 'Unamed: 0' which is categorical and which we remove.
# - the target column is 'y'  has values 1,2,3,4,5. Since only 1 is an epileptic seizure and we are only interested in detecting those, we map all the normal cases 2,3,4,5 onto 0
# 
# ### Sorting:
# - we sort the data for the target column
# 
# ### Split off Target
# We split the target off the rest of the data, since this is what the t-SNE algorithm expects

# In[ ]:


# Loading the data
X = pd.read_csv('../input/data.csv')
print("The data has {} observations and {} features".format(X.shape[0], X.shape[1]))


# In[ ]:


# Are there null values in the dataframe?
cols_null_counts = X.apply(lambda x: sum(x.isnull()))
print('number of columns with null values:', len(cols_null_counts[cols_null_counts != 0]))


# In[ ]:


# Any non numeric datatypes?
datatypes = X.dtypes
print('datatypes that are used: ', np.unique(datatypes.values))

# only the columns of type object concerns us
print('nr of columns for dtype object: ', len(datatypes.values[datatypes.values == 'object']))
print('Columns of type object are: ', [col for col in X.columns if X[col].dtype == 'object'])
X['Unnamed: 0'].values[:10]


# In[ ]:


# We drop the 'Unnamed: 0' column, maybe it is some internal adminstrative kind of information?
X.drop('Unnamed: 0', inplace=True, axis=1)

# we transform the target into 0 or 1: 0 for normal brain and 1 for the epileptic seizure
X['y'] = X['y'].apply(lambda x: 1 if x == 1 else 0)

# now we sort for the target
X.sort_values(by='y', inplace=True)

# We split the target off the fetures and store it seperately
y = X['y']
X.drop('y', inplace=True, axis=1)
assert 'y' not in X.columns

# make sure the target is binary now
assert set(y.unique()) == {0, 1}

# we also scale the data
X = scale(X) 


# ## 3. Applying the algorithm
# We are now ready for applying the algorithm:

# In[ ]:


# run the Algorithm
epileptic_proj = TSNE(random_state=RS).fit_transform(X)
epileptic_proj.shape


# ## 4. Visualization of the result
# In order to visualize the result we first build a function that produces the scatter plot from the data:
# - each observation is plotted to a plain
# - the color marks the observations as representing normal or epileptic brain activity

# In[ ]:


# building the scatter plot function: the target comes in as color, x is the data
def scatter(x, colors):
    """this function plots the result
    - x is a two dimensional vector
    - colors is a code that tells how to color them: it corresponds to the target
    """
    
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2)[::-1])

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)]
                   )
    
    ax.axis('off') # the axis will not be shown
    ax.axis('tight') # makes sure all data is shown
    
    # set title
    plt.title("Epilepsy detection", fontsize=25)
    
    # legend with color patches
    epilepsy_patch = mpatches.Patch(color=palette[1], label='Epileptic Seizure')
    normal_patch = mpatches.Patch(color=palette[0], label='Normal Brain Activity')
    plt.legend(handles=[epilepsy_patch, normal_patch], fontsize=10, loc=4)

    return f, ax, sc

# Now we call the scatter plot function on our data
scatter(epileptic_proj, y)


# ## 5. Interpretation of the Visualization
# 
# ### The normal zone is in the middle
# - there is a big "normal" area in the middle
# - the epileptic zone is on the borders
# - there two epleptic borders
# 
# ### Conclusions for choosing a machine learning algorithm
# - we can try kNearestNeighbor
# - randomForest or XgBoost might work
# - linear regression might not be very successful

# ## 6. Implementing K Nearest Neighbour Algorithm
# I want to compare K Nearest Neighbor Application on the original dataset with that applied to the t-SNE transformation: 
# - therefore I am building a function doing the scoring

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def apply_KnearestNeighbor(X):
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=seed)

    # fit model no training data
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    print(model)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y_test, predictions)
    print("Area Under the Receiver Operating Characteristic Curve: %.2f%%" % roc_auc)


# ## Applying KNearest Neighbor on the original dataset

# In[ ]:


apply_KnearestNeighbor(X)


# ## Applying KNearest Neighbor on the t-SNE transformation

# In[ ]:


apply_KnearestNeighbor(epileptic_proj)


# ### Result
# - it seems like the t-SNE Transformation produces even better results in combination with K Nearest Neighbor. That maybe the case, because it reduces the noise?

# ## 7. What's do you think?
# I am very much looking forward to Feedback on my work. 
