#!/usr/bin/env python
# coding: utf-8

# ## Visualization with t-SNE and Bokeh

# In[ ]:


# Basic
import numpy as np
import pandas as pd
# Plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
# Clustering
from sklearn.manifold import TSNE
from sklearn import cluster
# Bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
output_notebook()


# In[ ]:


df_train = pd.read_csv('../input/train.csv')  # load train
df_test = pd.read_csv('../input/test.csv')    # load test
y = df_train.pop('SalePrice')        # pop sales from train
df = pd.concat([df_train, df_test], axis=0).reset_index()  # train + test data
df.drop('index', axis=1, inplace=True)
print('Train')
df_train.info(max_cols=0)  # print summary about train dataframe
print('Test')
df_test.info(max_cols=0)   # print summary about test dataframe
print('All')
df.info(max_cols=0)        # print summary about all dataframe


# Let's see few lines from the main dataframe:

# In[ ]:


df.head(2)


# ## Processing the data
# Numeric and categorical features:

# In[ ]:


num_cols = df.select_dtypes(include=['float64', 'int64']).columns.values  # numerical
cat_cols = df.select_dtypes(include=['object']).columns.values            # categorical


# Print some info about numeric columns. For example, portion of NA's in a column.

# In[ ]:


print('Shape:', df[num_cols].shape)
nas = [x for x in df[num_cols].columns.values if df[x].isnull().sum() > 0]
print('Cols with NAs:', len(nas))
if len(nas)>0: 
    for x in nas: 
        print(x, ':{:.2f}% of NAs'.format(df[x].isnull().sum()/float(len(df))*100))


# Same for the categorical data:

# In[ ]:


print('Shape:', df[cat_cols].shape)
nas = [x for x in df[cat_cols].columns.values if df[x].isnull().sum() > 0]
print('Cols with NAs:', len(nas))
if len(nas)>0: 
    for x in nas: 
        print(x, ':{:.2f}% of NAs'.format(df[x].isnull().sum()/float(len(df))*100), 
              '; unique:', df[x].unique())


# Filling NA's in numeric columns with medians (by columns). Also, extracting ID column from the dataframe:

# In[ ]:


X_num = df[num_cols].fillna(df.median())  # X_num is the dataframe with numeric data
ids = X_num.drop('Id', axis=1)            # Get IDs and drop it from X_num


# Scale numeric data by StandardScaler:

# In[ ]:


scaler = StandardScaler()  # get a scaler
X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns.values) 
# X_num_scaled is the scaled numerical data


# Fill NA's in categorical data (using string NA):

# In[ ]:


X_cat = df[cat_cols].fillna('NA')


# Making dummy features for each categorical column:

# In[ ]:


list_of_dummies = []
for col in [cat_cols]:
    dum_col = pd.get_dummies(X_cat[col], prefix=col)
    list_of_dummies.append(dum_col)
X_cat = pd.concat(list_of_dummies, axis=1)
X_cat.head(2)


# Merging numeric and dummy features:

# In[ ]:


X = pd.concat([X_num_scaled, X_cat], axis=1)
X.info(max_cols=0)


# ## t-SNE
# Find 2-D representation of the data using t-SNE:

# In[ ]:


tsne = TSNE(init='pca', perplexity=40, learning_rate=1000, 
            early_exaggeration=8.0, n_iter=1000, random_state=0, metric='l2')
tsne_representation = tsne.fit_transform(X)


# ## Clustering

# In[ ]:


cl = cluster.AgglomerativeClustering(10)
cl.fit(tsne_representation);


# Let's plot 2D t-SNE representation and clusters:

# In[ ]:


plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
cmap = plt.cm.get_cmap('jet')
plt.scatter(tsne_representation[:len(y),0], tsne_representation[:len(y),1], 
            alpha=0.5, c=y, cmap=cmap, s=20)
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(tsne_representation[len(y):,0], tsne_representation[len(y):,1], 
            alpha=0.4, c=cl.labels_[len(y):], marker='s', s=20)


# ## Interactive plot with train and test data
# Here **square** = test objects, **circle** = train objects

# In[ ]:


source_train = ColumnDataSource(
        data=dict(
            x = tsne_representation[:len(y),0],
            y = tsne_representation[:len(y),1],
            desc = y,
            colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 
                      255*mpl.cm.jet(mpl.colors.Normalize()(y.values))],
            OverallQual = df['OverallQual'].iloc[:len(y)],
            GrLivArea = df['GrLivArea'].iloc[:len(y)],
            GarageCars = df['GarageCars'].iloc[:len(y)]
        )
    )

source_test = ColumnDataSource(
        data=dict(
            x = tsne_representation[len(y):,0],
            y = tsne_representation[len(y):,1],
            OverallQual = df['OverallQual'].iloc[len(y):],
            GrLivArea = df['GrLivArea'].iloc[len(y):],
            GarageCars = df['GarageCars'].iloc[len(y):]
        )
    )

hover_tsne = HoverTool(names=["test", "train"], tooltips=[("Price", "@desc"), 
                                 ("OverallQual", "@OverallQual"), 
                                 ("GrLivArea", "@GrLivArea"), 
                                 ("GarageCars", "@GarageCars")])
tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='Prices')

plot_tsne.square('x', 'y', size=7, fill_color='orange', 
                 alpha=0.9, line_width=0, source=source_test, name="test")
plot_tsne.circle('x', 'y', size=10, fill_color='colors', 
                 alpha=0.5, line_width=0, source=source_train, name="train")

show(plot_tsne)

