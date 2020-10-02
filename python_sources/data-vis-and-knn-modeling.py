#!/usr/bin/env python
# coding: utf-8

# # What trees do the City of Oakland like to plant?
# ## Statistics and predictions using species and size of wells
# ### Table of Contents
# 1. Introduction
# 1. Descriptive statistics 
# 1. Summary and data cleaning
# 1. Distribution, Central Tendancy, and Dispertion
# 1. KNN modelling using Neighborhood Component Analysis

# ## Introduction
# In this kernel I will performing basic data cleaning and visualization for the top species planted by the City of Oakland. In addition, I will perform a iterative learning approach to predicting the type of planted species by the size of the well dug by the city using K-nearest neighbors supervised learning method.

# ## Descriptive statistics
# ### Summary and data cleaning

# In[ ]:


#Kernel preperations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import NeighborhoodComponentsAnalysis as nca
from sklearn.pipeline import Pipeline as pipe

#Data import
df_base = pd.read_csv('../input/oakland-street-trees/oakland-street-trees.csv')
print(df_base.info())


# We are interested in the following parameters for this analysis. There are no current data descriptions on the data's homepage or kaggle, so assume the following,
# * WELLWIDTH - the width of the well (ft.)
# * WELLLENGTH - the length of the well (ft.)
# * SPECIES - scientific name of planted species
# The data also hase the structured variable 'Location 1' which contains highly specific information on where the tree was planted. This will make for a good data cleaning and mapping kernel later on.

# In[ ]:


#Basic summary statistics
print(df_base.describe())


# In[ ]:


#Start of data cleaning - returning the number of NaN values in the set
print(len(df_base) - df_base['WELLWIDTH'].count())


# In[ ]:


#Describing the missing data
print(df_base[df_base['WELLWIDTH'].isna()].describe())


# In[ ]:


#Dropping the observations since they do not contain useful information
df_base = df_base.dropna()

#Confirming that 5 observations were dropped (should expect 38,608 obs)
print(len(df_base))


# ### Distribution, Central Tendancy, and Dispertion (visual)

# In[ ]:


#Bar chart for top 20 species of trees
#Subsampling data into a pandas series of top planted tree species
df_topSpecies = df_base['SPECIES'].groupby(df_base['SPECIES']).count().sort_values(ascending=False).head(20)
fig = plt.figure(dpi=90)
df_topSpecies.plot.bar()
plt.style.use('seaborn-pastel')
plt.title('Most planted trees, by species')
plt.xlabel('Species')
plt.ylabel('Number of trees')
plt.show()

print("These top 20 species make up for", df_topSpecies.sum(), "of", len(df_base), "trees planted (or", df_topSpecies.sum()/len(df_base),"% of trees).")


# In[ ]:


#Boxplots showing wellwidth for the top 5 tree species (saving space)
#subample
df_tsg = df_base[(df_base.SPECIES == 'Liquidambar styraciflua') | (df_base.SPECIES == 'Platanus acerifolia') |(df_base.SPECIES == 'Pyrus calleryana cvs') | (df_base.SPECIES == 'Prunus cerasifera/blireiana') | (df_base.SPECIES == 'Lagerstroemia indica')]

#Checking that this new data set matches the desired output
print(len(df_tsg))
print(sum(df_topSpecies.head(5)))


# In[ ]:


fig = plt.subplots(1,2)

plt.subplot(121)
ww_bp = sns.boxplot(y='WELLWIDTH',x='SPECIES', data=df_tsg, width = 0.5, palette='colorblind')
plt.xticks(rotation=90)
plt.ylabel('Well Width (ft.)')

plt.subplot(122)
wl_bp = sns.boxplot(y='WELLLENGTH',x='SPECIES', data=df_tsg, width = 0.5, palette='colorblind')
plt.xticks(rotation=90)
plt.ylabel('Well Length (ft.)')
wl_bp.yaxis.tick_right()
wl_bp.yaxis.set_label_position("right")

plt.suptitle('Dispersion of well lengths and widths, top 5 species')
plt.show(fig)


# In[ ]:


#Scatter plot for well length and width
scat = sns.pairplot(data=df_tsg, x_vars='WELLLENGTH', y_vars='WELLWIDTH',kind='scatter',hue='SPECIES', height=4, aspect=2, palette='dark')
plt.xlabel('Well length (ft.)')
plt.ylabel('Well width (ft.)')
plt.title('Length vs. width of wells, top 5 species')


# ## KNN modelling using Neighborhood Component Analysis
# In this section we will use what information we have about well lengths and widths to accurately predict the type of species that is being planted. From the most previous graph, we can see that the area of the holes does not differ dramatically by the type of species when considering the top 5 species. This same graph can be done with all the species and shows that the distribution from the origin is similar to that above. This will likely cause our model to poor accuracy.
# 
# Due to the high number of species in the data set (196 unique species), I will continue to use the subsampled dataset containing only the top 20 planted species (around 70% of the data).

# In[ ]:


#Creating the subsample of the original dataframe conditional on being a top 20 species
species_top = df_topSpecies.index
df_knn = df_base[df_base['SPECIES'].isin(species_top)]
print('Number of species in this dataset:', len(df_knn['SPECIES'].unique()))
print('Number of observations in this dataset:', len(df_knn))
print('Porportion to the original dataset:', len(df_knn)/len(df_base))


# In our model, we restrict the number of features to two in order to avoid over-fitting the model,
# * Label/Target: Species
# * Feature: Width of the well
# * Feature: Length of the well
# 

# In[ ]:


#Using sklearn to encode our categorical data
from sklearn import preprocessing as pp
le = pp.LabelEncoder()
target = le.fit_transform(df_knn['SPECIES']) #Values of 0 through 19
#target = df_knn['target']


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import NeighborhoodComponentsAnalysis as nca
from sklearn.pipeline import Pipeline as pipe

#Splitting the data into training and testing (70:30)
X_train, X_test, y_train, y_test = tts(df_knn, target, test_size= .30, random_state = 42)


# In[ ]:


nca_dat = nca(random_state = 42)
knn_model = knn(n_neighbors = 3)
nca_pipe = pipe([('nca', nca_dat),('knn',knn_model)])
nca_pipe.fit(X_test[['WELLWIDTH','WELLLENGTH']],y_test)


# In[ ]:


score = nca_pipe.score(X_test[['WELLWIDTH','WELLLENGTH']],y_test)
print('The NCA pipeline calssification has an accuracy score of', round(score,4))


# ## Conclusions
# It is clear from this neighborhood components analysis, that this data is not heterogeneous enough to provide accurate estimations of tree species. With all 196 species being present, there is simply not enough variability in the area of the wells that are correlated to any certain species. For a more satisfying result, one could limit the sample to the top 2 or 3 species and be much more successful at predicting the species.
