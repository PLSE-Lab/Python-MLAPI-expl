#!/usr/bin/env python
# coding: utf-8

# # Approaching the Problem
# 
# ## The data
# So this dataset has the handtremours of 5 people. 
# 
# ## The questions
# Does it help to distinguish between these 5 people? Can it identify them?
# 
# ## Finding the answer
# When I approach a new classification problem my first question usually are:
# ### 1. How balanced are the classes?
# This can be answered very easily by plotting a bar graph on the class distribution
# ### 2. How much can they be set apart by the features provided?
# [t-SNE](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html**) is a very helpful algorithm in that regard since it tries to keep the distance between datapoints as much as possible by mapping them onto a plain where I can see them.
# 
# I will answer the 2 questions above with this notebook. Unfortuntely the answers are not very promising.

# # Import the needed Libraries
# We use numpy, pandas, sklearn matplotlib and seaborn.

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
# palette so that each person is having a color
palette = np.array(sns.color_palette("hls", 6))


# # Import the data and analyze it
# - this is just a small datafile in this case

# In[ ]:


data = pd.read_csv("../input/dataset.csv")
data.head()


# In[ ]:


data.info()


# In[ ]:


sns.countplot(x='ClassLabel', data=data, palette=palette[1:6])


# In[ ]:


data.describe()


# ## Conclusions
# - the data is clean: no missing values 
# - the label is equally distributed
# - the data could need some scaling

# # Prepare the data for t-SNE
# - sort the data by label (this is what the t-SNE algorithm expects)
# - we seperate input data and target

# In[ ]:


X = data.copy()

# now we sort for the target
X.sort_values(by='ClassLabel', inplace=True)

# We split the target off the features and store it separately
y = X['ClassLabel']
X.drop('ClassLabel', inplace=True, axis=1)

# make sure the target is not part of the input data any more
assert 'ClassLabel' not in X.columns

# make sure the target is as expected and turn it into an array
assert set(y.unique()) == {1, 2, 3, 4, 5}
y = np.array(y)

# we scale the data
X = scale(X) 


# # We now run the t-SNE algorithm mapping the features onto 2 dimensions

# In[ ]:


# run the Algorithm
handtremor_proj = TSNE(random_state=RS).fit_transform(X)
handtremor_proj.shape


# # We write a function to visualize the result

# In[ ]:


# choose the palette
palette = np.array(sns.color_palette("hls", 6))

# plot the result
def scatter_plot(x, colors, ax):
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')    
    return sc

# plot the legend
def legend_plot(font_size=14):
    patch1 = mpatches.Patch(color=palette[1], label='Person 1')
    patch2 = mpatches.Patch(color=palette[2], label='Person 2')
    patch3 = mpatches.Patch(color=palette[3], label='Person 3')
    patch4 = mpatches.Patch(color=palette[4], label='Person 4')
    patch5 = mpatches.Patch(color=palette[5], label='Person 5')
    plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], fontsize=font_size, loc=4)


# # Now we are ready to look at the result
# - what we expect are 5 equally distributed classes
# - we want to see how far the features are setting them apart from each other

# In[ ]:


f = plt.figure(figsize=(8, 8))
f.suptitle('Geometry of Handtremor for 5 persons', fontsize=20)
ax = plt.subplot(aspect='equal')
scatter_plot(handtremor_proj, y, ax)
legend_plot()


# # Disappointed?
# If this disappoints you, your are not alone. I was stunned: what did I do wrong here?
# Lets look at the different handtremors one by one, to find out, what is going on here:

# # Are all the projections on top of each other?
# - that would at least explain this picture
# 
# ## Let's try to take the pictures apart:

# In[ ]:


# finding the indexes for each person
persons = {}
for i in range(1, 6):
    persons[i] = np.where(y == i)    


# In[ ]:


# now we make a separate subfigure for each person
f, axs = plt.subplots(2, 3, figsize=(12,8))
axs[0,0] = scatter_plot(handtremor_proj[persons[1]], y[persons[1]], axs[0,0])
axs[1,0] = scatter_plot(handtremor_proj[persons[2]], y[persons[2]], axs[1,0])
axs[0,1] = scatter_plot(handtremor_proj[persons[3]], y[persons[3]], axs[0,1])
axs[1,1] = scatter_plot(handtremor_proj[persons[4]], y[persons[4]], axs[1,1])
axs[0,2] = scatter_plot(handtremor_proj[persons[5]], y[persons[5]], axs[0,2])
axs[-1, -1].axis('off')
legend_plot(font_size=20)


# # Conclusion
# - I am not sure whether this dataset is useful for identification.
# - Maybe the handtremors are too similar for healthy people, maybe they are not different enough to make for a satisfactory ID?

# # What is your opinion?
# I am very much hoping for feedback and a discussion on this datasets and my conclusions here!
