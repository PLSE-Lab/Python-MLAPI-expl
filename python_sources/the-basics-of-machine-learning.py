#!/usr/bin/env python
# coding: utf-8

# # Is This Mushroom Poisonous? - The basics of Machine Learning
# [Index](https://www.kaggle.com/veleon/in-depth-look-at-machine-learning-models)
# 
# Everyone has to start somewhere, usually the beginning. It is no different when you're learning Machine Learning. This Kernel is my attempt at showing the basics of Machine Learning.
# 
# We'll be using the Mushroom Classification dataset to determine if a mushroom we found is poisonous or edible.
# 
# <img src="https://cdn-images-1.medium.com/max/1600/1*_QGyIwpgq831xI54cIe_GQ.jpeg" alt="ML Process" width="600"/>
# ## Index
# 1. Importing Libraries & Data
# 2. Data Analysis & Cleaning
# 3. Training a Model
# 4. Visualize 
# 
# # 1. Importing Libraries & Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation
sns.set(style="darkgrid")
import matplotlib.pyplot as plt # data plotting

import warnings
warnings.simplefilter("ignore") # Remove certain warnings from Machine Learning Models

data = pd.read_csv('../input/mushrooms.csv')
data.head(2)


# # 2. Data Analysis & Cleaning
# Before we can train a model to help us determine if a mushroom is poisonous we have to look at what kind of data we have. After that we'll clean the data so it can be used by a model.

# In[ ]:


data.describe()


# We have a lot of different columns. We'll try to get some insight into what effect they have on the class of a mushroom by making some graphs.
# 
# Let's start by looking at the distribution of edible and poisonous mushrooms in our data. As you can see the distribution is almost 50/50, which is very nice when you are going to let a machine learning model use your data. This means it won't have to struggle to find correlations.

# In[ ]:


sns.countplot(data['class'])


# Most of the cap related columns seem fairly balanced between edible and poisonous. Notable are cap-shape k, cap-surface f and cap-color w for not having a fairly equal distribution.

# In[ ]:


fig, ax =plt.subplots(1,3, figsize=(15,5))
sns.countplot(x="cap-shape", hue='class', data=data, ax=ax[0])
sns.countplot(x="cap-surface", hue='class', data=data, ax=ax[1])
sns.countplot(x="cap-color", hue='class', data=data, ax=ax[2])
fig.tight_layout()
fig.show()


# The bruises are a much better indication if a mushroom is poisonous or not. You can clearly see the distribution in the plot. The odors are even more clear, only odor n has both poisonous and edible. Even then it's not really a fair distribution with edible being multiple times larger. These are both good columns to use for determining the class of a mushroom.

# In[ ]:


fig, ax =plt.subplots(1,2, figsize=(15,5))
sns.countplot(x="bruises", hue='class', data=data, ax=ax[0])
sns.countplot(x="odor", hue='class', data=data, ax=ax[1])
fig.tight_layout()
fig.show()


# Gills are a bit more evenly distributed again though there are some outliers in spacing, size and color

# In[ ]:


fig, ax =plt.subplots(1,4, figsize=(20,5))
sns.countplot(x="gill-attachment", hue='class', data=data, ax=ax[0])
sns.countplot(x="gill-spacing", hue='class', data=data, ax=ax[1])
sns.countplot(x="gill-size", hue='class', data=data, ax=ax[2])
sns.countplot(x="gill-color", hue='class', data=data, ax=ax[3])
fig.tight_layout()
fig.show()


# Again, the stalk columns are fairly evenly distributed. There are some outliers in surface and color that could be usefull for classification.

# In[ ]:


fig, ax =plt.subplots(2,3, figsize=(20,10))
sns.countplot(x="stalk-shape", hue='class', data=data, ax=ax[0,0])
sns.countplot(x="stalk-root", hue='class', data=data, ax=ax[0,1])
sns.countplot(x="stalk-surface-above-ring", hue='class', data=data, ax=ax[0,2])
sns.countplot(x="stalk-surface-below-ring", hue='class', data=data, ax=ax[1,0])
sns.countplot(x="stalk-color-above-ring", hue='class', data=data, ax=ax[1,1])
sns.countplot(x="stalk-color-below-ring", hue='class', data=data, ax=ax[1,2])
fig.tight_layout()
fig.show()


# Most of these columns are not very interesting, but ring-type has some useful information.

# In[ ]:


fig, ax =plt.subplots(2,2, figsize=(15,10))
sns.countplot(x="veil-type", hue='class', data=data, ax=ax[0,0])
sns.countplot(x="veil-color", hue='class', data=data, ax=ax[0,1])
sns.countplot(x="ring-number", hue='class', data=data, ax=ax[1,0])
sns.countplot(x="ring-type", hue='class', data=data, ax=ax[1,1])
fig.tight_layout()
fig.show()


# These last columns have some stronger outliers we can use for prediction. especially spore print color.

# In[ ]:


fig, ax =plt.subplots(1,3, figsize=(20,5))
sns.countplot(x="spore-print-color", hue='class', data=data, ax=ax[0])
sns.countplot(x="population", hue='class', data=data, ax=ax[1])
sns.countplot(x="habitat", hue='class', data=data, ax=ax[2])
fig.tight_layout()
fig.show()


# 

# Now that we know what our data looks like we can clean it. Let's start by turning the columns with only 2 different values into Booleans.

# In[ ]:


# Make column class True/False for isPoisonous
data['class'].replace('p', 1, inplace = True)
data['class'].replace('e', 0, inplace = True)

# Bruises: t = True / f = False
data['bruises'].replace('t', 1, inplace = True)
data['bruises'].replace('f', 0, inplace = True)


# Our machine learning models can't read characters (only integers and floats). So we'll have to make a column for every unique value. Pandas has a function for this, named get_dummies. Our DataFrame looks a bit different now.

# In[ ]:


# Encode the rest of the string data
data = pd.get_dummies(data)

pd.set_option("display.max_columns",200)
data.head(5)


# Let's make lists with the columns so we can make some correlation heatmaps.

# In[ ]:


Target = ['class']
bruisesColumn = ['bruises']
capColumns = list(data.columns[2:22])
odorColumns = list(data.columns[22:31])
gillColumns = list(data.columns[31:49])
stalkColumns = list(data.columns[49:82])
veilColumns = list(data.columns[82:87])
ringColumns = list(data.columns[87:95])
sporeColumns = list(data.columns[95:104])
populationColumns = list(data.columns[104: 110])
habitatColumns = list(data.columns[110:117])


# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(data[Target+odorColumns].corr(), annot=True)


# In[ ]:


plt.subplots(figsize=(10,10))
sns.heatmap(data[Target+populationColumns].corr(), annot=True)


# # 3. Training a Model
# Now that we have usable data we can start training our data. We'll be using the Decision Tree model since it's the easiest to visualise and understand.
# 
# First we have to create X and y DataFrames. Y DataFrames contain the data we predict, X DataFrames contain the data the model uses to predict y.

# In[ ]:


#Create X & y
X = data.iloc[:, 1:]
y = data['class']


# To test our model we split the data into training and testing data. This ensures that the model doesn't just memorize the data instead of finding correlations.

# In[ ]:


#Create Testing and Training Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Finally we can import and test our Decision Tree. We are using a Classifier version which gives us a Boolean variable ( 0 or 1 ). If we'd used a Regressor we would have gotten a number between 0 and 1.
# 
# To keep it simple we will limit the tree to a maximum depth of 5. Then we'll fit the training data. Fitting prepares the model for the real work, in this case the identification of X_test.

# In[ ]:


from sklearn import tree

dtc = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
dtc.fit(X_train, y_train)

dtc.score(X_test, y_test)


# Our final score is 0.954 which is really good! This means we are correct 95.4% of the time. To increase this we could remove the max_depth limit so the tree can grow bigger.
# 
# Now let's see what our Decision Tree looks like!
# The tree is built out of leaves and these leaves contain information about our model:
# * Condition of the leaf
# * Gini (or chance of incorrect measurement of a random training sample at that point)
# * The number of samples that passed during fitting
# * Class (or prediction) of the sample at that point

# In[ ]:


import graphviz
dot_data = tree.export_graphviz(dtc, feature_names=X.columns.values, class_names=['Edible', 'Poisonous'], filled=True )
graphviz.Source(dot_data) 


# # Conclusion 
# Now that you know how to start a Machine Learning project we'll go some more in depth into different Machine Learning Models in the next Kernels.
# ### Next Kernel
# [How Does Linear Regression Work?](https://www.kaggle.com/veleon/how-does-linear-regression-work)
# ### Back to Index
# [Index](https://www.kaggle.com/veleon/in-depth-look-at-machine-learning-models)
