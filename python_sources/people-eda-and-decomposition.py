#!/usr/bin/env python
# coding: utf-8

# Let's do some exploratory data analysis on the people data set, and see if it makes sense to use any decomposition techniques. I think there could be a couple good reasons why we might want to do this:
# 
# * If a lot of the characteristics of people are interdependent variables, we can consolidate them into single features so as to not muddy our final classifier inputs with repeat information.
# * There's potential for discovering latent features that are only evident by looking at multiple features together.
# * If you're like me, you might be pretty limited on computing resources and want to shrink the people data set before merging it in.

# In[ ]:


import pandas as pd

# Import the dataset
df = pd.read_csv("../input/people.csv")

# Print the first five rows
print(df.head())


# It looks like all the characteristics are denoted with a "char_" prefix. Let's use that to make sure that we don't drag along extra columns. Then we'll run a matrix of chi squared tests to get an idea of which features might be interdependent.

# In[ ]:


from scipy.stats import chisquare

# Create a list of characteristics
chars = [i for i in df.columns.values if "char_" in i]

# Create an empty list for appending flagged features
flags = []

# For each feature summarize frequencies of each other feature
for feat in df[chars]:
    group = df[chars].groupby(feat)
    for otherfeat in df[chars].drop(feat, axis=1):
        summary = group[otherfeat].count()
        
        # Run a chi squared test on the frequencies, and check if the p-value is less than 0.05
        if chisquare(summary)[1] < 0.05:
            
            # If so, flag both features
            flags.append(feat)
            flags.append(otherfeat)

# Remove duplicates by converting to a set at the end
flags = set(flags)

print("It looks like {}% of the characteristics might be related to one another.".format(len(flags)/len(chars)*100))


# Wow, 100%. Hopefully someone reviewing this can highlight if I implemented those tests wrong, or if chi squared wasn't the right test of choice. At any rate, I'm going to proceed on the assumption that we should definitely be using some decomposition techniques on the data set, if only just to reduce the repeated information. Let's use the scikit-learn implementation of PCA. Some success with PCA would confirm that the chi squared tests were useful.

# In[ ]:


# Convert to dummy variables
dums = pd.get_dummies(df[chars])

print("Before PCA the full size of the characteristics is {} features".format(len(dums.columns.values)))


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# As was suggested by dpace, let's also scale the features so that they're all in the range of 0 and 1
scaledums = MinMaxScaler().fit_transform(dums)

# Now we're ready for PCA. Let's just look at the first two principle components first
pca = PCA(n_components=2)
featurecomponents = pca.fit_transform(scaledums)

print(pca.explained_variance_ratio_)


# Awesome, scaling seems to fix the problem of char_38 dominating the first component and the overall variance of the dataset, thanks, dpace.

# In[ ]:


import numpy as np

# build a dictionary with the names of the components
components = {}
index = 0
for feature in dums.columns.values:
    components[feature] = [pca.components_[0][index]]
    index += 1
    
# Exclude all but the most extreme components, because there are a lot
sortedcomps = pca.components_[0]
sortedcomps.sort()
maxcap = sortedcomps[-3]
mincap = sortedcomps[2]
components = {i:x for i, x in components.items() if x >= maxcap or x <= mincap}
    
# Convert to dataframe
components = pd.DataFrame(components)

# Plot the most extreme components
components.plot(kind="bar", figsize=(12, 4))


# Interesting, and plotting the most extreme contributors to the first principle component, char_38 isn't among them. So scaling down char_38 was really critical.

# In[ ]:


# Plot the first two principle components
featurecomponents = pd.DataFrame(featurecomponents, columns=["Principle Component 1", "Principle Component 2"])
df["Principle Component 1"] = featurecomponents["Principle Component 1"]

featurecomponents.plot(kind="scatter", x="Principle Component 1", y="Principle Component 2", figsize=(12, 12), s=1)


# I'm curious now what the group_1 feature looks in terms of these first two principle components. I wonder if the groups were made from the characteristics, or something else. Just for curiosity's sake, let's plot a few groups.

# In[ ]:


# Add group_1 to the new data from pca
featurecomponents["group_1"] = df["group_1"]

# Get a list of groups to sample from
groupslist = list(set(featurecomponents["group_1"].tolist()))

# Pick a group and plot
group = featurecomponents[featurecomponents["group_1"]==groupslist[0]]
group.plot(kind="scatter", x="Principle Component 1", y="Principle Component 2", figsize=(3, 3))
print("There are {} data points in this group.".format(len(group.index)))


# In[ ]:


# Pick a group and plot
group = featurecomponents[featurecomponents["group_1"]==groupslist[5]]
group.plot(kind="scatter", x="Principle Component 1", y="Principle Component 2", figsize=(3, 3))
print("There are {} data points in this group.".format(len(group.index)))


# In[ ]:


# Pick a group and plot
group = featurecomponents[featurecomponents["group_1"]==groupslist[6]]
group.plot(kind="scatter", x="Principle Component 1", y="Principle Component 2", figsize=(3, 3))
print("There are {} data points in this group.".format(len(group.index)))


# Plotting the a few groups with at least two data points to them, I can't really tell if group_1 was created from clustering the characteristics. However, these first two principle components don't capture all of the information from the characteristics, and browsing through six data points is hardly conclusive. I or someone else will have to explore this more thoroughly.
# 
# As a final question, how many principle components do I need to merge into the train/test sets if I only care about a certain amount of the explained variance?

# In[ ]:


# Define a list of possible amounts of explained variance we might care about
cares = [i/100 for i in range(75, 100, 5)]

# Run the PCA with increased components until each care level is reached
for i in range (20, len(dums.columns.values)):
    pca = PCA(n_components=i)
    pca.fit(scaledums)
    try:
        if pca.explained_variance_ratio_.sum() > cares[0]:

            # If greater, print a statement and drop the first item off the list
            print("To explain {0} of the variance you'll need {1} components".format(cares[0], i))
            cares = cares[1:]
    except:
        break

