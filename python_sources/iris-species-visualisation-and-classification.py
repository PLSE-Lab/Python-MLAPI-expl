#!/usr/bin/env python
# coding: utf-8

# # **Iris Species Visualisation and Classification**
# 
# My first kernel - we all have to start somewhere!
# 
# Throughout this analysis, the labels are as follows: **0 is setosa, 1 is versicolor **and** 2 is virginica**.
# 
# ## Import the Required Libraries

# In[ ]:


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn import datasets, cross_validation, neighbors, svm
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


# ## Load the Data and View the Labels & Features
# 
# This does not retrieve the data set from Kaggle, however, it is the same data set.

# In[ ]:


iris = datasets.load_iris()

df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
                  columns = iris['feature_names'] + ['target'])

print("Number of rows: {}\n".format(len(df)))
print("Distribution of labels:\n{}".format(df['target'].value_counts(normalize=True)))
print("\nFeatures:")

df.drop(df.columns[4], axis=1).head()
# Drop the 'target' column as it contains the labels (0, 1 or 2). We only want to see the features.


# ## Plot the Data
# 
# Having seen the features in tabular form, plotting the data will help us understand the value of each feature.

# In[ ]:


for i in range (0, 4): # Loop through each feature column by index (0, 1, 2, 3).
    
    species_labels = [0, 1, 2]
    
    plt.rcParams["figure.figsize"] = [14, 5]
    ax = plt.subplot(2, 4, i+1) # Scatter graph on top.
    plt.scatter(df[df.columns[i]], df['target'], c=df['target'])
    if i is 0: plt.ylabel('species')
    plt.yticks(species_labels)
    
    plt.subplot(2, 4, i+5, sharex=ax) # Box plot below the scatter graph.
    cols = df[[df.columns[i], 'target']]
    data = [] # Group the data by species.
    for j in species_labels:
        data.append(cols[cols['target']==j][df.columns[i]])
    plt.boxplot(data, vert=False, labels=species_labels)
    plt.xlabel(df.columns[i])
    if i is 0: plt.ylabel('species')

plt.show()


# From the graphs above, we see that the sepal width/length has much more overlap between species than the petal width/length, suggesting that sepal width/length will be less useful.
# 
# To understand why overlap can be bad, we can use an example. If we have a sepal width value of 3 cm, it is of very limited use to us because we cannot say identify the species of the iris using that piece of information alone. It is possible for all three species to have a sepal width of 3 cm.
# 
# Looking at species 0 (setosa), we see that for petal width and length there is no overlap with the other species. Assuming this holds true for the population, we will be able to identify species 0 with 100% accuracy.
# 
# As there is a relatively significant amount of overlap for sepal width, we will not use this feature during classification. This could be said of sepal length, however, it may be of some value when trying to distinguish between species 1 and species 2.
# 
# ## Review the Relationships between Features
# 
# Let's look at the relationship between petal width/length and sepal width/length.

# In[ ]:


def relationship_subplot(col1, col2, pos):
    plt.rcParams["figure.figsize"] = [14, 5]
    plt.subplot(1,2,pos)
    plt.scatter(df[df.columns[col1]], df[df.columns[col2]], c=df['target'])
    plt.xlabel(df.columns[col1])
    plt.ylabel(df.columns[col2])
    patches = []
    for number, colour in enumerate(['#390540', '#3C9586', '#F3E552']):
        patches.append(mpatches.Patch(color=colour, label=number))
    plt.legend(handles=patches) # Manually create a legend (do let me know if this can be automated).
    return stats.pearsonr(df[df.columns[col1]], df[df.columns[col2]])

petal = relationship_subplot(0,1,1)
sepal = relationship_subplot(2,3,2)

plt.show()

print("Petal Correlation (PCC, p-value): {}".format(petal))
print("Sepal Correlation (PCC, p-value): {}".format(sepal))


# From the graphs and Pearson correlation coefficients above, we can see that there is a strong, positive correlation between petal width and petal length. This is not the case with sepal width and length.
# 
# Even though strong correlation suggests that there is not much use of having both petal width and length, we will include both for classification as there may be *some* value.
# 
# ## Prepare the Features and Labels for Training and Testing
# 
# Our analysis thus far suggests that all features apart from sepal width will assist (with varying effectiveness) with classification. We ruled out sepal width was ruled out because much of the data between species overlapped, making it an ineffective feature for classification.
# 
# We can now prepare the training and testing data using our features.

# In[ ]:


df.drop(df.columns[1], axis=1, inplace=True) # Remove sepal width.

X = df.drop(['target'], 1) # Features
y = df['target'] # Labels

X_train, X_test,y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

print("There are {} rows of training data and {} rows of testing data.".format(len(X_train),
                                                                               len(X_test)))


# ## Train and Test the Classifier

# In[ ]:


classifier = VotingClassifier([('lsvc', svm.LinearSVC()),
                               ('knn', neighbors.KNeighborsClassifier()),
                               ('rfor', RandomForestClassifier())])

classifier.fit(X_train, y_train)
confidence = classifier.score(X_test, y_test)

print("Accuracy: {}".format(confidence))


# Success!
