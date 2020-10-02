#!/usr/bin/env python
# coding: utf-8

# # Classificaiton of NIR spectra by Linear Discriminant Analysis in Python
# Learned from: https://nirpyresearch.com/classification-nir-spectra-linear-discriminant-analysis-python/
# Products or raw materials identification is one of the staples of NIR analysis in industrial processing. Identification of a product or substance - or detection of anomalies over the expected range - are usually accomplished by separating NIR spectra into different classes. In this post we'll work through exmaple of classification of NIR spectrea by LDA in Python.
# 
# ## What is LDA?
# PCA vs. LDA
# - PCA is unsupervised method, LDA is supervised method
# - PCA is to maximize the variances in the data, LDA has class labels and maximize the distance between those groups 
# This is a good picture showing the difference:
# <img src="https://nirpyresearch.com/wp-content/uploads/2018/11/PCAvsLDA-1024x467.png"></img>

# In[ ]:


import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
milk_powder = pd.read_csv('../input/milk-powder.csv')


# In[ ]:


milk_powder.head()


# In[ ]:


y = milk_powder.values[:, 1].astype('uint8')
X = milk_powder.values[:, 2:]

lda = LDA(n_components=2)
Xlda = lda.fit_transform(X, y)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# Define the labels for the plot legend
labplot = [f'Milk {i*10}% ' for i in range(11)]


# In[ ]:


# Scatter plot
unique = list(set(y))


# In[ ]:


unique


# In[ ]:


import numpy as np
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
plt.figure(figsize=(10, 10))
with plt.style.context('ggplot'):
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [Xlda[j, 0] for j in range(len(Xlda[:, 0])) if y[j] == u]
        yi = [Xlda[j, 1] for j in range(len(Xlda[:, 1])) if y[j] == u]
        plt.scatter(xi, yi, c = col, s=60, edgecolors='k', label=str(u))
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.legend(labplot, loc = 'lower right')
    plt.title('LDA')
    plt.show()


# ## LDA usage:
# The use of LDA is not much since we have known the labels. Therefore, it is used more in classification of a new instances.
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
# If you want it to be repeatable (for instance if you want to check the performance of the classifier on the same split by changing some other parameter)


# In[ ]:


lda = LDA()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(lda.score(X_test, y_test))


# # Cross validation scores
# Just to make sure that the accuracy is not due to chances, we will rule out this by using cross_val_score function.
# In this case we don't have to specify a test-train split. This is done automatically by specifying the number of "folds", that is the number of splits in our data. For instance, by specifying (cv = 4) we are dividing our data in 4 ports, train the classificer on three of them and use the last part for test. The entire procedure is then repeated by choosing a different test set.
# 
# In this way we can get an average of more than one run, which is guaranteed to give a better etimateion of the accuracy of our classifier.

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(LDA(), X, y, cv=4)


# In[ ]:


print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()*3))
print("Accuracy confidence interval (3*sigma) [%0.4f, %0.4f]" % (scores.mean()-scores.std()*3, min([scores.mean()+scores.std()*3, 1])))


# ## Dealing with colinearity
# Collinearity means tha tthe value of the spectra (the "samples" in machine learing lingo) at different wavelengths (the "features") are not indepenedent, but highly correlated. This is the reason for warning.
# 
# Therefore, PCA is one of the ways to remove collinearity problems. The individual principal components are always independent from one another, and by choosing a handful of PCA in our decomposition we are guaranteed that collinearity problem is eliminated.
# 
# ## Combining PCA and LDA
# PCA and LDA are two different beasts (they are not alternatives). So we can combine them.
# Step 1: Extract a handful of PCA components
# Step 2: Use them for classification

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=15)
Xpc = pca.fit_transform(X)
scores = cross_val_score(LDA(), Xpc, y, cv = 4)


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


print(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()*3))
print("Accuracy confidence interval (3*sigma) [%0.4f, %0.4f]" % (scores.mean()-scores.std()*3, min([scores.mean()+scores.std()*3, 1])))


# In[ ]:




