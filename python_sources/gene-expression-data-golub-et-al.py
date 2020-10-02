#!/usr/bin/env python
# coding: utf-8

# Gene expression analysis is a commonly used technique in molecular biology. In this analysis we are looking at the expression levels of multiple genes as determined by a DNA microarray and their roles in predicting cancer types. All of the subjects in the study had one of two types of cancer, ALL (acute lymphoblastic leukemia) or AML (acute myeloid leukemia). The data was used in 1999 by Golub et al. as a proof of concept study.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import itertools
import sklearn
import numpy as np


# ## Data Cleaning

# In[ ]:


train_df = pd.read_csv("../input/data_set_ALL_AML_train.csv")
test_df = pd.read_csv("../input/data_set_ALL_AML_independent.csv")
validation_df = pd.read_csv("../input/actual.csv")
train_df.head()


# The columns labeled call seem to represent probes used during DNA microarray analysis. These probes don't seem to provide any direct value in analysis and were dropped.

# In[ ]:


# removing all call columns from data frame
train_columns = [col for col in train_df if "call" not in col]
test_columns = [col for col in test_df if "call" not in col]
train_adjusted = train_df[train_columns]
test_adjusted = test_df[test_columns]
train_adjusted.head()


# In observing the data frames we can see that the genes accession numbers are listed as rows while the expression levels of each patient are listed as columns. This is not an ideal form for analysis as rows typically represent samples so the data frame was transposed.

# In[ ]:


#transposing data frames
transposed_train = train_adjusted.T
transposed_test = test_adjusted.T
transposed_train.head()


# In[ ]:


predictors = pd.concat([transposed_train, transposed_test], axis = 0)
predictors = predictors.drop(['Gene Description', 'Gene Accession Number'])
predictors.columns = transposed_train.iloc[0]


# ## Exploratory Data Analysis

# Below we can see the total proportion of AML to ALL cancer types, we can see  that the data frame has a larger proportion of patients with ALL than AML. We can also see that only slightly over 20 patients have AML so any findings gathered from EDA on the AML set are far from conclusive and should only be used as a guide for possible further investigation.

# In[ ]:


#resetting indices of both predictor and validation data frames so they can be combined
vd = validation_df.reset_index(drop = True)
pr = predictors.reset_index(drop = True)
#combining validation and predictor data frames
combined = pd.concat([pr, vd], axis = 1)
#finding most expressed genes in combined data dataframe
outcomes = combined.groupby('cancer').size()
outcomes.plot(kind = 'bar')


# Below we can see the values of the 10 most highly expressed genes in the study, as the data has been scaled these points represent relative values of the expression of these genes. The most expressed genes in the two samples differ and this may indicate the roles of these genes in causing the different cancer types. Genes encoding the RPL37a Ribosomal protein A are highly expressed in both. If compared to a control population this may possibly be used as a general indicator of either of these leukemia types. However genes encoding proteins such as Globin Beta are only more highly expressed in one of the cancer types and more reasearch into their function may lead to insight with regards to the mechanisms of the different leukemias and what disinguishes them from one another.

# In[ ]:


highest = combined.mean().abs().sort_values(ascending = False)
plt.figure(figsize=(10, 8))
highest.head(10).plot(kind = 'bar')
plt.title('10 Genes Highest Expression Levels Both Cancer Types')
plt.ylabel('Expression Levels (au)')


# In[ ]:


c_ALL = combined[combined.cancer == 'ALL']
highest_ALL = c_ALL.mean().abs().sort_values(ascending = False)
plt.figure(figsize=(10, 8))
highest_ALL.head(10).plot(kind = 'bar')
plt.title('10 Genes Highest Expression Levels ALL')
plt.ylabel('Expression Levels (au)')


# In[ ]:


c_AML = combined[combined.cancer == 'AML']
highest_AML = c_AML.mean().abs().sort_values(ascending = False)
plt.figure(figsize=(10, 8))
highest_AML.head(10).plot(kind = 'bar')
plt.title('10 Genes Highest Expression Levels AML')
plt.ylabel('Expression Levels (au)')


# ## Model Creation / Data Preparation

# In[ ]:


train_no_acc = transposed_train.drop(["Gene Accession Number","Gene Description"]).apply(pd.to_numeric)
test_no_acc = transposed_test.drop(["Gene Accession Number", "Gene Description"]).apply(pd.to_numeric)
predictors_no_acc = predictors.drop(['Gene Accession Number'], axis = 1).apply(pd.to_numeric)


# In[ ]:


#resetting indices for test and train data frames
train_no_acc = train_no_acc.reset_index(drop = True)
test_no_acc = test_no_acc.reset_index(drop = True)


# In[ ]:


#creating data frames for both test and train data validation
validation_train = validation_df[validation_df.patient <= 38].reset_index(drop = True)
validation_test = validation_df[validation_df.patient > 38].reset_index(drop = True)
validation_test.head()


# In[ ]:


# combining predictor and validation set data
train = pd.concat([validation_train, train_no_acc], axis = 1)
test = pd.concat([validation_test, test_no_acc], axis = 1)


# In[ ]:


#creating sample data frames from original for model creation
train_sample = train.iloc[:,2:].sample(n=200, axis=1)
test_sample = test.iloc[:,2:].sample(n=200, axis=1)
test_sample.head()


# In order to test whether the data should be scaled before further model creation it's distribution was analyzed using a histogram as well as a Kernel Density Estimation. The results of this analysis showed that although a large portion of the data was indeed centered at zero it was still right skewed. In order to ramify this a scaled version of the data was used for analysis. 

# In[ ]:


train_sample.plot(kind="hist", legend=None, bins=20, color='k')
train_sample.plot(kind="kde", legend=None);


# In[ ]:


from sklearn import preprocessing
scaled = pd.DataFrame(preprocessing.scale(train_sample))
scaled.plot(kind="hist", legend=None, bins=20, color='k')
scaled.plot(kind="kde", legend=None);


# ### PCA

# In order to test the potential value of PCA as a potential dimensionality reduction method before analysis it was performed on a sample and the cummulative variance was compared to different numbers of principle components. What is observed is that within 30 principle components over 90% of the data's variance is captured. This indicates that PCA will be effective in reducing the effects of the data's large feature number and creating a more accurate model.

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sample_scaled = StandardScaler().fit_transform(train_sample)
pca = PCA(n_components = 30)
pca.fit(sample_scaled)

cum_sum = pca.explained_variance_ratio_.cumsum()
cum_sum = cum_sum*100

fix, ax = plt.subplots(figsize = (8, 8))
plt.bar(range(30), cum_sum, color = 'r',alpha=0.5)
plt.title('PCA Analysis')
plt.ylabel('cumulative explained variance')
plt.xlabel('number of components')
plt.locator_params(axis='y', nbins=20)


# In[ ]:


#training and test samples are created and scaled for model creation
X_train = StandardScaler().fit_transform(train_no_acc)
X_test = StandardScaler().fit_transform(test_no_acc)
y_train = validation_train['cancer']
y_test = validation_test['cancer']


# ### Pairing PCA with Logistic Regression

# Below I've paired PCA with logistic regression in order to create a regression model using a sample with reduced dimensionality. From the results we see that the ideal number of components is less than 10. Below the accuracy of the regression is plotted compared to the number of components.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def pipeline_PCA_GLM(components):
    accuracy_chart = []
    for i in components:
        steps = [('pca', PCA(n_components = i)),
        ('estimator', LogisticRegression())]
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        accuracy_chart.append(accuracy_score(y_test,predictions))
    return accuracy_chart


# In[ ]:


n_components = range(1,30)
accuracy_chart = pipeline_PCA_GLM(n_components)


# In[ ]:


plt.figure(figsize=(10, 8))
plt.bar(n_components, accuracy_chart)
plt.ylim(0,1)
plt.xlim(0,30)
plt.locator_params(axis='y', nbins=20)
plt.locator_params(axis = 'x', nbins = 30)
plt.ylabel("Accuracy")
plt.xlabel("Number of Components")


# ### K-Nearest Neighbors

# In order to see if the data was organized in separate gaussian distributions the data was clustered using K-Nearest Neighbors. The results of this clustering was used to create a model that was fit to the test set. We can see that the ideal number of neighbors is less than 10 and seems to lay at roughly 2.

# In[ ]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
def knn_pred(train_predictors, train_outcome, k_range, test_predictors):
    #train_predictors and train_outcome should both be from training split while test_predictors should be from test split
    y_pred = []
    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(train_predictors, train_outcome)
        y_pred.append(knn.predict(test_predictors))
    return y_pred



# In[ ]:


#function compares KNN accuracy at different levels of K
def knn_accuracy(pred, k_range, test_outcome):
    #pred represents predicted values while test_outcome represents the values from the test set
    accuracy_chart = []
    for i in range(len(k_range)):
        accuracy_chart.append((sklearn.metrics.accuracy_score(test_outcome, pred[i])))
    return accuracy_chart
        


# In[ ]:


train_range = range(2, 20, 2)
sample_pred = knn_pred(X_train, y_train, train_range, X_test)
accuracy = knn_accuracy(sample_pred, train_range, y_test)
plt.figure(figsize=(10, 8))
plt.bar(train_range, accuracy)
plt.ylim(0,1)
plt.xlim(0,20)
plt.locator_params(axis='y', nbins=20)
plt.locator_params(axis = 'x', nbins = 10)
plt.ylabel("Accuracy")
plt.xlabel("Number of Neighborhoods")


# ### Conclusion

# The analysis of this DNA microarray data has shown some interesting results and seems relatively amenable to model creation. During PCA it was shown that the model achieves the highest accuracy when the number of components is two. If one were to research the roles and mechanisms of the main genes involved then it would be possible to find insight into the main factors leading to the two different cancer types as well as what proteins are generally linked to leukemia. The models created achieved accuracy levels within roughly 5% of those observed in the study from which the data was derived. Although modelling wasn't the main goal of this analysis these results indicate that the algorithms used could lead to a high degree of accuracy if fine tuned.
