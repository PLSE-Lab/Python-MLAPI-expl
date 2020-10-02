#!/usr/bin/env python
# coding: utf-8

# # **Predicting Fraud - Anomaly Detection**
# 
# Well, I am guessing you guys have probably had a look at the other kernels before coming to this one.
# So here's a few things I am assuming you might already know:
# - Data set is the result of PCA (Principal Component Analysis) transformation of the original credit card transactions. And features V1 to V28 are the resultant features of that transformation.
# - This dataset is highly skewed with only 492 frauds in 284807 transactions i.e. only 0.172 % of the dataset is contaminated.
# - There are no missing values.
# 
# Lets get on with the necessities:
# 

# In[ ]:


#Loading necessary libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import multivariate_normal


# In[ ]:


#Loading the dataset
df = pd.read_csv("../input/creditcard.csv")


# In[ ]:


#Exploring the dataset
print("Dataset is of shape: {}".format(df.shape))
print("Fraud cases: {}".format(len(df[df.Class==1])))
print("Normal cases: {}".format(len(df[df.Class==0])))
print("Contamination: {}".format((float(len(df[df.Class==1]))/len(df))*100))
df.describe()


# ## **t-SNE Visualization**
# t-SNE (t-distributed stochastic neighbor embedding) is a visualization technique used to visualize high dimensional data. It starts by calculating the probability of similarity of points in high-dimensional space and probability of similarity of points in the corresponding low-dimensional space. It then uses a gradient descent method to minimize the error (sum of difference of proabilities of high and corresponding low dimension).
# 
# If you want to know more, then here's a pretty nice explanation of what t-SNE does: http://www.youtube.com/watch?v=NEaUSP4YerM
# 
# We can not infer much from t-SNE, it is just used here for better visualization of our dataset:

# In[ ]:


#Sampling the dataset for tsne algorithm
tsne_data_fraud = df[df.Class==1]
tsne_data_normal = df[df.Class==0].sample(frac=0.05, random_state=1)
print(tsne_data_fraud.shape)
print(tsne_data_normal.shape)


# t-SNE is very time consuming and it directly relates to the amount of the data given to it. Performing it on 284807 transactions is not practical given the gig i have, hence, I am only taking a fraction of it, which corresponds to around 15000 examples.

# In[ ]:


#Data engineering for tsne
tsne_data = tsne_data_fraud.append(tsne_data_normal, ignore_index=True)
tsne_data = shuffle(tsne_data)
label = tsne_data.iloc[:, -1]
tsne_data = tsne_data.iloc[:, :30]
tsne_data = tsne_data.astype(np.float64)

standard_scaler = StandardScaler()
tsne_data = standard_scaler.fit_transform(tsne_data)

print(tsne_data.shape)
print(label.shape) 


# In[ ]:


#Performing dimension reduction (tsne)
tsne = TSNE(n_components=2, random_state=0)
tsne_data = tsne.fit_transform(tsne_data)


# In[ ]:


#Making final changes to the resulted data from tsne
print(tsne_data.shape)
tsne_plot = np.vstack((tsne_data.T, label))
tsne_plot = tsne_plot.T
print(tsne_plot.shape)


# In[ ]:


#Plotting the tsne results
tsne_plot = pd.DataFrame(data=tsne_plot, columns=("V1", "V2", "Class"))
sb.FacetGrid(tsne_plot, size=6, hue="Class").map(plt.scatter, "V1", "V2").add_legend()


# You can see how most of the anomilies are clustered together in groups. So we need an efficient anomaly detection technique and train it to identify those points as outliers and hence brand them as anomalies.
# 
# ## **Multivariate Gaussian Anomaly Detection:**
# I will be using multivariate gaussian anomaly detection. For those who don't know what it is, kindly follow the "Machine Learning by Andrew Ng" course on coursera, you will find anomaly detection in  week 9 of the course (its a pretty nice course :).
# 
# Here's a summary of what is to be done in this anomaly detection algorithm:
# * Divide the dataset into 3 parts: training set(only normal cases), cross-validation set(normal + fraud) and test set(normal + fraud).
# * Fit the model p(x) by finding the mean and co-variance matrix using the training set.
# * Use cross-validation set to find an optimal epsilon.
# * Apply the model on test set. For any new data, its p(x) will be calculated and it will be marked as an anomaly if it is less than epsilon.
# 

# In[ ]:


#Visualizing each feature separately
df.hist(figsize=(20,20), bins=50, color="green", alpha=0.5)
plt.show()


# As the number of features is quite less, there is no specific need to select and drop the irrelevent features.
# And as for creating new features from the existing ones, Multivariate Gaussian itlself detects the correlation among features so no need for that too !

# In[ ]:


#Creating train, cross-validation and test set
df_fraud = shuffle(df[df.Class==1])
df_normal = shuffle(df[df.Class==0].sample(n=280000))
print(df_fraud.shape)
print(df_normal.shape)
df_train = df_normal.iloc[:240000, :].drop(labels = ["Class", "Time"], axis = 1)
df_cross = shuffle(df_normal.iloc[240000:260000, :].append(df_fraud.iloc[:246, :]))
Y_cross = df_cross.loc[:, "Class"]
df_cross = df_cross.drop(labels = ["Class", "Time"], axis = 1)
df_test = shuffle(df_normal.iloc[260000:, :].append(df_fraud.iloc[246:, :]))
Y_test = df_test.loc[:, "Class"]
df_test = df_test.drop(labels = ["Class", "Time"], axis = 1)
print(df_train.shape)
print(df_cross.shape)
print(Y_cross.shape)
print(df_test.shape)
print(Y_test.shape)


# In[ ]:


#Defining fuctions to calculate mean, cov and gaussian probablities
def mean_variance(data):
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    return mean, cov

def gaussian_dist(data, mean, cov):
    prob = multivariate_normal.pdf(data, mean=mean, cov=cov)
    return prob


# In[ ]:


#Fitting the model for train, cross and test set using mean and cov from train_set
mean, cov = mean_variance(df_train)
print(mean.shape)
print(cov.shape)
prob_train = gaussian_dist(df_train, mean, cov)
prob_cross = gaussian_dist(df_cross, mean, cov)
prob_test = gaussian_dist(df_test, mean, cov)

print(prob_train.shape)
print(prob_cross.shape)
print(prob_test.shape)


# Below is the optimization function for epsilon, normally you would start by setting *max_e = prob_train.max()*
# and then* keep decreasing max_e* or *keep increasing min_e* depending on the results untill you reach to an optimal balance between recall and precision or simply a better f1 score.
# But it isn't practical to show all the steps here so I am just skipping to the end:

# In[ ]:


#Using cross-validation set to find the optimum epsilon
def optimize_for_epsilon(prob_train, prob_cross, Y_cross):
    best_f1 = 0
    max_e = 2.062044871798754e-79
    min_e = prob_train.min()
    step = (max_e - min_e) / 1000
    
    for e in np.arange(prob_cross.min(), max_e, step):
        Y_cross_pred = prob_cross < e
        precision, recall, f1_score, support = prfs(Y_cross, Y_cross_pred, average="binary")
        print("for epsilon: {}".format(e))
        print("f1_score: {}".format(f1_score))
        print("recall: {}".format(recall))
        print("precision: {}".format(precision))
        print("support: {}".format(support))
        print()
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epsilon = e
            recall = recall
        
    return best_f1, best_epsilon, recall

best_f1, best_epsilon, recall = optimize_for_epsilon(prob_train, prob_cross, Y_cross)
print(best_f1, best_epsilon, recall)
    


# In[ ]:


#Predicting the anomalies on test_set using the optimal epsilon from above results
Y_test_pred = prob_test < best_epsilon
precision, recall, f1_score, ignore = prfs(Y_test, Y_test_pred, average="binary")
print("epsilon: {}".format(best_epsilon))
print("f1_score: {}".format(f1_score))
print("recall: {}".format(recall))
print("precision: {}".format(precision))
# print("support: {}".format(support))


# *Note: Values mentioned below might vary a bit with the above results as the model gives a bit different results in different epochs.*
# 
# We were able to achieve f1 score of 0.71 and were able to catch almost 80% of the fraud cases (0.7926 recall) with around 65% precision. We can increase f1 score even more but then recall will reduce as there will be a trade off between recall and precision and i personally think catching more of the frauds even if we get some false positives along with is more important rather than improving precision and missing lots of fraud cases.

# In[ ]:




