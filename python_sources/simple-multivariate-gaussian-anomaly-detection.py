#!/usr/bin/env python
# coding: utf-8

# We create a ML pipeline for anomaly detection. The anomaly detection approach is suited best in this case as on the one hand the data is highly skewed. On the other hand an ordinary training method might detect fraudulent behaviour similar to the one that it has been trained on but fail to detect new anomalies.
# 
# The first step is to read in and get a feeling for the data.

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')
data.head()


# + 28 numerical features without further information -> check individual distribution, we expect them to be Gaussian
# + one sorted feature "Time" 
# + one positive numerical feature "Amount" (most likely not Gaussian distributed) 

# In[ ]:


data.describe()


# + The PCA features seem to be ordered by standard deviation
# + Values are centered around zero and scaled within one order of magnitude

# We check the distribution of the normal data.

# In[ ]:


normal_data = data.loc[data["Class"] == 0]
fraud_data = data.loc[data["Class"] == 1]


# In[ ]:


plt.figure();


# In[ ]:


matplotlib.style.use('ggplot')
pca_columns = list(data)[1:-2]
normal_data[pca_columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2));


# + For first approximation features can be taken to be Gaussian distributed
# + Note that some features have outliers in the normal data (we might want to engineer features that look at these separately later if they are)
# + feature V1 seems to behave a little differently and has the most deviance from normal distribution on first sight
# 
# Analysis of the amount of withdrawal
# ===

# In[ ]:


normal_data["Amount"].loc[normal_data["Amount"] < 500].hist(bins=100);


# + The vast majority of withdrawals are in the sub 100$ range
# + We expect the the fraudulent amounts to be higher 

# In[ ]:


print("Mean", normal_data["Amount"].mean(), fraud_data["Amount"].mean())
print("Median", normal_data["Amount"].median(), fraud_data["Amount"].median())


# In[ ]:


fraud_data["Amount"].hist(bins=100);


# + Interestingly the median is lower but the mean is higher for the fraudulent cases.  This suggests there are some high value oriented criminals and some that focus on withdrawals "below the radar" to avoid detection
# + This makes the amount of withdrawal another feature
# 
# Analysis of time withdrawal
# ====
# + First we see if there are clusters of time with many normal/fraudulent withdrawals

# In[ ]:


normal_data["Time"].hist(bins=100);


# The two days the data is collected in are clearly visible in the normal transactions, being more or less constant during the day and falling sharply during the night.
# 
# Are the criminals more active during the day or creatures of the night?

# In[ ]:


fraud_data["Time"].hist(bins=50);


# + There seem to be some accumulation of frauds at the beginning of the night
# + In general the day night activities are way less pronounced in the fraudulent cases

# Finally we check the correlations between the features to decide what anomaly detection model we start out with

# In[ ]:


normal_pca_data = normal_data[pca_columns]
fraud_pca_data = fraud_data[pca_columns]
plt.matshow(normal_pca_data.corr());


# + There are correlations (albeit rather small) in the first 18 principal components
# + As we have >200000 data points calculating the co-variance matrix will run into memory problems
# + We start with uncorrelated Gaussians

# Multivariate Gaussian analysis of the normal data
# ===
# + First focus on the PCA features and see where it takes us
# + Numpy built-in function for the covariance matrix cant handle 200000+ -> implement our own
# + We start to train the model now, so split the data into training, cross validation and test set. The fraudulent cases are split between validation and test set.

# In[ ]:


num_test = 75000
shuffled_data = normal_pca_data.sample(frac=1)[:-num_test].values

X_train = shuffled_data[:-2*num_test]

X_valid = np.concatenate([shuffled_data[-2*num_test:-num_test], fraud_pca_data[:246]])
y_valid = np.concatenate([np.zeros(num_test), np.ones(246)])

X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[246:]])
y_test = np.concatenate([np.zeros(num_test), np.ones(246)])


# In[ ]:


def covariance_matrix(X):
    m, n = X.shape 
    tmp_mat = np.zeros((n, n))
    mu = X.mean(axis=0)
    for i in range(m):
        tmp_mat += np.outer(X[i] - mu, X[i] - mu)
    return tmp_mat / m


# In[ ]:


cov_mat = covariance_matrix(X_train)


# In[ ]:


cov_mat_inv = np.linalg.pinv(cov_mat)
cov_mat_det = np.linalg.det(cov_mat)
def multi_gauss(x):
    n = len(cov_mat)
    return (np.exp(-0.5 * np.dot(x, np.dot(cov_mat_inv, x.T))) 
            / (2. * np.pi)**(n/2.) 
            / np.sqrt(cov_mat_det))


# In[ ]:


from sklearn.metrics import confusion_matrix

def stats(X_test, y_test, eps):
    predictions = np.array([multi_gauss(x) <= eps for x in X_test], dtype=bool)
    y_test = np.array(y_test, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    recall = tp / (tp + fn)
    prec = tp / (tp + fp)
    F1 = 2 * recall * prec / (recall + prec)
    return recall, prec, F1


# To get an estimate for the threshold eps we take the maximum of the fraudulent cases. 
# 
# We should expect a recall of 100%

# In[ ]:


eps = max([multi_gauss(x) for x in fraud_pca_data.values])
print(eps)


# In[ ]:


recall, prec, F1 = stats(X_valid, y_valid, eps)
print("For a boundary of:", eps)
print("Recall:", recall)
print("Precision:", prec)
print("F1-score:", F1)


# To compare different thresholds we score them with the F1 score against the cross validation data set

# In[ ]:


validation = []
for thresh in np.array([1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]) * eps:
    recall, prec, F1 = stats(X_valid, y_valid, thresh)
    validation.append([thresh, recall, prec, F1])


# In[ ]:


x = np.array(validation)[:, 0]
y1 = np.array(validation)[:, 1]
y2 = np.array(validation)[:, 2]
y3 = np.array(validation)[:, 3]
plt.plot(x, y1)
plt.title("Recall")
plt.xscale('log')
plt.show()
plt.plot(x, y2)
plt.title("Precision")
plt.xscale('log')
plt.show()
plt.plot(x, y3)
plt.title("F1 score")
plt.xscale('log')
plt.show()


# Conclusion
# ---
# + High recall means (extremely) low precision
#     + this might be ok if on spot security measures are cheaply implemented. For example an extra verification online for cases that seem suspicious.
#     + problematic if the flagged cases have to be reviewed by hand. 
#     
#     
# + Need to include Time and Amount into data
#     + The Amount especially is important as most likely steal a high amount of money should be penalized stronger 
#     
#     
# + Need to analyze data further to check if non-fraudulent outliers play an important role
#     + Check if non-fraudulent outliers cluster in different manner -> engineer feature
#     
# The two last cases are problematic within the Multivariate Gaussian approach as the provided data are not normal distributed.

# Follow up
# ==
# Looking at the first couple of principal component dimensions we see that the data is rather awkardly scattered. In particular the positive cases are not real outliers but seem to cluster in certain areas. This makes a supervised classifier algorithm usable.

# In[ ]:


data.plot.scatter("V1","V2", c="Class")
data.plot.scatter("V2","V3", c="Class")
data.plot.scatter("V1","V3", c="Class")


# A multivariate gaussian draws ellipses around the negative data points. From the above pictures it is evident that any ellipse with a large recall also must have low precision. In particular as the ellipses are not learned per se.

# 