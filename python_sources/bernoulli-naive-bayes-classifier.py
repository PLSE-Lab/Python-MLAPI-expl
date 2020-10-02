#!/usr/bin/env python
# coding: utf-8

# # Load accuracy score, confusion matrix, seaborn and naive bayes library for sklearn

# In[ ]:


from sklearn import naive_bayes
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


# # Load cancer data from the input folder

# In[ ]:


CancerData = pd.read_csv("../input/wdbc-data/wdbc.data").values


# # Plot some of the features using seaborn

# We can see that the features follow a Gaussian distribution

# In[ ]:


fig, axes = plt.subplots(4,3)
fig.set_size_inches(18.5, 6.5)
fig.tight_layout()
sns.distplot(CancerData[:,2],ax=axes[0][0]).set_title("Feature 1")
sns.distplot(CancerData[:,3],ax=axes[0][1]).set_title("Feature 2")
sns.distplot(CancerData[:,4],ax=axes[0][2]).set_title("Feature 3")
sns.distplot(CancerData[:,5],ax=axes[1][0]).set_title("Feature 4")
sns.distplot(CancerData[:,6],ax=axes[1][1]).set_title("Feature 5")
sns.distplot(CancerData[:,7],ax=axes[1][2]).set_title("Feature 6")
sns.distplot(CancerData[:,8],ax=axes[2][0]).set_title("Feature 7")
sns.distplot(CancerData[:,9],ax=axes[2][1]).set_title("Feature 8")
sns.distplot(CancerData[:,10],ax=axes[2][2]).set_title("Feature 9")
sns.distplot(CancerData[:,11],ax=axes[3][0]).set_title("Feature 10")
sns.distplot(CancerData[:,12],ax=axes[3][1]).set_title("Feature 11")
axes[3][2].hist(CancerData[:,1])
axes[3][2].set_title("Diagnosis")


# # Since we use Bernoulli Naive Bayes the input features should be binary in nature. We use Binarizer to convert the above Gaussian distribution to Bernoulli distribution. We can also use the binarize parameter in BernoulliNB class instead of this. 
# # Here the threshold to binarize the feature is the mean value of that particular feature

# In[ ]:


CancerDataBinary = np.zeros((568,30))
for i in range(2,32,1):
    binarizer = preprocessing.Binarizer(threshold = statistics.mean(CancerData[:,i])).fit([CancerData[:,i]])  # fit does nothing
    CancerDataBinary[:,i-2] = binarizer.transform([CancerData[:,i]])


# # Plot the above features - Bernoulli distribution

# In[ ]:


fig, axes = plt.subplots(4,3)
fig.set_size_inches(18.5, 6.5)
fig.tight_layout()
sns.distplot(CancerDataBinary[:,0],ax=axes[0][0]).set_title("Feature 1")
sns.distplot(CancerDataBinary[:,1],ax=axes[0][1]).set_title("Feature 2")
sns.distplot(CancerDataBinary[:,2],ax=axes[0][2]).set_title("Feature 3")
sns.distplot(CancerDataBinary[:,3],ax=axes[1][0]).set_title("Feature 4")
sns.distplot(CancerDataBinary[:,4],ax=axes[1][1]).set_title("Feature 5")
sns.distplot(CancerDataBinary[:,5],ax=axes[1][2]).set_title("Feature 6")
sns.distplot(CancerDataBinary[:,6],ax=axes[2][0]).set_title("Feature 7")
sns.distplot(CancerDataBinary[:,7],ax=axes[2][1]).set_title("Feature 8")
sns.distplot(CancerDataBinary[:,8],ax=axes[2][2]).set_title("Feature 9")
sns.distplot(CancerDataBinary[:,9],ax=axes[3][0]).set_title("Feature 10")
sns.distplot(CancerDataBinary[:,10],ax=axes[3][1]).set_title("Feature 11")
axes[3][2].hist(CancerData[:,1])
axes[3][2].set_title("Diagnosis")


# # Split the data to train and test data

# In[ ]:


X = CancerDataBinary
y = CancerData[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# # Create a Bernoulli Naive Bayes classifier model and fit the data

# In[ ]:


BernoulliNBModel = naive_bayes.BernoulliNB()
BernoulliNBModel.fit(X_train, y_train)


# # Predict the test data

# In[ ]:


y_test_predicted = BernoulliNBModel.predict(X_test)


# # Print confusion matrix and accuracy

# In[ ]:


print(confusion_matrix(y_test, y_test_predicted))
accuracy_score(y_test, y_test_predicted)

