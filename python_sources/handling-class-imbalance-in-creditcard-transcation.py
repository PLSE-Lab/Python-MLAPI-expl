#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Credit card fraud detection
# 
# We will in this notebook will work on credit card fraud detection. One of the major issues in the cases like this is class imbalance problem. Out of all transction there will be around 1 or 2 percent of transaction which is fraudulant and remaining 98% as non-fraudulant. This creates a major issues in classification models. so will ssee various ways to handle such imbalance

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load data
dataset = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
dataset.head()


# In[ ]:


dataset.shape


# In[ ]:


# check for infos
dataset.info()


# In[ ]:


# check for missing values 
dataset.isna().sum()


# There is no missing values on our dataset and all the features are numeric

# In[ ]:


# we drop time feature from our dataset
data = dataset.copy()
data = data.drop('Time', axis = 1)
data.head()


# ### Lets observe how our data is distributed.

# In[ ]:


data.hist(figsize = (20,20))
plt.show()


# Our data are **skewed**

# ### Feature scaling
# 
# we can clearly see in our dataset the feature 'Amount is quite different from remaining feature. i mean its range is quite different (big one) than remaining. This can be issue for out classification algorithms so we need to scale it.  we can either scale whole dataset or only this "Amount" one since other features are in close ranges to each others.

# In[ ]:


amount = pd.DataFrame(data['Amount'])
amount.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data['Amount'] = sc.fit_transform(amount)
#data['Amount'] = data['scaleAmount']

data.head()


# ### Checking the target variable

# In[ ]:


data.Class.value_counts()


# In[ ]:


sns.countplot(data.Class)


# **Looking above plot it clearly indicates the class imbalance **
# 
# It is always recommended to check for class imbalance in Classification problems.
# 
# Class imbalance is very common in Machine Learning especially in classification.Standard accuracy measure for example accuracy_score is no longer reliable performance evaluation metrics instead we need to opt for other metrics.
# 
# **Some techniques to deal with class imbalance problems in classification are:**
# 1. Collecting more data
# 2. Considering alternative performance metrics.
# 
#     Like i said aarlier standard performance metrics like accuracy_score is no            longer reilable because they tends to classify the majority labels. Insted we 
#      can go with metrics like f1-scpre, recall, precision, ROC curve etc.
# 3.Resampling the dataset
#   Two common sampling methods are: under-sampling and oversampling.
# 4. Use tree based algorithms: (definately worth trying)
# 5. SMOTE algorithms
#    many more.....
#    
#   One thing to note is that there is no magic tricks to know which techniques like mentioned above is best one.We can say they all are good and they all are bad. They all have their own benefits and drawbacks. We need to test all or as much as possible above techniques in our dataset and classification algorithm and chosse the best one.

# ### LOgistic regression in imbalaced data
# 
# We will try to apply logistic regression in imbalanced data and see how it performs.

# In[ ]:


# Creating feature and target vectors
X = data.drop('Class', axis = 1)
Y = data.Class

X.shape, Y.shape


# In[ ]:


# spliting data into train set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)


y_pred_train = logreg.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print("Accuracy score: ", accuracy_score(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))


# With training and test set accuracy almost **100**% but recall score of label/class **1** is **0.62** . So this model is not satisfying.

# In[ ]:





# ### Over-Sampling technique

# Over-sampling is the process of randomly duplicating observations from the minority class making it proportiate to majority class.
# 
# Approach: we will first separate data into majority class and minority class and than we will oversample the minority class observation with replacement. and then we will combine majority class observation and oversampled minority class observation. Thats it.

# In[ ]:


from sklearn.utils import resample

# separate the maority and minority class observation
data_major = data[data['Class'] == 0]
data_minor = data[data['Class'] == 1]

# over-sample the minority class observations
data_minor_oversample = resample(data_minor, replace = True, n_samples=284315, random_state = 0)

# finally combine the majority class observation and oversampled minoiry class observation
data_oversampled = pd.concat([data_major, data_minor_oversample])


# In[ ]:


# class label count after oversampled.we will see that minoity class now is proportionate to majority class
data_oversampled['Class'].value_counts()


# In[ ]:


# again lets splt our over samoled data into feature and traget variables
X = data_oversampled.drop('Class', axis = 1)
Y = data_oversampled.Class

# lets split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

# model building
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# Lets evaluate our model
y_pred_train = logreg.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print("Accuracy score: ", accuracy_score(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))


# Accuracy is found to be **0.95** and recall of minority is **0.92** which is quite good. So our model is performing quite well.Also test set abd training and testing accuracies are very comparable so no much issues of overfitting and underfitting.

# ### Under-Sampling Techniques

# Under-sampling involves randomly removing observations from the majority class to prevent its signal from dominating the learning algorithm.
# In simple word we will randomly remove data from majority class obdervation and make it equally proportionate or actionable proportional with minority class observation.

# In[ ]:


from sklearn.utils import resample

# separate majority and minority class observation
data_major = data[data['Class'] == 0]
data_minor = data[data['Class'] == 1]

# perform undersampling in majority class data
data_major_undersample = resample(data_major, replace = False, n_samples=492, random_state = 0)

# finally concat the minority class data and undersampled majority class data
data_undersampled = pd.concat([data_minor, data_major_undersample])


# In[ ]:


# class lbel count after undersampling
data_undersampled.Class.value_counts()


# In[ ]:


# lets create feature and target variables from above undersampled data
X = data_undersampled.drop('Class', axis = 1)
Y = data_undersampled.Class

# lets split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)

# model building
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

# Lets evaluate our model
y_pred_train = logreg.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

print("Accuracy score: ", accuracy_score(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))


# Accuracy comes out to be **0.94** and recall for minority class **1** is **0.90**.
# Even though our model is performing well for undersampled data but we have loss quite amount data from majority class. So above model may not be ideal model.

# ### SMOTE Technique
# 
# 

# **SMOTE** (synthetic minority oversampling technique) is one of the most commonly used **oversampling** methods to solve the imbalance problem.
# It aims to balance class distribution by randomly increasing minority class examples by replicating them.
# **SMOTE** and **Oversampling** are closely related.

# In[ ]:


print('Initially the class distribution counts as below')
data.Class.value_counts()


# Here 0 denotes non-fraudulant transaction and 1 denotes fraudulant dataset.It's the clear indication of class imbalace.

# In[ ]:


# creating feature and target vectors
x = data.drop('Class', axis = 1)
y = data.Class

x.shape, y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
x_train.shape, x_test.shape


# In[ ]:


print('Label counts in splited y_train before applying SMOTE algorithm:\n')
print("counts of label '1': {}".format(sum(y_train == 1))) 
print("counts of label '0': {} \n".format(sum(y_train == 0))) 


# In[ ]:


from imblearn.over_sampling import SMOTE 
smote = SMOTE(random_state = 0) 
x_trainN, y_trainN = smote.fit_sample(x_train, y_train.ravel()) 

print('Lets see the sample size again')
print('\nx_train size: ', x_trainN.shape)
print(' y_train size: ', y_trainN.shape)


# In[ ]:


print('Label counts in splited y_train AFTER applying SMOTE algorithm:\n')
print("counts of label '1': {}".format(sum(y_trainN == 1))) 
print("counts of label '0': {} \n".format(sum(y_trainN == 0))) 


# In[ ]:


# model building
logreg = LogisticRegression() 
logreg.fit(x_trainN, y_trainN.ravel()) 
y_pred = logreg.predict(x_test) 


# Lets evaluate our model
y_pred_train = logreg.predict(x_trainN)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_trainN, y_pred_train)))

print("Accuracy score: ", accuracy_score(y_test,y_pred))
print('\n')
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))  


# here accuracy comes out to be **0.98** and recall forminority class 1 is **0.92**
# This model is performing quite well.
# There is slight case of underfitting which can be dealt by tuning hyperparameter. Since we are using logistic regression model, our hyperparameter is **'c'**

# In[ ]:





# ## More update as well as improvement of above notebook will be done soon!!
# 
# WE have till now only worked with logistic regression  and applied it to various class imbalanced handling techniques.Soon more **tree based algorithms, svm,**  will be added as well. And we will observe and compare performance.
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




