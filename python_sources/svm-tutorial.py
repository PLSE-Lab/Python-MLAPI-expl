#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns; sns.set()
from sklearn import svm


# Load the data

# In[ ]:


file_name = '../input/SAheart.data'
data = pd.read_csv(file_name, sep=',', index_col=0)
print(len(data))
data.head()


# Need to convert that categorical variable into binary indicators.

# In[ ]:


data['famhist'] = data.famhist.apply(lambda x: x == 'Present')


# First thing to do with a machine learning problem is visualize the data and get a feel for the distributions. Start from the preponderance of coronary heart disease (the response) in the data.

# In[ ]:


data.chd.value_counts()


# So it's an uneven dataset. On this basis we define our random baseline accuracy as majority class, and is therefore $302/462 = 0.6536$.

# I like a [joint plot](https://seaborn.pydata.org/generated/seaborn.jointplot.html) for comparing the distributions of features with a binary response.[](http://)

# In[ ]:


for feature in data.columns:
    if feature == 'chd':
        continue
    sns.jointplot(x=feature, y='chd', data=data, kind='kde')
    plt.title('Correlation between %s and Heart Disease' % feature)
    plt.show()


# Looks like age should be a nice predictor. Using all these features is probably going to be a bit messy. Let's try and simplify the problem and just pick two for now.

# In[ ]:


chd = data[data.chd == True]
nchd = data[data.chd == False]
y = plt.scatter(chd.age.values, chd.adiposity.values, c='r')
n = plt.scatter(nchd.age.values, nchd.adiposity.values, c='b')
plt.xlabel('age')
plt.ylabel('adiposity')
plt.legend((y, n), ['chd', 'healthy'])
plt.show()


# Create a train-test split.

# In[ ]:


n_test = int(math.ceil(len(data) * 0.3))
random.seed(42)
test_ixs = random.sample(list(range(len(data))), n_test)
train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]
train = data.iloc[train_ixs, :]
test = data.iloc[test_ixs, :]
print(len(train))
print(len(test))


# Train a model using default hyperparameters. Refer to [the documentation for the SVM classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) for details.

# First separate the features from the response.

# In[ ]:


#features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']
features = ['adiposity', 'age']
response = 'chd'
x_train = train[features]
y_train = train[response]
x_test = test[features]
y_test = test[response]


# After the test split, recalculate the random baseline.

# In[ ]:


1. - y_test.mean()


# Now we can try and fit our model.

# In[ ]:


model = svm.SVC(gamma='scale')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
np.mean(y_pred == y_test)


# That's OK. But, the default hyperparameter settings are almost never the best. The main hyperparameter we use to control bias-variance is `C`, so let's just manually play with some values to get a feel.

# In[ ]:


best_acc = 0.
best_c = None
for c in np.linspace(0.1, 1.0):
    model = svm.SVC(C=c, gamma='scale')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = np.mean(y_pred == y_test)
    if acc > best_acc:
        best_acc = acc
        best_c = c
print(best_acc)
print(best_c)


# ## Beat My Baseline
# 
# Consider:
# - more hyperparameters (listed [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))
# - more features, different combinations thereof

# In[ ]:


# practice here, using the code above as a template

