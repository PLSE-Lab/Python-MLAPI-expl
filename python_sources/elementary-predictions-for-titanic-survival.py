#!/usr/bin/env python
# coding: utf-8

# *The aim of this notebook is to compare an elementary prediction with one made by a machine learning algorithm, and to see how much the accuracy increases.*

# ##Data##

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load data
data = pd.read_csv("../input/train.csv")


# In[ ]:


# Take a look at the data
data.head()


# In[ ]:


data.info()


# ##Training and validation sets##

# Divide data into training and validation sets (80/20), and set the target.

# In[ ]:


traindata = data.sample(frac = 0.8)
valind = list(set(data.index) - set(traindata.index))
valdata = data.loc[valind, :]
traintarget = data.loc[traindata.index, 'Survived']
valtarget = data.loc[valind, 'Survived']


# In[ ]:


traintarget.value_counts()


# Examine features that might be relevant.

# In[ ]:


traindata['Sex'].value_counts()


# In[ ]:


traindata['Pclass'].value_counts()


# ##Finding relevant features##

# Let's crosstabulate to find dependencies:

# In[ ]:


sex = traindata['Sex']
pd.crosstab(traintarget, sex)


# It looks like women were more likely to survive than men. Let's try to quantify this by mutual information score.

# In[ ]:


from sklearn.metrics import mutual_info_score
mutual_info_score(traintarget, sex)


# Let's do the same with the 'Pclass' feature.

# In[ ]:


pclass = traindata['Pclass']
pd.crosstab(traintarget, pclass)


# MI works best when the number of classes are equal.

# In[ ]:


target12 = traindata[traindata['Pclass'].isin([1,2])]['Survived']
target13 = traindata[traindata['Pclass'].isin([1,3])]['Survived']
target23 = traindata[traindata['Pclass'].isin([2,3])]['Survived']
pclass12 = traindata[traindata['Pclass'].isin([1,2])]['Pclass']
pclass13 = traindata[traindata['Pclass'].isin([1,3])]['Pclass']
pclass23 = traindata[traindata['Pclass'].isin([2,3])]['Pclass']


# We then evaluate the MI scores...

# In[ ]:


mutual_info_score(target12, pclass12)


# In[ ]:


mutual_info_score(target13, pclass13)


# In[ ]:


mutual_info_score(target23, pclass23)


# ...and see that the class divide between 1 and 3 is more relevant than between 1 and 2, or 2 and 3.

# Finally, let's crosstabulate 'Sex' against 'Pclass':

# In[ ]:


pd.crosstab(sex, pclass)


# In[ ]:


sex12 = traindata[traindata['Pclass'].isin([1,2])]['Sex']
sex13 = traindata[traindata['Pclass'].isin([1,3])]['Sex']
sex23 = traindata[traindata['Pclass'].isin([2,3])]['Sex']


# In[ ]:


mutual_info_score(sex12, pclass12)


# In[ ]:


mutual_info_score(sex13, pclass13)


# In[ ]:


mutual_info_score(sex23, pclass23)


# These MI scores indicate that there's hardly any dependence between 'Sex' and 'Pclass'.

# **Conclusion: 
# Survival depends more heavily on 'Sex' than 'Pclass'.
# Women and first class passengers are more likely to survive. 
# Moreover, 'Sex' and 'Pclass' are not very strongly dependent.**

# ## Two elementary predictions ##

# First prediction: Women survive, others don't.

# In[ ]:


pred1 = 1 * valdata['Sex'].isin(['female'])


# For scoring our prediction we compute its accuracy and the confusion matrix.

# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


print("Accuracy: {0:.2f}".format(accuracy_score(valtarget, pred1)))


# Confusion matrix:

# In[ ]:


C = confusion_matrix(valtarget, pred1)
C


# In[ ]:


TN = C[0,0]
FN = C[1,0]
TP = C[1,1]
FP = C[0,1]


# In[ ]:


print("Precision: {0:.2f}".format(TP / (TP + FP)))


# In[ ]:


print("Recall: {0:.2f}".format(TP / (TP + FN)))


# The scores are quite good for such a simple predictor, but let's see what we missed:

# In[ ]:


ind = valdata[valtarget != pred1].index
df = valdata.loc[ind, ['Survived', 'Sex' ,'Pclass']]
df


# In[ ]:


pd.crosstab(df['Survived'], df['Pclass'])


# It looks like we've overlooked the relevance of 'Pclass'. 

# Second prediction: Women survive, except the ones in third class.

# Let's first resample our training and validation data.

# In[ ]:


traindata = data.sample(frac = 0.8)
valind = list(set(data.index) - set(traindata.index))
valdata = data.loc[valind, :]
traintarget = data.loc[traindata.index, 'Survived']
valtarget = data.loc[valind, 'Survived']


# In[ ]:


pred1 = 1 * valdata['Sex'].isin(['female'])
pred2 = 1 * valdata['Pclass'].isin([3])
pred2 = 1 - pred2
pred2 = pred1 * pred2


# Let's evaluate our new prediction:

# In[ ]:


print("Accuracy: {0:.2f}".format(accuracy_score(valtarget, pred2)))


# In[ ]:


C = confusion_matrix(valtarget, pred2)
C


# In[ ]:


TN = C[0,0]
FN = C[1,0]
TP = C[1,1]
FP = C[0,1]


# In[ ]:


print("Precision: {0:.2f}".format(TP / (TP + FP))); print("Recall: {0:.2f}".format(TP / (TP + FN)))


# With the second prediction we reached higher precision (less false positives).

# To improve the accuracy of our prediction, we'll include more features and use the K-Nearest Neighbors algorithm with a handcrafted distance function.

# ## K-Nearest Neighbors ##

# We need to include more features; let's take a look at 'Fare':

# In[ ]:


import matplotlib.pyplot as plt
data['Fare'].plot.hist(bins=40)
plt.show()


# It seems reasonable to encode 'Fare' into a discrete variable as follows:

# In[ ]:


data['FareD'] = np.nan
data['FareD'][data['Fare'] <= 5] = 0
data['FareD'][(data['Fare'] > 5) & (data['Fare'] <= 10)] = 1
data['FareD'][(data['Fare'] > 10) & (data['Fare'] <= 15)] = 2
data['FareD'][(data['Fare'] > 15) & (data['Fare'] <= 20)] = 3
data['FareD'][(data['Fare'] > 20) & (data['Fare'] <= 30)] = 4
data['FareD'][(data['Fare'] > 30) & (data['Fare'] <= 50)] = 5
data['FareD'][(data['Fare'] > 50) & (data['Fare'] <= 80)] = 6
data['FareD'][data['Fare'] > 80] = 7


# In[ ]:


data['FareD'].plot.hist(bins=7)
plt.show()


# The 'Age' feature is missing a lot of data. Let's see how far we can get without imputation, and discretize the feature as follows:

# In[ ]:


data['AgeD'] = -1
data['AgeD'][data['Age'] <= 5] = 0
data['AgeD'][(data['Age'] > 5) & (data['Age'] <= 10)] = 1
data['AgeD'][(data['Age'] > 10) & (data['Age'] <= 20)] = 2
data['AgeD'][(data['Age'] > 20) & (data['Age'] <= 30)] = 3
data['AgeD'][(data['Age'] > 30) & (data['Age'] <= 40)] = 4
data['AgeD'][(data['Age'] > 40) & (data['Age'] <= 50)] = 5
data['AgeD'][(data['Age'] > 50) & (data['Age'] <= 60)] = 6
data['AgeD'][(data['Age'] > 60) & (data['Age'] <= 70)] = 7
data['AgeD'][data['Age'] > 70] = 8


# The two missing 'Embarked' values can be imputed as in a few other Titanic notebooks here.

# In[ ]:


data['Embarked'][61] = 'C'
data['Embarked'][829] = 'C'


# A copy of data before label encoding.

# In[ ]:


datacopy = data.copy()


# In[ ]:


columns = ['Sex', 'Pclass', 'Embarked', 'FareD', 'AgeD']
target = data['Survived']
data = data[columns]


# In[ ]:


from sklearn.preprocessing import LabelEncoder
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])


# We define a custom made distance function to be used in K-Nearest Neighbors. The weights reflect our understanding of feature importances.
# 
#  - 'Sex': Most important feature (two values)
#  - 'Pclass': Second most important feature, squared Euclidean distance (difference between classes 1 and 3 is important)
#  - 'Embarked': Categorical feature, least important, no order
#  - 'FareD': Categorical feature, no order (might also be reasonable to use Euclidean distance)
#  - 'AgeD': Categorical feautre, no order (also reasonable with Euclidean distance)

# In[ ]:


def customdist(x,y):
    w = {'Sex': 5, 'Pclass': 1, 'Embarked': 1, 'FareD': 1, 'AgeD': 3}
    return w['Sex'] * (x[0] != y[0]) + w['Pclass'] * abs(x[1] - y[1])**2 + w['Embarked'] * (x[2] != y[2]) + w['FareD'] * (x[3] != y[3]) + w['AgeD'] * (x[4] != y[4])


# To see how a nearest neighbors classifier performs, let's sample some training and validation data.

# In[ ]:


traindata = data.sample(frac = 0.8)
valind = list(set(data.index) - set(traindata.index))
valdata = data.loc[valind, :]
traintarget = target.loc[traindata.index]
valtarget = target.loc[valind]


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors = 4, algorithm = 'ball_tree', metric = customdist)
kn.fit(traindata, traintarget)
pred = kn.predict(valdata)


# In[ ]:


print("Accuracy: {0:.2f}".format(accuracy_score(valtarget, pred)))


# In[ ]:


C = confusion_matrix(valtarget, pred)
C


# What did we miss?

# In[ ]:


ind = valdata[valtarget != pred].index
print(datacopy.loc[ind, ['Survived', 'Sex' ,'Pclass', 'Embarked', 'FareD', 'AgeD']])


# Looking at the list of incorrect predictions, it's not immediately clear how to adjust the parameters for better results.

# ##Cross-validation##

# To get a more reliable score, let's use cross-validation.

# In[ ]:


kn = KNeighborsClassifier(n_neighbors = 4, algorithm = 'ball_tree', metric = customdist)


# In[ ]:


from sklearn.model_selection import cross_val_score
cvscores = cross_val_score(kn, data, target, cv=5)
print("Accuracy: {0:.2f} (+/-) {1:.2f}".format(cvscores.mean(),cvscores.std() * 2))


# **Concluding remark: The accuracy didn't increase much (if at all) compared to the elementary predictions. It is clear that survival is not fully predictable so that there's always a component of pure chance. However, by clever feature engineering and careful imputation of missing data, it is likely that one could perform better.**
