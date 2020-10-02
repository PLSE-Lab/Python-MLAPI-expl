#!/usr/bin/env python
# coding: utf-8

# Python Data Visualizations - Credit Card Data  

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


# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the Credit Card dataset, which is in the "../input/" directory
creditcard = pd.read_csv("../input/creditcard.csv") # the creditcard dataset is now a Pandas DataFrame

# Let's see what's in the creditcard data - Jupyter notebooks print the result of the last thing you do
creditcard.head()

# Press shift+enter to execute this cell


# In[ ]:


# Let's see how many examples we have of each case
creditcard["Class"].value_counts()


# In[ ]:


import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[ ]:


columns=creditcard.columns
# The labels are in the last column ('Class'). Simply remove it to obtain features columns

features_columns=columns.delete(len(columns)-1)

features = creditcard[features_columns]

labels=creditcard['Class']


# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, 
                                                                            labels, 
                                                                            test_size=0.2, 
                                                                            random_state=0)


# In[ ]:


oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)


# In[ ]:


# verify new data set is balanced
len(os_labels[os_labels==1])


# In[ ]:


clf=RandomForestClassifier(random_state=0)
clf.fit(os_features,os_labels)


# In[ ]:


# perform predictions on test set
actual=labels_test
predictions=clf.predict(features_test)


# In[ ]:


confusion_matrix(actual,predictions)


# In[ ]:


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)


# In[ ]:


import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:


# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Credit Card features V1 and V2.
creditcard.plot(kind="scatter", x="V1", y="V2")


# In[ ]:


# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="V1", y="V2", data=creditcard, size=5)


# In[ ]:


# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(creditcard, hue="Class", size=5)    .map(plt.scatter, "V1", "V2")    .add_legend()


# In[ ]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Class", y="V1", data=creditcard)


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="V1", data=creditcard)
ax = sns.stripplot(x="Class", y="V1", data=creditcard, jitter=True, edgecolor="gray")


# In[ ]:


# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
sns.violinplot(x="Class", y="V1", data=creditcard, size=6)


# In[ ]:


# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(creditcard, hue="Class", size=6)    .map(sns.kdeplot, "V1")    .add_legend()


# In[ ]:


# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by Class
creditcard.boxplot(by="Class", figsize=(12, 6))


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="Amount", data=creditcard)
ax = sns.stripplot(x="Class", y="Amount", data=creditcard, jitter=True, edgecolor="gray")


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="Time", data=creditcard)
ax = sns.stripplot(x="Class", y="Time", data=creditcard, jitter=True, edgecolor="gray")


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="V2", data=creditcard)
ax = sns.stripplot(x="Class", y="V2", data=creditcard, jitter=True, edgecolor="gray")


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="V3", data=creditcard)
ax = sns.stripplot(x="Class", y="V3", data=creditcard, jitter=True, edgecolor="gray")


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="V4", data=creditcard)
ax = sns.stripplot(x="Class", y="V4", data=creditcard, jitter=True, edgecolor="gray")


# In[ ]:


# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
ax = sns.boxplot(x="Class", y="V5", data=creditcard)
ax = sns.stripplot(x="Class", y="V5", data=creditcard, jitter=True, edgecolor="gray")

