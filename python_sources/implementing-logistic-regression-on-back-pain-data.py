#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings                        # to hide error messages(if any)
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/Dataset_spine.csv')

#Renaming the columns
df = df.rename(columns = {'Col1':'pelvic_incidence',
                          'Col2':'pelvic tilt',
                            'Col3':'lumbar_lordosis_angle',
                            'Col4':'sacral_slope',
                            'Col5':'pelvic_radius',
                            'Col6':'degree_spondylolisthesis',
                            'Col7':'pelvic_slope',
                            'Col8':'Direct_tilt',
                            'Col9':'thoracic_slope',
                            'Col10':'cervical_tilt',
                            'Col11':'sacrum_angle',
                            'Col12':'scoliosis_slope',
                            'Class_att':'label'})

#Removing the unnecessary colum('Unnamed: 13')
df = df.drop('Unnamed: 13', axis = 1)
df.head()


# Giving a numerical value to the label attribute:<br/>
#     **0** for **normal** <br/>
#     **1** for **abnormal**

# In[ ]:


def label_values(label):
    if label == 'Abnormal':
        return 1
    elif label == 'Normal':
        return 0


df['label_value'] = df['label'].apply(label_values)


# In[ ]:


df.head()


# A new column called label value is added

# Checking for null values in the data frame

# In[ ]:


df.isnull().sum()


# There are no empty/missing values in the data frame

# In[ ]:


df.shape


# In[ ]:


df.dtypes


# ## Data set summary

# In[ ]:


df.describe()


# The features described in the above data set are:<br/>
# 
# **1. count** tells us the number of NoN-empty rows in a feature.
# 
# **2. mean** tells us the mean value of that feature.
# 
# **3. std** tells us the Standard Deviation Value of that feature.
# 
# **4. min** tells us the minimum value of that feature.
# 
# **5. 25%**, **50%**, and **75%** are the percentile/quartile of each features.
# 
# **6. max** tells us the maximum value of that feature.

# In[ ]:


#Generating heatmap
plt.figure(figsize = (20,12))
sns.heatmap(df.corr(), annot = True, cmap = 'Paired')
plt.show()


# In[ ]:


sns.pairplot(df, hue = 'label')
plt.show()


# In[ ]:


#Count of the attribures
sns.countplot(x = 'label', data = df)
plt.show()


# In[ ]:


# importing the sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing


# In[ ]:


#Listing features to be used for prediction
features = ['pelvic_incidence','pelvic tilt',
'lumbar_lordosis_angle','sacral_slope',
'pelvic_radius',
'degree_spondylolisthesis',
'pelvic_slope',
'Direct_tilt',
'thoracic_slope',
'cervical_tilt',
'sacrum_angle',
'scoliosis_slope']


# In[ ]:


#Drawing box plots for the various featues in X
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
for idx, feat in enumerate(features):
    ax = axes[int(idx / 4), idx % 4]
    sns.boxplot(x='label', y=feat, data=df, ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout();


# In[ ]:


X = df[features]
y =df['label_value']


# In[ ]:


#Splitting the data set into training and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)


# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# In[ ]:


# Storing the predicted values in y_pred for X_test
y_pred = logreg.predict(X_test)


# In[ ]:


#Generating the cofusion matrix
cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
cnf_matrix


# In[ ]:


class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:


print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred))


# In[ ]:


print('Precision Score: ',metrics.precision_score(y_test,y_pred))


# In[ ]:


print('Recall Score: ',metrics.recall_score(y_test,y_pred))

