#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)


# In[ ]:


credit_df = pd.read_csv(r"../input/creditcardfraud/creditcard.csv")
credit_df.head()


# ### Data understanding:

# In[ ]:


counts = credit_df['Class'].value_counts()


# In[ ]:


counts.plot(kind='bar')
plt.title("Class count")
plt.xlabel("Class")
plt.ylabel("Value Count")
plt.show()


# There is only 0.173 percent values from class one. Data is highly unblanced. We can approch this issue using number of techniques lets see those 
# 1. Under Sampling
# 2. Over Sampling
# 3. Synthetic Minority oversmapling Technique(SMOTE)
# 4. AdaSYN

# ### 1. Undersampling
# <br>
# In this case we remove excess data points from the majority class. The Huge data loss in above condition where majority class have almost 2,84,315 data points.  

# In[ ]:


class_0_df = credit_df[credit_df["Class"] == 0]
class_1_df = credit_df[credit_df["Class"] == 1]


# In[ ]:


count_class_0, count_class_1 = credit_df["Class"].value_counts()


# In[ ]:


class_0_df_under_samp = class_0_df.sample(count_class_1)
credit_df_under = pd.concat([class_1_df, class_0_df_under_samp], axis=0)
counts = credit_df_under['Class'].value_counts()
counts.plot(kind='bar')
plt.title("Class count")
plt.xlabel("Class")
plt.ylabel("Value Count")
plt.show()


# In[ ]:


print("We lost {} % of data due to undersampling". format(100 - credit_df_under.shape[0]/credit_df.shape[0]*100))


# ### 2. Oversampling

# Using this method we can assign weigth to randomly choosen data points from the minority class and duplicate the same. 

# In[ ]:


class_1_df_over_sample = class_1_df.sample(count_class_0, replace=True)
credit_df_over = pd.concat([class_1_df_over_sample, class_0_df], axis=0)
counts = credit_df_over['Class'].value_counts()
counts.plot(kind='bar', )
plt.title("Class count")
plt.xlabel("Class")
plt.ylabel("Value Count")
plt.show()


# In[ ]:


fig=plt.figure(figsize=(12,6))

plt.subplot(121)
sns.scatterplot(x="V1",y="V2", hue="Class", data=credit_df)
plt.title('Before Sampling')
plt.subplot(122)
sns.scatterplot(x="V1",y="V2", hue="Class", data=credit_df_over)
plt.title('After Sampling')
plt.show()


# We does not see any changes beacuse as mensioned above it just duplicate the minority class data points. while plotting data points from minority class overlap each other.

# ### SMOTE(Synthetic Minority Over-Sampling Technique)
# 
# As name suggest it is over-sampling method. It create synthetic (Non duplicate) sample of the minority class. First it find the n-nearest neighbour in the minority class for each of the sample in the class. Then it draw line between those and generate random points on the line.<br>
# 
#   ***SMOTE first selects a minority class instance at a random and finds its k nearest minority class neighbors. The synthetic instance is then created by choosing one of the k nearest neighbors b at random and connecting a and b to form a line segment in the feature space. The synthetic instances are generated as a convex combination of the two chosen instances a and b.***

# In[ ]:


feature_column = list(credit_df.columns)
feature_column = feature_column[:-1]
class_labels = credit_df.Class
features = credit_df[feature_column]


# In[ ]:


#train test split
feature_train, feature_test, class_train, class_test = train_test_split(features, class_labels, 
                                                                        test_size = 0.3, random_state=0)


# In[ ]:


print("Test value counts")
print(class_test.value_counts(),"\n")
print("Train value counts")
print(class_train.value_counts())


# In[ ]:


feature_train["class"] = class_train


# Applying the SMOTE oversampling 

# In[ ]:


SMOTE_Oversampler=SMOTE(random_state=0)
SOS_features,SOS_labels=SMOTE_Oversampler.fit_sample(feature_train,class_train)


# In[ ]:


SOS_features['class'] = SOS_labels


# In[ ]:


fig=plt.figure(figsize=(12,6))

plt.subplot(121)
sns.scatterplot(x="V1",y="V3", hue="class", data=feature_train)
plt.title('Before Sampling')
plt.subplot(122)
sns.scatterplot(x="V1",y="V3", hue="class", data=SOS_features)
plt.title('After Sampling')
plt.show()


# ### ADASYN

# Its a improved version of Smote. Instead of all the sample being linearly correlated to the parent they have a little more variance in them i.e they are bit scattered.<br>
# <br>
# ADASYN is to use a weighted distribution for different minority class examples according to their level of difficulty in learning, where more synthetic data is generated for minority class examples that are harder to learn compared to those minority examples that are easier to learn. As a result, the ADASYN approach improves learning with respect to the data distributions in two ways: 
# 1.  Reducing the bias introduced by the class imbalance, and 
# 2.  Adaptively shifting the classification decision boundary toward the difficult examples. 

# In[ ]:


from imblearn.over_sampling import ADASYN

Adasyn_Oversampler=ADASYN()
AOS_features,AOS_labels=Adasyn_Oversampler.fit_sample(feature_train,class_train)


# In[ ]:


AOS_features['class'] = AOS_labels


# In[ ]:


fig=plt.figure(figsize=(12,6))

plt.subplot(121)
sns.scatterplot(x="V1",y="V3", hue="class", data=feature_train)
plt.title('Before Sampling')
plt.subplot(122)
sns.scatterplot(x="V1",y="V3", hue="class", data=AOS_features)
plt.title('After Sampling')
plt.show()


# ### SMOTE vs ADASYN

# In[ ]:


fig=plt.figure(figsize=(12,6))

plt.subplot(121)
sns.scatterplot(x="V1",y="V3", hue="class", data=SOS_features)
plt.title('SMOTE Oversampling')
plt.subplot(122)
sns.scatterplot(x="V1",y="V3", hue="class", data=AOS_features)
plt.title('AdaSYN Oversampling')
plt.show()

