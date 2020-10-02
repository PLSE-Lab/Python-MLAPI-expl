#!/usr/bin/env python
# coding: utf-8

# **The goal of this kernel was to experiment with artificially generating realistic data to increase the size of the training set and to (hopefully) improve the performance of a model. Please upvote if you found this to be interesting and/or helpful. Feel free to implement this technique in your solutions as long as you give credit to this kernel ;)**

# **Standard processing procedure for any dataset:**
# * Reading data with pandas
# * Checking nulls
# * Dropping columns/rows
# * Label encoding
# * Etc.

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# **Dropping most of the columns to simplify the dataset and to show proof of concept**

# In[ ]:


df_short = df[['LotArea','HouseStyle','YearBuilt','SalePrice']]


# In[ ]:


df_short.head()


# In[ ]:


df_short.hist('SalePrice', bins=30)


# **Converted label into bins to make a classification task. However, this method can be used effectivily with regression tasks too!**

# In[ ]:


df_short['SalePrice'] = pd.cut(df['SalePrice'].values, bins=[0, 100000, 200000, 300000, df_short['SalePrice'].max()], labels=["0", "1", "2", "3"])
df_short.head()


# In[ ]:


df_short['HouseStyle'].value_counts()


# In[ ]:


df_short.isnull().values.any()


# **Note: the method does not support one-hot encoded data at this moment(explanation later on)**

# In[ ]:


#df_short = pd.get_dummies(df_short, columns=['HouseStyle'])


# In[ ]:


df_short.head()


# In[ ]:


def MinMaxScale(column): # you can use minmaxscaler from sklearn library
    minimum = column.min()
    maximum = column.max()
    print("Minimum for column: ", minimum)
    print("Maximum for column: ", maximum)
    return (column - column.min())/(column.max()-column.min())


# In[ ]:


df_short['LotArea'] = MinMaxScale(df_short['LotArea'])


# In[ ]:


df_short['YearBuilt'] = MinMaxScale(df_short['YearBuilt'])


# In[ ]:


df_short['SalePrice'] = df_short['SalePrice'].astype(np.float)


# **Label Encoding**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df_short['HouseStyle'] = enc.fit_transform(df['HouseStyle'])
df_short['HouseStyle'] = MinMaxScale(df_short['HouseStyle'])


# **Final processed dataset**

# In[ ]:


df_short.head()


# **Making training and testing sets to show effectiveness of data generation**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_short.drop('SalePrice',axis=1), df_short['SalePrice'], test_size=0.2, random_state=42)


# **Obtaining benchmark accuracies across several models on original data**

# In[ ]:


original_scores = []

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron

models = [KNeighborsClassifier(n_neighbors=7),
          LogisticRegression(),
          GaussianNB(),
          LinearSVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(max_depth=5),
          AdaBoostClassifier(),
          GradientBoostingClassifier(),
          LinearDiscriminantAnalysis()]

for Model in models:
    Model.fit(X_train, y_train)
    y_pred = Model.predict(X_test)
    print(type(Model).__name__, 'accuracy is',round(accuracy_score(y_pred,y_test)*100), "%")
    original_scores.append(accuracy_score(y_pred,y_test))

print(models[np.argmax(np.array(original_scores))])


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# **Graph of distribution of original data**

# In[ ]:


sns.scatterplot(x=df_short["LotArea"], y=df_short["YearBuilt"], hue=df_short["SalePrice"].astype(np.float))


# **Generation of Mock Dataset**
# /
# **Hint: This is the interesting part ;)**

# ![Intuition image](http://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/12/Scatter-Plot-of-Blobs-Test-Classification-Problem.png)

# **Intuition**
# 1. Data points that are plotted on a n-dimensional graph are located in relatively distinguishable clusters(as seen in the example image above)
# 2. The numerical data of each feature can be described as having a certain distribution
# 
# To realistically reproduce or expand on a dataset, the generated data must follow the distribution set by the original set and must be located in the same clusters.
# 
# My proposed solution that accomplishes this goal:
# * Generates random values from the original distribution using bins and a probability assignment to each interval
# * Uses distance to the centroids of clusters to determine whether a generated data point is unrealistic
# 
# A "checking" mechanism such as the distance to the centroids of clusters is essential to the success of this method as simply generating features from their distributions assumes that each feature is independent and can have a value across its entire range regardless of the value of any other feature.
# 
# A challenge to this method is assigning labels. Labels cannot be chosen from a distribution like other data because they depend greatly on the other features. The best model from the benchmarks on the original dataset above is used to assign labels to generated rows. This is a downside to the method as a part of the generated data is mislabeled.
# 
# **Warning: this process may be slow for datasets with a large number of features and records. It is in no way optimized for performance or for gpus.

# In[ ]:


import random
from scipy.spatial import distance

def euclidean(dataframe, point):
    return (dataframe - np.array(point)).pow(2).sum(1).pow(0.5)

def GenerateDataset(dataframe, label_col_name):
    num_records = dataframe.shape[0]
    print("Number of Records to Produce: ", num_records)
    labels = dataframe[label_col_name].unique()
    centroids = []
    max_distances = []
    
    for label in labels:
        df_label = dataframe[dataframe[label_col_name]==label]
        centroid = calculate_centroid(df_label, label_col_name)
        centroids.append(centroid)
        distances = euclidean(df_label.drop(label_col_name, axis=1), centroid)
        max_distances.append(distances[np.argmax(distances)])
        
    num_generated = 0
    df_drop = dataframe.drop(label_col_name,axis=1)  
    df_result = pd.DataFrame()
    print("Starting Generation of Feature Data")
    while num_generated < num_records:
        record = []
        for column in df_drop.columns:
            record.append(select_from_distribution(df_drop[[column]],column))
        
        for i in range(len(centroids)):
            if distance.euclidean(centroids[i], record) > max_distances[i]:
                pass
            else:     
                df_result = df_result.append(pd.DataFrame(record).T)
                num_generated += 1
                if num_generated % 100 == 0:
                    print("Progress: ", num_generated, "/", num_records, " rows generated")
                break
    print("Finished Generation of Feature Data")
    df_result.columns = df_drop.columns
    model = KNeighborsClassifier(n_neighbors=7) # get best model from previous step
    print("Starting Label Assignment Using ", type(model).__name__)
    model.fit(dataframe.drop(label_col_name,axis=1),dataframe[label_col_name])
    df_result[label_col_name] = model.predict(df_result)
    print("Finished")
    return df_result

def calculate_centroid(dataframe, label_col_name):
    df = dataframe.drop(label_col_name,axis=1)
    coord = []
    for column in df.columns:
        coord.append(df[column].median())  
    return coord

def select_from_distribution(df, column):
    mx = df[column].max()
    mn = df[column].min()
    bins = np.linspace(mn,mx,100)
    df[column] = pd.cut(df[column], bins)
    list_counts = df[column].value_counts().tolist()
    list_prob = [x / df[column].size for x in list_counts]
    bin_names = df[column].value_counts().index.tolist()
    select_range = random.choices(bin_names, list_prob)[0]
    return random.uniform(select_range.left, select_range.right)


# In[ ]:


df = GenerateDataset(df_short, 'SalePrice')


# In[ ]:


df.head()


# **Distribution of original data**

# In[ ]:


lm=sns.scatterplot(x=df_short["LotArea"], y=df_short["YearBuilt"], hue=df_short["SalePrice"])
axes = lm.axes
axes.set_xlim(-0.1,1.1)
axes.set_ylim(-0.1,1.1)


# **Distribution of generated data**

# In[ ]:


lm=sns.scatterplot(x=df["LotArea"], y=df["YearBuilt"], hue=df["SalePrice"])
axes = lm.axes
axes.set_xlim(-0.1,1.1)
axes.set_ylim(-0.1,1.1)


# As you can see by the graphs, the generated data effectively matches the distribution of the original without producing unreasonable outliers. Lets test the performance of the generated data on models. If the average model performance is not impacted by training on the generated data, the method successfully imitated the original dataset. As we will see, while the accuracy of some models decreases, others experience a noticable performance increase when trained on the generated data and evaluated on the test set of the original data.

# In[ ]:


new_scores = []
for Model in models:
    Model.fit(df.drop('SalePrice',axis=1), df['SalePrice'])
    y_pred = Model.predict(X_test)
    print(type(Model).__name__, 'accuracy is',round(accuracy_score(y_pred,y_test)*100), "%")
    new_scores.append(accuracy_score(y_pred,y_test))


# In[ ]:


sum(np.array(new_scores)-np.array(original_scores))/(len(original_scores))


# In[ ]:


print("Original Scores: ", np.array(original_scores))


# In[ ]:


print("New Scores: ", np.array(new_scores))


# **Thank you for reading! Once again, please upvote if you found this to be interesting and/or helpful and feel free to implement this technique in your solutions as long as you give credit to this kernel!**
