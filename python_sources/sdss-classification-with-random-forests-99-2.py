#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Founded by the University of Chicago, the [Sloan Digital Sky Survey (SDSS)](http://https://www.sdss.org/) uses a wide-angle optical telescope at the Apache Point Observatory in New Mexico, United States to create multi-colored 3D maps of a third of our sky. The survey includes spectral information on all detected astronomical objects which include stars, galaxies, and quasars. Data is periodically and publically released. This notebook will cover Data Release 16 (DR16).
# 
# This data set contains 100,000 labelled samples and 17 features. However, clever feature engineering allows me to reduce our number of features and yield classification accuracies above 99%. For this exercise, I'll compare two ML models -- deep neural networks and random forests -- on their ability to classify stars, galaxies, and quasars in the SDSS DR16 data set.
# 
# 
# **Approach**
# 1. Data exploration
# 2. Feature Engineering
# 3. Construct ML Models (DNN and Random Forest)
# 4. Test and evaluate models
# 
# For organization, I like to separate my data exploration and ML Python scripts.
# 
# # Data exploration
# 
# Off the bat, I recognized that there are a few features that bear no physical significance whatsoever: objid, specobjid, run, rerun, camcol, and field. I'll omit these features from further discussion.
# 
# Let's generate the following figures:
# * Pie chart of labels
# * Equitorial coordinates of observations
# * Histogram distributions of redshift, plate, modified julien day, and fiber id
# * Histogram distributions of Thuan-Gunn Astronomic Magnitude system

# In[ ]:


"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Purpose:    Data Exploration

@author:    Kevin Trinh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pieChart(sdss_df):
    '''Plot a pie chart for label count.'''
    label_counts = sdss_df['class'].value_counts()
    colors = ['skyblue', 'red', 'gold']
    fig1, ax1 = plt.subplots()
    ax1.pie(label_counts, labels=['Galaxy', 'Stars', 'Quasars'],
            autopct='%1.2f%%', startangle=45, colors=colors)
    ax1.axis('equal')
    plt.title('SDSS Object Classes')
    plt.show()

def distribution(sdss_df, axes, feature, row):
    '''Plot the distribution of a space object w.r.t. a given feature.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    for i in range(len(labels)):
        label = labels[i]
        ax = sns.distplot(sdss_df.loc[sdss_df['class']==label, feature], 
                          kde=False, bins=30, ax=axes[row, i], color=colors[i])
        ax.set_title(label)
        if (i == 0):
            ax.set(ylabel='Count')
            
def equitorial(sdss_df, row):
    '''Plot equitorial coordinates of observations.'''
    labels = np.unique(sdss_df['class'])
    colors = ['skyblue', 'gold', 'red']
    label = labels[row]
    sns.lmplot(x='ra', y='dec', data=sdss_df.loc[sdss_df['class']==label],
               hue='class', palette=[colors[row]], scatter_kws={'s': 2}, 
               fit_reg=False, height=4, aspect=2)
    plt.ylabel('dec')
    plt.title('Equitorial coordinates')
    

def main():

    # read in SDSS data
    filepath = '../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv'
    sdss_df = pd.read_csv(filepath, encoding='utf-8')

    # define lists of relevant features
    geo = ['ra', 'dec']
    nonugriv = ['redshift', 'plate', 'mjd', 'fiberid']
    ugriv = ['u', 'g', 'r', 'i', 'z']

    # plot pie chart of label count
    pieChart(sdss_df)

    # plot equitorial coordinates of observations
    for row in range(3):
        equitorial(sdss_df, row)
        plt.show()
    
    # plot the distribution of non-geo and non-ugriv features
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 14))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(nonugriv)):
        feat = nonugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()
        
    # plot the distribution of ugriv features
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for row in range(len(ugriv)):
        feat = ugriv[row]
        distribution(sdss_df, axes, feat, row)
    plt.show()

main()


# I noticed a few interesting things about my figures.
# 
# First, my dataset consists of mostly galaxies and stars, though quasars are a sizeable minority (10.58% of samples). A skilled data scientist (which I am not) may factor this in when creating curated batches for ML models to improve model performance.
# 
# Second, the equitorial coordinates of stars are distributed differently than that of galaxies and quasars. This could be due to the Milky Way's shape as a barred spiral galaxy. We're more likely to see stars from our own solar system if we make observations along the plane of our galaxy. These coordinates could be useful in separating stars from other classes.
# 
# Third, redshift appears to be very telling of the observed object's class. The axes of each class are drastically different. This makes sense because redshift results from observing objects travelling away from the observer at fast speeds. Stars travel away from us fast; galaxies travel away from us at faster rates; quasars travel away from us at even faster rates.
# 
# Fourth, the bulk of galaxies observed have low plate numbers and fiber IDs relative to quasars and stars. 
# 
# Fifth, the distributions among classes for the ugriv variables look the same, and differences between classes are present but subtle. It is worth noting that each ugriv feature (i.e. band) is ordinally related to each other, so some additional processing of these features should be done prior to any ML.
# 
# # Machine Learning (Training)
# 
# Note: The below script is broken into two parts: 1) feature engineering, training, and validation, and 2) plots and predictions.
# 
# I shuffled, partitioned, and reformated my data to feed into my RFC. While I have three different labels, I can convert them to arbitrary integers given that I'm working with decision trees (i.e. no dummy variables or one-hot-encoding needed). I scaled all features and used three principal component analysis (PCA) features to replace the u, g, r, i, and z features.
# 
# Next, I trained my RFCs using 200 trees and obtained a validation accuracy of 98.4%.

# In[ ]:


"""
Project:    Classification of Sloan Digital Sky Survey (SDSS) Objects
Phase:      Feature engineering and ML classification

Algorithm:  Random Forest

Steps:      1) Import libraries
            2) Read, shuffle, and partition data
            3) Restructure data as inputs for DNN
            4) Feature Engineering
            5) Create and train DNN
            6) Make predictions on validation sets
            7) Fine-tune models for highest performance on validation set
            8) Make predictions on test set
            9) Evaluate model with confusion matrix

@author:    Kevin Trinh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns


# read in and shuffle SDSS data
filename = '../input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv'
sdss_df = pd.read_csv(filename, encoding='utf-8')
sdss_df = sdss_df.sample(frac=1)

# drop physically insignificant columns
sdss_df = sdss_df.drop(['objid', 'specobjid', 'run', 'rerun', 'camcol',
                        'field'], axis=1)


# partition SDSS data (60% train, 20% validation, 20% test)
train_count = 60000
val_count = 20000
test_count = 20000

train_df = sdss_df.iloc[:train_count]
validation_df = sdss_df.iloc[train_count:train_count+val_count]
test_df = sdss_df.iloc[-test_count:]


# obtain feature dataframes
X_train = train_df.drop(['class'], axis=1)
X_validation = validation_df.drop(['class'], axis=1)
X_test = test_df.drop(['class'], axis=1)


# encode labels as arbitrary integer classes
le = LabelEncoder()
labels = le.fit_transform(sdss_df['class'])

y_train = labels[:train_count]
y_validation = labels[train_count:train_count+val_count]
y_test = labels[-test_count:]

# scale features
scaler = StandardScaler()
scaler.fit(X_train) # fit scaler to training data only
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_validation.columns)

# apply principal component analysis to wavelength intensities
pca = PCA(n_components=3)
dfs = [X_train, X_validation, X_test]
for i in range(len(dfs)):
    df = dfs[i]
    ugriz = pca.fit_transform(df[['u', 'g', 'r', 'i', 'z']])
    df = pd.concat((df, pd.DataFrame(ugriz)), axis=1)
    df.rename({0: 'PCA1', 1: 'PCA2', 2: 'PCA3'}, axis=1, inplace=True)
    df.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)
    dfs[i] = df
X_train, X_validation, X_test = dfs

# create a random forest model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
preds = rfc.predict(X_validation)
model_acc = (preds == y_validation).sum().astype(float) / len(preds) * 100

print('Validation Accuracy: %3.5f' % (model_acc))


# # Machine Learning (Testing)
# 
# I then evaluated my model against the test set, measured the importance of each feature (which is a nice property of decision trees), and plotted a confusion matrix. Unlike my training and validation set, I ran the following code ONCE for good practice so that my test data can truly be considered 'unseen' data.
# 
# My model yielded a test accuracy of 99.2% which is better than my validation performance. As previously predicted, redshift is highly correlated with the astronomical object label (58.3% importance). The next three most important features are the PCA features, though these features are much less influential than redshift.
# 
# According to my confusion matrix:
# 
# * Galaxies have a precision of 99.5% and recall of 99.0%.
# * Quasars have a precision of 95.4% and recall of 98.2%.
# * Stars have a precision of 99.8% and 99.8%.
# 
# In other words, my model performs much better on galaxies and stars than quasars. This makes sense because quasars are the minority in my 100,000 samples.

# In[ ]:


# evaluate the random forest
preds = rfc.predict(X_test)
model_acc = (preds == y_test).sum().astype(float) / len(preds) * 100

print('Test Accuracy: %3.5f\n' % (model_acc))

# analyze feature importance in random forest model
importances = pd.DataFrame({
    'Feature': X_validation.columns,
    'Importance': rfc.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances = importances.set_index('Feature')
print(importances)

# plot confusion matrices for both models
labels = np.unique(sdss_df['class'])

ax = plt.subplot(1, 1, 1)
ax.set_aspect(1)
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', xticklabels = labels,
            yticklabels = labels, cbar_kws={'orientation': 'horizontal'})
plt.xlabel('Actual values')
plt.title('Random Forest Classifier')

plt.show()

# compute precision and recall
precision = np.diag(cm) / np.sum(cm, axis = 0)
recall = np.diag(cm) / np.sum(cm, axis = 1)

print('Recall: ', precision)
print('Precision: ', recall)


# # Conclusion
# 
# My RFC obtained a 99.2% classification accuracy with fairly low run-time. The precision and recall is also generally high, thus demonstrating the strength of my model.
# 
# I've also tackled this classification problem using [deep neural network](https://www.kaggle.com/ktrinh/sdss-classification-with-deep-neural-networks), though I got a lower accuracy with a significantly longer run-time.
# 
# If you'd like to check out my other Kaggle kernels or learn my about my background, feel free to check out my [website](http://www.kevinttrinh.com/data-science/)!
