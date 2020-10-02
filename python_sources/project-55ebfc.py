#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# indicates that we want our plots to be shown in our notebook and not in a sesparate viewer
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_path='./../input/nassCDS.csv'
data=pd.read_csv(data_path)


# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


#drop the column unnamed because if of no relievant 
data.drop('Unnamed: 0', axis=1, inplace=True)
#replace the incorrect values with correct ones
data['dvcat'].replace('1-9km/h','1-9',inplace=True)
data.head()


# In[ ]:


#filling the missing with the median value.
data.fillna( data.median(),inplace=True )
data.isnull().sum()


# In[ ]:


def create_label_encoder_dict(df):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    for column in df.columns:
        # Only create encoder for categorical data types
        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':
            label_encoder_dict[column]= LabelEncoder().fit(df[column])
    return label_encoder_dict


# In[ ]:


label_encoders = create_label_encoder_dict(data)
print("Encoded Values for each Label")
print("="*32)
for column in label_encoders:
    print("="*32)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_))


# In[ ]:


# Apply each encoder to the data set to obtain transformed values
data2 = data.copy() # create copy of initial data set
for column in data2.columns:
    if column in label_encoders:
        data2[column] = label_encoders[column].transform(data2[column])
print("Transformed data set")
print("*"*20)
data2.head()


# plotting a bar chart of the number of vehicles that have an air bag deployed depending on the year of the car

# In[ ]:


data_agg=data2.groupby(["yearVeh"],as_index=False).agg({"deploy": "sum"})
ax=data_agg.plot('yearVeh', 'deploy', kind='bar', figsize=(17,5), color='#86bf91', zorder=2, width=0.85)
ax.set_xlabel("Year", labelpad=20, size=12)
# Set y-axis label
ax.set_ylabel("deploy", labelpad=20, size=12)
ax.legend_.remove()


# Bar plot of people who wore a seatbelt vs those who died

# In[ ]:


data_agg=data2.groupby(["dead"],as_index=False).agg({"seatbelt": "sum"})
ax=data_agg.plot('dead', 'seatbelt', kind='bar', figsize=(17,5), color='blue', zorder=2, width=0.85)
ax.set_xlabel("dead", labelpad=20, size=12)
# Set y-axis label
ax.set_ylabel("seatbelt", labelpad=20, size=12)
ax.legend_.remove()


# In[ ]:


data_agg=data2.groupby(["yearVeh"],as_index=False).agg({"airbag": "sum"})
ax=data_agg.plot('yearVeh', 'airbag', kind='bar', figsize=(17,5), color='#86bf91', zorder=2, width=0.85)
ax.set_xlabel("Year", labelpad=20, size=12)
# Set y-axis label
ax.set_ylabel("airbag present", labelpad=20, size=12)
ax.legend_.remove()


# Injury Severity Count

# In[ ]:


acc_count = data2.groupby(data2.injSeverity).injSeverity.count().plot(kind = 'bar')


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


#comparing the dead between the deployment of airbag
ax=data['ageOFocc'].plot(kind='hist')
ax.set_xlabel("Age of Occupant", labelpad=20, size=12)


# In[ ]:


sns.countplot(x='dead',hue='seatbelt', data=data)


# In[ ]:


sns.countplot(x='dead',hue='airbag', data=data)


# In[ ]:


x_data= data2[['airbag','seatbelt','deploy','frontal', 'sex', 'weight','ageOFocc', 'yearacc', 'yearVeh',
       'injSeverity']]
y_data=data2['dead']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30)


# In[ ]:


# Import linear model package (has several regression classes)
from sklearn.linear_model import LogisticRegression


# In[ ]:


# Create an instance of linear regression
reg = LogisticRegression()


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


test_predicted = reg.predict(X_test)
test_predicted


# In[ ]:


reg.coef_


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score


# In[ ]:


classification_report(y_test,test_predicted)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


#
cm = confusion_matrix(y_test,test_predicted)


# In[ ]:





# In[ ]:


#explaining how accurate is the predictions
score=accuracy_score(y_test,test_predicted)


# In[ ]:


# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))


# In[ ]:


#Explained variance score: 1 is perfect prediction
# R squared
print('Variance score: %.2f' % r2_score(y_test, test_predicted))


# **Confusion matrix given a visualization of the logistic regression**

# In[ ]:


plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[ ]:


dead_count = data.groupby(data.dead).dead.count().plot(kind = 'bar')


# **Aim : Use a decision tree to identify a seatelt will save me whether or not an airbag is present**

# 

# In[ ]:


treedata = data[['dead','airbag','seatbelt', 'deploy', 'injSeverity']]
treedata


# In[ ]:


label_encoders = create_label_encoder_dict(treedata)
print("Encoded Values for each Label")
print("="*32)
for column in label_encoders:
    print("="*32)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)


# # Apply each encoder to the data set to obtain transformed values
# 

# In[ ]:


treedata2 = treedata.copy() # create copy of initial data set
for column in treedata2.columns:
    if column in label_encoders:
        treedata2[column] = label_encoders[column].transform(treedata2[column])

print("Transformed data set")
print("="*32)
treedata2


# In[ ]:


# separate our data into dependent (Y) and independent(X) variables
X_data = treedata2[['airbag','seatbelt','deploy']]
Y_data = treedata2['dead']


# In[ ]:


# import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[ ]:


clf = DecisionTreeClassifier(max_depth=90, criterion='entropy') 


# In[ ]:


# build classifier
clf.fit(X_data, Y_data)


# In[ ]:


pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])


# In[ ]:


import graphviz
dot_data = tree.export_graphviz(clf,out_file=None, 
                                feature_names=X_data.columns, 
                         class_names=label_encoders[Y_data.name].classes_,  
                         filled=True, rounded=True,  proportion=True,
                                node_ids=True, #impurity=False,
                         special_characters=True)


# In[ ]:


graph = graphviz.Source(dot_data) 
graph


# In[ ]:


k=(clf.predict(X_data) == Y_data) # Determine how many were predicted correctly


# In[ ]:


k.value_counts()


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm=confusion_matrix(Y_data, clf.predict(X_data), labels=Y_data.unique())
cm


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


plot_confusion_matrix(cm,data2['dead'].unique())


# Trying Neural Networks

# In[ ]:


# Apply each encoder to the data set to obtain transformed values
neuraldata = data.copy() # create copy of initial data set
for column in data2.columns:
    if column in label_encoders:
        neuraldata[column] = label_encoders[column].transform(neuraldata[column])

print("Transformed data set")
print("="*32)
data2.head(15)


# In[ ]:


# separate our data into dependent (Y) and independent(X) variables
X_data = neuraldata[['yearVeh','airbag','weight']]
Y_data = neuraldata['deploy'] # actually department column


# In[ ]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)


# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


#Create an instance of linear regression
reg = MLPClassifier()
#reg = MLPClassifier(hidden_layer_sizes=(8,120))


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


reg.n_layers_ # Number of layers utilized


# In[ ]:


# Make predictions using the testing set
test_predicted = reg.predict(X_test)
test_predicted


# In[ ]:


k=(reg.predict(X_test) == y_test) # Determine how many were predicted correctly


# In[ ]:


k.value_counts()


# In[ ]:


cm=confusion_matrix(y_test, reg.predict(X_test), labels=y_test.unique())
cm


# In[ ]:


plt.figure(figsize=(9,16))
plot_confusion_matrix(cm,data['deploy'].unique())

