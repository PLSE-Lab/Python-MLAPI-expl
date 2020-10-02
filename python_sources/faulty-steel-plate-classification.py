#!/usr/bin/env python
# coding: utf-8

# # **Faulty Steel Plate Classification**
# 
# For the complex dataset below we have used KNN and Random Forest for classication of dataset. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# # Analyzing Dataset
# * Reading ***faults.csv*** file
# * Describing dataframe

# In[ ]:


df = pd.read_csv("../input/faults.csv")
df.describe().T


# # Pre-processing and Arranging Dataset
# 1. Dropping redundant column *'TypeOfSteel_A400'*
# 2. Dropping hot encoding classes and creating one class column

# In[ ]:


from xlsxwriter.utility import xl_rowcol_to_cell
conditions=[(df['Pastry'] == 1) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0), (df['Pastry'] == 0) & (df['Z_Scratch'] == 1)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 1)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 1)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 1)& (df['Bumps'] == 0)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 1)& (df['Other_Faults'] == 0),(df['Pastry'] == 0) & (df['Z_Scratch'] == 0)& (df['K_Scatch'] == 0)& (df['Stains'] == 0)& (df['Dirtiness'] == 0)& (df['Bumps'] == 0)& (df['Other_Faults'] == 1)]
choices = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df['class'] = np.select(conditions, choices)
#Dropping redundant column
#Dropping Hot Encoding Classes
drp_cols=['TypeOfSteel_A400','Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df.drop(choices, inplace=True,axis = 1)
df


# # Pre-prepossing and Arranging Dataset Continued..
# 3. Describing the color code of errors
# >         Error               Color
# >     1. 'Pastry'            'Red'
# >     2. 'Z_Scratch'         'Blue'
# >     3. 'K_Scatch'          'Green'
# >     4. 'Stains'            'Black'
# >     5. 'Dirtiness'         'Pink'
# >     6. 'Bumps'             'Brown'
# >     7. 'Other_Faults'      'Gold'     
# 4. Plotting a scatter graph for general overview of errors in individual steel plates
# 5. Displaying the graph for each attribute
# 6. Save the figure

# In[ ]:


color_code = {'Pastry':'Red', 'Z_Scratch':'Blue', 'K_Scatch':'Green', 'Stains':'Black', 'Dirtiness':'Pink', 'Bumps':'Brown', 'Other_Faults':'Gold'}
color_list = [color_code.get(i) for i in df.loc[:,'class']]
pd.plotting.scatter_matrix(df.loc[:, df.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.3,
                                       s = 50)
plt.show()
plt.savefig("figure_1.png")


# # Overview of total errors in the dataset

# In[ ]:


sns.countplot(x="class", data=df)
df.loc[:,'class'].value_counts()


# # Implementation of K-Nearest Neighbor Classifier
# 1. Diving dataset into training and testing dataset
# 2. Taking test size of 30% - Random shuffle
# 3. Checking accuracy from k = 1 to 27
# 4. Plot K-value vs Accuracy graph which shows the score of Training Accuracy and Testing Accuracy
# 5. Extracting the K-value producing the best accurracy
# 
# **K in general is called a hyperparameter and we need to choose the K that gives best performace.**
# 
# **Below, K is ranged from 1 to 27(excluded) and accuracy is found for each K value.**
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
x,y = df.loc[:,df.columns != 'class'], df.loc[:,'class']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 2,shuffle=True)
# Model complexity
neig = np.arange(1, 27)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 27(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('k value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# # Model complexity:
# 1. K is 12 which is small. Therefore, model is a complex model that can lead to overfitting. It means that model memozizes the trainning sets and cannot predict test set with good accuracy.
# 2. If k is big, model that is less complex model can lead to underfit. (Not the case here)
# 3. As you can see in plot, when K is 1 it memozize train sets and cannot give good accuracy on test set (overfit). Also if K is 12, model is leading to overfitting. Again, accuracy is not enough **as it looks like the model is just guessing** the test dataset even when K=12 is the best performance it gives. 
# 
# **Since the dataset is sparse and the error varies extremely i.e Other_Faults is at 673 and Dirtiness is at 55, it is better to try to get a better accuracy after stratifying the trainning dataset sample and checking the best performance.**

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size = 0.3,random_state = 2,shuffle=True)
# Model complexity
neig = np.arange(1, 27)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 27(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('k value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# # **Again, we can see that K has improved just by little but still the model is overfitting **
# Moving on to Random Forest Classifier

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier

x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,test_size = 0.3,random_state = 8)
trees = np.arange(1, 50)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(trees):
    # k from 1 to 27(exclude)
    rf = RandomForestClassifier(random_state = 8, n_estimators=k, min_samples_split=2)
    # Fit with rf
    rf.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(rf.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(rf.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(trees, test_accuracy, label = 'Testing Accuracy')
plt.plot(trees, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('No. of trees VS Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.xticks(trees)
plt.show()

rf = RandomForestClassifier(random_state = 8, n_estimators=48, min_samples_split=2)
y_pred = rf.fit(x_train,y_train).predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d") 
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print('Classification report: \n',classification_report(y_test,y_pred))
print("Best accuracy is {} with No. of trees = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# In random forests classifier, each tree in the ensemble is built from a sample drawn with replacement from the training set. In addition, when splitting a node during the construction of the tree, the split that is chosen is no longer the best split among all features. Instead, the split that is picked is the best split among a random subset of the features. As a result of this randomness, the bias of the forest usually slightly increases (with respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than compensating for the increase in bias, hence yielding an overall better model.
# 
# >** Random Forest Classifier performs best with this complex dataset giving an approx accuracy of 80.1% with 48 trees.**
# 

# In[ ]:




