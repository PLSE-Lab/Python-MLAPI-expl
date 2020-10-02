#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from subprocess import check_output
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re
import matplotlib.pyplot as plt


# # Import libraries

# In[ ]:


#read dataset
data = pd.read_csv('../input/mushrooms.csv')
   # chose path where .data file present


# # Data Preprocessing: 

# In[ ]:


data.head()  #to find first 5 values


# in class e for 'edible', and p for 'poisonous'.

# In[ ]:


data.columns 


# In[ ]:


data.info()


# MISSING ATTRIBUTE: In the dataset, there is an attribute with missing values. Its name is stalk-root. The missing value is represented by "?". 
# 

# In[ ]:


data.describe()


# In[ ]:


data['stalk-root'].value_counts(dropna=False)


# now we can see that 'stalk-root' has 2480 missing value.
#     

# # How to handle this missing value?

# # strategy 1):

# In[ ]:


#data["stalk-root"].replace(["?"], ["b"], inplace= True)   # use b(it is mode) 


# # Strategy 2):

# In[ ]:


from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()
for column in data.columns:
    data[column] = labelencoder.fit_transform(data[column])


# In[ ]:


data['stalk-root'].value_counts(dropna=False)


# In[ ]:


data['stalk-root'].describe()


# In[ ]:


data["stalk-root"] = data["stalk-root"].astype(object)


# In[ ]:


#data["stalk-root"][::].replace(0, 1.109565) 
#1.109565 = mean 


# In[ ]:


data["stalk-root"] = data["stalk-root"].astype(int)


# In[ ]:


import seaborn as sns
plt.figure(figsize=(16,10))
sns.heatmap(data.corr(), annot=True);


# by observing heatmap, we can see that veil-type is not contributing to dataset.
# 

# In[ ]:


data = data.drop('veil-type', axis=1)


# # Strategy 3)

# In[ ]:


data=data.drop(["stalk-root"],axis=1)


# In[ ]:


data.head()


#  in class 0 for 'edible' and 1 for 'poisonous'.

# In[ ]:


data.info()


# dataset does not included any object, now we are ready to build our model.

# # Feature selection

# In[ ]:


X = data.drop(['class'], axis=1)  #remove target from train dataset
y = data['class'] # test dataset with target 


# In[ ]:


# divide dataset into 50% train, and other 50% test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# # Decision Tree classifier

# we use **BINARY SPLITS**.

# # i )The Gini impurity

# In[ ]:


clf1 = DecisionTreeClassifier(criterion = "gini",                # model design 
            random_state = 100,max_depth=2, min_samples_leaf=5, ) 
# split dataset into depth 2(0,1,2)
# stop dataset when leaf is 5 . 


# In[ ]:


clf1 = clf1.fit(X_train, y_train)  #training the model 


# In[ ]:


y_pred = clf1.predict(X_test)  # prediction on test dataset 


# In[ ]:


print('accuracy of train dataset is',clf1.score(X_train, y_train))


# In[ ]:


print('accuracy of test dataset is',clf1.score(X_test, y_test))


# In[ ]:


from sklearn.metrics import classification_report
print("Decision tree Classification report", classification_report(y_test, y_pred))


#  precision score of class 0 is 0.87, and class 1 is 0.96

#  Recall score of class 0 is 0.97, and class1 is 0.85

#  f1- score of class 0  is 0.92 and class1 is 0.90. 

# In[ ]:


confusion_matrix(y_test, y_pred)


# 2048  are classify in class 0 , 1656  classify in class 1 and 358 items does not classify on any class. 

# In[ ]:


cfm=confusion_matrix(y_test, y_pred)
sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[ ]:


import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
dot_data = export_graphviz(clf1, out_file='tree1.dot',
                          feature_names=X.columns,
                          filled=True, rounded = True, 
                          special_characters= True,
             class_names=['0','1']  )
graph = graphviz.Source(dot_data)


# In[ ]:


os.system('dot -Tpng tree1.dot -o tree1.png')


# In[ ]:


from IPython.display import Image
Image(filename="tree1.png", height=1000, width=1000)


# # ii)The GINI Impurity:

# In[ ]:


clf2 = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=5, min_samples_leaf=15, ) 
clf2 = clf2.fit(X_train, y_train)   
# split dataset into depth 5
# stop dataset when leaf is 15. 


# In[ ]:


y_pred = clf2.predict(X_test)


# In[ ]:


print('accuracy of train dataset is',clf2.score(X_train, y_train))


# In[ ]:


print('accuracy of test dataset is',clf2.score(X_test, y_test))


# In[ ]:


from sklearn.metrics import classification_report
print("Decision tree Classification report", classification_report(y_test, y_pred))


#  precision score of class 0 is 0.98, and class 1 is 0.96

#  recall score of class 0 is 0.96 and class 1 is 0.98

# f1 score of class 0 and class 1 is 0.97

# In[ ]:


confusion_matrix(y_test, y_pred)


# 2034  are classify in class 0 , 1913  classify in class 1 and 115 items does not classify on any class. 

# In[ ]:


cfm=confusion_matrix(y_test, y_pred)
sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[ ]:


import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
dot_data = export_graphviz(clf2, out_file='tree2.dot',
                          feature_names=X.columns,
                          filled=True, rounded = True, 
                          special_characters= True,
             class_names=['0','1']  )
graph = graphviz.Source(dot_data)


# In[ ]:


os.system('dot -Tpng tree2.dot -o tree2.png')


# In[ ]:


from IPython.display import Image
Image(filename="tree2.png", height=1000, width=1000)


# # iii)The entropy impurity: 

# In[ ]:


clf3 = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 10) 
clf3 = clf3.fit(X_train, y_train)
# split dataset into depth 3(0,1,2,3)
# stop dataset when leaf is 10 . 


# In[ ]:


y_pred = clf3.predict(X_test)


# In[ ]:


print('accuracy of train dataset is',clf3.score(X_train, y_train))


# In[ ]:


print('accuracy of test dataset is',clf3.score(X_test, y_test))


# In[ ]:


print("Decision tree Classification report", classification_report(y_test, y_pred))


#  precision score of class 0 is 0.97, and class 1 is 0.94

#  recall score of class 0 is 0.94, and class 1 is 0.97

# F1-score score of class 0 is 0.96, and class 1 is 0.95

# In[ ]:


confusion_matrix(y_test, y_pred)


# 1983  are classify in class 0 , 1896  classify in class 1 and 183 items does not classify on any class. 

# In[ ]:


cfm=confusion_matrix(y_test, y_pred)
sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[ ]:


import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
dot_data = export_graphviz(clf3, out_file='tree3.dot',
                          feature_names=X.columns,
                          filled=True, rounded = True, 
                          special_characters= True,
             class_names=['0','1']  )
graph = graphviz.Source(dot_data)


# In[ ]:


os.system('dot -Tpng tree3.dot -o tree3.png')


# In[ ]:


from IPython.display import Image
Image(filename="tree3.png", height=1000, width=1000)


# # iv)The entropy impurity

# In[ ]:


clf4 = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 10, min_samples_leaf = 20) 
clf4 = clf4.fit(X_train, y_train)
# split dataset into depth 3(0,1,2,3)
# stop dataset when leaf is 10 . 


# In[ ]:


y_pred = clf4.predict(X_test)


# In[ ]:


print('accuracy of train dataset is',clf4.score(X_train, y_train))


# In[ ]:


print('accuracy of test dataset is',clf4.score(X_test, y_test))


# In[ ]:


print("Decision tree Classification report", classification_report(y_test, y_pred))


# Precision, recall and f1 score is 1.00 which is 100%

# In[ ]:


confusion_matrix(y_test, y_pred)


# 2111  are classify in class 0 , 1949  classify in class 1 and only 2  items does not classify on any class. 

# In[ ]:


cfm=confusion_matrix(y_test, y_pred)
sns.heatmap(cfm, annot=True, linewidth=5, cbar=None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('predicted label')


# In[ ]:


import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
dot_data = export_graphviz(clf4, out_file='tree4.dot',
                          feature_names=X.columns,
                          filled=True, rounded = True, 
                          special_characters= True,
             class_names=['0','1']  )
graph = graphviz.Source(dot_data)


# In[ ]:


os.system('dot -Tpng tree4.dot -o tree4.png')


# In[ ]:


from IPython.display import Image
Image(filename="tree4.png", height=1000, width=1000)


#  Decision Tree classifier accuracy is 100%, this is the problem of overfitting-->
#     
#  this problem salved by K-Fold.    

# In[ ]:





# In[ ]:




