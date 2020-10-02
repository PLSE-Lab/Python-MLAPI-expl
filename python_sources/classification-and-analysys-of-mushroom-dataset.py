#!/usr/bin/env python
# coding: utf-8

# >> **IMPORTING THE REQUIRED MODULES **

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#Loading the dataset..
df = pd.read_csv('../input/mushroom.csv')


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


# converting the data from categorical to ordinal ..
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])


# In[ ]:


#checking the information of the dataset......
df.info()


# In[ ]:


#dropping the column "veil-type" is 0 
df=df.drop(["veil-type"],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


#Question a:
plt.figure()
pd.Series(df['edible']).value_counts().sort_index().plot(kind = 'bar')
plt.ylabel("Count")
plt.xlabel("edible")
plt.title('Number of poisonous/edible mushrooms (0=edible, 1=poisonous)');


# So we can observe fromthe above graph that the classification of different type of mushrooms (Edible vs non edible ), not let us check the chances of survival from this graph.

# In[ ]:


plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1, annot=True)
plt.yticks(rotation=0);


# This is a heatmap which shows the correlation between the variables 

# In[ ]:


df.corr()


# **Based on the above data visualization, we can use machine learning on the dataset because the dataset becomes balanced now **

# > Let us use the Random Forest Classifier on this data to check the accuracy 

# **Usually the least correlating variable is the most important one for classification. In this case, "gill-color" has -0.53 so let's look at it closely:**

# In[ ]:


df[['edible', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='edible', ascending=False)


# In[ ]:


#Looking closely at the feature 'gill-color'
new_var=df[['edible', 'gill-color']]
new_var=new_var[new_var['gill-color']<=3.5]
sns.factorplot('edible', col='gill-color', data=new_var, kind='count', size=2.5, aspect=.8, col_wrap=4);


# In[ ]:


new_var=df[['edible', 'gill-color']]
new_var=new_var[new_var['gill-color']>3.5]

sns.factorplot('edible', col='gill-color', data=new_var, kind='count', size=2.5, aspect=.8, col_wrap=4);


# I would prefer using the Random Forest classifier as well as the Decesion Tree Models to get the prediction because trees are easy to understand .

# # Splitting the data into test and train (Using Random forest classifier)

# In[ ]:


X=df.drop(['edible'], axis=1)
Y=df['edible']


# In[ ]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)


# In[ ]:


# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(X_train, Y_train)

# Print the score of the fitted random forest
print(my_forest.score(X, Y)*100)
acc_randomforest=(my_forest.score(X, Y)*100)


# # Confusion matrix

# In[ ]:


def plot_confusion_matrix(df, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df, cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plot_confusion_matrix(df)


# >> So we have a good model that has 100% accuracy in predicting if a person will survive or not.

# # Let us use a decision tree classifier for the same dataset to compare between the two machine learning models 

# In[ ]:


clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)


# > By all methods examined before the feature that is most important is "gill-color".

# In[ ]:


features_list = X.columns.values
feature_importance = clf.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(5,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()


# In[ ]:


X=df.drop(['edible'], axis=1)
Y=df['edible']
y_pred=clf.predict(X_test)


# In[ ]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.1)


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred7 = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree


# In[ ]:


cfm=confusion_matrix(Y_test, y_pred)

sns.heatmap(cfm, annot = True,  linewidths=.5, cbar =None)
plt.title('Decision Tree Classifier confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label');


# # So we can see that Using Both the models the accuracy is 100% 

# # Using KNN 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=3, n_neighbors=10, p=2, 
                           weights='uniform')
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_test)
acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
acc_knn


# In[ ]:


objects = ('Decision Tree', 'Random Forest','KNN Model ')
x_pos = np.arange(len(objects))
accuracies1 = [acc_decision_tree, acc_randomforest, acc_knn]
    
plt.bar(x_pos, accuracies1, align='center', alpha=0.5, color='b')
plt.xticks(x_pos, objects, rotation='vertical')
plt.ylabel('Accuracy')
plt.title('Classifier Outcome')
plt.show()


# >>** So if my friend who is a data analalyst as well as a Machine Learning enthu , says that using neural network with 3 layers containg 3 layers with 10 nodes each will give me a better accuracy then as we can see above that using both decision tree classifier as well as random forest models we get the accuracy as 100 % and using the KNN model the accuracy is 99.75 %**

# In[ ]:





# In[ ]:




