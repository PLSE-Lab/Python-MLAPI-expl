#!/usr/bin/env python
# coding: utf-8

# ## This kernel demonstrates the impact of decision tree depth on the classification accuracy
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
warnings.filterwarnings('ignore')
import graphviz
from IPython.display import Image  
from subprocess import call
from sklearn.metrics import accuracy_score
get_ipython().system('pip install pydotplus')
import pydotplus


# > ## EDA and data preparation
# 
# ### Load data
# 

# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv', sep=',')
X_test_data = pd.read_csv('../input/titanic/test.csv', sep=',')
y_test_data = pd.read_csv('../input/titanic/gender_submission.csv', sep=',')


# In[ ]:


train_data.head()


# ### Missing data filling
# (Reference: https://www.kaggle.com/mknorps/titanic-with-decision-trees)

# In[ ]:


train_data.isnull().sum()


# In[ ]:


X_test_data.isnull().sum()


# In[ ]:


# filling NaN in "Embarked" and "Fare"

train_data['Embarked'].fillna(value='S',inplace=True) # S is most popular value 
mean_Fare = train_data["Fare"].mean()
train_data['Fare'].fillna(value=mean_Fare,inplace=True)
    
X_test_data['Embarked'].fillna(value='S',inplace=True) # S is most popular value 
mean_Fare = X_test_data['Fare'].mean()
X_test_data['Fare'].fillna(value=mean_Fare,inplace=True)


# In[ ]:


from random import choices
# filling NaN in "Age" 
x = train_data['Age'].dropna()
hist, bins = np.histogram( x,bins=15)

bin_centers = 0.5*(bins[:len(bins)-1]+bins[1:])
probabilities = hist/hist.sum()

#dictionary with random numbers from existing age distribution
train_data['Age_rand'] = train_data['Age'].apply(lambda v: np.random.choice(bin_centers, p=probabilities))
Age_null_list = train_data[train_data['Age'].isnull()].index
train_data.loc[Age_null_list,'Age'] = train_data.loc[Age_null_list,'Age_rand']
    
# filling NaN in "Age" 
x = X_test_data['Age'].dropna()
hist, bins = np.histogram( x,bins=15)

bin_centers = 0.5*(bins[:len(bins)-1]+bins[1:])
probabilities = hist/hist.sum()

#dictionary with random numbers from existing age distribution
X_test_data['Age_rand'] = X_test_data['Age'].apply(lambda v: np.random.choice(bin_centers, p=probabilities))
Age_null_list = X_test_data[X_test_data['Age'].isnull()].index
X_test_data.loc[Age_null_list,'Age'] = X_test_data.loc[Age_null_list,'Age_rand']


# In[ ]:


# Gender
genders = {'male': 1, 'female': 0}
train_data['Sex'] = train_data['Sex'].apply(lambda s: genders.get(s))
X_test_data['Sex'] = X_test_data['Sex'].apply(lambda s: genders.get(s))


# In[ ]:


# Embarkment
embarkments = {'U': 0, 'S': 1, 'C': 2, 'Q': 3}
train_data['Embarked'] = train_data['Embarked'].apply(lambda e: embarkments.get(e))
X_test_data['Embarked'] = X_test_data['Embarked'].apply(lambda e: embarkments.get(e))


# In[ ]:


plt.subplots(figsize = (10,10))
data = train_data.loc[:,['Survived','Pclass', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Age', 'Sex']]
sns.heatmap(data.corr(),
            annot=True,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.show()


# In[ ]:


train_data, valid_data = train_test_split(train_data, test_size=0.2)


# In[ ]:


X_train = train_data.loc[:,['Pclass', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Age', 'Sex']]
X_valid = valid_data.loc[:,['Pclass', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Age', 'Sex']]
X_test = X_test_data.loc[:,['Pclass', 'SibSp', 'Parch', 'Embarked', 'Fare', 'Age', 'Sex']]

y_train = train_data.loc[:,['Survived']]
y_valid = valid_data.loc[:,['Survived']]
y_test = y_test_data.loc[:,['Survived']]


# In[ ]:


X_train.describe()


# In[ ]:


X_valid.describe()


# In[ ]:


X_test.describe()


# ## Lets build 10 models with different depth and compare the results

# In[ ]:


X_train_matrix = X_train.to_numpy()
X_valid_matrix = X_valid.to_numpy()
X_test_matrix = X_test.to_numpy()
y_train_matrix = y_train.to_numpy()
y_valid_matrix = y_valid.to_numpy()
y_test_matrix = y_test.to_numpy()


# In[ ]:


acc_train_array = []
acc_valid_array = []
for depth in range(1,11):
    decision_tree = tree.DecisionTreeClassifier(max_depth = depth)
    decision_tree = decision_tree.fit(X_train_matrix, y_train_matrix)
    y_pred = decision_tree.predict(X_train_matrix)
    acc_train = accuracy_score(y_train_matrix, y_pred)
    acc_train_array.append(acc_train)
    y_pred = decision_tree.predict(X_valid_matrix)
    acc_valid = accuracy_score(y_valid_matrix, y_pred)
    acc_valid_array.append(acc_valid)
    dot_data = tree.export_graphviz(decision_tree, 
                                out_file=None,
                                max_depth = depth,
                                filled=True, 
                                rounded=True,                                
                                special_characters=True,
                                class_names = ['Died', 'Survived'],
                                feature_names = X_train.columns.values)
    pydot_graph = pydotplus.graph_from_dot_data(dot_data)
    pydot_graph.write_png('../working/tree_depth_' + str(depth) + '.png')
fig, ax = plt.subplots(figsize = (12,6))
print('For train set:')
print(acc_train_array)
print('For valid set:')
print(acc_valid_array)
ax.set_xlabel('depth')
ax.set_ylabel('accuracy')
ax.plot(range(1,11), acc_train_array)
ax.plot(range(1,11), acc_valid_array)
plt.show()


# In[ ]:


print('{:^10}{:^20}{:^20}'.format('depth','train accuracy','valid accuracy'))
for i in range(10):
    print('{:^10}{:^20.5}{:^20.5}'.format(str(i+1), str(acc_train_array[i]), str(acc_valid_array[i])))


# ### As we see, starting from depth 6 the model is overfited and model the depth 5 shows the best accuracy on the validation dataset.

# In[ ]:


Image('../working/tree_depth_1.png')


# In[ ]:


Image('../working/tree_depth_2.png')


# In[ ]:


Image('../working/tree_depth_3.png')


# In[ ]:


Image('../working/tree_depth_4.png')


# In[ ]:


Image('../working/tree_depth_5.png')


# In[ ]:


Image('../working/tree_depth_6.png')


# In[ ]:


Image('../working/tree_depth_7.png')


# In[ ]:


Image('../working/tree_depth_8.png')


# In[ ]:


Image('../working/tree_depth_9.png')


# In[ ]:


Image('../working/tree_depth_10.png')


# In[ ]:


decision_tree = tree.DecisionTreeClassifier(max_depth = 5)
decision_tree = decision_tree.fit(X_train_matrix, y_train_matrix)
# Predicting results for test dataset
y_pred = decision_tree.predict(X_test_matrix)
submission = pd.DataFrame({
        'PassengerId': [ int(x) for x in X_test_data.loc[:,['PassengerId']].to_numpy()],
        'Survived': y_pred
    })
submission.to_csv('../working/submission.csv', index=False)

