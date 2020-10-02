#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[118]:


train_data = pd.read_csv("../input/train.csv")
# first five train_data samples
train_data.head()


# In[119]:


# summary of the training dataset
train_data.info()


# In[120]:


# changing the datatype of the Pclass column
train_data['Pclass'] = train_data['Pclass'].astype('object')


# In[121]:


train_data.isna().sum()


# In[122]:


# filling the nan values in the age column with mean age
train_data['Age'] = train_data['Age'].fillna(round(train_data['Age'].mean()))
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])


# In[123]:


survival = pd.crosstab(train_data['Survived'], train_data['Sex'], normalize = True)
print(survival)
survival.plot(kind = 'Bar', stacked = True)


# From the above plot we can say that about 40% of the pasengers survived the accident. Out of the 40%, about 26% were females and the rest were males.

# ### Survival by class

# In[124]:


sns.countplot(y = "Pclass", hue = "Survived", data = train_data)


# ### Survival on the place embarked

# In[125]:


sns.countplot(y= 'Embarked', hue = "Survived", data = train_data)


# ### Relationship between different numeric data

# In[126]:


numerical = list(set(['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']))
corr_matrix = train_data[numerical].corr()
sns.heatmap(corr_matrix, annot = True)


# # Categorizing the training data into feature and target variable

# In[127]:


X = train_data.drop(["Survived", "Name", "Cabin", "Ticket", 'Sex', 'Embarked'], axis = 1)
embarked_sex = train_data[['Sex', 'Embarked']]
embarked_sex = pd.get_dummies(embarked_sex, prefix = ["Sex", "Embarked"])
y = train_data.Survived
X = pd.concat([X,embarked_sex], axis = 1)
X.head()


# ## Dividing the training data furthur into training and testing data

# In[128]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.40)


# ### Using the Random Forest Classification

# In[129]:


my_model1 = RandomForestClassifier(random_state = 1)
pipeline1 = make_pipeline(my_model1)
pipeline1.fit(X_train, y_train)
y_val1 = pipeline1.predict(X_test)
print(accuracy_score(y_test, y_val1))


# In[130]:


rf_model_on_full_data = RandomForestClassifier(random_state = 1)
rf_model_on_full_data.fit(X,y)


# In[131]:


test_data = pd.read_csv("../input/test.csv")
test_data = test_data.drop(["Name", "Cabin", "Ticket"], axis = 1)
test_data = pd.get_dummies(test_data, prefix = ["Sex", "Embarked"])
test_data.head()


# In[132]:


test_data.isna().sum()


# In[133]:


test_data['Age'] = test_data['Age'].fillna(round(test_data.Age.mean()))
test_data['Fare'] = test_data['Fare'].fillna(round(test_data.Fare.mean()))


# In[134]:


test_predictions1 = rf_model_on_full_data.predict(test_data)


# In[135]:


output1 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions1})
output1.to_csv("submission1.csv", index = False)


# ### Using KNN Classification

# In[136]:


my_scalar = StandardScaler()
my_model2 = KNeighborsClassifier(n_neighbors = 10, algorithm = 'ball_tree',
                              leaf_size = 20, weights = 'uniform', metric = 'manhattan')
pipeline2= make_pipeline(my_scalar, my_model2)
pipeline2.fit(X_train, y_train)
y_val2 = pipeline2.predict(X_test)
print(accuracy_score(y_test, y_val2))


# In[137]:


knn_on_full_model = KNeighborsClassifier(n_neighbors = 10, algorithm = 'ball_tree',
                              leaf_size = 20, weights = 'uniform', metric = 'manhattan')
knn_pipe = make_pipeline(my_scalar, knn_on_full_model)
knn_pipe.fit(X, y)


# In[138]:


test_predictions2 = knn_pipe.predict(test_data)


# In[139]:


output2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions2})
output2.to_csv("submission2.csv", index = False)


# ### Using decision trees

# In[140]:


my_model3 = DecisionTreeClassifier(criterion = 'gini', random_state = 17, max_depth = 6)
pipeline3 = make_pipeline(my_model3)
pipeline3.fit(X_train, y_train)
y_val3 = pipeline3.predict(X_test)
print(accuracy_score(y_test, y_val3))


# In[141]:


tree_param = {'max_depth': range(1,11),
             'max_features': range(4,8)}
tree_grid = GridSearchCV(my_model3, tree_param, cv = 5, n_jobs = -1, verbose = True)
tree_grid.fit(X_train, y_train)
print(tree_grid.best_params_)
print(tree_grid.best_score_)
print(accuracy_score(y_test, tree_grid.predict(X_test)))


# In[142]:


tree_model_on_full_data = DecisionTreeClassifier(criterion = 'gini',random_state = 1)
tree_param = {'max_depth': range(1,11),
             'max_features': range(4,8)}
tree_grid = GridSearchCV(tree_model_on_full_data, tree_param, cv = 5, n_jobs = -1, verbose = True)
tree_grid.fit(X, y)
print(tree_grid.best_params_)
print(tree_grid.best_score_)


# In[143]:


test_predictions3 = tree_grid.predict(test_data)


# In[144]:


output3 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions3})
output3.to_csv("submission3.csv", index = False)

