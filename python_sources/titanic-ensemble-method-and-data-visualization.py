#!/usr/bin/env python
# coding: utf-8

# Hello everyone, for this kernel I porpose a simple emsemble method to improve the prediction of who survive or die in Titanic incident
# 
# These are the steps for this work
# 
# 1. **Cleaning and Standalize data**
# 2. **Data Exploration/Visualization**
# 3. **Use Machine Learning methods to predict the target value in data**
# 4. **Put all methods together to get a better prediction**
# 
# So let's get start!

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np

from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

np.random.seed = 5


# In[ ]:


data = pd.read_csv("../input/titanic/train.csv")
data.head()


# **We are going use a method from Pandas called .info() to verify if our dataset have null values, or if some atribute in dataset is important for our model or not.**

# In[ ]:


data.info()


# > <h3>1. **Cleaning and Standalize data**</h3>
# 
# For cleaning the data we don't use the PassengerId, Cabin, Ticket, Name because they don't have relevance for our model and we delete rows in Embarked that have NULL values

# In[ ]:


train_dataset = data

train_dataset.drop("PassengerId", axis=1, inplace=True)
train_dataset.drop("Cabin", axis=1, inplace=True)
train_dataset.drop("Name", axis=1, inplace=True)
train_dataset.drop("Ticket", axis=1, inplace=True)

train_dataset.head()


# For standalize the data we use a simple arithmetic mean in the Age atributte to substitute NULL values in the dataset

# In[ ]:


train_dataset.replace(to_replace = np.nan, value = round(train_dataset.mean(), 0), inplace=True)


# In[ ]:


train_dataset.fillna(method ='pad', inplace=True)


# In[ ]:


train_dataset.info()


# In[ ]:


train_dataset.head()


# In[ ]:


x_train = train_dataset.iloc[:, 1:7]
y_train = train_dataset.iloc[:, 0]


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# In[ ]:


x_train.head()


# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')

x_test = test

x_test.drop("PassengerId", axis=1, inplace=True)
x_test.drop("Cabin", axis=1, inplace=True)
x_test.drop("Name", axis=1, inplace=True)
x_test.drop("Ticket", axis=1, inplace=True)


x_test.replace(to_replace = np.nan, value = round(train_dataset.mean(), 0), inplace=True)
x_test.fillna(method ='pad', inplace=True)

le = LabelEncoder()

le.fit(x_test['Sex'])
x_test["Sex"] = le.transform(x_test['Sex'])


# In[ ]:


le = LabelEncoder()
le.fit(x_test['Embarked'])
x_test["Embarked"] = le.transform(x_test['Embarked'])


# > <h3>2. **Data Exploration/Visualization**</h3>
# 
# For visualization, we're going see socio-economic aspects and the difference between mens and womens that survived in incident

# In[ ]:


sns.heatmap(train_dataset[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(16,12),ncols=2)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_dataset, ax = ax[0]);
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=x_test, ax = ax[1]);
ax1.set_title("Training Set", fontsize = 18)
ax2.set_title('Test Set',  fontsize = 18)


# In[ ]:


pal = {'male':"green", 'female':"Pink"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
                 y = "Survived", 
                 data=train_dataset, 
                 palette = pal,
                 linewidth=5,
                 order = ['female','male'],
                 capsize = .05,
                )

plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25,loc = 'center', pad = 40)
plt.ylabel("% of passenger survived", fontsize = 15, )
plt.xlabel("Sex",fontsize = 15);


# In[ ]:


plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=train_dataset, 
            linewidth=5,
            capsize = .1

           )
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25, pad=40)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right. 
plt.xticks(val, labels);


# In[ ]:


le = LabelEncoder()
le.fit(x_train['Sex'])
x_train["Sex"] = le.transform(x_train['Sex'])


# In[ ]:


x_train.head()


# In[ ]:


y_test = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


x_test.drop("Embarked", axis=1, inplace=True)


# In[ ]:


scale = MinMaxScaler(feature_range=(0, 1))
scale.fit(x_train)
scaled_train = scale.transform(x_train)
scaled_train


# In[ ]:


scale = MinMaxScaler(feature_range=(0, 1))
scale.fit(x_test)
scaled_test = scale.transform(x_test)
scaled_test


# <h3>3. **Use Machine Learning methods to predict the target value in data**</h3>
# 
# 1 - K-Nearest Neightboors
# 2 - Decision Tree
# 3 - Multi Layer Perceptron
# 4 - SVM
# 5 - Naive Bayes
# 
# 

# In[ ]:


clf_knn = KNeighborsClassifier()
clf_tree = DecisionTreeClassifier()
clf_mlp = MLPClassifier()
clf_svm = SVC()
clf_naive = GaussianNB()


# In[ ]:


clf_knn.fit(scaled_train, y_train)
prediction_knn = clf_knn.predict(scaled_test)
print("Accuracy {0:.2f}%".format(accuracy_score(y_test['Survived'], prediction_knn) * 100))
print("=================================")
print(classification_report(y_test['Survived'], prediction_knn))
plot_confusion_matrix(conf_mat=confusion_matrix(y_target=y_test['Survived'], 
                      y_predicted=prediction_knn))


# In[ ]:


clf_tree.fit(scaled_train, y_train)
prediction_tree = clf_tree.predict(scaled_test)
print("Accuracy {0:.2f}%".format(accuracy_score(y_test['Survived'], prediction_tree) * 100))
print("=================================")
print(classification_report(y_test['Survived'], prediction_tree))
plot_confusion_matrix(conf_mat=confusion_matrix(y_target=y_test['Survived'], 
                      y_predicted=prediction_tree))


# In[ ]:


clf_mlp.fit(scaled_train, y_train)
prediction_mlp = clf_mlp.predict(scaled_test)
print("Accuracy {0:.2f}%".format(accuracy_score(y_test['Survived'], prediction_mlp) * 100))
print("=================================")
print(classification_report(y_test['Survived'], prediction_mlp))
plot_confusion_matrix(conf_mat=confusion_matrix(y_target=y_test['Survived'], 
                      y_predicted=prediction_mlp))


# In[ ]:


clf_svm.fit(scaled_train, y_train)
prediction_svm = clf_svm.predict(scaled_test)
print("Accuracy {0:.2f}%".format(accuracy_score(y_test['Survived'], prediction_svm) * 100))
print("=================================")
print(classification_report(y_test['Survived'], prediction_svm))
plot_confusion_matrix(conf_mat=confusion_matrix(y_target=y_test['Survived'], 
                      y_predicted=prediction_svm))


# In[ ]:


clf_naive.fit(scaled_train, y_train)
prediction_naive = clf_naive.predict(scaled_test)
print("Accuracy {0:.2f}%".format(accuracy_score(y_test['Survived'], prediction_naive) * 100))
print("=================================")
print(classification_report(y_test['Survived'], prediction_naive))
plot_confusion_matrix(conf_mat=confusion_matrix(y_target=y_test['Survived'], 
                      y_predicted=prediction_naive))


# <h3>4. **Put better methods together to get a better prediction**</h3>

# In[ ]:


ensemble = VotingClassifier(estimators=[('MLP', clf_mlp), ('SVM', clf_svm), ('Naive', clf_naive)], voting='hard')


# In[ ]:


ensemble.fit(scaled_train, y_train)


# In[ ]:


prediction_ensemble = ensemble.predict(scaled_test)


# In[ ]:


print("Accuracy {0:.2f}%".format(accuracy_score(y_test['Survived'], prediction_ensemble) * 100))
print("=================================")
print(classification_report(y_test['Survived'], prediction_ensemble))
plot_confusion_matrix(conf_mat=confusion_matrix(y_target=y_test['Survived'], 
                      y_predicted=prediction_ensemble))


# In[ ]:


submission = pd.DataFrame({ 'PassengerId': y_test['PassengerId'],
                            'Survived': prediction_ensemble })
submission.to_csv("submission.csv", index=False)


# In[ ]:




