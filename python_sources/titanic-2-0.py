#!/usr/bin/env python
# coding: utf-8

# This is my first practise machine learning problem.
# Start date: 2020/06/05

# In[ ]:



# all import statements
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling 

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# # Understand data
# First take a statistical glance of data
# 

# In[ ]:


# read the data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_x = train_data.loc[:, train_data.columns != "Survived"]
train_y = train_data.loc[:, train_data.columns == "Survived"]
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_x = test_data
full_data = train_x.append(test_x)
print ("full data shape",full_data.shape)
print ("train_x shape",train_x.shape)


# In[ ]:


full_data[full_data['Age'].isnull()]


# In[ ]:


ftest_pid = test_data["PassengerId"]
ftest_pid


# In[ ]:


test_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


test_data.describe()


# In[ ]:


# import warnings
# warnings.filterwarnings('ignore')
# profile = pandas_profiling.ProfileReport(train_data)
# profile


# In[ ]:


# heat map of features
plt.rcParams['figure.figsize'] = (7, 7)
plt.style.use('ggplot')

sns.heatmap(train_data.corr(), annot = True, cmap = 'Wistia')
plt.title('Heatmap for the Dataset', fontsize = 20)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15,5)
sns.distplot(train_data['Fare'], kde=False, rug=True, bins=90)
plt.title('Distribution of fare', fontsize=20)
plt.show()


# In[ ]:


plt.rcParams['figure.figsize'] = (15,5)
sns.distplot(train_data['Age'], kde=False, rug=True)
plt.title('Distribution of age', fontsize=20)
plt.show()


# Below boxplot tells us distribution of data with respect to target.

# In[ ]:


# box plot of fare
plt.rcParams['figure.figsize'] = (15,8)
sns.boxplot(train_data['Survived'], full_data['Fare'])
plt.title('box plot of fare with target')
plt.show()


# Above boxplot shows us:
# * survived -> paid high fare
# * Lot of outliers 

# In[ ]:


# boxen plot of Age
plt.rcParams['figure.figsize'] = (15,8)
sns.boxenplot(train_data['Survived'], full_data['Age'])
plt.title('boxen plot of age with target')
plt.show()


# In[ ]:


# pair plot of different feature
tmp_df = train_data[['Survived', 'Fare', 'Age']]
sns.pairplot(tmp_df)
plt.show()


#  For categorical data, 
#  * we can use swarm plot
#  * box plots with hue

# In[ ]:



# bar graph with hue
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train_data)
plt.show()


# In[ ]:


sns.catplot(x="Survived", y="Fare", hue="Sex", kind="swarm", data=train_data)
plt.show()


# In[ ]:


# sns.relplot(x="SibSp", y="Ticket", hue="Survived", data=train_data)
# plt.show()


# In[ ]:


full_data.iloc[891:895,:]


#  # Fill the missing data 
# Use sklearn.SimpleImputer to fill the missing values with (mean, median, mode) of the data <br>
# SimpleImputer: Uses single column values to calculate that column NaN's  <br>
# IterativeImputer: Uses entire set of feature dimentions to calculate the missing value. <br>
# NOTE: we can also use imputer for categorical (text) based features

# In[ ]:


# simpleImpute of Age feature
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy='mean')
imp = imp.fit(train_data[["Age"]])
# full_data["Age"] = pd.DataFrame(imp.transform(full_data[["Age"]]))
# test_data["Age"] = pd.DataFrame(imp.transform(test_data[["Age"]]))
# sns.distplot(full_data["Age"], kde=False, rug=True)
# plt.show()
tmp_age = pd.DataFrame(imp.transform(full_data[["Age"]]))
tmp_age = tmp_age.rename(columns={0:"Age"})
full_data.drop(['Age'],axis=1,inplace=True)


# In[ ]:


full_data.iloc[415:420,:]


# In[ ]:


result = pd.concat([full_data, tmp_age], axis=1, join='inner')


# In[ ]:


full_data = result
full_data


# In[ ]:


# # IterativeImpute
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# imp = IterativeImputer(max_iter=10, random_state=0)
# imp.fit(train_x[["Age"]])
# tmp_age1 = pd.DataFrame(imp.transform(train_data[["Age"]]))
# tmp_age2 = pd.DataFrame(imp.transform(test_data[["Age"]]))
# sns.distplot(tmp_age1, kde=False, rug=True)
# plt.show()


# # Drop any unnessasary columns
# useless columns such as name, id, etc. 
# These columns doesn't help ML algo in any way.<br>
# Here, Parch and Ticket feature has a lot in common. Both 76% filled by single passengers, 14%

# In[ ]:


full_data=full_data.drop(columns=["PassengerId","Name", "Cabin", "Ticket"])
full_data.columns


# # Transform cat values in to new features
# This transformation enables ml algorithms to better understand the data <br>
# From now on use full_data

# In[ ]:


# using one hot encoder from pandas.get_dummies
onehot_embark = pd.get_dummies(full_data["Embarked"], prefix="Embark")
onehot_pclass = pd.get_dummies(full_data["Pclass"], prefix="Pclass")
# full_data.drop(columns=["Embarked","Pclass"],axis=1,inplace=True)
# full_data = pd.concat([full_data, onehot_embark, onehot_pclass], axis=1,join='inner')
# full_data


# In[ ]:


result = pd.concat([full_data, onehot_embark, onehot_pclass], axis=1,join='inner')
result


# In[ ]:


full_data = result.copy()


# In[ ]:


full_data.drop(columns=["Embarked","Pclass"],axis=1,inplace=True)
full_data


# In[ ]:


# modify cat string values of Sex column to int values
full_data.loc[full_data.Sex=="male","Sex"] = 0
full_data.loc[full_data.Sex=="female", "Sex"] = 1
full_data


# # Normalize the data
# Normalize all the feature vectors. Here we used standard normalization <br>
# If we don't normalize, we get this result
# >        Training Accuracy : 0.9839486356340289
#         Testing Accuracy : 0.8283582089552238
#                       precision    recall  f1-score   support
#                     0       0.84      0.89      0.86       164
#                    1       0.81      0.73      0.77       104
#             accuracy                           0.83       268
#            macro avg       0.82      0.81      0.82       268
#         weighted avg       0.83      0.83      0.83       268
#         
# If we normalize, we get following result
# 
# >             Training Accuracy : 0.9839486356340289
#             Testing Accuracy : 0.8022388059701493
#                           precision    recall  f1-score   support
#                        0       0.84      0.84      0.84       167
#                        1       0.74      0.74      0.74       101
#                 accuracy                           0.80       268
#                macro avg       0.79      0.79      0.79       268
#             weighted avg       0.80      0.80      0.80       268
# 
# There is lot of overfitting going on. 
# 
# 

# In[ ]:


tmp_full_data = full_data.copy()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
tmp_age = pd.DataFrame(tmp_full_data["Age"][0:891])
full_age_0 = pd.DataFrame(tmp_full_data["Age"][:])
scaler.fit(tmp_age)
full_age_1 = pd.DataFrame(scaler.transform(full_age_0))
full_age_1 = full_age_1.rename(columns={0:"Age"})
full_age_1


# In[ ]:


sns.distplot(full_age_0)
plt.show()


# In[ ]:


sns.distplot(full_age_1)
plt.show()


# In[ ]:


full_data.drop(columns=["Age"], axis=1, inplace=True)
full_data


# In[ ]:


full_data = pd.concat([full_data, full_age_1], axis=1, join="inner")
full_data


# In[ ]:


tmp_full_data = full_data.copy()


# In[ ]:


scaler = StandardScaler()
tmp_fare = pd.DataFrame(tmp_full_data[["Fare","SibSp","Parch"]][0:891])
tmp_fare
full_fare_0 = pd.DataFrame(tmp_full_data[["Fare","SibSp","Parch"]][:])
scaler.fit(tmp_fare)
full_fare_1 = pd.DataFrame(scaler.transform(full_fare_0))
full_fare_1 = full_fare_1.rename(columns={0:"Fare",1:"SibSp",2:"Parch"})
full_fare_1


# In[ ]:


full_data.drop(columns=["Fare","SibSp","Parch"], axis=1, inplace=True)
full_data = pd.concat([full_data, full_fare_1], axis=1, join="inner")
full_data


# In[ ]:


full_data_copy = full_data.copy()
full_data.describe()


# # split the data in to train, validation, test set
# Out of all the train data split it into 60%train, 20%test, 20%test. <br>
# but here we dont have much train data and it's my first practise. so Train-> 70%train, 30%Test.

# In[ ]:


from sklearn.model_selection import train_test_split
#following is the final test
x_ftest = full_data.iloc[891:,:]
print ("shape of x_ftest",x_ftest.shape)
# (418,11)
xdf = pd.DataFrame(full_data.iloc[:891,:])
ydf = train_y
x_train,x_test,y_train, y_test = train_test_split(xdf,ydf,test_size=0.25,random_state=None)
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_test :", y_test.shape)


# # Modelling 
# Here we need to try out different ML models and analyze their performance.
# 

# In[ ]:


# Random forests is well known for classification problems.
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# model = RandomForestClassifier(n_estimators=100)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)

# # evaluating the model
# print("Training Accuracy :", model.score(x_train, y_train))
# print("Testing Accuracy :", model.score(x_test, y_test))

# # cofusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.rcParams['figure.figsize'] = (5, 5)
# sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

# # classification report
# cr = classification_report(y_test, y_pred)
# print(cr)


# In[ ]:


# random forest algorithm is overfitting. So, lets use logistic regression with regularization.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = model.predict(x_test)
# evaluating the model
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# cofusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, annot_kws = {'size':15}, cmap = 'PuBu')

# classification report
cr = classification_report(y_test, y_pred)
print(cr)


# In[ ]:


y_fpred = pd.DataFrame(model.predict(x_ftest))
y_fpred = y_fpred.rename(columns={0:"Survived"})
y_fpred


# In[ ]:


fresult = pd.concat([ftest_pid, y_fpred],axis=1,join='inner')


# In[ ]:


fresult


# In[ ]:


fresult.to_csv("result_with_norm_log_reg.csv", index=False)


# In[ ]:




