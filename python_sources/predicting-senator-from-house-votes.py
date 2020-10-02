#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.float_format = '{:.2f}'.format
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 14


# In[3]:


feature=  ['Class','handicapped-infants', 'water-project-cost-sharing', 
                    'adoption-of-the-budget-resolution', 'physician-fee-freeze',
                    'el-salvador-aid', 'religious-groups-in-schools',
                    'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
                    'mx-missle', 'immigration', 'synfuels-corporation-cutback',
                    'education-spending', 'superfund-right-to-sue', 'crime',
                    'duty-free-exports', 'export-administration-act-south-africa']

votes= pd.read_csv("../input/house-votes-84.data.txt",na_values=['?'], names=feature)


# In[4]:


# filling missing values with a = Did not vote

votes.fillna('a', inplace = True)

#def fillna(col):
 #   col.fillna(col.value_counts().index[0], inplace=True)
  #  return col
#votes=votes.apply(lambda col:fillna(col))


# In[5]:


votes.head()


# In[6]:



votes_original=votes.copy() 


# In[7]:


votes.columns, votes.shape


# In[8]:


# null values in attributes. 

votes.isnull().sum()


# In[9]:


# Print data types for each variable 
print(votes.dtypes)


# In[10]:


# assigning numerical values to categories

votes.replace(('a','n','y'), (0,-1,1), inplace=True)

votes.replace(('democrat', 'republican'), (1, 0), inplace=True)


# In[11]:


# Print data types for each variable 
print(votes.dtypes)


# In[12]:


votes.head()


# In[13]:


votes.describe()


# In[14]:


votes.shape


# In[15]:


# Correlation Matrix

corr = votes.corr()
corr


# # Strongly Negatively Correalated 
# 
# Class & Physican fee freeze (-0.91) <br>
# Class & el-salvador aid (-0.71) <br>
# Class & eduation spending (-0.69) <br>
# el-salvador aid & aid-to-nicaraguan-contras (-0.83) <br>
# el-salvador aid & mx-missle (-0.78) <br>
# el-salvador aid & anti-satellite-test-ban (-0.69) <br>
# aid-to-nicaraguan-contras & Physican fee freeze (-0.69) <br>
# 
# # Strongly Positively correlated
# 
# Class & adoption-of-the-budget-resolution (0.74) <br>
# adoption-of-the-budget-resolution & aid-to-nicaraguan-contras (0.7) <br>
# physician-fee-freeze & el-salvador aid (0.75) <br>
# physician-fee-freeze & crime (0.70) <br>
# anti-satellite-test-ban & aid-to-nicaraguan-contras (0.72) <br>
# aid-to-nicaraguan-contras & mx-missle (0.74) <br>
# education-spending &  Physical fee freeze (0.69) <br>

# In[16]:


# correlation matriix visualization 

f, ax = plt.subplots(figsize=(30, 18)) 
sns.heatmap(corr, vmax=1, square=True,annot=True, fmt=".2f")


# # Exploratory Data Analysis 

# In[17]:


sns.countplot(x='Class', data=votes)
plt.title('Class:Republican=0, Democrat =1')
votes['Class'].value_counts()


# In[86]:



plt.rcParams['figure.figsize'] = (20, 18)    
plt.subplot(2, 3, 1)
sns.countplot(votes['handicapped-infants'], color = 'violet')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 2)
sns.countplot(votes['water-project-cost-sharing'], color = 'blue')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 3)
sns.countplot(votes['adoption-of-the-budget-resolution'], color = 'green')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 4)
sns.countplot(votes['physician-fee-freeze'], color = 'red')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 5)
sns.countplot(votes['el-salvador-aid'], color = 'purple')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 6)
sns.countplot(votes['religious-groups-in-schools'], color = 'orange')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.show()


# In[87]:



plt.rcParams['figure.figsize'] = (20, 18)    
plt.subplot(2, 3, 1)
sns.countplot(votes['anti-satellite-test-ban'], color = 'violet')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 2)
sns.countplot(votes['aid-to-nicaraguan-contras'], color = 'blue')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 3)
sns.countplot(votes['mx-missle'], color = 'green')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 4)
sns.countplot(votes['immigration'], color = 'red')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 5)
sns.countplot(votes['synfuels-corporation-cutback'], color = 'purple')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 3, 6)
sns.countplot(votes['education-spending'], color = 'orange')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.show()


# In[88]:


plt.rcParams['figure.figsize'] = (15, 15)    
plt.subplot(2, 2, 1)
sns.countplot(votes['superfund-right-to-sue'], color = 'violet')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 2, 2)
sns.countplot(votes['crime'], color = 'blue')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 2, 3)
sns.countplot(votes['duty-free-exports'], color = 'green')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.subplot(2, 2, 4)
sns.countplot(votes['export-administration-act-south-africa'], color = 'red')
plt.title('No=-1,Yes=1,Undecided=0')
plt.xticks(rotation = 45)

plt.show()


# Attributes which do not add values:<b>
# 
# 
# water-project-cost-sharing,<br>
# el-salvador-aid<br>
# mx-missle <br>
# immigration<br>
# superfund-right-to-sue<br>

# In[89]:


print(votes.dtypes)


# # Deep Dive into Class voting

# In[147]:


plt.rcParams['figure.figsize'] = (20, 15) 

#plt.subplot(8, 2, 1)
sns.catplot(x='handicapped-infants', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 2)
sns.catplot(x='water-project-cost-sharing', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 3)
sns.catplot(x='adoption-of-the-budget-resolution', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 4)
sns.catplot(x='physician-fee-freeze', col='Class', kind='count', data=votes);


# In[148]:


plt.rcParams['figure.figsize'] = (20, 15) 
#plt.subplot(8, 2, 5)
sns.catplot(x='el-salvador-aid', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 6)
sns.catplot(x='religious-groups-in-schools', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 7)
sns.catplot(x='anti-satellite-test-ban', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 8)
sns.catplot(x='aid-to-nicaraguan-contras', col='Class', kind='count', data=votes);


# In[149]:


plt.rcParams['figure.figsize'] = (20, 15) 

#plt.subplot(8, 2, 9)
sns.catplot(x='mx-missle', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 10)
sns.catplot(x='immigration', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 11)
sns.catplot(x='synfuels-corporation-cutback', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 12)
sns.catplot(x='education-spending', col='Class', kind='count', data=votes);


# In[150]:


plt.rcParams['figure.figsize'] = (20, 15) 

#plt.subplot(8, 2, 13)
sns.catplot(x='crime', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 14)
sns.catplot(x='superfund-right-to-sue', col='Class', kind='count', data=votes);


#plt.subplot(8, 2, 15)
sns.catplot(x='duty-free-exports', col='Class', kind='count', data=votes);

#plt.subplot(8, 2, 16)
sns.catplot(x='export-administration-act-south-africa', col='Class', kind='count', data=votes);


# In[94]:


# dropping the attributes to avoid multicollinearty 

votes_drop = votes.drop(['physician-fee-freeze','el-salvador-aid','education-spending','aid-to-nicaraguan-contras','adoption-of-the-budget-resolution'], axis=1)

votes_drop.shape


# In[95]:


votes_drop.head()


# # Decision Tree 

# In[96]:


#x=votes.iloc[:, :1]
#y=votes.iloc[:, 1:12]

X = votes_drop.drop(['Class'], axis=1)
y = votes_drop["Class"]


# In[97]:


# Stratified sampling

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101,stratify=y)


# In[98]:


# Importing the packages for Decision Tree Classifier

from sklearn import tree
tree_one = tree.DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=101, min_samples_leaf=3, class_weight="balanced")  #, class_weight="balanced"
tree_one


# In[99]:


# Fitting the decision tree model on your features and label

tree_one = tree_one.fit(X_train, y_train)


# In[100]:


# The feature_importances_ attribute make it simple to interpret the significance of the predictors you include

list(zip(X_train.columns,tree_one.feature_importances_))


# We can see that 'Crime' is an important  attribute in the model while predicting whether senator is republican or democrat.

# In[101]:


# The accuracy of the model on Train data

print(tree_one.score(X_train, y_train))


# The accuracy of the model on Test data

print(tree_one.score(X_test, y_test))


# We can that model is not much overfitted beacuse the model variation is not beyond the limit of 5%.

# In[102]:


# Visualize the decision tree graph

with open('tree.dot','w') as dotfile:
    tree.export_graphviz(tree_one, out_file=dotfile, feature_names=X_train.columns, filled=True)
    dotfile.close()
    
    
from graphviz import Source

with open('tree.dot','r') as f:
    text=f.read()
    plot=Source(text)
plot


# # Decision Tree model prediction

# In[103]:


y_pred = tree_one.predict(X_test)


# # Evaluation of Decision Tree

# In[104]:


from sklearn.metrics import confusion_matrix, classification_report

df_confusion = confusion_matrix(y_test, y_pred)
df_confusion


# In[105]:


plt.rcParams['figure.figsize'] = (10, 6) 
cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion,cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,
            fmt='d')


# **Result:**
# 
# TrueNegative(TN) = 48 cases, which are republican and predicted as republican as well.
# 
# TruePositive(TP) = 70 cases, which are democrat and predicted as democrat as well.
# 
# FalseNegative(FN) = 8 cases, which are actually republican but predicted as democrat.
# 
# FalsePositive(FP) = 18 cases, which are actually democrat but predicted as republican.

# In[106]:



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Parameter Tuning 

# In[107]:


# Setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 101, class_weight='balanced')
tree_two = tree_two.fit(X_train, y_train)

#Print the score of both the decision tree

print("New Decision Tree Accuracy: ",tree_two.score(X_train, y_train))
print("Original Decision Tree Accuracy",tree_one.score(X_train,y_train))


# We have improved our model by fine tuning the hyperparameters. 

# In[108]:


# The accuracy of the model on Train data

print(tree_two.score(X_train, y_train))


# The accuracy of the model on Test data

print(tree_two.score(X_test, y_test))


# As we can see that the variation between the train and test data is significant. So, we can infer that our model is suffering from the overfitting. Though the accuracy score is better than the previous model (tree one_test: 0.816). It mainly because of the hyperparmater i.e. the increase in the max_depth, min_sample split due to which it become too much attached to training data and increases the level of complexity. Instead of being generic model it become more centric to specific conditions as we increases the number of levels. <br>

# In[109]:


# Building confusion matrix of our improved model
predict = tree_two.predict(X_test)
df_confusion_new = confusion_matrix(y_test, predict)
df_confusion_new


# In[110]:


cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion_new, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,
            fmt='d')


#  

# If we look our above model it is predicting (tree two, TP =74) more accureate than the previous model (tree one, TP =70). Therefore the recall and F1 score increased in this model compared to previous model. 

# In[111]:



from sklearn.metrics import classification_report
print(classification_report(y_test, predict))


# # Decision Tree with Grid Search

# In[112]:


# Different parameters we want to test

max_depth = [5,10,15] 
criterion = ['gini', 'entropy']
min_samples_split = [5,10,15]


# In[113]:


# Importing GridSearch

from sklearn.model_selection import GridSearchCV


# In[114]:


# Building the model

tree_three = tree.DecisionTreeClassifier(class_weight="balanced")

# Cross-validation tells how well a model performs on a dataset using multiple samples of train data
grid = GridSearchCV(estimator = tree_three, cv=3, 
                    param_grid = dict(max_depth = max_depth, criterion = criterion, min_samples_split=min_samples_split), verbose=2)


# In[115]:


grid.fit(X_train,y_train)


# In[116]:


# Best accuracy score

print('Avg accuracy score across 54 models:', grid.best_score_)


# In[117]:


# Best parameters for the model

grid.best_params_


# In[118]:


# Building the model based on new parameters

tree_three = tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 10, random_state=42, min_samples_split=5, class_weight="balanced")


# In[119]:


tree_three.fit(X_train,y_train)


# In[120]:


# Accuracy Score for new model

print ("DT_three accuracy Train:",tree_three.score(X_train,y_train))


# It is noticed that our accuracy score increase from 0.88 to 0.95 by following the best parameters generated by the grid search

# In[121]:


# The accuracy of the model on Test data

print("DT_three accuracy Test:",tree_two.score(X_test, y_test))


# The model is having a overfitting issue. 

# In[122]:


# Building confusion matrix of our improved model
pred_three = tree_three.predict(X_test)
df_confusion_three = confusion_matrix(y_test, pred_three)
df_confusion_three


# In[123]:


cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion_three, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,
            fmt='d')


# In[124]:



from sklearn.metrics import classification_report
print(classification_report(y_test, pred_three))


# In[125]:


test = pd.read_csv("../input/testing.csv")


# In[126]:


submission = pd.DataFrame({'id':test['id'],'predicted':pred_three})


# In[127]:



submission.to_csv("submission_DTgrid.csv", index=False)
submission.head()


# # Random Forest  

# In[128]:


# Building and fitting Random Forest

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(criterion = 'gini',  n_estimators = 100, max_depth = 10,random_state = 101, class_weight="balanced")


# In[129]:


rf_forest = forest.fit(X_train, y_train)


# In[130]:


# Print the accuracy score of the fitted random forest

print("RF Accuracy Train:", rf_forest.score(X_train, y_train))
print("RF Accuracy Test:", rf_forest.score(X_test, y_test))


# As we can see the model is suffering from overfitting. The main reason for this is having a 10 level of max depth, which seems to be huge and result in having a complex decision tree. We can reduce the overfitting by reducing the maxdepth.

# #  Random Forest Model Prediction

# In[131]:


# Making predictions

pred_rf = rf_forest.predict(X_test)


# In[132]:


list(zip(X_train.columns,rf_forest.feature_importances_))


# Similar to Decision tree the random forest also point toward 'crime' as a important attribute to predict the senator class i.e. democrat or republican.

# In[133]:


df_confusion_rf = confusion_matrix(y_test, pred_rf)
df_confusion_rf


# In[134]:


cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion_rf, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,
            fmt='d')


# In[135]:



from sklearn.metrics import classification_report
print(classification_report(y_test, pred_rf))


# In[136]:


submission = pd.DataFrame({'id':test['id'],'predicted':pred_rf})


# In[137]:



submission.to_csv("submission_RForest.csv", index=False)
submission.head()


# # Naive Bayes 

# In[138]:


from sklearn import naive_bayes

clf = naive_bayes.GaussianNB()
model=clf.fit(X_train, y_train)


# * The Gaussian method is consider because we are dealing with a classification problem and it assumes that features follow a normal distribution.

# In[139]:


# Print the accuracy score of the fitted random forest

print("NB Accuracy Train:", model.score(X_train, y_train))
print("NG Accuracy Test:", model.score(X_test, y_test))


# In[140]:


pred_NB=model.predict(X_test)
print(pred_NB)


# In[141]:


df_confusion_NB = confusion_matrix(y_test, pred_NB)
df_confusion_NB


# In[142]:


cmap = sns.cubehelix_palette(15, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(df_confusion_NB, cmap = cmap,xticklabels=['Prediction 0','Prediction 1'],yticklabels=['Actual 0','Actual 1'], annot=True,
            fmt='d')


# In[143]:



from sklearn.metrics import classification_report
print(classification_report(y_test, pred_NB))


# The F1score of Naive bayes in comparison with other model is slightly better (0.9), which means the naive bayes model correctly identify 90% positive results to get the true positive rate.

# In[144]:


submission = pd.DataFrame({'id':test['id'],'predicted':pred_NB})


# In[145]:



submission.to_csv("submission_NB.csv", index=False)
submission.head()


# # END
