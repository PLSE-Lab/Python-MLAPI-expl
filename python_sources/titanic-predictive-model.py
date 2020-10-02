#!/usr/bin/env python
# coding: utf-8

# ## 1. Import libraries 

# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings ('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Read the data

# In[13]:


train= pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')


# In[14]:


train.head()


# ## 3. Exploratory analysis 
# ### Visualize the data
# * The majority of passengers were between 20 and 40 years old, traveling alone (no family, siblings, parents, kids) 
# * The ones who did not survive were predominately men, who were 3rd class passengers

# In[15]:


sns.distplot(train['Age'].dropna(), bins = 30)


# In[16]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))
sns.countplot(data=train, x='Survived', hue='Sex', ax=ax1)
sns.countplot(data=train, x='Survived', hue='Pclass', ax=ax2)


# In[17]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))
sns.countplot(data=train, x='Survived', hue='Parch', ax=ax1)
sns.countplot(data=train, x='Survived', hue='SibSp', ax=ax2)


# ### Explore the data
# In this section, we'll be exploring the data, looking for missing values, filling them in, discarding data that is redundant. 

# In[18]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Find missing values
total_miss_train = train.isnull().sum()
perc_miss_train = total_miss_train/train.isnull().count()*100
missing_data_train = pd.DataFrame(({'Total missing train':total_miss_train,
                            '% missing':perc_miss_train}))
missing_data_train.sort_values(by='Total missing train',ascending=False).head(2)


# For the missing 'Age' data, we will impute values based on the median value in each class, as shown below:

# In[19]:


sns.set(style="darkgrid")
plt.figure (figsize=(12,7))
sns.boxplot(data=train, y='Age', x='Pclass')


# In[20]:


def impute_age (cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else: return Age


# In[21]:


train['Age']= train[['Age', 'Pclass']].apply(impute_age, axis=1)


# The missing 'Embarked' values will be treated as a dummy variable, and so will 'Sex', while data that is redundant and cannot be of use for the model, such as 'PassengerId', 'Name', 'Cabin' and 'Ticket' will not be used as features, as shown furhter down. 

# In[22]:


sex=pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,sex,embark],axis=1)


# In[23]:


train.head()


# Similarly for the 'test' dataset, we impute the missing 'Age' values and work similarly for the other variables.

# In[24]:


sns.set(style="darkgrid")
plt.figure (figsize=(12,7))
sns.boxplot(data=test, y='Age', x='Pclass')


# In[25]:


def impute_age (cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return 42
        elif Pclass==2:
            return 27
        else:
            return 25
    else: return Age


# In[26]:


test['Age']= test[['Age', 'Pclass']].apply(impute_age, axis=1)


# In[27]:


sex=pd.get_dummies(test['Sex'], drop_first=True)
embark = pd.get_dummies(test['Embarked'], drop_first=True)
test = pd.concat([test,sex,embark],axis=1)


# In[28]:


test.head()


# The following variables will be used as features for the model that will predict for the target variable 'Survived'. 

# In[29]:


features = ['Pclass','Age','SibSp', 'Parch', 'Fare', 'male', 'Q', 'S']
target = ['Survived']


# In[30]:


# Check for missing values in the test dataset
train[features].isnull().sum()


# In[31]:


# Check for missing values in the test dataset
test[features].isnull().sum()


# In[32]:


# Fill them in with 0
test[features]=test[features].replace(np.NAN, 0)
#test["Survived"] = ""


# Indeed, the feautre variables seem be reasonably correlated with the target variable 'Survived', as shown from the correlation matrix below.

# In[33]:


data_correlation = train.corr()
mask = np.array(data_correlation)
mask[np.tril_indices_from(mask)] = False
fig = plt.subplots(figsize=(20,10))
sns.heatmap(data_correlation, mask=mask, vmax=1, square=True, annot=True)


# ## 4. Model Selection
# In this section, we test the performance of seven different classification on the training data: DecisionTreeClassifier, LogisticRegression, SGDClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier. 

# In[34]:


# Model Selection
from sklearn.model_selection import train_test_split

X = train[features]
y = train[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101)
#X_train = train[features]
#y_train = train[target]
#X_test = test[features]


# In more detail:
# * A 10-fold CV was chosen as resampling method, while the area under the Receiver Operating Characteristic curve (ROC_AUC) was chosen as a scoring criterion. The GradientBoostingClassifier was found to outperform the other models. 
# 
# * Other criteria (not presented here) were also tested, such as the F1 score, the Recall, the Precision and the Accuracy of the model. It was observed that the metrics F1 score, Recall and Precision demonstrated similar classification reports to the results obrained by the ROC_AUC cirterion, while Accuracy performed worse. This is to be expected, as Accuracy can in principle be misleading because it does not account for class imbalance in the 'test' dataset. 

# In[35]:


#Selection of algorithm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

models = [LogisticRegression(),
          SGDClassifier(),
          DecisionTreeClassifier(), 
          GradientBoostingClassifier(),
          RandomForestClassifier(),
          BaggingClassifier(),
          svm.SVC(),
          GaussianNB()]

def test_algorithms(model):
    kfold = model_selection.KFold(n_splits=10, random_state=101)
    predicted = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    print(predicted.mean())
    
for model in models:
    test_algorithms(model)


# ## 5. Hyperparameter tuning 

# Next, we tuned the hyperparameters based on the best performance obtained. A preliminary analysis is shown here for the learning rate, the number of estimators and the max depth. 

# In[36]:


from sklearn.metrics import roc_curve, auc
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
train_results = []
test_results = []
for eta in learning_rates:
   model = GradientBoostingClassifier(learning_rate=eta)
   model.fit(X_train, y_train)
   train_pred = model.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(learning_rates, train_results, 'b', label='Train AUC')
line2, = plt.plot(learning_rates, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.show()


# In[37]:


n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200, 300, 500]
train_results = []
test_results = []
for estimator in n_estimators:
   model = GradientBoostingClassifier(n_estimators=estimator)
   model.fit(X_train, y_train)
   train_pred = model.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# In[38]:


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   model = GradientBoostingClassifier(max_depth=max_depth)
   model.fit(X_train, y_train)
   train_pred = model.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = model.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# In what follows, parameter optimization is done automatically for a range of values chosen based on the aforemetioned preliminary analysis. The best parameters, along with the best score are printed.

# In[39]:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':[10,20,30,50, 100, 200, 300],'max_depth':[3,5,7,9,11,13,14], 'learning_rate': [0.1, 0.15, 0.2, 0.25, 0.3]}
grid_rf = GridSearchCV(GradientBoostingClassifier(),param_grid,cv=10,scoring='roc_auc').fit(X_train,y_train)
print('Best parameter: {}'.format(grid_rf.best_params_))
print('Best score: {:.2f}'.format((grid_rf.best_score_)))


# ## 6. Model Performance

# For 'learning_rate': 0.15, 'max_depth': 3, 'n_estimators': 20, we re-run the model and calculate the precision, recall, f1-score and the roc_auc of the 'test' dataset.

# In[40]:


gbc = GradientBoostingClassifier(max_depth=3, n_estimators=20,learning_rate=0.15)
gbc.fit(X_train,y_train)
y_pred = gbc.predict(X_test)


# In[41]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[42]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# As a next step, we plot the ROC_AUC of the model, which comes up to 86.3%.

# In[43]:


from sklearn.metrics import roc_curve, auc, roc_auc_score
# predict probabilities
probs = gbc.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()


# Next, we plot the learning curve, which shows the validation and training score of an estimator for varying numbers of training samples. It is a tool to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error. Since both the validation score and the training score converge to a value with increasing size of the training set, we will not benefit much from more training data.

# In[44]:


from sklearn.model_selection import learning_curve
train_sizes, train_scores, valid_scores = learning_curve(GradientBoostingClassifier(max_depth=3, n_estimators=20, learning_rate=0.15), X_train, y_train, cv=3, scoring='roc_auc')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure()
plt.plot(train_sizes,valid_scores_mean,label='valid')
plt.plot(train_sizes,train_scores_mean,label='train')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.3,color="g")
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,valid_scores_mean + valid_scores_std, alpha=0.3, color="b")
plt.xlabel('Number of samples')
plt.ylabel('ROC_AUC')
plt.legend()


# ## 7. Predictions

# As a final step, we use the 'test' dataseet that was provided to make predictions.

# In[45]:


gbc = GradientBoostingClassifier(max_depth=3, n_estimators=20,learning_rate=0.15)
gbc.fit(train[features],train[target])
y_pred = gbc.predict(test[features])


# In[46]:


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
submission.head()


# In[47]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# In[ ]:




