#!/usr/bin/env python
# coding: utf-8

# # EARLY DIABETES PREDICTION : PIMA INDIAN DIABETES

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import Perceptron
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import os

print(os.listdir("../input"))


# In[ ]:


df=pd.read_csv("../input/diabetes.csv")


# In[ ]:


df.head()


# In[ ]:


df.isna().any() # No NAs


# In[ ]:


print(df.dtypes)


# In[ ]:


dataset2 = df.drop(columns = ['Outcome'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# As we have seen in above histograms there are zero entries in features **Glucose,BloodPressure,SkinThickness,Insulin and BMI**.So, we will fill it with median so that there will be uniform distribution of data before fitting into the machine learning models.

# In[ ]:


# Calculate the median value for BMI
median_bmi = df['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
df['BMI'] = df['BMI'].replace(
    to_replace=0, value=median_bmi)


# In[ ]:


median_bloodp = df['BloodPressure'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
df['BloodPressure'] = df['BloodPressure'].replace(
    to_replace=0, value=median_bloodp)


# In[ ]:


# Calculate the median value for PlGlcConc
median_plglcconc = df['Glucose'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
df['Glucose'] = df['Glucose'].replace(
    to_replace=0, value=median_plglcconc)


# In[ ]:


# Calculate the median value for SkinThick
median_skinthick = df['SkinThickness'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
df['SkinThickness'] = df['SkinThickness'].replace(
    to_replace=0, value=median_skinthick)


# In[ ]:


# Calculate the median value for SkinThick
median_skinthick = df['Insulin'].median()
# Substitute it in the SkinThick column of the
# dataset where values are 0
df['Insulin'] = df['Insulin'].replace(
    to_replace=0, value=median_skinthick)


# So,I have transformed most of the columns having zero entries, except some value such as number of times pregnant can make sense to be zero .Now lets watch the distribution how it looks like now.

# In[ ]:


dataset2 = df.drop(columns = ['Outcome'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# # Correlation with Target variable

# In[ ]:


dataset2.corrwith(df.Outcome).plot.bar(
        figsize = (20, 10), title = "Correlation with Outcome", fontsize = 15,
        rot = 45, grid = True)


# As seeing from the above graph, it is inferred that Target variable which is Outcome is highly correlated with Glucose, BMI, Age and Pregnancies.
# While Blood pressure and Skin Thickness is not much correlated with target variable but we as of now we will not remove these features and train the model with all the features.

# # Finding Correlation among feature variables

# In[ ]:


## Correlation Matrix


# Compute the correlation matrix
corr = dataset2.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


X = df.drop(['Outcome'],axis=1)
y = df['Outcome']


# # Splitting the Dataset

# Splitting the dataset is a very important step for supervised machine learning models. This step split the dataset into Train set and Test Set.We will use Train set to train the model (ignoring the column target variable), then we use the trained model to make predictions on new data (which is the test dataset, not part of the training set) and compare the predicted value with the pre assigned label.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 42)


# # Feature Scaling

# As there is uneven distribution of data in the dataset, so we will perform Normalization or feature scaling so that all the features in Training and Test set have values between 0 and 1. So, that machine learning models provide greater accuracy. 

# In[ ]:


min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train


# In[ ]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# # Applying Machine Learning Models

# **1. Applying Machine Learning Models**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logi = LogisticRegression()
logi.fit(X_train_scaled, y_train)


# In[ ]:



y_predict = logi.predict(X_test_scaled)
from sklearn.metrics import  accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, y_predict)

acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

results = pd.DataFrame([['Logistic Regression', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results


# In[ ]:


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_scaled, y_train)


# In[ ]:


y_predict = xgb_classifier.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['XGBOOST', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train_scaled, y_train)


# In[ ]:


y_predict = random_forest.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['Random Forest', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


sgd = SGDClassifier(max_iter=1000)

sgd.fit(X_train_scaled, y_train)
y_predict = sgd.predict(X_test_scaled)


# In[ ]:


roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['SGD', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


adaboost =AdaBoostClassifier()
adaboost.fit(X_train_scaled, y_train)
y_predict = adaboost.predict(X_test_scaled)


# In[ ]:


roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['Adaboost', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


gboost =GradientBoostingClassifier()
gboost.fit(X_train_scaled, y_train)
y_predict = gboost.predict(X_test_scaled)


# In[ ]:


roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['Gboost', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_scaled, y_train)
y_predict = knn.predict(X_test_scaled)


# In[ ]:


roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['KNN7', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


from sklearn.svm import SVC 


svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test_scaled)
roc=roc_auc_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

model_results = pd.DataFrame([['SVC Linear', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# # Voting Classifier
# 
# Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms.
# 
# It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.
# 
# We will be using weighted Voting Classifier. We will assign weights to the classifiers according to their accuracies. So the classifier with single accuracy will be assigned the highest weight and so on.
# 
# <img src="https://image.ibb.co/bEOhML/majority-voting.png">****

# In[ ]:


clf1=LogisticRegression()
clf2 = RandomForestClassifier()
clf3=AdaBoostClassifier()
clf4=XGBClassifier()
clf5=SGDClassifier(max_iter=1000,loss='log')
clf6=KNeighborsClassifier(n_neighbors = 7)
clf7=GradientBoostingClassifier()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ada', clf3),('xgb',clf4),('sgd',clf5),('knn',clf6),('gboost',clf7)], voting='soft', weights=[1,1,2,2,1,3,2])
eclf1.fit(X_train_scaled,y_train)


# In[ ]:


eclf_predictions = eclf1.predict(X_test_scaled)
acc = accuracy_score(y_test, eclf_predictions)
prec = precision_score(y_test, eclf_predictions)
rec = recall_score(y_test, eclf_predictions)
f1 = f1_score(y_test, eclf_predictions)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, eclf_predictions)
model_results = pd.DataFrame([['Voting Classifier ', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# The maximum accuracy we have achieved using Voting weighted Average classifier is 75.32% which is equivalent to XGBoost.

# # Stacking
# 
# Stacking is a way of combining multiple models, that introduces the concept of a meta learner. It is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types. 
# 
# In stacking, the combining mechanism is that the output of the classifiers (Level 0 classifiers) will be used as training data for another classifier (Level 1 classifier) to approximate the same target function. Basically, you let the Level 1 classifier to figure out the combining mechanism.
# 
# The point of stacking is to explore a space of different models for the same problem. The idea is that you can attack a learning problem with different types of models which are capable to learn some part of the problem, but not the whole space of the problem. So you can build multiple different learners and you use them to build an intermediate prediction, one prediction for each learned model. Then you add a new model which learns from the intermediate predictions the same target. This final model is said to be stacked on the top of the others, hence the name. Thus you might improve your overall performance, and often you end up with a model which is better than any individual intermediate model.
# 
# <img src="https://image.ibb.co/d5ySo0/stackingclassification-overview.png">

# In[ ]:


clf1=LogisticRegression()
clf2 = RandomForestClassifier()
clf3=AdaBoostClassifier()
clf4=XGBClassifier()
clf5=SGDClassifier(max_iter=1000,loss='log')
clf6=GradientBoostingClassifier()
knn=KNeighborsClassifier(n_neighbors = 7)


sclf = StackingClassifier(classifiers=[clf1,clf2, clf3, clf4,clf5,clf6], 
                          meta_classifier=knn)

print('10-fold cross validation:\n')

for clf, label in zip([clf1,clf2, clf3, clf4,clf5,clf6, sclf], 
                      ['Logistic Regression'
                       'Random Forest', 
                       'Adaboost',
                          'XGB','SGD','Gradient',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_test_scaled, y_test,
                                              cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# So, from stacking we have achieved accuracy of about 74%.

# In[ ]:


import time
parameters = {
        'min_child_weight': [1, 5,7, 10],
        'max_depth': [2,3, 5,7,10,12],
        'n_estimators':[10,50,100,200]
        }

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = xgb_classifier, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 5,
                           n_jobs = -1)

t0 = time.time()
grid_search.fit(X_train_scaled, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_predictions = grid_search.predict(X_test_scaled)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, grid_predictions)


# In[ ]:


sns.heatmap(cm, annot=True)


# In[ ]:


acc = accuracy_score(y_test, grid_predictions)
prec = precision_score(y_test, grid_predictions)
rec = recall_score(y_test, grid_predictions)
f1 = f1_score(y_test, grid_predictions)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, grid_predictions)
model_results = pd.DataFrame([['XGBoost Optimized', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:


import time
parameters = {
         'C':[0.1, 1, 10, 100,1000],
        'gamma':[1, 0.1, 0.01, 0.001,0.0001],
    'kernel':['rbf','linear']
        }

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = svc_model, # Make sure classifier points to the RF model
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 5,
                           n_jobs = -1)

t0 = time.time()
grid_search.fit(X_train_scaled, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_predictions = grid_search.predict(X_test_scaled)
acc = accuracy_score(y_test, grid_predictions)
prec = precision_score(y_test, grid_predictions)
rec = recall_score(y_test, grid_predictions)
f1 = f1_score(y_test, grid_predictions)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, grid_predictions)
model_results = pd.DataFrame([['SVC Optimized', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# We have achieved 6% accuracy increase in SVM with parameter tuning from 64% to 72% whereas other metric parameters also achieved good result.

# In[ ]:


import lightgbm
train_data = lightgbm.Dataset(X_train_scaled, label=y_train)
test_data = lightgbm.Dataset(X_test_scaled, label=y_test)


#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_bin': 200,
    'boosting': 'gbdt',
    'num_leaves': 10,
    'bagging_freq': 20,
    'learning_rate': 0.003,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# In[ ]:


y_predict = model.predict(X_test_scaled)
#convert into binary values
for i in range(0,154):
    if y_predict[i]>=.5:       # setting threshold to .5
       y_predict[i]=1
    else:  
       y_predict[i]=0
    
acc = accuracy_score(y_test, y_predict)
prec = precision_score(y_test, y_predict)
rec = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
from sklearn.metrics import roc_auc_score
roc=roc_auc_score(y_test, y_predict)
model_results = pd.DataFrame([['Light GBM', acc,prec,rec, f1,roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score','ROC'])
results = results.append(model_results, ignore_index = True)
results


# In[ ]:




