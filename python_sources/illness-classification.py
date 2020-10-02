#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
import warnings
warnings.simplefilter("ignore")


# ## Data Preprocessing 

# ### Importing Dataset

# In[ ]:


df = pd.read_csv('../input/toy-dataset/toy_dataset.csv')
df.head()


# ### Count on basis of "Yes" and "No" Class in Target Variable 

# In[ ]:


df["Illness"].value_counts()


# In[ ]:


df[df["Illness"] == 'No'].count()


# In[ ]:


df[df["Illness"] == 'No'].count()


# ### DataFrame information

# In[ ]:


df.info()


# ### Encoding Categorical Variable 

# In[ ]:


df.loc[df['Gender']=='Male','Gender'] = 1
df.loc[df['Gender']=='Female','Gender'] = 0
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb_results = lb.fit_transform(df['City'])
new_df = pd.DataFrame(lb_results, columns=lb.classes_)
df = pd.concat([df, new_df], axis = 1)


# #### In Below code block output we can check all categorical variable are encoded as numerical value

# In[ ]:


df.head()


# ### Seperating Feature matrix and target variable

# In[ ]:


X = df.drop(['Illness','Number', 'City'], axis=1)
y = df.Illness


# ### Encoding Target Variable using LabelEncoder

# In[ ]:


encoder = LabelEncoder()
y = pd.Series(encoder.fit_transform(y), name='Illness')


# ### Splitting data into train_set and test_set

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)


# ### concatenate training data together so that we can seperate the data on basis of classes in target column

# In[ ]:


X = pd.concat([X_train, y_train], axis=1)


# ### Now we are seperating the data on the basis of majority and minority classes to apply oversampling

# In[ ]:


not_ill = X[X.Illness== 0]
ill = X[X.Illness== 1]


# ### We are trying to upsample minority class using scikit learn's resample class

# In[ ]:


ill_upsampled = resample(ill,
                         replace=True, # sample with replacement
                         n_samples=len(not_ill), # match number in majority class
                         random_state=100) # reproducible results


# ### Now we will combining the majority class and the upsampled minority class 

# In[ ]:


upsampled = pd.concat([not_ill, ill_upsampled])
upsampled.Illness.value_counts()


# ### Below Function will check the training performance and test performance.

# In[ ]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test)))) 


# ## Training and Testing Model

# ### Training Logisitic Regression on Oversampled Dataset

# In[ ]:


y_train = upsampled.Illness
X_train = upsampled.drop('Illness', axis=1)
clf = LogisticRegression(solver='liblinear').fit(X_train, y_train)


# ### Training performance

# In[ ]:


print_score(clf, X_train, y_train, X_test, y_test, train=True)


# ### Test Performance

# In[ ]:


print_score(clf, X_train, y_train, X_test, y_test, train=False)


# ### As resampling doesn't made any big effect we are trying SMOTE oversampling which actually loops through the existing, real minority instance. At each loop iteration, one of the K closest minority class neighbours is chosen and a new minority instance is synthesised somewhere between the minority instance and that neighbour.

# In[ ]:


# setting up testing and training sets
X = df.drop(['Illness','Number', 'City'], axis=1)
y = df.Illness
encoder = LabelEncoder()
y = pd.Series(encoder.fit_transform(y), name='Illness')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)


# ### SMOTE oversampling

# In[ ]:


sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)


# #### Visualizing Balanced Training data after Smote Sampling 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(y_train)
plt.title('Balanced training data')
plt.show()


# ### Training Random Forest over Smote Oversampling data set

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)


# In[ ]:


print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)


# In[ ]:


print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# ### Still we are not getting desired as for class 1 there is only 8% precision and recall is too less,so we are using GridSearch so that we can get best parameter

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[ ]:


rf_clf = RandomForestClassifier(random_state=42)
params_grid = {"max_depth": [3, None],
               "min_samples_split": [2, 3, 5],
               "min_samples_leaf": [1, 3, 5],
               "bootstrap": [True, False],
               "criterion": ['entropy']}


# In[ ]:


grid_search = GridSearchCV(rf_clf, params_grid,
                           n_jobs=-1, cv=3,
                           verbose=1, scoring='accuracy')


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_score_


# In[ ]:


grid_search.best_estimator_.get_params()


# In[ ]:


print_score(grid_search, X_train, y_train, X_test, y_test, train=True)


# In[ ]:


print_score(grid_search, X_train, y_train, X_test, y_test, train=False)


# ### Our Model is overfit as we can see it performing well on training set but on test set it is having issues with the class 1.To Overcome this we will try it with Extra-Trees Ensemble

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


xt_clf = ExtraTreesClassifier(random_state=42, min_samples_leaf=3, min_samples_split=2)


# In[ ]:


xt_clf.fit(X_train, y_train)


# In[ ]:


print_score(xt_clf, X_train, y_train, X_test, y_test, train=True)


# In[ ]:


print_score(xt_clf, X_train, y_train, X_test, y_test, train=False)


# ### StratifiedKfold with Random Forest classifier

# In[ ]:


# setting up testing and training sets
X = df.drop(['Illness','Number', 'City'], axis=1)
y = df.Illness
encoder = LabelEncoder()
y = pd.Series(encoder.fit_transform(y), name='Illness')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
kf = StratifiedKFold(n_splits=3, random_state=100)
cross_val_f1_score_lst = []
cross_val_accuracy_lst = []
cross_val_recall_lst = []
cross_val_precision_lst = []

for train_index_ls, validation_index_ls in kf.split(X_train, y_train):
    # keeping validation set apart and oversampling in each iteration using smote 
    train, validation = X_train.iloc[train_index_ls], X_train.iloc[validation_index_ls]
    target_train, target_val = y_train.iloc[train_index_ls], y_train.iloc[validation_index_ls]
    sm = SMOTE(random_state=100)
    X_train_res, y_train_res = sm.fit_sample(train, target_train)
    print (X_train_res.shape, y_train_res.shape)
    
    # training the model on oversampled 4 folds of training set
    rf = RandomForestClassifier(n_estimators=100, random_state=100)
    rf.fit(X_train_res, y_train_res)
    # testing on 1 fold of validation set
    validation_preds = rf.predict(validation)
    cross_val_recall_lst.append(recall_score(target_val, validation_preds))
    cross_val_accuracy_lst.append(accuracy_score(target_val, validation_preds))
    cross_val_precision_lst.append(precision_score(target_val, validation_preds))
    cross_val_f1_score_lst.append(f1_score(target_val, validation_preds))
print ('Cross validated accuracy: {}'.format(np.mean(cross_val_accuracy_lst)))
print ('Cross validated recall score: {}'.format(np.mean(cross_val_recall_lst)))
print ('Cross validated precision score: {}'.format(np.mean(cross_val_precision_lst)))
print ('Cross validated f1_score: {}'.format(np.mean(cross_val_f1_score_lst)))


# ### We have tried to build the best model to predict Illness based on Income, Age, Gender and City but  as we have shown earlier in notebook that the data is skewed so we have tried random upsampling and Smote oversampling and we have trained our model on the sampled data but our model is overfit as we have tried different techniques but the result or evaluation metrics not upto the mark we will try doing undersampling the majority class because of the availability issue uploading it till this. Because training the models are taking too much time so it was not possible to use more_splits. 

# # Visualization Variable Importance

# In[ ]:


from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
xgbmodel = XGBClassifier()
xgbmodel.fit(X_train, y_train)
feat_importances = pd.Series(xgbmodel.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')


# #  To operationalize the model we have to take the below steps:
# ## 1. First of all we have to save the best model in pickle format and put it in a database.
# ## 2. We have to create a GUI using python Django or Flask framework and there it will ask a person about the details like city, Gender, Income and Age after that when they hit enter button there will be a script which will load the model from pickle format and do the prediction , that predicted output will be shown to user.

# ## There are other ways to operationalize the model using Google cloud ML Engine and other clouds have the options.
