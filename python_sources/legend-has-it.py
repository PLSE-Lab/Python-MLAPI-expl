#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[ ]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd, xgboost, numpy as np, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
# imports
from sqlalchemy import create_engine
# import psycopg2
from sklearn import linear_model
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
get_ipython().run_line_magic('matplotlib', 'inline')
import re
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import datetime
from itertools import combinations
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc , roc_auc_score,confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import operator
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


# ### Access Data

# In[ ]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
ps = train


# In[ ]:


# Function to tokenise whole speech into sentences
def tok(speech):
    speech = tokenize.sent_tokenize(speech)
    return speech
ps['sent'] = ps['text'].apply(tok)
ps.head()


# In[ ]:


# Transpose sentences to 1 row per sentence
testing = ps.drop('text',axis=1)
new = (testing['sent'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sent').join(testing[['president','year']], how='left'))


# In[ ]:


# Encoding President Labels and reset index
new['pres_id'] = new['president'].factorize()[0]
new = new.reset_index()
new = new.drop(['index'],axis= 1)


# In[ ]:


# Test out if sentence is correct
new['sent'][0]


# ### Feature Engineering

# In[ ]:


# Additional Features
new['char_count'] = new['sent'].apply(len)
new['word_count'] = new['sent'].apply(lambda x: len(x.split()))
new['word_density'] = new['char_count'] / (new['word_count']+1)
new['punctuation_count'] = new['sent'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
new['title_word_count'] = new['sent'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
new['upper_case_word_count'] = new['sent'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

#Parts of Speech tagging
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}
# Function to apply parts of speech
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt
new['noun_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'noun'))
new['verb_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'verb'))
new['adj_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'adj'))
new['adv_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'adv'))
new['pron_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'pron'))


# In[ ]:


# Save base engineered feature set
new.to_csv('sentence_data.csv')
new = pd.read_csv('sentence_data.csv')


# ### Pre Processing

# In[ ]:


def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation + '0123456789')]).lower()
    return s

new['sent'] = new['sent'].apply(remove_punctuation)
# make lower case
new = pd.read_csv('sentence_data.csv')
new = new.drop(['Unnamed: 0'],axis =1)


# In[ ]:


new.head()


# ### Modeling

# NLP

# In[ ]:


# Initiate object to vectorize of text and mapping of additional features to that sparse matrix
mapper = DataFrameMapper([
     ('sent', TfidfVectorizer(stop_words='english',
                        ngram_range=(1, 2),
#                        min_df=0.1,
                       max_df=0.8,
                       max_features=500
#                         tfidf=True,
#                         smooth=True
                       )),
     ('char_count', None),
     ('word_count',None),
     ('word_density', None),
     ('punctuation_count', None),
     ('title_word_count', None),
     ('upper_case_word_count', None),
     ('noun_count', None),
     ('verb_count', None),
     ('adj_count', None),
     ('adv_count', None),
     ('pron_count', None),
 ])


# In[ ]:


# Apply mapping from cell above
features = mapper.fit_transform(new)
categories = new['pres_id']
 
# Split the data between train and test
X_train, X_test, y_train, y_test = train_test_split(features,categories,test_size=0.3,train_size=0.7, random_state = 42)
 
# Test run to see if features are working
# clf = RandomForestClassifier(random_state=0)
# clf.fit(features, categories)
# predicted = clf.predict(X_test)
# print(X_test,y_test, predicted)


# In[ ]:


X_train.shape


# In[ ]:


# For XG Boost

data_dmatrix = xgboost.DMatrix(data=features,label=categories)
names = [
#     'Logistic Regression', 
#          'Nearest Neighbors', 
#          'Linear SVM',
#          'RBF SVM', 
#          'Naive Bayes',
#          'LDA',
#          "QDA",          
#          "Decision Tree",
         "XG Boost",
         "Random Forest" 
#          "AdaBoost", 
#          "Neural Net"
]

classifiers = [
#     LogisticRegression(), 
#     KNeighborsClassifier(n_neighbors=10),
#     SVC(kernel="linear"),
#     SVC(kernel="rbf"),    
#     GaussianNB(),    
#     LinearDiscriminantAnalysis(),
#     QuadraticDiscriminantAnalysis(),    
#     DecisionTreeClassifier(),
    xgboost.XGBClassifier(),
    RandomForestClassifier(n_estimators=10)
#     AdaBoostClassifier(learning_rate=0.01),
#     MLPClassifier(learning_rate=0.001)    
]


# In[ ]:


results = []

models = {}
confusion = {}
class_report = {}

for name, clf in zip(names, classifiers):    
    print ('Fitting {:s} model...'.format(name))
    run_time = get_ipython().run_line_magic('timeit', '-q -o clf.fit(X_train, y_train)')
    
    print ('... predicting')
    y_pred = clf.predict(X_train)   
    y_pred_test = clf.predict(X_test)
    
    print ('... scoring')
    accuracy_train  = metrics.accuracy_score(y_train, y_pred)
    precision_train = metrics.precision_score(y_train, y_pred,average='weighted')
    recall_train    = metrics.recall_score(y_train, y_pred,average='weighted')
    accuracy_test  = metrics.accuracy_score(y_test, y_pred_test)
    precision_test = metrics.precision_score(y_test, y_pred_test,average='weighted')
    recall_test    = metrics.recall_score(y_test, y_pred_test,average='weighted')
    
    f1_train        = metrics.f1_score(y_train, y_pred,average='weighted')    
    f1_test   = metrics.f1_score(y_test, y_pred_test,average='weighted')
    cohen_kappa = cohen_kappa_score(y_test,y_pred_test)
    
    # save the results to dictionaries
    models[name] = clf    
#     confusion[name] = metrics.confusion_matrix(y_train, y_pred)
#     class_report[name] = metrics.classification_report(y_train, y_pred)
    confusion[name] = metrics.confusion_matrix(y_test, y_pred_test)
    class_report[name] = metrics.classification_report(y_test, y_pred_test)
    
    results.append([name, 
                    accuracy_train, precision_train, recall_train,
                    accuracy_test, precision_test, recall_test,
                    f1_train,
                    f1_test,
                    cohen_kappa,
                    run_time.best])

    
results = pd.DataFrame(results, columns=['Classifier', 
                                         'Accuracy_train', 'Precision_train', 'Recall_train',
                                         'Accuracy_test', 'Precision_test', 'Recall_test',
                                         'F1 Train', 
                                         'F1 Test',
                                         'Cohen Kappa test',
                                         'Train Time'])
results.set_index('Classifier', inplace= True)
print('Donezo')


# ### Model Evaluation

# In[ ]:


results.sort_values('F1 Test', ascending=False)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
results.sort_values('F1 Test', ascending=False, inplace=True)
results.plot(y=['F1 Test'], kind='bar', ax=ax[0], xlim=[0,1.1])
results.plot(y='Train Time', kind='bar', ax=ax[1])


# In[ ]:


# Confusion Matrix
confusion['Random Forest']


# In[ ]:


# Classification Report
print(class_report['Random Forest'])


# In[ ]:


# Cross Validation
model = models['Random Forest']
print(cross_val_score(models['Random Forest'], features,categories))


# In[ ]:


cv = []
for name, model in models.items():
    print(name)
    scores = cross_val_score(model, X=features, y=categories, cv=10)
    print("Accuracy: {:0.2f} (+/- {:0.2f})".format(scores.mean(), scores.std()))
    cv.append([name, scores.mean(), scores.std() ])
    print('                                             ')
cv = pd.DataFrame(cv, columns=['Model', 'CV_Mean', 'CV_Std_Dev'])
cv.set_index('Model', inplace=True)


# In[ ]:


cv.plot(y='CV_Mean', yerr='CV_Std_Dev',kind='bar', ylim=[0, 1])


# ### Hyperparameter Tuning

# In[ ]:


# # Parameter Grid
param_grid = {
                'n_estimators' : [300, 350],
#             'bootstrap' : [True,False],
              'min_samples_leaf' :[2,3],
#               'learning_rate' : [0.01],
                 'max_depth':[10,15],
#                 'min_child_weight' : [1,2,3],
#                 'objective':['multi:softmax'],
                "max_features" : [250,350]
             }


# In[ ]:


grid_search = GridSearchCV(RandomForestClassifier(),param_grid,cv=3)


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


y_pred_gs = grid_search.predict(X_test)


# In[ ]:


# some metrics
print('accuracy score')
print(accuracy_score(y_test,y_pred_gs))
print('\n')
print('confusion_matrix')
print(confusion_matrix(y_test,y_pred_gs))
print('\n')
print('classification_report')
print(classification_report(y_test,y_pred_gs))
print('\n')
print('cohen_kappa_score')
print(cohen_kappa_score(y_test,y_pred_gs))


# ### Prediction and Submission

# In[ ]:


test.head()


# In[ ]:


test.head()
new = test


# In[ ]:


new['sent'] = new['text']
new.drop('text',axis=1,inplace=True)


# In[ ]:


new.head(2)


# ### Feature Engineering

# In[ ]:


# Additional Features
new['char_count'] = new['sent'].apply(len)
new['word_count'] = new['sent'].apply(lambda x: len(x.split()))
new['word_density'] = new['char_count'] / (new['word_count']+1)
new['punctuation_count'] = new['sent'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
new['title_word_count'] = new['sent'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
new['upper_case_word_count'] = new['sent'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

#Parts of Speech tagging
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}
# Function to apply parts of speech
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt
new['noun_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'noun'))
new['verb_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'verb'))
new['adj_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'adj'))
new['adv_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'adv'))
new['pron_count'] = new['sent'].apply(lambda x: check_pos_tag(x, 'pron'))


# ### Pre Processing

# In[ ]:


def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation + '0123456789')]).lower()
    return s

new['sent'] = new['sent'].apply(remove_punctuation)
# make lower case


# In[ ]:


new.head()


# In[ ]:


# Initiate object to vectorize of text and mapping of additional features to that sparse matrix
mapper = DataFrameMapper([
     ('sent', TfidfVectorizer(stop_words='english',
                        ngram_range=(1, 2),
#                        min_df=0.1,
                       max_df=0.8,
                       max_features=500
#                         tfidf=True,
#                         smooth=True
                       )),
     ('char_count', None),
     ('word_count',None),
     ('word_density', None),
     ('punctuation_count', None),
     ('title_word_count', None),
     ('upper_case_word_count', None),
     ('noun_count', None),
     ('verb_count', None),
     ('adj_count', None),
     ('adv_count', None),
     ('pron_count', None),
 ])


# In[ ]:


features = mapper.fit_transform(new)
# categories = new['pres_id']


# In[ ]:


grid_search.predict(features)


# In[ ]:


test_final = pd.read_csv('../input/test.csv')


# In[ ]:


test_final['president'] = grid_search.predict(features)
test_final.drop('text', axis =1, inplace = True)


# In[ ]:


test_final


# In[ ]:


# Output Final File
test_final.to_csv('test_output.csv')


# In[ ]:




