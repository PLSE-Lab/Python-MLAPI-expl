#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import comet_ml for versioning
#from comet_ml import Experiment

#experiment = Experiment(api_key="toXXRULujXVmmXBW2zMcncxEI",
#                        project_name="general", workspace="shyken")


# # Kaggle Classification Challenge

# This notebook contains Randomized Grid Search so Please run this notebook cell by cell as to avoid having to wait a long time for outputs feel free to share changes/improvements.

# How i structured this is to give a step by step of how i built
# the model that the highest f1 score we have on kaggle and also 
# includes an example of hyperparameter tuning.

# In[ ]:


get_ipython().system('pip install comet_ml')


# In[ ]:


# import comet_ml in the top of your file
from comet_ml import Experiment
    


# In[ ]:



# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="toXXRULujXVmmXBW2zMcncxEI",
                     project_name="kaggle", workspace="shyken")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# For printing option and text color
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# In[ ]:


df_sub = pd.read_csv('../input/climate-change-belief-analysis/sample_submission.csv')
df_test = pd.read_csv('../input/climate-change-belief-analysis/test.csv')
test = df_test.set_index('tweetid')
df_train = pd.read_csv('../input/climate-change-belief-analysis/train.csv')
train = df_train.set_index('tweetid')


# In[ ]:


test.head()


# In[ ]:


test.shape


# In[ ]:


train.head()


# In[ ]:


train.shape


# ## Exploditory Data Analysis

# In[ ]:


print(f'Missing values in train dataset:\n{train.isna().sum()}\n')
print(f'Missing values in test dataset:\n{test.isna().sum()}')


# In[ ]:


# Checking for Empty messages in both train and test datasets

blanks_test = []
for tID,msg in test.itertuples():
    if msg.isspace == True:
        blanks_test.append(tID)

blanks_train = []
for tID,sent,msg in train.itertuples():
    if msg.isspace == True:
        blanks_test.append(tID)


# In[ ]:


print(f'No. of empty messages in train: {len(blanks_train)}\n')
print(f'No. of empty messages in test: {len(blanks_test)}')


# **Both the train and test datasets are clean and ready for modelling** 

# ___

# Different `sentiment` classes types and their corrisponding descriptions

# <img src="class_description.png">

#  

#   

# In[ ]:


# Count of classes in sentiment 

sns.set(style="darkgrid")
ax = sns.countplot(x='sentiment', data=df_train)


# In[ ]:


print(color.BOLD +'Percentage of a particular `Class` in the train dataset\n'+ color.END)
print(f'Class 2 ~ News \n{round((df_train.sentiment.value_counts()[2]/len(df_train))*100,2)} %\n')
print(f'Class 1 ~ Pro \n{round((df_train.sentiment.value_counts()[1]/len(df_train))*100,2)} %\n')
print(f'Class 0 ~ Neutral \n{round((df_train.sentiment.value_counts()[0]/len(df_train))*100,2)} %\n')
print(f'Class -1 ~ Anti \n{round((df_train.sentiment.value_counts()[-1]/len(df_train))*100,2)} %')


# From this we observe that the classses are unbalnced and we also observe that the Pro Class has the highest count and is about 2x the second highest class. We expect that this this class will be the most correctly classified out of all other classes.

# ## Model Building

# ### Library Imports for model building

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# Stop words from scikit-learn
from sklearn.feature_extraction import text


# In[ ]:


sk_stop_words = list(text.ENGLISH_STOP_WORDS)


# In[ ]:


costom_stop_words = ['and','of','a','an','the','in',
                   'to','&','@','am','the','were',
                   'what','where','how','why','about',
                   'all','at','be','but','by','from',
                   'got','had','hadn\'t','has','have',
                   'having''he','i','i\'ll','i\'m',
                   'in','is','it','it\'s','just','me',
                   'my','na','she','so','that','them',
                   'they','this','was','with','she',
                   'so','that','them','they','this',
                   'was','(','[','{',')',']','}']


# In[ ]:


X = train.iloc[:,-1]
y = train.iloc[:,:-1].values


# In[ ]:


# Splitting train dataset into train subset(for training model) and validation subset(for model evaluation)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[ ]:


vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)


# ### Linear Support Vector Classifier

# #### Training LinearSVC

# In[ ]:


# A Pipeline that first imploys tokenazation and vectorization and then trains model using Linear-SVC using train data.
cl_pipe = Pipeline([('vectorizer', TfidfVectorizer()),
                     ('linearS', LinearSVC()),])


# In[ ]:


# Fitting/training model
cl_pipe.fit(X_train, y_train) 


# #### Accessing LinearSVC Acurracy

# In[ ]:


# Predicting test subset of validation data
y_pred = cl_pipe.predict(X_test)


# In[ ]:


# LinearSVC confusion matrix
LSVC_confusion = pd.DataFrame(confusion_matrix(y_test,y_pred),
                  index=['Anti','Neutral','Pro','News'],
                  columns=['Anti','Neutral','Pro','News'])

LSVC_confusion


# In[ ]:


# Classification Report matrix 
print(classification_report(y_test,y_pred))


# The important metric to note is the f1-score (macro avg) since its the
# metric used on kaggle for scoring. The score obtained on the notebook will
# not be the same as the one you will obtain on kaggle, the kaggle score
# will usually be higher.The score on the notebook is helpful on gauging 
# your margins and comparing different models on the notebook.

# #### Predictions on test dataset  using LinearSVC

# In[ ]:


y_pred_test = cl_pipe.predict(test.message)


# In[ ]:


y_pred_test


# In[ ]:


# Predictions in submission kaggle format
linearSVC_submission1 = pd.DataFrame({'tweetid': test.index, 
                           'sentiment': y_pred_test})


# In[ ]:


linearSVC_submission1


# In[ ]:


# Saving to .csv file
linearSVC_submission1.to_csv('LSVC_01.csv',index = False)


# #### Save Model

# In[ ]:


import pickle

# Here i am saving th entire pipeline,if there's a need i'll save the vecorizer and model separate.
# Saving the pipeline in the same name as the submission file helps keep track of everything.

model_save_path = "LSVC_01.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(cl_pipe,file)


# ### K-Nearest Neighbors Classifier

# #### Training KNN Classifier

# In[ ]:


# Instantiate KNN model
KNN = KNeighborsClassifier(n_neighbors=10)


# In[ ]:


# KNN pipeline
KNN_pipe = Pipeline([('vectorizer', TfidfVectorizer()),('KNN',KNN)])


# In[ ]:


# Fitting/Training KNN model
KNN_pipe.fit(X_train,y_train.ravel())


# #### Accessing KNN accuracy

# In[ ]:


# Predicting validation subset
y_pred_KNN = KNN_pipe.predict(X_test)


# In[ ]:


# KNN confusion matrix
KNN_confusion = pd.DataFrame(confusion_matrix(y_test,y_pred_KNN),
                  index=['Anti','Neutral','Pro','News'],
                  columns=['Anti','Neutral','Pro','News'])

KNN_confusion


# In[ ]:


# Classification report matrix
print(classification_report(y_test,y_pred_KNN))


# #### Predictions on test dataset using KNN

# In[ ]:


y_KNN = KNN_pipe.predict(test.message)


# In[ ]:


y_KNN


# In[ ]:


# Predictions in submission kaggle format
KNN_submission1 = pd.DataFrame({'tweetid': test.index, 
                           'sentiment': y_KNN})

KNN_submission1


# In[ ]:


# Saving to .csv file
KNN_submission1.to_csv('KNN_01.csv',index = False)


# #### Save Model

# In[ ]:


model_save_path = 'KNN_01.pkl'
with open(model_save_path,'wb') as file:
    pickle.dump(KNN_pipe,file)


# ### Random forest Classifier

# #### Training Random forest Classifier

# In[ ]:


# Instantiating Random forest Classifier
RFC = RandomForestClassifier(n_estimators=100)


# In[ ]:


# Random forest pipeline
RFC_pipe = Pipeline([('vectorizer', TfidfVectorizer()),
                     ('RFC',RFC)])


# In[ ]:


# Fitting/Training model
RFC_pipe.fit(X_train,y_train.ravel())


# #### Accessing RFC accuracy

# In[ ]:


# Predicting validation subset
y_pred_RFC = RFC_pipe.predict(X_test)


# In[ ]:


# RFC Confusion matrix
RFC_confusion = pd.DataFrame(confusion_matrix(y_test,y_pred_RFC),
                  index=['Anti','Neutral','Pro','News'],
                  columns=['Anti','Neutral','Pro','News'])


# In[ ]:


# Classification report matrix
print(classification_report(y_test,y_pred_RFC))


# #### Hyperparameter Tunning for RFC

# In[ ]:


# Selecting Parameter to tune

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_state = list(range(0,43))

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'random_state':random_state}


# In[ ]:


# Re-instantiate Random Classifier for tunning
rfc = RandomForestClassifier()


# In[ ]:


# Instantiate Random Search
RFC_Random = RandomizedSearchCV(estimator=rfc, 
                               param_distributions=random_grid,
                               n_iter=5, cv=5, verbose=2, random_state=42)


# Before we do Random Search on the RFC model we need to vectorize our Predictor variable since we are doing a randomized grid search on the RFC model and not on it's pipeline, although it is possible to do a randomised grid search on the pipeline.

# The reason for it takes this particuler Random Search takes so much time is the number of parameters we are tunning and also it runs `n_iter` by `cv` models. Hyperparameter tuning for some other models might much less time to run.

# In[ ]:


#vectorizer = TfidfVectorizer()
#X_vec = vectorizer.fit_transform(X_train)


# In[ ]:


# Fitting Random Search on train subset
#RFC_Random.fit(X_vec,y_train.ravel())


# In[ ]:


# Retrieving parameters for the RFC model that performed best
#RFC_Random.best_params_


# In[ ]:


# Re-instiate RFC with best parameters
rfc_boosted = RandomForestClassifier(random_state=8,
                                     n_estimators=1800,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     max_features='auto',
                                     max_depth=60,
                                     bootstrap=False)


# In[ ]:


# Pipeline of RFC model with best parameters
boosted_pipe = Pipeline([('vectorizer', TfidfVectorizer()),('RFC',rfc_boosted)])


# In[ ]:


# Fitting/Training model
boosted_pipe.fit(X_train,y_train.ravel())


# In[ ]:


# Predicting validation subset
y_pred_boost = boosted_pipe.predict(X_test)


# In[ ]:


# RFC random search confusion matrix
RFC_RS_confusion = pd.DataFrame(confusion_matrix(y_test,y_pred_boost),
                  index=['Anti','Neutral','Pro','News'],
                  columns=['Anti','Neutral','Pro','News'])
RFC_RS_confusion


# In[ ]:


# Classification report matrix
print(classification_report(y_test,y_pred_boost))


# #### Predictions on test dataset  using RFC_RS

# In[ ]:


y_boost = boosted_pipe.predict(test.message)


# In[ ]:


y_boost


# In[ ]:


# Predictions in submission kaggle format
RFC_RS_submission1 = pd.DataFrame({'tweetid': test.index, 
                           'sentiment': y_boost})

RFC_RS_submission1 


# In[ ]:


# Saving to .csv file
RFC_RS_submission1.to_csv('RandomForest_boosted_01.csv',index = False)


# #### Save Model

# In[ ]:


model_save_path = 'RandomForest_boosted_01.pkl'
with open(model_save_path,'wb') as file:
    pickle.dump(boosted_pipe,file)


# In[ ]:


experiment.end()

experiment.display()

