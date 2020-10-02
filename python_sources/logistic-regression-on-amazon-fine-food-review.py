#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression on Amazon Fine Food Review

# ## Loading necessary libraries

# In[ ]:


import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


# ## Loading already prepared corpus

# In[ ]:


df = pd.read_csv("../input/amazon-fine-food-corpus/corpus.csv")
df.head()


# ## Since dataset is very large so taking out 120K points to work

# In[ ]:


df1k = df.loc[:119999,:]
print("Shape:- ",df1k.shape)
print(df1k.head())
df1k['Score'].value_counts()


# ## Function for time based splitting into train and test dataset

# In[ ]:


from sklearn.model_selection import TimeSeriesSplit
def timesplit(x,y):
    ts = TimeSeriesSplit(n_splits = 4)
    for train_index,test_index in ts.split(x):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
    return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = timesplit(df1k["Text"].values,df1k["Score"].values)


# ### Feature Importance Function

# In[ ]:


def imp_features(model,classifier):
    voc = model.get_feature_names()
    w = list(classifier.coef_[0])
    pos_coef = []
    neg_coef = []
    pos_words = []
    neg_words = []
    for i,c in enumerate(w):
        if c > 0:
            pos_coef.append(c)
            pos_words.append(voc[i])
        if c < 0:
            neg_coef.append(abs(c))
            neg_words.append(voc[i])
    pos_df = pd.DataFrame(columns = ['Words','Coef'])
    neg_df = pd.DataFrame(columns = ['Words','Coef'])
    pos_df['Words'] = pos_words
    pos_df['Coef'] = pos_coef
    neg_df['Words'] = neg_words
    neg_df['Coef'] = neg_coef
    pos_df = pos_df.sort_values("Coef",axis = 0,ascending = False).reset_index(drop=True)
    neg_df = neg_df.sort_values("Coef",axis = 0,ascending = False).reset_index(drop=True)
    print("Shape of Positive dataframe:- ,",pos_df.shape)
    print("Shape of Negative dataframe:- ",neg_df.shape)
    print("Top ten positive predictors:- \n",pos_df.head(10))
    print("\nTop ten negative predictors:- \n",neg_df.head(10))


# ## Bag of Words Implementation

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bow_train = cv.fit_transform(x_train)
print("Shape of BOW vector:- ",bow_train.shape)


# ### Performing Standardization on train data

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
bow_train = sc.fit_transform(bow_train)


# ### Parameter tuning using GridSearchCV

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nclassifier = LogisticRegression(penalty = \'l2\',solver = \'sag\')\nparam_grid = {"C":[0.01,0.1,1,5,10,50]}\ngs = GridSearchCV(classifier,param_grid,cv = 5,scoring = \'f1_micro\',n_jobs = -1)\ngs.fit(bow_train,y_train)\nprint("Best parameter:- ",gs.best_params_)\nprint("Best score:- ",gs.best_score_)')


# ### Applying the model on test data with optimal value of C

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import confusion_matrix\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import f1_score\nbow_test = cv.transform(x_test)\nbow_test = sc.transform(bow_test)   # Standardizing the test data\nclassifier = LogisticRegression(C=1,penalty = \'l2\',solver = \'sag\',n_jobs = -1)\nclassifier.fit(bow_train,y_train)\ny_pred = classifier.predict(bow_test)\nprint("BOW test accuracy:- ",accuracy_score(y_test,y_pred))\nprint("F1 score:- ",f1_score(y_test,y_pred,average=\'micro\'))\nprint("Training accuracy:- ",accuracy_score(y_train,classifier.predict(bow_train)))')


# ### Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)


# ### Checking Multicollinearity using peturbation test

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from scipy.sparse import csr_matrix\ncoef = classifier.coef_    #weight vector of original classifier\ne = 0.02 # introducing small error in the training dataset\nbow_train_pert = csr_matrix(bow_train,dtype=np.float64)\nbow_train_pert[np.nonzero(bow_train_pert)]+=e\nclassifier_pert = LogisticRegression(C=1,penalty = \'l2\',solver = \'sag\',n_jobs = -1)\nclassifier_pert.fit(bow_train_pert,y_train)\ncoef_pert = classifier_pert.coef_\ncoef_diff = coef_pert - coef\nprint("Average difference in weight vectors:- ",np.mean(coef_diff))')


# #### Differences in the coefficients of the peturbed model and original model are very less so weight vector of the classifier can be considered for the feature importance.

# ### Feature importance:- Top ten predictors from each class

# In[ ]:


imp_features(cv,classifier)


# ### Error plots

# In[ ]:


y_pred_prob = classifier.predict_proba(bow_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()


# ### =====================================================================================

# ### Tfidf implementation

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True)
tfidf_train = tfidf.fit_transform(x_train)
print("Shape of tfidf_train:- ",tfidf_train.shape)


# ### Standardizing the data

# In[ ]:


sc = StandardScaler(with_mean = False)
tfidf_train = sc.fit_transform(tfidf_train)


# ### Parameter tuning using GridSearchCV

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nclassifier = LogisticRegression(penalty = \'l1\',solver = \'liblinear\',class_weight = \'balanced\')\nparam_grid = {"C":[0.01,0.1,1,10,50]}\ngs = GridSearchCV(classifier,param_grid,cv = 5,scoring = \'f1\',n_jobs = -1)\ngs.fit(tfidf_train,y_train)\nprint("Best parameter:- ",gs.best_params_)\nprint("Best score:- ",gs.best_score_)')


# ### Testing the model on test data with optimal C.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import confusion_matrix\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import f1_score\ntfidf_test = tfidf.transform(x_test)\ntfidf_test = sc.transform(tfidf_test)   # Standardizing the test data\nclassifier = LogisticRegression(C=0.01,penalty = \'l1\',solver = \'liblinear\',class_weight = \'balanced\')\nclassifier.fit(tfidf_train,y_train)\ny_pred = classifier.predict(tfidf_test)\nprint("Tfdif test accuracy:- ",accuracy_score(y_test,y_pred))\nprint("F1 score:- ",f1_score(y_test,y_pred))\nprint("Training accuracy:- ",accuracy_score(y_train,classifier.predict(tfidf_train)))')


# ### Testing the model with class_weight as None

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.metrics import confusion_matrix\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.metrics import f1_score\ntfidf_test = tfidf.transform(x_test)\ntfidf_test = sc.transform(tfidf_test)   # Standardizing the test data\nclassifier = LogisticRegression(C=0.01,penalty = \'l1\',solver = \'liblinear\')\nclassifier.fit(tfidf_train,y_train)\ny_pred = classifier.predict(tfidf_test)\nprint("Tfdif test accuracy:- ",accuracy_score(y_test,y_pred))\nprint("F1 score:- ",f1_score(y_test,y_pred))\nprint("Training accuracy:- ",accuracy_score(y_train,classifier.predict(tfidf_train)))')


# ### Confusion Matrix of model with class_weight as None

# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)


# ### Confusion Matrix of model with class_weight='balanced'

# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)


# ### Feature Importance:- Top ten predictors of each class

# In[ ]:


print("No. of features with zero coefficients:- ",tfidf_train.shape[1]-np.count_nonzero(classifier.coef_))
imp_features(tfidf,classifier)


# ### Error Plots of model with class_weight as 'balanced'

# In[ ]:


y_pred_prob = classifier.predict_proba(tfidf_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()


# ### Error plot of model with class_weight as None

# In[ ]:


y_pred_prob = classifier.predict_proba(tfidf_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()


# ### =====================================================================================

# ### Word2Vec

# In[ ]:


#Function to create list of sentences
def sent_list(x):
    list_of_sent = []
    for sent in tqdm(x):
        words = []
        for w in sent.split():
            words.append(w)
        list_of_sent.append(words)
    return list_of_sent


# In[ ]:


#implementing word2vec
from gensim.models import Word2Vec
sent_train = sent_list(x_train)
w2v = Word2Vec(sent_train,size=50,min_count=2,workers=4)


# ### Average Word2Vec implementation:-

# In[ ]:


#Function to create avg word2vec vector
def avgw2v(x):
    avgw2v_vec = []
    for sent in tqdm(x):
        sent_vec = np.zeros(50)
        count = 0
        for word in sent:
            try:
                vec = w2v.wv[word]
                sent_vec+=vec
                count+=1
            except:
                pass
        sent_vec/=count
        avgw2v_vec.append(sent_vec)
    return avgw2v_vec


# In[ ]:


#Creating average word2vec training data
avgw2v_train = np.array(avgw2v(sent_train))
print("Shape of avg word2vec train data:- ",avgw2v_train.shape)


# #### Standardizing the data

# In[ ]:


sc = StandardScaler()
avgw2v_train = sc.fit_transform(avgw2v_train)


# #### Applying GridSearchCV to find optimal hyperparameter

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nclassifier = LogisticRegression(penalty = \'l2\',solver=\'sag\',class_weight = \'balanced\')\nparam_grid = {"C":[0.01,0.1,1,10,50]}\ngs = GridSearchCV(classifier,param_grid,cv = 5,scoring = \'f1\',n_jobs = -1)\ngs.fit(avgw2v_train,y_train)\nprint("Best parameter:- ",gs.best_params_)\nprint("Best score:- ",gs.best_score_)')


# #### Testing the model on the test data 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sent_test = sent_list(x_test)\navgw2v_test = np.array(avgw2v(sent_test))\navgw2v_test = sc.transform(avgw2v_test)\nclassifier = LogisticRegression(C=1,penalty = \'l2\',solver = \'sag\',class_weight = \'balanced\')\nclassifier.fit(avgw2v_train,y_train)\ny_pred = classifier.predict(avgw2v_test)\nprint("Avg Word2Vec test accuracy:- ",accuracy_score(y_test,y_pred))\nprint("F1 score:- ",f1_score(y_test,y_pred))\nprint("Training accuracy:- ",accuracy_score(y_train,classifier.predict(avgw2v_train)))')


# ### Testing the model with class_weight as None

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sent_test = sent_list(x_test)\navgw2v_test = np.array(avgw2v(sent_test))\navgw2v_test = sc.transform(avgw2v_test)\nclassifier = LogisticRegression(C=1,penalty = \'l2\',solver = \'sag\')\nclassifier.fit(avgw2v_train,y_train)\ny_pred = classifier.predict(avgw2v_test)\nprint("Avg Word2Vec test accuracy:- ",accuracy_score(y_test,y_pred))\nprint("F1 score:- ",f1_score(y_test,y_pred))\nprint("Training accuracy:- ",accuracy_score(y_train,classifier.predict(avgw2v_train)))')


# ### Confusion matrix of model with class_weight as None

# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)


# #### Confusion Matrix of model with class_weight as 'balanced'.

# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)


# #### Error Plots:-

# In[ ]:


y_pred_prob = classifier.predict_proba(avgw2v_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()


# ### ==================================================================================

# ### Tfidf Word2Vec implementation

# In[ ]:


#Function for creating tfidf weighted Word2Vec
def tfidfw2v(x):
    dictionary = dict(zip(tfidf.get_feature_names(),list(tfidf.idf_)))
    tfidf_w2v_vec = []
    i=0
    for sent in tqdm(x):
        sent_vec = np.zeros(50)
        weights = 0
        for word in sent:
            try:
                vec = w2v.wv[word]
                tfidf_value = dictionary[word]*sent.count(word)
                sent_vec+=(tfidf_value*vec)
                weights+=tfidf_value
            except:
                pass
        sent_vec/=weights
        tfidf_w2v_vec.append(sent_vec)
        i+=1
    return tfidf_w2v_vec


# In[ ]:


tfidfw2v_train = np.array(tfidfw2v(sent_train))
print("Shape of tfidf avgw2v train vector:- ",tfidfw2v_train.shape)


# #### Standardizing the data

# In[ ]:


sc = StandardScaler()
tfidfw2v_train = sc.fit_transform(tfidfw2v_train)


# #### Applying GridSearchCV to find optimal hyperparameter

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nclassifier = LogisticRegression(penalty = \'l2\',solver=\'sag\',class_weight = \'balanced\')\nparam_grid = {"C":[0.1,1,10,50,100]}\ngs = GridSearchCV(classifier,param_grid,cv = 5,scoring = \'f1\',n_jobs = -1)\ngs.fit(tfidfw2v_train,y_train)\nprint("Best parameter:- ",gs.best_params_)\nprint("Best score:- ",gs.best_score_)')


# #### Applying model on test data with optimal hyperparameter

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sent_test = sent_list(x_test)\ntfidfw2v_test = np.array(tfidfw2v(sent_test))\ntfidfw2v_test = sc.transform(tfidfw2v_test)\nclassifier = LogisticRegression(C=10,penalty = \'l2\',solver = \'sag\',class_weight = \'balanced\')\nclassifier.fit(tfidfw2v_train,y_train)\ny_pred = classifier.predict(tfidfw2v_test)\nprint("Tfidf Word2Vec test accuracy:- ",accuracy_score(y_test,y_pred))\nprint("F1 score:- ",f1_score(y_test,y_pred))\nprint("Training accuracy:- ",accuracy_score(y_train,classifier.predict(tfidfw2v_train)))')


# #### Confusion Matrix

# In[ ]:


cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:- \n",cm)
sns.heatmap(cm,annot = True)


# ### Error Plots:-

# In[ ]:


y_pred_prob = classifier.predict_proba(tfidfw2v_test)
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred_prob[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred_prob[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()


# ### ===================================================================================

# ## Results:-
# 
# | Model | Hyperparameter(C) | Training Accuracy | Test Accuracy |
# | - | - | - | - |
# | Bag of Words | 1 | 89.7 | 83.6 |
# |Tfidf | 0.01 | 78, 89.7 | 65.57, 83.8 |
# | Avg Word2Vec | 1 | 50.98, 87.8 | 50, 84.5 |
# | Tfidf Word2Vec | 10 | 50.72 | 49.79 |

# ### Conclusion:-
# #### Losgistic Regression has given overall good results with Bag of Words and Tfidf vectors.
# 
# #### Logisitic Regression is not at all performing good when applying on Word2Vec vectors. When i am not using class_weight='balanced' it is classifying all points as postive and when i am using class_weight='balanced' it is giving very less accuracy.

# ### ===================================================================================
