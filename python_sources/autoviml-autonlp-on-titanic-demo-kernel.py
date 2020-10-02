#!/usr/bin/env python
# coding: utf-8

# # Demonstrating autoviml with Titanic Data set
# ###  This Kernel demonstrates how to use a new library called "Autoviml" for creating multiple models on any data set with just one line of code.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv',index_col=None)
print(train.shape)
test = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=None)
print(test.shape)
subm = pd.read_csv('/kaggle/input/titanic/gender_submission.csv',index_col=None)
print(subm.shape)


# !pip install autoviml

# In[ ]:


get_ipython().system('pip install autoviml')


# In[ ]:


from autoviml.Auto_ViML import Auto_ViML


# In[ ]:


target = 'Survived'


# # We are going to build a Logistic Regression model at first
# Set the Boosting_Flag below to None. This signifies to Auto_ViML that it must not use any Boosting models and hence it will use a simple Linear Model to build. You must provide it train, target and test. The rest of the variables are assumed.

# In[ ]:


#### Set Boosting_Flag = None, KMeans_Featurizer = True, scoring_parameter='balanced-accuracy'   ###########
m, feats, trainm, testm = Auto_ViML(train, target, test,
                            sample_submission=subm,
                            scoring_parameter='balanced-accuracy', KMeans_Featurizer=True,
                            hyper_param='GS',feature_reduction=True,
                             Boosting_Flag=None,Binning_Flag=True,
                            Add_Poly=0, Stacking_Flag=True,Imbalanced_Flag=False,
                            verbose=0)


# In[ ]:


subm[target] = testm[target+'_predictions'].astype(int).values


# In[ ]:


filename='sample_submission_log.csv'
savefile = '/kaggle/working/Survived/sample_submission_log.csv'
savefile


# In[ ]:


subm.to_csv(savefile,index=False)


# Notice that Auto_ViML built a solid Logistic Regression model by automatically removing the ID and Ticket variables. It also automatically converted the Categorical variables into Numeric variables. The regular Accuracy is 81% and balanced accuracy is 80% which is quite good.  
# ## Now let use NLP on "Name" column using Auto_NLP

# In[ ]:


from autoviml.Auto_NLP import Auto_NLP


# In[ ]:


score_type = 'accuracy'
modeltype = 'Classification'
nlp_column = 'Name'


# In[ ]:


train_nlp, test_nlp, best_nlp_transformer, _ = Auto_NLP(
                nlp_column, train, test, target, score_type,
                modeltype,top_num_features=50, verbose=0)


# With NLP Columns added to train and test, 
# ## Now let us Try two Boosting algorithms: "CatBoost" on the NLP-added data set 
# First we need to set the Boosting_Flag to "CatBoost" to get "CatBoost" which is the default Boosting algorithm.

# Notice how XGBoost has done a better job by providing better results: Accuracy at 83% and Balanced-Accuracy at 82%
# ## Now let us try CatBoost by setting the Boosting_Flag = "CatBoost"
# This gets us a Score of 79.45% on Titanic Leaderboard -> Not Bad but let us try one more thing. Let us Try Imbalanced Class handling

# In[ ]:


m4, feats4, trainm4, testm4 = Auto_ViML(train_nlp, target, test_nlp, 
                                    sample_submission=subm,
                                    scoring_parameter='balanced-accuracy',
                                    hyper_param='GS',feature_reduction=True,
                                     Boosting_Flag="CatBoost",Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,Imbalanced_Flag=False, 
                                    verbose=2)


# CatBoost is no better than XGBoost in this case. However, only by submitting results can be be sure whether our models performed in the real world as well as they did in Cross Validation and the held out data set.
# ## This shows how you can build 4 fast models using Auto_ViML. There are so many flags that you can set in the calls above to get better Results. We will let you tune these results on your own as Homework!

# In[ ]:


subm[target] = testm4['Survived_CatBoost_predictions'].values.astype(int)
subm.head()


# In[ ]:


subm.to_csv('sample_submission4.csv',index=False)


# In[ ]:


m2, feats2, trainm2, testm2 = Auto_ViML(train_nlp, target, test_nlp, 
                                    sample_submission=subm,
                                    scoring_parameter='balanced-accuracy',
                                    hyper_param='GS',feature_reduction=True,
                                     Boosting_Flag=None,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,
                                        Imbalanced_Flag=False, 
                                    verbose=2)


# In[ ]:


subm[target] = (testm2['Survived_proba_1']>0.5).astype(int).values
subm.head()


# In[ ]:


subm.to_csv('sample_submission2.csv',index=False)


# In[ ]:


m3, feats3, trainm3, testm3 = Auto_ViML(train_nlp, target, test_nlp, 
                                    sample_submission=subm,
                                    scoring_parameter='balanced-accuracy',
                                    hyper_param='RS',feature_reduction=True,
                                     Boosting_Flag=True,Binning_Flag=False,
                                    Add_Poly=0, Stacking_Flag=False,
                                        Imbalanced_Flag=False, 
                                    verbose=2)


# In[ ]:


subm[target] = (testm3['Survived_proba_1']>0.5).astype(int).values
subm.head()


# In[ ]:


subm.to_csv('sample_submission3.csv',index=False)


# In[ ]:




