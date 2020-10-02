#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[ ]:





# In[ ]:


df_treino_original = pd.read_csv("train_data.csv")

df_treino = df_treino_original[(df_treino_original['default'].notnull())]


# In[ ]:


df_treino.info()


# In[ ]:


print("credit_limit: ", df_treino['credit_limit'].mean())
print("ok_since: ", df_treino['ok_since'].mean())
print("n_bankruptcies: ", df_treino['n_bankruptcies'].mean())
print("n_defaulted_loans: ", df_treino['n_defaulted_loans'].mean())
print("n_issues: ", df_treino['n_issues'].mean())

sub_credit_limit = df_treino['credit_limit'].mean()
sub_ok_since = df_treino['ok_since'].mean()
sub_n_bankruptcies = df_treino['n_bankruptcies'].mean()
sub_n_defaulted_loans = df_treino['n_defaulted_loans'].mean()
sub_n_issues = df_treino['n_issues'].mean()


# In[ ]:


df_treino['credit_limit'].fillna(sub_credit_limit, inplace=True)
df_treino['ok_since'].fillna(sub_ok_since, inplace=True)
df_treino['n_bankruptcies'].fillna(sub_n_bankruptcies, inplace=True)
df_treino['n_defaulted_loans'].fillna(sub_n_defaulted_loans, inplace=True)
df_treino['n_issues'].fillna(sub_n_issues, inplace=True)


# In[ ]:


df_treino.info()


# In[ ]:


df_treino['reason'].fillna('NA1', inplace=True)
df_treino['sign'].fillna('NA2', inplace=True)
df_treino['gender'].fillna('NA3', inplace=True)
df_treino['facebook_profile'].fillna('NA4', inplace=True)
df_treino['job_name'].fillna('NA5', inplace=True)


# In[ ]:


df_treino.info()


# In[ ]:





# In[ ]:





# In[ ]:


df_treino_tratada = DataFrameImputer().fit_transform(df_treino)
#df_treino_tratada = df_treino
df_treino_tratada=df_treino_tratada.reset_index()


# In[ ]:





# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# In[ ]:


print('score_1:' ,len(df_treino_tratada.groupby('score_1').size()))
print('score_2:' ,len(df_treino_tratada.groupby('score_2').size()))
print('reason:' ,len(df_treino_tratada.groupby('reason').size()))
print('sign:' ,len(df_treino_tratada.groupby('sign').size()))
print('gender:' ,len(df_treino_tratada.groupby('gender').size()))
print('state:' ,len(df_treino_tratada.groupby('state').size()))
print('zip:' ,len(df_treino_tratada.groupby('zip').size()))
print('channel:' ,len(df_treino_tratada.groupby('channel').size()))
print('job_name:' ,len(df_treino_tratada.groupby('job_name').size()))
print('real_state:' ,len(df_treino_tratada.groupby('real_state').size()))
print('facebook_profile:' ,len(df_treino_tratada.groupby('facebook_profile').size()))


# In[ ]:


df_zip = df_treino_tratada['zip'].value_counts().reset_index(name='count').rename(columns={'index': 'zip'})
df_reason = df_treino_tratada['reason'].value_counts().reset_index(name='count').rename(columns={'index': 'reason'})
df_job_name = df_treino_tratada['job_name'].value_counts().reset_index(name='count').rename(columns={'index': 'job_name'})

df_zip_count = pd.merge(df_treino_tratada,df_zip,on=['zip'], how='left' )['count'].rename("df_zip_count")
df_reason_count = pd.merge(df_treino_tratada,df_reason,on=['reason'], how='left' )['count'].rename("df_reason_count")
df_job_name_count = pd.merge(df_treino_tratada,df_job_name,on=['job_name'], how='left' )['count'].rename("df_job_name_count")


# In[ ]:


df_FINAL = pd.concat([df_treino_tratada[['default','score_3','score_4','score_5','score_6','risk_rate','amount_borrowed',                                   'borrowed_in_months','credit_limit','income','ok_since','n_bankruptcies',                                   'n_defaulted_loans','n_accounts','n_issues']],                      pd.get_dummies(df_treino_tratada.score_1), pd.get_dummies(df_treino_tratada.score_2),                      pd.get_dummies(df_treino_tratada.sign), pd.get_dummies(df_treino_tratada.gender),                      pd.get_dummies(df_treino_tratada.real_state), pd.get_dummies(df_treino_tratada.facebook_profile),                      df_zip_count, df_reason_count, df_job_name_count], axis=1)


# In[ ]:


X, y = df_FINAL.loc[:, df_FINAL.columns != 'default'],df_FINAL['default']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_FINAL.loc[:, df_FINAL.columns != 'default'],                                                    df_FINAL['default'], test_size=0.3, random_state=57)


# In[ ]:


print (X_train.shape)
print (X_test.shape)
print (Y_train.shape)
print (Y_test.shape)


# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# Necessario utilizar outro pacote
from mlxtend.classifier import StackingClassifier
import numpy as np
from xgboost.sklearn import XGBClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

clf1_params = {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 1}
clf1 = GradientBoostingClassifier(**clf1_params)

clf2_params = {'learning_rate': 0.3, 'n_estimators': 150}
clf2 = AdaBoostClassifier(**clf2_params)

clf3_params = {'colsample_bytree': 0.9, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 7,                       'n_estimators': 200, 'nthread': 4, 'objective': 'binary:logistic', 'seed': 57, 'silent': 1,                       'subsample': 0.7, 'base_score': 0.8}
clf3 = XGBClassifier(**clf3_params)

eclf = VotingClassifier(estimators=[('gradBos',clf1),  ('adaBoost', clf2),  ('xgboost',clf3)], 
                        voting='soft')



# Ajustando cada modelo separadamente
for clf, label in zip([clf1,clf2,  clf3, eclf], ['gradBos', 'adaBoost', 'xgboost', 'Voting']):
    scores = cross_val_score(clf, X_train, Y_train, cv=4, scoring='roc_auc')
    print("roc_auc: %0.4f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#
# roc_auc: 0.7711 (+/- 0.01) [gradBos]
# roc_auc: 0.7737 (+/- 0.01) [adaBoost]
# roc_auc: 0.7724 (+/- 0.01) [xgboost]
# roc_auc: 0.7727 (+/- 0.01) [Voting]


# In[ ]:


eclf.fit(X_train, Y_train)

eclf_final = eclf.predict_proba(X_test)[:,1]


# In[ ]:


from sklearn.metrics import roc_auc_score


print("AUC: ", roc_auc_score(Y_test, eclf_final))

# Melhor XBoost: AUC:  0.7636338573553124


# ## Stacking

# In[ ]:




clf1_params = {'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'subsample': 1}
clf1 = GradientBoostingClassifier(**clf1_params)

clf2_params = {'learning_rate': 0.3, 'n_estimators': 150}
clf2 = AdaBoostClassifier(**clf2_params)

clf3_params = {'colsample_bytree': 0.9, 'learning_rate': 0.05, 'max_depth': 4, 'min_child_weight': 7,                       'n_estimators': 200, 'nthread': 4, 'objective': 'binary:logistic', 'seed': 57, 'silent': 1,                       'subsample': 0.7, 'base_score': 0.8}
clf3 = XGBClassifier(**clf3_params)

lr = LogisticRegression()


# Define que a regressao logistica utilizara a previsao dos 5 modelos
sclf = StackingClassifier(classifiers=[ clf1, clf2, clf3] ,meta_classifier=lr)


for clf, label in zip([ clf1, clf2, clf3,sclf], 
                      [
                       'gradBos', 
                       'adaBoost',
                       'xgboost',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X_train, Y_train, 
                                              cv=3, scoring='roc_auc')
    print("roc_auc: %0.4f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))


# ## O melhor foi o Ada separadamento

# In[ ]:


from xgboost.sklearn import XGBClassifier


clf2_params = {'learning_rate': 0.3, 'n_estimators': 150}
model = AdaBoostClassifier(**clf2_params)
print(1)

model.fit(X_train, Y_train)
print(1)

xgb_proba = model.predict_proba(X_test)[:,1]
print(1)


# In[ ]:


print("AUC: ", roc_auc_score(Y_test, xgb_proba))


# ### Aplicando na base teste

# In[ ]:


df_teste = pd.read_csv("teste_data.csv")


# In[ ]:


df_teste.info()


# In[ ]:


df_teste['credit_limit'].fillna(sub_credit_limit, inplace=True)
df_teste['ok_since'].fillna(sub_ok_since, inplace=True)
df_teste['n_bankruptcies'].fillna(sub_n_bankruptcies, inplace=True)
df_teste['n_defaulted_loans'].fillna(sub_n_defaulted_loans, inplace=True)
df_teste['n_issues'].fillna(sub_n_issues, inplace=True)

df_teste['reason'].fillna('NA1', inplace=True)
df_teste['sign'].fillna('NA2', inplace=True)
df_teste['gender'].fillna('NA3', inplace=True)
df_teste['facebook_profile'].fillna('NA4', inplace=True)
df_teste['job_name'].fillna('NA5', inplace=True)

df_teste.info()


# In[ ]:


#df_treino_tratada = DataFrameImputer().fit_transform(df_teste)
#df_treino_tratada=df_treino_tratada.reset_index()
df_treino_tratada=df_teste


# In[ ]:





# In[ ]:


#df_zip = df_treino_tratada['zip'].value_counts().reset_index(name='count').rename(columns={'index': 'zip'})
#df_reason = df_treino_tratada['reason'].value_counts().reset_index(name='count').rename(columns={'index': 'reason'})
#df_job_name = df_treino_tratada['job_name'].value_counts().reset_index(name='count').rename(columns={'index': 'job_name'})

df_zip_count = pd.merge(df_treino_tratada,df_zip,on=['zip'], how='left' )['count'].rename("df_zip_count")
df_reason_count = pd.merge(df_treino_tratada,df_reason,on=['reason'], how='left' )['count'].rename("df_reason_count")
df_job_name_count = pd.merge(df_treino_tratada,df_job_name,on=['job_name'], how='left' )['count'].rename("df_job_name_count")


# In[ ]:


df_FINAL = pd.concat([df_treino_tratada[['score_3','score_4','score_5','score_6','risk_rate','amount_borrowed',                                   'borrowed_in_months','credit_limit','income','ok_since','n_bankruptcies',                                   'n_defaulted_loans','n_accounts','n_issues']],                      pd.get_dummies(df_treino_tratada.score_1), pd.get_dummies(df_treino_tratada.score_2),                      pd.get_dummies(df_treino_tratada.sign), pd.get_dummies(df_treino_tratada.gender),                      pd.get_dummies(df_treino_tratada.real_state), pd.get_dummies(df_treino_tratada.facebook_profile),                      df_zip_count, df_reason_count, df_job_name_count], axis=1)


# In[ ]:


np.where(np.isnan(df_FINAL))


# In[ ]:


df_FINAL_2 = np.nan_to_num(df_FINAL)


# In[ ]:


np.where(np.isnan(df_FINAL_2))


# In[ ]:





# In[ ]:


xgb_classifier_y_prediction = model.predict_proba(df_FINAL_2)


# In[ ]:


df_probs=pd.DataFrame(xgb_classifier_y_prediction[:,1], columns=['prob'])


# In[ ]:


df_SEND = pd.concat([df_teste[['ids']], df_probs[['prob']]], axis=1)


# In[ ]:


df_SEND.to_csv('send_3_1.csv',index=False)


# In[ ]:





# In[ ]:




