#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import *
import sklearn
import pandas as pd
import numpy as np
import lightgbm as lgb


# In[ ]:


train = pd.read_csv('../input/training_variants')
test = pd.read_csv('../input/stage2_test_variants.csv')
trainx = pd.read_csv('../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('../input/stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[ ]:


train = pd.merge(train, trainx, how='left', on='ID').fillna('')
train = train.drop(["ID"], axis=1)

test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values


# In[ ]:


df_variants_test = pd.read_csv('../input/test_variants', usecols=['ID', 'Gene', 'Variation'])
df_text_test = pd.read_csv('../input/test_text', sep='\|\|', engine='python', 
                           skiprows=1, names=['ID', 'Text'])
df_variants_test['Text'] = df_text_test['Text']
df_test = df_variants_test

# read stage1 solutions
df_labels_test = pd.read_csv('../input/stage1_solution_filtered.csv')
df_labels_test['Class'] = pd.to_numeric(df_labels_test.drop('ID', axis=1).idxmax(axis=1).str[5:])

# join with test_data on same indexes
df_test = df_test.merge(df_labels_test[['ID', 'Class']], on='ID', how='left').drop('ID', axis=1)
df_test = df_test[df_test['Class'].notnull()]

# join train and test files
df_stage_2_train = pd.concat([train, df_test])
df_stage_2_train.reset_index(drop=True, inplace=True)
train = df_stage_2_train
df_stage_2_train.info()


# In[ ]:


y = train['Class'].values
train = train.drop(['Class'], axis=1)
train.info()


# In[ ]:


df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)


# In[ ]:


#commented for Kaggle Limits
for i in range(56):
   df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
   df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')


# In[ ]:


gen_var_lst = sorted(list(train.Gene.unique()) + list(train.Variation.unique()))
print(len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0


# In[ ]:


#commented for Kaggle Limits
for gen_var_lst_itm in gen_var_lst:
   if i_ % 100 == 0: print(i_)
   df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
   i_ += 1


# In[ ]:


for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = preprocessing.LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = preprocessing.LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 


# In[ ]:


train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]

class cust_regression_vals(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        x = x.drop(['Gene', 'Variation','ID','Text'],axis=1).values
        return x

class cust_txt_col(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        return x[self.key].apply(str)


# In[ ]:


print('Pipeline...')
fp = pipeline.Pipeline([
    ('union', pipeline.FeatureUnion(
        n_jobs = -1,
        transformer_list = [
            ('standard', cust_regression_vals()),
            ('pi1', pipeline.Pipeline([('Gene', cust_txt_col('Gene')), ('count_Gene', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 10))), ('tsvd1', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            ('pi2', pipeline.Pipeline([('Variation', cust_txt_col('Variation')), ('count_Variation', feature_extraction.text.CountVectorizer(analyzer=u'char', ngram_range=(1, 10))), ('tsvd2', decomposition.TruncatedSVD(n_components=20, n_iter=25, random_state=12))])),
            #commented for Kaggle Limits
            ('pi3', pipeline.Pipeline([('Text', cust_txt_col('Text')), ('tfidf_Text', feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))), ('tsvd3', decomposition.TruncatedSVD(n_components=50, n_iter=25, random_state=12))]))
        ])
    )])

train = fp.fit_transform(train); print(train.shape)
test = fp.transform(test); print(test.shape)

y = y - 1 #fix for zero bound array


# In[ ]:


denom = 0
for i in range(2):
    params = {
        'learning_rate': 0.01,
        'max_depth': 6,
        'num_leaves': 50,
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 9
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.15, random_state=i)
    train_set = lgb.Dataset(x1, y1)
    valid_set = lgb.Dataset(x2, y2)
    
    model = lgb.train(params, train_set, 1000, [valid_set], ["val"], verbose_eval=50, early_stopping_rounds=50)
    score1 = metrics.log_loss(y2, model.predict(x2), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(test)
        preds += pred
    else:
        pred = model.predict(test)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)


# In[ ]:


preds /= denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)
submission.head()


# In[ ]:


submission.info()

