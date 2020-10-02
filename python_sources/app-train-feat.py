#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.chdir('/kaggle/input/home-credit-default-risk')
#os.chdir('/Users/xianglongtan/Desktop/kaggle')
#print(os.getcwd())
#print(os.listdir())
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_activity = 'all'
# Any results you write to the current directory are saved as output.
app_train = pd.read_csv('application_train.csv')
#app_train = pd.read_csv('application_train.csv')
#app_train.head()
app_test = pd.read_csv('application_test.csv')
#app_test = pd.read_csv('application_test.csv')
#app_test.head()


# In[2]:


os.chdir('../imputed')
#print(os.listdir())
train_and_test_imputed = pd.read_csv('train_and_test_imputed.csv')


# In[3]:


#os.chdir('/Users/xianglongtan/Desktop/kaggle/submission')
os.chdir('/kaggle/working')
os.getcwd()


# In[4]:


# Check missing value
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',1000)
#print(train_and_test_imputed.isnull().any().sum())
#train_and_test_imputed = train_and_test_imputed.drop('Unnamed: 0',axis=1)
#train_and_test_imputed.head(10)

# Process features 
## Encoding categorical data
### Transform Education
map_educ = {'Secondary / secondary special':int(2),'Higher education':int(4),'Incomplete higher':int(3),'Lower secondary':int(1),'Academic degree':int(5)}
train_and_test_imputed['NAME_EDUCATION_TYPE'] = train_and_test_imputed['NAME_EDUCATION_TYPE'].map(map_educ)
#train_and_test_imputed['NAME_EDUCATION_TYPE'].head()
train_and_test_imputed['NAME_EDUCATION_TYPE'] = train_and_test_imputed['NAME_EDUCATION_TYPE'].fillna(0)
#train_and_test_imputed['NAME_EDUCATION_TYPE'].head()

### One-hot encoding categorical features
one_hot = 1
if one_hot == 1:
    tnt_imp_dum = pd.get_dummies(train_and_test_imputed)
    hour = pd.get_dummies(tnt_imp_dum['HOUR_APPR_PROCESS_START'],prefix='HOUR')
    tnt_imp_dum = pd.concat([tnt_imp_dum,hour],axis=1).drop('HOUR_APPR_PROCESS_START',axis=1)
    tnt_imp_dum.head()
    ## Normalize numeric features
    # Combine these (average)
    #train_and_test['MEAN_CNT_SOCIAL_CIRCLE'] = (train_and_test['OBS_30_CNT_SOCIAL_CIRCLE']+train_and_test['DEF_30_CNT_SOCIAL_CIRCLE']+train_and_test['OBS_60_CNT_SOCIAL_CIRCLE']+train_and_test['DEF_60_CNT_SOCIAL_CIRCLE'])/4
    # Transform to years
    #train_and_test['TIME_LAST_PHONE_CHANGE'] = train_and_test['DAYS_LAST_PHONE_CHANGE ']/365
    # round counting features
    col = [x for x in tnt_imp_dum.columns if x.startswith('AMT_REQ_CREDIT_BUREAU')]
    tnt_imp_dum[col] = tnt_imp_dum[col].apply(lambda x: round(x))
    # extract columns that have more than 2 unique values and the maximum large than 1 and normalized
    cols = [x for x in tnt_imp_dum.columns if len(tnt_imp_dum[x].value_counts())>2 and np.nanmax(np.abs(tnt_imp_dum[x].values)) > 1]
    cols.remove('SK_ID_CURR')
    cols.remove('Unnamed: 0')
    from sklearn import preprocessing
    tnt_imp_dum[cols] = preprocessing.normalize(tnt_imp_dum[cols],axis=0)
    #standardscaler = preprocessing.StandardScaler()
    #tnt_imp_dum[cols] = standardscaler.fit_transform(tnt_imp_dum[cols])
    #for col in cols:
        #tnt_imp_dum[col] = preprocessing.normalize(tnt_imp_dum[col])
else:
    col = [x for x in tnt_imp_dum.columns if x.startswith('AMT_REQ_CREDIT_BUREAU')]
    tnt_imp_dum[col] = tnt_imp_dum[col].apply(lambda x: round(x))
    col = [x for x in tnt_imp_dum.columns if tnt_imp_dum[x].dtype == 'float64']
    #print(col)
    #from sklearn import preprocessing
    #normalizer = preprocessing.Normalizer(norm='l2')
    #tnt_imp_dum[col] = normalizer.fit_transform(tnt_imp_dum[col])


# # Split training set and test set

# In[5]:


train_X = tnt_imp_dum.loc[0:len(app_train)-1,:]
test_X = tnt_imp_dum.loc[len(app_train):(len(app_train)+len(app_test)),:]
train_Y = app_train['TARGET']
train_X = train_X.drop(['Unnamed: 0','SK_ID_CURR'],axis=1)
train_X.head()


# In[6]:


test_ID = pd.DataFrame(test_X['SK_ID_CURR'])
test_ID = test_ID.reset_index().drop('index',axis=1)
test_X = test_X.drop(['Unnamed: 0','SK_ID_CURR'],axis=1)
#test_X.head()
#test_ID.head()


# In[7]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
import time
from plotnine import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
seed = 0
X_train,X_val,y_train,y_val = train_test_split(train_X,train_Y,random_state=seed,stratify=train_Y)


# In[8]:


flag = None
if flag == 'KNN':
    start = time.time()
    model = KNeighborsClassifier()
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_knn1.csv')
elif flag == 'SVM':
    start = time.time()
    model = SVC()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('only_nonan_svm1.csv')
elif flag == 'logit':
    start = time.time()
    model = LogisticRegression(C=2,max_iter=200,random_state=seed)
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_logit1.csv')
elif flag == 'RF':
    start = time.time()
    model = RandomForestClassifier(n_estimators=120,random_state=seed)
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_rf1.csv')
elif flag == 'XGB':
    start = time.time()
    model = XGBClassifier(max_depth = 6, n_estimators = 200)
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y,eval_metric='auc')
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_xgb1.csv')
elif flag == 'ET':
    start = time.time()
    model = ExtraTreesClassifier()
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_ext1.csv')
elif flag == 'AB':
    start = time.time()
    model = AdaBoostClassifier(base_estimator = LogisticRegression(C=2,max_iter=200,random_state=seed),random_state=seed, learning_rate=0.8)
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_ab1.csv')
elif flag == 'BC':
    start = time.time()
    model = BaggingClassifier(base_estimator=LogisticRegression(C=2,max_iter=200,random_state=seed),n_estimators = 20, random_state=seed)
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_bc1.csv')
elif flag == 'GNB':
    start = time.time()
    model = GaussianNB(prior=[24825/(282686+24825),282686/(282686+24825)])
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_gnb1.csv')
elif flag == 'DT':
    start = time.time()
    model = DecisionTreeClassifier()
    #model.fit(X_train, y_train)
    #pred_train = model.predict(X_train)
    #pred_val = model.predict(X_val)
    model.fit(train_X,train_Y)
    end  = time.time()
    #print('F1 score of training set:',f1_score(pred_train, y_train, average='weighted'))
    #print('F1 score of test set:',f1_score(pred_val, y_val, average='weighted'))
    print('Done! Time spent:',end-start)
    pred_test = pd.DataFrame(model.predict_proba(test_X)).loc[:,1]
    result = pd.concat([test_ID,pred_test],axis=1)
    result.columns = ['SK_ID_CURR','TARGET']
    result = result.set_index('SK_ID_CURR')
    result.to_csv('app_train_dt1.csv')
else:
    pass


# In[9]:


result.head(20)


# In[24]:





# In[ ]:


# CV for logistic regression
cv_logit = 0
if cv_logit == 1
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    model = LogisticRegression(class_weight='balanced')
    param_grid = dict(C=[0.1,0.5,1,1.5,2],
                     penalty=['l1','l2'])

    kfold = StratifiedKFold(n_splits = 5, shuffle=True,random_state = seed)
    grid_search = GridSearchCV(model, param_grid, scoring = 'f1', cv=kfold)
    start = time.time()
    grid_result = grid_search.fit(train_X,train_Y)
    end = time.time()
    print(end-start)


# In[ ]:


# CV for XGB
cv_xgb = 0
if xv_xgb == 1:
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    model = XGBClassifier(class_weight='balanced')
    learning_rate = [0.0001, 0.001,0.01]
    max_depth = [2,4,6,8]
    subsample = [0.2, 0.4, 0.6, 0.8, 1.0]
    colsample_bytree = [0.2, 0.4, 0.6, 0.8, 1.0]
    n_estimator = range(100, 500, 50)

    param_grid = dict(learning_rate = learning_rate,
                     #max_depth = max_depth,
                     #subsample = subsample,
                     #colsample_bytree = colsample_bytree,
                     n_estimator = n_estimator)
    kfold = StratifiedKFold(n_splits = 10, shuffle=True,random_state = seed)
    grid_search = GridSearchCV(model, param_grid, scoring = 'f1', cv=kfold)
    start = time.time()
    grid_result = grid_search.fit(train_X,train_Y)
    end = time.time()
    print(end-start)


# In[10]:


NN = 1
if NN == 1:
    import tensorflow as tf

    # Hyperparam
    LR = 0.0001 # learning rate
    ITERATION = 5000
    BATCH_SIZE = 20000
    KEEP_PROB = 0.7
    NUM_FEAT = 264
    NUM_CLASS = 2

    class DataIter2():
        def __init__(self, X,Y):
            self.X = X
            self.Y = Y
            self.size = len(self.X)
            self.epochs = 0
            self.df = pd.concat([X,Y],axis=1)
            self.shuffle()
        def shuffle(self):
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.cursor = 0
        def next_batch(self, n):
            if self.cursor + n > self.size:
                #res = self.df.iloc[self.cursor:self.size]
                self.epochs += 1
                self.shuffle()
            res = self.df.iloc[self.cursor:(self.cursor+n)]
            self.cursor += n
            return x, res['y'], res['length'],res.index

    class DataIter():
        def __init__(self, X,Y):
            self.X = X
            self.Y = Y
            self.size = len(self.X)
            self.epochs = 0
            self.df = pd.concat([X,Y],axis=1)
            self.pos = self.df.loc[self.Y == 1]
            self.neg = self.df.loc[self.Y == 0]
        def next_batch(self,n):
            #X_train,X_val,y_train,y_val = train_test_split(X,Y,test_size = n/self.size,random_state=seed,stratify=Y)
            #res = pd.concat([X_val,y_val],axis=1)
            pos_sample = self.pos.sample(n, replace=True)
            neg_sample = self.neg.sample(n, replace=True)
            res = pd.concat([neg_sample, pos_sample],axis=0)
            return res

    class DataIter2():
        def __init__(self, X,Y):
            self.X = X
            self.Y = Y
            self.size = len(self.X)
            self.epochs = 0
            self.df = pd.concat([X,Y],axis=1)
            self.shuffle()
        def shuffle(self):
            self.df = self.df.sample(frac=1).reset_index(drop=True)
            self.cursor = 0
        def next_batch(self, n):
            if self.cursor + 2*n > self.size:
                #res = self.df.iloc[self.cursor:self.size]
                self.epochs += 1
                self.shuffle()
            res = self.df.iloc[self.cursor:(self.cursor+2*n)]
            self.cursor += n
            return res


    # build graph
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32,[BATCH_SIZE*2, NUM_FEAT])
    y = tf.placeholder(tf.int32,[BATCH_SIZE*2])
    keep_prob = tf.constant(KEEP_PROB)
    nn_inputs = tf.layers.dense(x, units = round(1.5*NUM_FEAT), kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid)# hidden layer 1
    nn_inputs = tf.layers.dropout(nn_inputs, KEEP_PROB,training=True)
    #print(nn_inputs.get_shape)
    nn_inputs = tf.layers.dense(x, units = round(2*NUM_FEAT), kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid) # hidden layer 2
    nn_inputs = tf.layers.dropout(nn_inputs, KEEP_PROB,training=True)
    #print(nn_inputs.get_shape)
    nn_inputs = tf.layers.dense(x, units = round(1*NUM_FEAT), kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid) # hidden layer 3
    nn_inputs = tf.layers.dropout(nn_inputs, KEEP_PROB,training=True)
    #print(nn_inputs.get_shape)
    nn_inputs = tf.layers.dense(x, units = 0.5*NUM_FEAT, kernel_initializer = tf.truncated_normal_initializer(),activation = tf.nn.sigmoid) # hidden layer 4
    nn_inputs = tf.layers.dropout(nn_inputs, KEEP_PROB,training=True)
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [0.5*NUM_FEAT, NUM_CLASS],initializer = tf.truncated_normal_initializer())
        b = tf.get_variable('b', [NUM_CLASS],initializer = tf.constant_initializer(0.0))
    logits = tf.matmul(nn_inputs, W)+b
    #print(logits.get_shape)
    preds = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(preds,1), tf.int32)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = preds))
    precision, precision_op = tf.metrics.precision(y,prediction)
    #print(precision.get_shape)
    recall, recall_op = tf.metrics.recall(y,prediction)
    #print(recall.get_shape)
    f1score = 2*precision*recall/(precision+recall)
    train_step = tf.train.AdamOptimizer(LR).minimize(loss)

    # session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        tr = DataIter2(train_X,train_Y)
        for i in range(ITERATION):
            batch = tr.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
            if i%200 == 0:
                _,prec = sess.run([precision,precision_op], feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
                _,rec = sess.run([recall,recall_op], feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
                f1s = sess.run(f1score, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
                los = sess.run(loss, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})
                print('losss after',i,'round',los)
                print('precision after',i,'round',prec)
                print('recall after',i,'round',rec)
                print('F1 score after',i,'round:',f1s)
                #print('\n----------------------------------\n')
                print('logits:\n',sess.run(logits,feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})[0:10])
                print('preds:\n',sess.run(preds, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})[0:10])
                print('prediction:\n',sess.run(prediction, feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})[0:10])
                print('y:\n',sess.run(y,feed_dict={x:batch.iloc[:,0:-1], y:batch['TARGET']})[0:10])
                print('\n----------------------------------\n')
        cursor = 0
        while cursor <= len(test_X):
            if cursor+2*BATCH_SIZE <= len(test_X):
                te = test_X.iloc[cursor:cursor+2*BATCH_SIZE]
            else:
                te = pd.concat([test_X.iloc[cursor:len(test_X)],test_X.iloc[0:2*BATCH_SIZE-len(test_X.iloc[cursor:len(test_X)])]])
            results = sess.run(preds, feed_dict={x:te})
            if cursor == 0:
                prediction_test = pd.DataFrame(data=results, columns=['0','TARGET'])
            else:
                prediction_test = pd.concat([prediction_test, pd.DataFrame(data=results,columns=['0','TARGET'])])
            cursor += 2*BATCH_SIZE
    result = prediction_test.iloc[0:len(test_X)]
    pred_test = pd.DataFrame(result['TARGET']).reset_index()
    result_final = pd.concat([test_ID, pred_test],axis=1).drop('index',axis=1)
    result_final.columns = ['SK_ID_CURR','TARGET']
    result_final = result_final.set_index('SK_ID_CURR')
    result_final.to_csv('app_feat_NN1.csv')


# In[11]:


result_final.head()


# In[12]:





# In[ ]:




