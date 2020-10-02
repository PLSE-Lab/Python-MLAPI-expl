import pandas as pd
import xgboost as xgb
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from collections import Counter

EMPTY = 'nan'
#train and test files:    

print('loading')  
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print('preprocessing')
train.fillna(EMPTY, inplace=True)
test.fillna(EMPTY, inplace=True)

train.question1 = train.question1.map(lambda x: x.rstrip())
train.question2 = train.question2.map(lambda x: x.rstrip())
test.question1 = test.question1.map(lambda x: x.rstrip())
test.question2 = test.question2.map(lambda x: x.rstrip())

#train.loc[train['question1'].map(lambda x: len(x) < 2),'question1'] = EMPTY
#train.loc[train['question2'].map(lambda x: len(x) < 2),'question2'] = EMPTY
#test.loc[test['question1'].map(lambda x: len(x) < 2),'question1'] = EMPTY
#test.loc[test['question2'].map(lambda x: len(x) < 2),'question2'] = EMPTY

# Engineering level 1:

# LENGHT of questions 1 and 2:
print('len features') 
train['lenQ1'] = train.question1.apply(lambda x: len(x))
train['lenQ2'] = train.question2.apply(lambda x: len(x))

test['lenQ1'] = test.question1.apply(lambda x: len(x))
test['lenQ2'] = test.question2.apply(lambda x: len(x))


#Dif in two lenghs

train['dif_len']=abs(train.lenQ1-train.lenQ2)
test['dif_len']=abs(test.lenQ1-test.lenQ2)

#Character length of questions without spaces

train['char_lenQ1']=train.question1.apply(lambda x: len(''.join(set(x.replace(' ','')))))
train['char_lenQ2']=train.question2.apply(lambda x: len(''.join(set(x.replace(' ','')))))
test['char_lenQ1']=test.question1.apply(lambda x: len(''.join(set(x.replace(' ','')))))
test['char_lenQ2']=test.question2.apply(lambda x: len(''.join(set(x.replace(' ','')))))

#Number of words

train['nWordsQ1']=train.question1.apply(lambda x: len(x.split()))
train['nWordsQ2']=train.question2.apply(lambda x: len(x.split()))

test['nWordsQ1']=test.question1.apply(lambda x: len(x.split()))
test['nWordsQ2']=test.question2.apply(lambda x: len(x.split()))

#Common words netween question 1 and 2
print('common words') 
train['common_words']=train.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)
test['common_words']=test.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)



#From Anokas without oversample the negative class. https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb



stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words = [w for w in q1words.keys() if w in q2words]
    R = 2 * len(shared_words) / (len(q1words) + len(q2words))
    return R
    
print('w m share')    
train['train_word_match'] = train.apply(word_match_share, axis=1, raw=True)
test['train_word_match'] = test.apply(word_match_share, axis=1, raw=True)


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)
test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist()).astype(str)
     
eps = 5000 
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}



     
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / (np.sum(total_weights) + 1e-8)
    return R

train['tfidf_word_match'] = train.apply(tfidf_word_match_share, axis=1, raw=True)
test['tfidf_word_match'] = test.apply(tfidf_word_match_share, axis=1, raw=True)





#XGB function
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 4
    param['silent'] = 0
    #param['num_class'] = 2
    param['eval_metric'] = "logloss"
    param['min_child_weight'] = 7
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    param['booster']='gbtree'
    #param['early_stopping_rounds'] = 2
    #param['max_delta_step']=1
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
    
 
features_to_use  = ['lenQ1','lenQ2', 'dif_len', 'char_lenQ1', 'char_lenQ2', 'nWordsQ1', 'nWordsQ2',
       'common_words','train_word_match','tfidf_word_match']

train_X = np.array(train[features_to_use])
test_X = np.array(test[features_to_use])


train_y = np.array(train['is_duplicate'])
print(train_X.shape, test_X.shape)   

#Cross validation


#cv_scores = []
#kf = model_selection.KFold(n_splits=7, shuffle=True, random_state=2016)
#for dev_index, val_index in kf.split(range(train_X.shape[0])):
 #       dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]
  #      dev_y, val_y = train_y[dev_index], train_y[val_index]
   #     preds, model = runXGB(dev_X, dev_y, val_X, val_y)
    #    cv_scores.append(log_loss(val_y, preds))
     #   print(cv_scores)
      #  break
      
preds, model = runXGB(train_X, train_y, test_X, num_rounds=300)
out_df = pd.DataFrame(preds)
out_df.columns = ["is_duplicate"]
out_df["test_id"] = test.test_id.values
out_df.to_csv("submision.csv", index=False)






