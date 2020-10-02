# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from sklearn import model_selection
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import Normalizer

from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# import the datasets
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market, news) = env.get_training_data()
news.shape

top_10_idx = np.argsort(market["returnsClosePrevRaw1"])[-10:]
days = env.get_prediction_days()
(market, news, predictions_template_df) = next(days)
#market.head()


# check if there are abnormal prices changes in a single day

market['price_diff'] = market['close'] - market['open']
market['close/open'] = market['close'] / market['open']
market['assetCode_mean_open'] = market.groupby('assetCode')['open'].transform('mean')
market['assetCode_mean_close'] = market.groupby('assetCode')['close'].transform('mean')

# replace abnormal data record

for i, row in market.loc[market['close/open'] >= 1.5].iterrows():
    if np.abs(row['assetCode_mean_open'] - row['open']) > np.abs(row['assetCode_mean_close'] - row['close']):
        market.iloc[i,5] = row['assetCode_mean_open']
    else:
        market.iloc[i,4] = row['assetCode_mean_close']
    

for i, row in market.loc[market['close/open'] <= 0.5].iterrows():
    if np.abs(row['assetCode_mean_open'] - row['open']) > np.abs(row['assetCode_mean_close'] - row['close']):
        market.iloc[i,5] = row['assetCode_mean_open']
    else:
        market.iloc[i,4] = row['assetCode_mean_close']
        

fill_cols = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']
              
market = market.sort_values(by = ['assetCode','time'], ascending=[True, True])
        

for i in market[fill_cols]:
    market[i] = market[i].fillna(method = 'ffill')

for i in market[fill_cols]:
    market[i] = market[i].fillna(market[i].mean())


market = market.drop(['price_diff','close/open','assetCode_mean_open','assetCode_mean_close'], axis = 1)
market.info
### work on news data

news_df=news.loc[news["time"]>="2016-01-01 00:00:00+0000"].reset_index(drop=True)
news_df.shape

#lda modeling 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

cv = CountVectorizer(min_df = 2,
                     max_features = 10,
                     analyzer = "word",
                     ngram_range = (1, 1),
                     stop_words = "english")
count_vectors = cv.fit_transform(news_df["headline"])
count_vectors.shape

lda_model = LatentDirichletAllocation(n_components = 5, 
                                      # we choose a small n_components for time convenient
                                      learning_method = "online",
                                      max_iter = 20,
                                      random_state = 32)

news_topics = lda_model.fit_transform(count_vectors)

# Words in each topics
n_top_words = 10
topic_summaries = []
topic_word = lda_model.components_
vocab = cv.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(" ".join(topic_words))
    print("Topic {}: {}".format(i, " | ".join(topic_words)))

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 32, n_iter = 500)
tsne_lda = tsne_model.fit_transform(news_topics)
news_topics = np.matrix(news_topics)
doc_topics = news_topics/news_topics.sum(axis = 1)

lda_keys = []
for i, tweet in enumerate(news["headline"]):
    lda_keys += [doc_topics[i].max()]

tsne_lda_df = pd.DataFrame(tsne_lda, columns = ["x", "y"])
tsne_lda_df["headline"] = news["headline"].values
tsne_lda_df["assetCodes"] = news["assetCodes"].values
tsne_lda_df["assetName"]=news["assetName"]
tsne_lda_df["relevance"] = news["relevance"].values
tsne_lda_df["sentimentNegative"]= news["sentimentNegative"]
tsne_lda_df["sentimentNeutral"]=news["sentimentNeutral"]
tsne_lda_df["sentimentPositive"]=news["sentimentPositive"]
tsne_lda_df["topics"] = lda_keys
tsne_lda_df["time"]=news["time"]
tsne_lda_df["firstCreated"]=news["firstCreated"]
tsne_lda_df["sentimentClass"]=news["sentimentClass"]
tsne_lda_df["sourceTimestamp"]=news["sourceTimestamp"]

pd.set_option('max_columns',40)
tsne_lda_df.head(2)
tsne_lda_df["topics"][:1]
type(tsne_lda_df["time"])
type(news["time"])

tsne_lda_df.info()
news.info()


def news_prep(news_df):
    news_df['assetCodes'] = news_df['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_df

news_df = news_prep(tsne_lda_df)
news_df.head(2)

def unstack_assetcodes(news_df):
    codes = []
    indexes = []
    for i, values in news_df['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    return index_df

news_index = unstack_assetcodes(news_df)
news_index.head(3)

def merge_news_index(news_df, index_df):
    news_df['news_index'] = news_df.index.copy()

    news_unstack = index_df.merge(news_df, how = 'left', on = 'news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack

merge_news = merge_news_index(news_df, news_index)
merge_news.info()

def group_news_value(news_df):
    news_df['date'] = news_df.firstCreated.dt.date
    group_values = news_df.groupby(['assetCode', 'date']).agg('mean')
    group_values.reset_index(inplace = True)

    return group_values

news_train = group_news_value(merge_news)
news_train.info()



def data_prep(market_train,news_train):
     #market_train.time = market_train.time.dt.date
     #news_train.time = news_train.time.dt.hour
     #news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
     #news_train.firstCreated = news_train.firstCreated.dt.date
     #news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
     #news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
     kcol = ['firstCreated', 'assetCodes']
     news_train = news_train.groupby(kcol, as_index=False).mean()
     market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], 
                             right_on=['firstCreated', 'assetCodes'])
     lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
     market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    
    
     market_train = market_train.dropna(axis=0)
    
     return market_train

market_train = data_prep(market, tsne_lda_df)
market_train.shape


# combined.info()
# combined.isna().sum()
# combined.shape

target = (combined.returnsOpenNextMktres10 >= 0).astype('int8')

feat_cols = [c for c in combined.columns if c not in['assetCode', 'assetCodes', 'assetCodesLen','assetName', 'assetCodeT','firstCreated','time_x','time_y','date','universe','assetCodeLen','returnsOpenNextMktres10']]

X = combined[feat_cols]

X = X.fillna(X.mean())

train, test = model_selection.train_test_split(X.index.values, test_size = 0.15, random_state=11)

lgb = LGBMClassifier(
    n_jobs = 4,
    ojective='binary',
    boosting='gbdt',
    learning_rate = 0.05,
    max_depth = 8,
    num_leaves = 80,
    n_estimators = 200,
    bagging_fraction = 0.8,
    feature_fraction = 0.9)

lgb.fit(X.loc[train],target.loc[train])

print("lgb accuracy : %f" % \
      accuracy_score(lgb.predict(X.loc[test]),
                     target.loc[test]))
                     
print("lgb AUC : %f" % \
      roc_auc_score(target.loc[test].values,
                    lgb.predict_proba(X.loc[test])[:, 1]))
                    

### submission

n_days = 0

for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
    n_days += 1
    if n_days % 50 == 0:
        print (n_days, end = ' ')
    
    market_obs_df = data_prep(market_obs_df,news_obs_df)
    # market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_test = market_obs_df[feat_cols]
    X_test = X_test.fillna(X_test.mean()).values
    lp = lgb.predict_proba(X_test)
    confidence = 2 * lp[:,1] - 1
    
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'], 'confidence':confidence})
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    
    env.predict(predictions_template_df)

env.write_submission_file()

