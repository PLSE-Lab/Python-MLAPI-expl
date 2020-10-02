# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from string import punctuation

import matplotlib.pyplot as plt
from pylab import plot, show, subplot, specgram, imshow, savefig

RS = 23
ROUNDS = 315

print("Started")
np.random.seed(RS)
input_folder = '../input/'


def text_to_wordlist(text):
    # Clean the text, with the option to remove stop_words and to stem words.
    '''
    # Clean the text
    text = re.sub(r"what\'s", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"0k", " 0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" uk ", " england ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iphone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"iii", "3", text)
    text = re.sub(r"the us", "america", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r"^[A-Za-z0-9\']", " ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, shorten words to their stems
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    '''
    # Return a list of words
    return text.split()


def train_xgb(X, y, params):
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
    x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RS)

    xg_train = xgb.DMatrix(x, label=y_train)
    xg_val = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(xg_train, 'train'), (xg_val, 'eval')]
    return xgb.train(params, xg_train, ROUNDS, watchlist)


def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def main():
    params = {}
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params["subsample"] = 0.5
    params['eta'] = 0.15
    params['max_depth'] = 7
    params["colsample_bytree"] = 0.5
    params['silent'] = 1
    params['seed'] = RS
    params["gamma"] = 0.0
    params["reg_alpha"] = 0.3
    params["reg_lambda"] = 0.3
    params["min_child_weight"] = 1

    df_train = pd.read_csv(input_folder + 'train.csv')
    df_test = pd.read_csv(input_folder + 'test.csv')
    print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

    print("Features processing, be patient...")

    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(count, eps=10000, min_count=2):
        return 0 if count < min_count else 1 / (count + eps)

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    words = text_to_wordlist(" ".join(train_qs).lower())
    # words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    stops = set(stopwords.words("english"))

    def word_shares(row):
        q1 = set(text_to_wordlist(str(row['question1']).lower()))
        q1words = q1.difference(stops)
        if len(q1words) == 0:
            return '0:0:0:0:0:0'

        q2 = set(text_to_wordlist(str(row['question2']).lower()))
        q2words = q2.difference(stops)
        if len(q2words) == 0:
            return '0:0:0:0:0:0'

        q1stops = q1.intersection(stops)
        q2stops = q2.intersection(stops)

        shared_words = q1words.intersection(q2words)
        shared_weights = [weights.get(w, 0) for w in shared_words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
        sumtw = np.sum(total_weights) ;
        if sumtw == 0:
            R1 = 0  # tfidf share
        else:
            R1 = np.sum(shared_weights) / sumtw  # tfidf share

        R21 = len(shared_words) / (len(q1words))  # count share
        R22 = len(shared_words) / (len(q2words))  # count share
        R31 = len(q1stops) / len(q1words)  # stops in q1
        R32 = len(q2stops) / len(q2words)  # stops in q2
        return '{}:{}:{}:{}:{}:{}'.format(R1, R21,R22, len(shared_words), R31, R32)

    df = pd.concat([df_train, df_test])
    df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

    x = pd.DataFrame()

    x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
    x['shared_count1'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))
    x['shared_count2'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
    x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
    x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
    x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']
    x['shared_sum'] =  x['shared_count1'] + x['shared_count2']
    x['shared_dif'] = x['shared_count1'] - x['shared_count2']
    x['shared_mul'] = x['shared_count1']*x['shared_count2']
    x['cross_mul1'] = x['shared_count1'] * x['stops1_ratio']
    x['cross_mul2'] = x['shared_count2'] * x['stops2_ratio']
    x['cross_idf'] = x['word_match'] *x['tfidf_word_match'] ;
    x['sum_idf'] = x['word_match'] + x['tfidf_word_match'] ;
    x['dif_idf'] = x['word_match'] - x['tfidf_word_match'] ;

    x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
    x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
    x['diff_len'] = x['len_q1'] - x['len_q2']

    x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

    x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
    x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
    x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

    x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
    x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
    x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

    x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
    x['duplicated'] = df.duplicated(['question1', 'question2']).astype(int)

    # ... YOUR FEATURES HERE ...

    feature_names = list(x.columns.values)
    create_feature_map(feature_names)
    print("Features: {}".format(feature_names))

    x_train = x[:df_train.shape[0]]
    x_test = x[df_train.shape[0]:]
    y_train = df_train['is_duplicate'].values
    del x, df_train

    if 1:  # Now we oversample the negative class - on your own risk of overfitting!
        pos_train = x_train[y_train == 1]
        neg_train = x_train[y_train == 0]

        print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))
        p = 0.165
        scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
        while scale > 1:
            neg_train = pd.concat([neg_train, neg_train])
            scale -= 1
        neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
        print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

        x_train = pd.concat([pos_train, neg_train])
        y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
        del pos_train, neg_train

    print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
    clr = train_xgb(x_train, y_train, params)
    preds = predict_xgb(clr, x_test)

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = preds
    sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)

    print("Features importances...")
    importance = clr.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    ft = pd.DataFrame(importance, columns=['feature', 'fscore'])

    ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
    plt.gcf().savefig('features_importance.png')


main()
print("Done.")