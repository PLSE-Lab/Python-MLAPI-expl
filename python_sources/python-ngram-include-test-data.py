# This Python 3 environment comes with many helpful analytics libraries installed
# This script would give public score about 0.15+


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def train_ngram_pred(input_lists, ngram_n):
    ngram_table = dict()

    return train_ngram_pred_cont(ngram_table, input_lists, ngram_n)


def train_ngram_pred_cont(ngram_table, input_lists, ngram_n):
    for l in input_lists:
        ngram = list()
        for c in l:
            if len(ngram) == ngram_n:
                ngram_key = tuple(ngram)
                pred = ngram_table.get(ngram_key)
                if pred:
                    pred_freq = pred.get(c)
                    if pred_freq:
                        pred[c] = pred_freq + 1
                    else:
                        pred[c] = 1
                else:
                    ngram_table[ngram_key] = (dict({c: 1}))

                ngram.pop(0)
            ngram.append(c)

    return ngram_table


def predict_ngram_pred(input_lists, ngram_table, ngram_n):
    predicts = list()

    for l in input_lists:
        if len(l) < ngram_n:
            predicts.append(None)
        else:
            ngram_key = tuple(l[-ngram_n:])
            pred = ngram_table.get(ngram_key)
            if pred:
                max_occur_count = 0
                max_occur = None
                for c, count in pred.items():
                    if count > max_occur_count:
                        max_occur = c
                        max_occur_count = count
                predicts.append(max_occur)
            else:
                predicts.append(None)

    return predicts


dataset = pd.read_csv('../input/train.csv')
dataset['Integers'] = dataset['Sequence'].str.split(',').apply(lambda x: list(map(int, x)))

testset = pd.read_csv('../input/test.csv')
testset['Integers'] = testset['Sequence'].str.split(',').apply(lambda x: list(map(int, x)))

train = dataset['Integers'].values
test = testset['Integers'].values

ngram_preds = list()
# ngram up to 10
for ngram_n in range(10, 1, -1):
    # ngram with training data
    ngram_table = train_ngram_pred(train, ngram_n)
    # continue with testing data as well
    ngram_table = train_ngram_pred_cont(ngram_table, test, ngram_n)
    # make prediction
    ngram_pred = predict_ngram_pred(test, ngram_table, ngram_n)
    ngram_preds.append(ngram_pred)

# take first ngram predicion (then, one with highest frequency), else take integer 1
y_pred = list()
for i in range(0, len(test)):
    pred = None
    for j in range(0, len(ngram_preds)):
        if ngram_preds[j][i]:
            pred = ngram_preds[j][i]
            break
    if not pred:
        pred = 1
    y_pred.append(pred)

df = pd.DataFrame({'Id': testset['Id'], 'Last': y_pred})
df.to_csv('submission_ngram.csv', index=False, header=True)
