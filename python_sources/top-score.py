import codecs
import string
import gc
import nltk
import random

import numpy as np
import pandas as pd

from collections import defaultdict
from nltk.corpus import stopwords
from vowpalwabbit import pyvw
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def vw_fit(inputFile, numPasses, otherArgs):
    vw = pyvw.vw(otherArgs)
    for p in range(numPasses):
        print('pass', (p+1))
        h = open(inputFile, 'r')
        for l in h.readlines():
            ex = vw.example(l)
            vw.learn(ex)
            ex.finish()

        h.close()
    return vw

def vw_predict(vw, inputFile, outputFile):
    h = open(inputFile, 'r')
    out = open(outputFile, 'w')
    preds = []
    for l in h.readlines():
        ex = vw.example(l)
        print(vw.predict(ex), file=out)
        ex.finish()

    h.close()
    out.close()

def stem_text(text, stemmer, trans):
    return ' '.join((x for x in text.translate(trans).split()))
    
def stem_texts(filename, n_jobs=1, verbose=0):
    stemmer = nltk.stem.snowball.EnglishStemmer()
    trans = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    
    r = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(stem_text)(row, stemmer, trans) for row in open(filename, encoding='utf-8'))
    return r

def calc_roc_auc(true, pred):
    true = np.array([float(x) for x in open(true)])
    pred = np.array([float(x) for x in open(pred)])
    print(true.shape, pred.shape)
    return roc_auc_score(true, pred)
    
def format_for_submit(pred):
    pred = np.array([float(x) for x in open(pred)])
    resdf = pd.DataFrame(pred, columns=['Probability'])
    resdf.index += 1
    resdf.to_csv('vw_res.csv', index_label='Id')

    
texts = stem_texts('../input/x_train.txt', n_jobs=8)
print('Stemming done!')
print('Collected {}'.format(gc.collect()))

target = pd.read_csv(open('../input/y_train.csv', encoding='utf-8'))['Probability']

texts_train, texts_test, y_train, y_test = train_test_split(texts, target, test_size=0.001, random_state=228)

del texts
del target
print('Collected {}'.format(gc.collect()))

for mode, x, y in (('train', texts_train, y_train), ('test', texts_test, y_test)):
    f = open(mode + '.vw', 'w')
    for i, text in enumerate(x):
        print('{} | {}'.format(1 if y.iloc[i] == 1.0 else -1, text), file=f)
    f.close()
    if mode == 'test':
        f = open('y_test_clean', 'w')
        for e in y.tolist():
            print(int(e), file=f) 
        f.close()
 
del texts_train
del texts_test
del y_train
del y_test
print('Collected {}'.format(gc.collect()))
   
vw = vw_fit('train.vw', 13, '--link=logistic --random_seed 228 --loss_function logistic --learning_rate 0.843763562344744 --l1 3.81427762457755e-09 --l2 4.84808788797216e-11 -b 29 --ngram 3 --skips 1 --hash all')

vw_predict(vw, 'test.vw', 'pred_test')

print('Score: ', calc_roc_auc('y_test_clean', 'pred_test'))


texts = stem_texts('../input/x_test.txt', n_jobs=8)
print('Submit: Stemming done!')
print('Submit: Collected {}'.format(gc.collect()))

f = open('submit.vw', 'w')
for text in texts:
    print('1 | ' + text, file=f)
f.close()

del texts
print('Submit: Collected {}'.format(gc.collect()))

vw_predict(vw, 'submit.vw', 'pred_submit')
format_for_submit('pred_submit')