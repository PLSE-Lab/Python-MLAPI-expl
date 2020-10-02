import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from nltk.stem.porter import PorterStemmer
import re

stemmer = PorterStemmer()
regex1 = re.compile('[^a-zA-Z]')

def stem(word):
    return stemmer.stem(word)

def Jaccard(row):
    words0 = regex1.sub(' ', row[0])
    words1 = regex1.sub(' ', row[1])

    words0 = set(words0.lower().split(' '))
    words1 = set(words1.lower().split(' '))

    normalizer = max(float(len(words0 | words1)), 1.0)
    return len(words0 & words1) / normalizer

def length(string):
    words = regex1.sub(' ', string)
    words = [stem(word) for word in words.split(' ') if len(word) > 1]
    return len(set(words))

train = pd.read_csv(os.path.join("..", "input", "train.csv")).fillna(" ")
rows = train.shape[0]
train = train.loc[np.random.choice(train.index, round(0.4*rows), replace=False)]

# we dont need ID columns
train = train.drop('id', axis=1)
y = train.median_relevance.values
print(itemfreq(y))

train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

train['queryCount'] = train['query'].apply(length)
train['titleCount'] = train['product_title'].apply(length)
train['Jaccard'] = train[['query', 'product_title']].apply(Jaccard, axis=1)
train['label'] = y
train = train.drop(['query', 'product_title', 'product_description'], axis=1)
print(train.describe())

fig = plt.figure()
pg = sns.pairplot(train, hue='label', size=2.5)
pg.savefig('pairplot.png')

