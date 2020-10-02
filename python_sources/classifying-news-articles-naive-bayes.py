import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import hstack
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

DATA_PATH = '../input/uci-news-aggregator.csv'


def create_dataset(data_path):
    raw_data = pd.read_csv(data_path)
    # Labels
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(raw_data['CATEGORY'])

    # Features
    # Title
    title_enc = CountVectorizer(min_df=1e-6, max_df=0.1, stop_words='english')
    title_bow = title_enc.fit_transform(raw_data['TITLE'])
    # Url
    url_enc = CountVectorizer(max_features=50, max_df=0.5, stop_words='english')
    url_bow = url_enc.fit_transform(raw_data['URL'])
    # Publisher
    publisher_enc = CountVectorizer(max_features=1000)
    publisher_categories = publisher_enc.fit_transform(raw_data['PUBLISHER'].astype(str))
    # Story
    # Do not use story as a feature as I will be needed as a grouping factor during cross-validation
    # Hostname
    hostname_enc = FeatureHasher(n_features=30, input_type='string', non_negative=True)
    hostname_categories = hostname_enc.fit_transform(raw_data['HOSTNAME'])
    # Timestamp
    weekday = (raw_data['TIMESTAMP'] / 1000).apply(datetime.fromtimestamp).apply(lambda d: d.strftime('%w'))
    weekday = weekday.values.reshape((-1, 1)).astype(int)
    x = hstack((title_bow, url_bow, publisher_categories, hostname_categories, weekday))

    return x, y, raw_data['STORY']


if __name__ == '__main__':
    x, y, g = create_dataset(DATA_PATH)
    clf = MultinomialNB(alpha=0.5, fit_prior=True)
    scores = cross_val_score(clf, x, y, cv=5, groups=g)
    print('Accuracy | Mean: %.4f\tStdev: %.4f' % (np.mean(scores), np.std(scores)))
