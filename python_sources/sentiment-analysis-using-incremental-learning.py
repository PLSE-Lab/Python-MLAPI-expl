import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import pickle

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


def path_viewer(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def preprocessor(text):
    """
    Discards all http-links and usernames. (very basic)
    """
    
    patt_usr = r'\@[\w\._]+'
    patt_lnk = r'http[s]*:[\w\S]+'
    usr_removed =  re.sub(patt_usr, '', text).strip()
    return re.sub(patt_lnk, '', usr_removed).strip()


def csv_linter(path=None, new_file=None):
    """
    Lints (cleans the csv dataset) according to the context
    Use this often, so that the model is trained on shuffled dataset.
    """
    
    df = pd.read_csv(path or r'/kaggle/input/sentiment140/'
                             r'training.1600000.processed.noemoticon.csv',
                             encoding='latin1', header=None)
    
    # remove useless columns
    df.drop(columns=[1, 2, 3, 4], inplace=True)
    df.columns = ('sentiment', 'tweet')

    # remove @usernames and http[s]://links.
    df.tweet = df.tweet.apply(preprocessor)

    # shuffle the indexes
    df = df.reindex(np.random.permutation(df.index), index=False)

    # save as new file
    df.to_csv(new_file or r'/kaggle/working/processed.csv', encoding='utf-8')


def trainer(path, chunk=10000, rows=16e5, t_ratio=0.6):
    """
    Train the model with small chunks (incremental)
    
    path    : path to csv data
    chunk   : chunk of data to be read in
    rows    : row count in csv file
    t_ratio : train/text ratio
    """
    
    class_ = np.array([0, 4])
    
    df_chunk = pd.read_csv(path, chunksize=chunk)
    loops = int((rows * t_ratio) / chunk)

    vect, sgd = HashingVectorizer(decode_error='ignore',
                                  n_features=2**21,
                                  preprocessor=None,
                                  ngram_range=(1, 2)), \
        SGDClassifier(loss='log', n_jobs=-1)

    # will achieve the same (~0.75) accuracy
    # benifit : Online out of core learning
    for i in range(loops):
        df = df_chunk.get_chunk()
        y_tr, X_tr = df['sentiment'].values, df['tweet'].values.astype('U')

        X_tr = vect.fit_transform(X_tr)
        sgd.partial_fit(X_tr, y_tr, classes=class_)

    return (sgd, vect, df_chunk)


# use this after training or loading the model
def predictor(text, sgd, vect):
    """
    Toy Predictor for the given dataset
    """

    labels = {0:'sad', 4:'happy'}
    to_vect = vect.transform([text])
    label = sgd.predict(to_vect)[0]
    
    return labels[label]


if __name__ == "__main__":
    path_to_csv = '/kaggle/working/processed.csv'

    # lint the dataset
    csv_linter()

    # train the model so carefully built :)
    sgd, vect, left_data = trainer(path_to_csv)
    df = left_data.get_chunk()

    # get the trainable data
    X_test, y_test = df['tweet'].values.astype('U'), df['sentiment'].values

    X_test = vect.transform(X_test)
    y_pred = sgd.predict(X_test)

    # achieves a score of 75.79%
    print('Accuracy of model is: {:3}%'.format(100 * accuracy_score(y_test, y_pred)))

    # optional (but you can use it)
    # sgd.fit(X_test, y_test)
    pickle.dump(sgd, open('/kaggle/working/model_sgd.pkl', 'wb'), protocol=4)
    pickle.dump(vect, open('/kaggle/working/model_vect.pkl', 'wb'), protocol=4)
