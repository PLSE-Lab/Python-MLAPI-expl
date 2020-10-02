# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from collections import namedtuple

DATA_FOLDER = "../input"
SEED = 9

def read_data(train_data_perc=0.8):
    train_data_file = DATA_FOLDER + "/" + "train.csv"

    all_data = pd.read_csv(train_data_file)
    X = all_data[["qid", "question_text"]]
    y = all_data[["target"]]
    
    X["num_words"] = X["question_text"].apply(lambda x: math.log(len(str(x).split())))
    X["num_unique_words"] = X["question_text"].apply(lambda x: math.log(len(set(str(x).split()))))
    X["num_chars"] = X["question_text"].apply(lambda x: math.log(len(str(x))))
    X["num_punctuations"] = X["num_unique_words"].apply(lambda x: math.log(len([c for c in str(x)
                                                                       if c in string.punctuation])) )    

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=train_data_perc,
                                                        random_state=SEED)

    return X_train, y_train, X_test, y_test

def read_test_data():
    test_data_file = DATA_FOLDER + "/" + "test.csv"

    all_data = pd.read_csv(test_data_file)
    X = all_data[["qid", "question_text"]]

    X["num_words"] = X["question_text"].apply(lambda x: math.log(len(str(x).split())))
    X["num_unique_words"] = X["question_text"].apply(lambda x: math.log(len(set(str(x).split()))))
    X["num_chars"] = X["question_text"].apply(lambda x: math.log(len(str(x))))
    X["num_punctuations"] = X["num_unique_words"].apply(lambda x: math.log(len([c for c in str(x)
                                                                       if c in string.punctuation])) )

    return X
    
class QuestionSincerityEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, classifier, vectorizer, X_train, X_test=None):
        self.classifier = classifier
        self.vectorizer = vectorizer
        if X_test is None:
            self.vectorizer.fit(X_train["question_text"])
        else:
            self.vectorizer.fit(X_train["question_text"].values.tolist() +
                                          X_test["question_text"].values.tolist())

    def fit(self, X, y):
        X_transformed = self.vectorizer.transform(X["question_text"])
        #X_train = sp.sparse.hstack((X_transformed, X[["num_words", "num_punctuations"]].values),
        #                     format='csr')

        self.classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X["question_text"])
        #X_train = sp.sparse.hstack((X_transformed, X[["num_words", "num_punctuations"]].values),
        #                     format='csr')        
        prediction = self.classifier.predict(X_transformed)
        return pd.DataFrame({"qid": X["qid"],
                             "prediction": prediction})

    def predict_proba(self, X):
        X_transformed = self.vectorizer.transform(X["question_text"])
        #X_train = sp.sparse.hstack((X_transformed, X[["num_words", "num_punctuations"]].values),
        #                     format='csr')        
        prediction = self.classifier.predict_proba(X_transformed)[:,1]
        return pd.DataFrame({"qid": X["qid"],
                             "prediction": prediction})

    def score(self, X, y):
        best_score = -1
        y_pred = self.predict_proba(X)
        for threshold in np.arange(0.05, 0.801, 0.01):
            threshold = np.round(threshold, 2)
            model_f1_score = f1_score(y_true=y["target"],
                                      y_pred=(y_pred["prediction"] > threshold).astype(int))
            if model_f1_score > best_score:
                best_score = model_f1_score

        return best_score

def get_classifier(args):
    classifier = args.classifier
    if classifier == "logreg":
        return LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=-1, random_state=SEED)
    elif classifier == "naivebayes":
        return MultinomialNB()
    elif classifier == "svm":
        return SVC(random_state=seed, max_iter=1000, probability=True, gamma='scale')
    elif classifier == "gradientboost":
        return GradientBoostingClassifier()
    elif classifier == "lightgbm":
        return lgbm.sklearn.LGBMClassifier(learning_rate=0.05,
                                           n_estimators=200,
                                           n_jobs=-1,
                                           random_state=SEED)
    else:
        raise AttributeError("Unknown model %s" % classifier)


def get_vectorizer(args):
    vectorizer_name = args.vectorizer
    ngram_range_start = args.ngram_range_start
    ngram_range_end = args.ngram_range_end
    min_df = args.min_df
    if vectorizer_name == "count":
        return CountVectorizer(ngram_range=(ngram_range_start, ngram_range_end),
                               min_df=min_df,
                               dtype=np.float32)
    elif vectorizer_name == "tfidf":
        return TfidfVectorizer(ngram_range=(ngram_range_start, ngram_range_end),
                               min_df=min_df)
    else:
        raise AttributeError("Unknown vectorizer %s" % vectorizer_name)


def fit_model(model, X_train, y_train):
    return model.fit(X_train["question_text"], y_train)


def evaluate_model(model, X):
    prediction = model.predict_proba(X["question_text"])[:,1]
    return pd.DataFrame({"qid": X["qid"],
                         "prediction": prediction})



def fit_and_eval_model_with_validation_data(args):
    X_train, y_train, X_test, y_test = read_data()
    vectorizer = get_vectorizer(args)
    classifier = get_classifier(args)
    model = QuestionSincerityEstimator(classifier, vectorizer, X_train, X_test)
    model = model.fit(X_train, y_train)
    predictions_proba = model.predict_proba(X_test)
    test_qid_target = zip(X_test["qid"], y_test["target"])
    train_qid_pred = zip(predictions_proba["qid"], predictions_proba["prediction"])
    for (test_qid, target), (train_qid, pred) in zip(test_qid_target, train_qid_pred):
        assert test_qid == train_qid
    score = model.score(X_test, y_test)
    print("Model score: %s" % score)
    best_threshold = -1
    best_score = -1
    for threshold in np.arange(0.05, 0.801, 0.01):
        threshold = np.round(threshold, 2)
        model_f1_score = f1_score(y_true=y_test["target"],
                                  y_pred=(predictions_proba["prediction"] > threshold).astype(int))
        if model_f1_score > best_score:
            best_score = model_f1_score
            best_threshold = threshold
        print("F1 score at threshold %s: %s" % (threshold, model_f1_score))
    print("F1 score at best threshold %s: %s" % (best_threshold, best_score))
    return best_threshold, best_score


def fit_and_eval_model(args, threshold=0.2):
    X_train, y_train, X_t, y_t = read_data(train_data_perc=1)
    vectorizer = get_vectorizer(args)
    classifier = get_classifier(args)
    model = QuestionSincerityEstimator(classifier, vectorizer, X_train, X_t)
    model = model.fit(X_train, y_train)
    X_test = read_test_data()
    predictions_proba = model.predict_proba(X_test)
    labels = (predictions_proba["prediction"] > threshold).astype(int)
    preds = pd.DataFrame({
        "qid": X_test["qid"],
        "prediction": labels
    })
    preds.to_csv("submission.csv", index=False)

Arguments = namedtuple("Arguments", ["classifier", "vectorizer", 
                                     "ngram_range_start", "ngram_range_end", "min_df"])


#args = Arguments("logreg", "count", 1, 2, 3)
#print('Training with arguments', args)
#fit_and_eval_model(args, 0.21)
