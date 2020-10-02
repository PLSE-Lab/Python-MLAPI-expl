from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score

def get_train_data():
    print('Loading train file...'),
    df_train = pd.read_csv('../input/train.csv')
    X = df_train.drop('label', 1)
    Y = df_train['label']
    print('Done')

    return X, Y


def train_with_classifier(cls, X, Y):

    cls.fit(X, Y)


def create_submission(cls, X, Y):

    X_test = pd.read_csv('../input/test.csv')
    Y_pred = cls.predict(X_test)

    X_test['label'] = Y_pred

    X_test = X_test[['label']]
    X_test['imageId'] = X_test.index + 1

    X_test.to_csv('scikit-learn-sub.csv', index=False)


def main():
    X, Y = get_train_data()

    classifiers_names = ['RandomForest_200',
                         'RandomForest_1000',
                         'GaussianNB',
                         'SVM_linear',
                         'SVM_poly']

    classifiers = [RandomForestClassifier(n_estimators=200),
                   RandomForestClassifier(n_estimators=1000),
                 #  GaussianNB(),
                 #  SVC(kernel="linear"),
                 #  SVC(kernel="poly")
                 ]

    print('Fitting %d classifiers' % len(classifiers))

    cv_scores = []
    cls_count = 1
    print('0%'),
    for cls in classifiers:
        scores = cross_val_score(cls, X, Y)
        cv_scores.append(np.mean(scores))
        print('\r' + str(int((cls_count / len(classifiers)) * 100)) + '%'),
        cls_count += 1

    plt.bar(range(len(classifiers)), cv_scores, align='center')
    plt.xticks(range(len(classifiers)), classifiers_names)
    plt.show()


if __name__ == '__main__':
    main()






