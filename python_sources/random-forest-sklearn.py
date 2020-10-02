import sys

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_train_data(path=None, train_size=0.8):
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    df = pd.read_csv("../input/train.csv")
    X = df.values.copy()
    np.random.shuffle(X)
    
    # Test training and validations sets
    # Note in this case training data has -
    # predictors: starting from second column to end
    # targets: in first column
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #    X[1:, 1:], X[1:, 0], train_size=train_size,
    #    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:], X[:, 0], train_size=train_size,
        )
    print(" -- Loaded data.")
    print("Training set has {0[0]} rows and {0[1]} columns".format(X.shape))

    return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))

def load_test_data(path=None, train_size=0.8):
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    df = pd.read_csv("../input/test.csv")
    X = df.values
    
    X_test = X[:, :]
    print("Test set has {0[0]} rows and {0[1]} columns".format(X.shape))
    return X_test.astype(float)

def train(n_estimators):
    X_train, X_valid, y_train, y_valid = load_train_data()

    # Setup classification trainer
    n_estimators=50  # number of trees, increase this to beat benchmark :-)
    clf = RandomForestClassifier(n_estimators=n_estimators,n_jobs=4)

    # Start training
    print(" -- Start training Random Forest Classifier.")
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_valid)
    print(" -- Finished training.")

    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_valid)
    assert (encoder.classes_ == clf.classes_).all()

    # return clf, encoder, score
    return clf, encoder

def make_submission(clf, encoder, path='my_submission.csv'):
    X_test = load_test_data()

    # y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    # ImageId = range(1,len(y_pred)+1)

    with open(path, 'w') as f:
        f.write("ImageId,Label\n")
        ImageId = 1
        for pred in y_pred:
            f.write("%d,%d" %(ImageId,int(pred)))
            f.write('\n')
            ImageId = ImageId+1
    print(" -- Wrote submission to file {}.".format(path))

def main():
    print(" - Start.")
    clf,encoder = train(1400)
    make_submission(clf, encoder)
    
    # Write to the log:
    # print("Training set has {0[0]} rows and {0[1]} columns".format(X.shape))
    # print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
    # Any files you write to the current directory get shown as outputs
    print(" - Finished.")

if __name__ == '__main__':
    main()