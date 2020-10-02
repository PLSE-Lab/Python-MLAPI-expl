'''
@Link: https://www.kaggle.com/hideki1234/digit-recognizer/randomforest-of-tree-and-accuracy/code
@Date Aug-24-2016

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import time
from sklearn import svm

def main():
    print('Loading training data...')
    data = pd.read_csv('../input/train.csv')
    X_train = data.values[:, 1:].astype(float)
    y_train = data.values[:, 0]

    print('X_train = ', len(X_train))
    print('y_train = ', len(y_train))

    scores = list()
    scores_std = list()

    print('Start learning...')
    n_trees = [100, 300, 500]
    for n_tree in n_trees:
        print('n_tree = ', n_tree)
        recognizer = RandomForestClassifier(n_tree)
        try:
            score = cross_validation.cross_val_score(recognizer, X_train, y_train)
        except Exception as e:
            print('Exception', e)

        scores.append(np.mean(score))
        scores_std.append(np.std(score))
        print('scores = ', scores)
        print('scores_std = ', scores_std)

    sc_array = np.array(scores)
    std_array = np.array(scores_std)
    print('Score: ', sc_array)
    print('Std: ', std_array)

    plt.figure(figsize = (4, 3))
    plt.plot(n_trees, scores)
    plt.plot(n_trees, sc_array + std_array, 'b--')
    plt.plot(n_trees, sc_array - std_array, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('# of trees')
    plt.savefig('cv_trees.png')
    plt.show()


if __name__ == '__main__':
    main()
