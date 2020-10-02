#!/usr/bin/env python3


'''
# Randomforest - plot: # of trees - accuracy
#
# @Author: Hideki Ikeda
# @Date 7/11/15
# modified by dixhom 4/16/16
'''

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

def main():
    # loading training data
    print('Loading training data')
    t1 = time.time()
    data = pd.read_csv('../input/train.csv')
    t2 = time.time()
    print('Done loading data ({0:.3f}sec)'.format(t2-t1))
    X_tr = data.values[:, 1:].astype(float)
    y_tr = data.values[:, 0]

    scores = list()
    scores_std = list()

    print('Start learning...')
    n_trees = [int(np.power(10,x/2)) for x in range(6)]
    for n_tree in n_trees:
        print("number of trees : {0}".format(n_tree))
        t1 = time.time()
        recognizer = RandomForestClassifier(n_tree)
        score = cross_val_score(recognizer, X_tr, y_tr)
        scores.append(np.mean(score))
        scores_std.append(np.std(score))
        t2 = time.time()
        print('Done processing {0} trees ({1:.3f}sec)'.format(n_tree, t2-t1))

    sc_array = np.array(scores)
    std_array = np.array(scores_std)
    print('Score: ', sc_array)
    print('Std  : ', std_array)

    plt.figure(figsize=(4,3))
    plt.plot(n_trees, scores)
    plt.plot(n_trees, sc_array + std_array, 'b--')
    plt.plot(n_trees, sc_array - std_array, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('# of trees')
    plt.savefig('cv_trees.png')
    plt.show()


if __name__ == '__main__':
    main()