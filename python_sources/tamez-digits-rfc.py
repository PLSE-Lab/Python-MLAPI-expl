#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright 2016 Danny Tamez <zematynnad@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Use Machine Learning to Identify Handwritten Digits
"""

import numpy as np

from pandas import read_csv

from sklearn.ensemble import RandomForestClassifier


def main():
    training = read_csv('../input/train.csv')
    testing = read_csv('../input/test.csv')
    training = np.array(training)
    features_test = np.array(testing)
    features_train = training[:, 1:] 
    labels_train = training[:, 0]

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    with open('predictions.csv', 'w') as fd: 
        fd.write('ImageId,Label\n')
        for i, pred in enumerate(predictions):
            fd.write('{},{}\n'.format(i + 1, pred))


if __name__ == "__main__":
    main()