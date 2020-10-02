# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:55:01 2017

@author: Nusri
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report
from sklearn.cross_validation import train_test_split


dat = pd.read_csv("../input/creditcard.csv")

dat.head()

dat.describe()

count = dat.groupby(["Class"])["Time"].count()
count

count.plot.bar()
plt.xlabel("Class")
plt.ylabel("Freq")

class1 = dat.loc[dat["Class"] == 1]
class1



dat["normalizedAmt"] = StandardScaler().fit_transform(dat["Amount"].reshape(-1,1))
dat_norm = dat.drop(["Amount", "Time"], axis = 1)

x = dat.loc[dat["Class"] == 1]
y = dat.loc[dat["Class"] == 0]

fraud_count = len(dat[dat.Class == 1])
fraud_count



fraud_index = np.array(dat[dat.Class == 1].index)
secure_index = np.array(dat[dat.Class != 1].index)

random_secure_index = np.random.choice(secure_index, fraud_count, replace = False)
len(random_secure_index)

selected_record_index = np.concatenate([random_secure_index, fraud_index])

selected_records = dat.loc[selected_record_index,:]

x_dat = dat.ix[:, dat.columns != 'Class']
y_dat = dat.ix[:, dat.columns == 'Class']

x_undersample = selected_records.ix[:, selected_records.columns != 'Class']
y_undersample = selected_records.ix[:, selected_records.columns == 'Class']


x_sel = selected_records[selected_records.Class == 1]
y_sel = selected_records[selected_records.Class == 0]

x_sel_len = len(selected_records[selected_records.Class == 1])
y_sel_len = len(selected_records[selected_records.Class == 0])





x_train, x_test, y_train, y_test = train_test_split(x_dat
                                                    , y_dat, train_size = 0.7, random_state = 0)


x_train_sel, x_test_sel, y_train_sel, y_test_sel = train_test_split(x_undersample,y_undersample, 
                                                                    train_size = 0.7, 
                                                                    random_state = 0)




def printing_Kfold_scores(x_train_data,y_train_data):
    fold = KFold(len(y_train_data),5,shuffle=False) 

    c_param_range = [0.01,0.1,1,10,100]

    results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    j = 0
    for c_param in c_param_range:
        print('C parameter: ', c_param)
        print('')

        recall_accs = []
        for iteration, indices in enumerate(fold,start=1):

            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C = c_param, penalty = 'l1')

            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
            # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
            lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', iteration,': recall score = ', recall_acc)

        # The mean value of those recall scores is the metric we want to save and get hold of.
        results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']
    
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    
    return best_c

best_c = printing_Kfold_scores(x_train_sel,y_train_sel)

# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
# dataset
lr = LogisticRegression(C = best_c, penalty = 'l1')
lr.fit(x_train_sel,y_train_sel.values.ravel())
y_pred = lr.predict(x_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)


