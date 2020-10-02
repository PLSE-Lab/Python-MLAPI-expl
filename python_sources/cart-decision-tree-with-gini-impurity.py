# package source: https://github.com/kopylovvlad/decision_tree_kv
import decision_tree_kv
from typing import List, Dict, Tuple, Union
import csv
import random
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#print(os.listdir("../input"))

#
# algorithm:
# *) open csv-file
# *) shuffle data
# *) divide data into two subsets (data_1, data_2) by value of Dataset column (1,2)
# *) take 80% of data_1 and data_2 for train data set
# *) take 30% of data_1 and data_2 for test data set
# *) train decision tree
# *) check accuracy
# *) to repeat 2 more times
# *) calculate final accuracy


accuracy_list = []
for d in range(3):
    print('Experiment #'+str(d+1)+' start ...')
    
    # variables for subset: train and test
    train_data = []
    test_data = []
    
    # data with label equal 1 or 2
    data_1 = []
    data_2 = []
    
    # open scv-file
    file_data = list(csv.reader(open('../input/indian_liver_patient.csv', newline='')))
    
    # shuffle data
    random.shuffle(file_data)
    
    # collect datawith label 1,2 to data_1,data_2
    for row in file_data:
        # ignore head-row
        if row[0] == 'Age':
            continue
    
        label = row[len(row)-1]
        tmp_row = []
    
        for j in range(len(row)):
            if j == 1 or j == 9:
                tmp_row.append(row[j])
            else:
                # convert string to float
                tmp_row.append(float(row[j]))
    
        if label == '1':
            data_1.append(tmp_row)
        elif label == '2':
            data_2.append(tmp_row)
        else:
            raise BaseException('Undefined label')
    
    #
    # 80% data with 1 and 2 labels will be at train subset
    # 30% data with 1 and 2 labels will be at test subset
    # 
    
    i = 0
    for row in data_1:
        i += 1
        if i < len(data_1) * 0.8:
            train_data.append(row)
        else:
            test_data.append(row)
    i = 0
    for row in data_2:
        i += 1
        if i < len(data_2) * 0.8:
            train_data.append(row)
        else:
            test_data.append(row)
    
    print('Train data amount: %d' % len(train_data))
    print('Test data amount: %d' % len(test_data))
    
    #
    # train decision tree
    # It is CART decision tree with Gini impurity 
    #
    
    tree = decision_tree_kv.buildtree(train_data, scoref=decision_tree_kv.giniimpurity)
    
    #
    # test accuracy
    #
    
    stat = {'correct': 0, 'not_correct': 0}
    for test_row in test_data:
        real_label = test_row[len(test_row) - 1]
        row_without_label = test_row[0:len(test_row)-1]
    
        r = decision_tree_kv.classify(row_without_label, tree)
        label_from_tree = list(r.keys())[0]
    
        if label_from_tree == real_label:
            stat['correct'] += 1
        else:
            stat['not_correct'] += 1
    
    print('Test data amount: %d' % (stat['correct'] + stat['not_correct']))
    print('Correct items: %d' % stat['correct'])
    print('Not correct items: %d' % stat['not_correct'])
    divider = (stat['correct'] + stat['not_correct'])/100
    accuracy = 0
    if not divider == 0:
        accuracy = stat['correct']/divider
    print('Accuracy '+str(accuracy)+'%')
    accuracy_list.append(accuracy)
    print('Experiment #'+str(d+1)+' end')
    print('')

final_accuracy = sum(accuracy_list) / float(len(accuracy_list))

print('final accuracy: %f' % final_accuracy)
# 62-68%
    
    
