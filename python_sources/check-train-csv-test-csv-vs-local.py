# ***********
# a quick check if my local files still match the online files
# ***********

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
from collections import Counter
import matplotlib.pyplot as plt
import operator

print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

df_train = df_train.fillna('NULLVALUES')
df_test = df_test.fillna('NULLVALUES')

def eda(df):
    print ("IS_DUPLICATE 1 Count = %s , IS_DUPLICATE 0 Count = %s" 
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))
    
    question_ids_combined = df.qid1.tolist() + df.qid2.tolist()
    
    print ("Unique question IDs = %s" %(len(np.unique(question_ids_combined))))
    
    question_ids_counter = Counter(question_ids_combined)
    question_appearing_more_than_once = [i for i in question_ids_counter.values() if i > 1]
    print ("Count of question IDs appearing more than once = %s" %(len(question_appearing_more_than_once)))
    
    
eda(df_train)

train_q1_unique = np.unique(df_train['question1'])
train_q2_unique = np.unique(df_train['question2'])
train_question_combined = np.hstack([train_q1_unique, train_q2_unique])
print("Unique Train question text sentences = %s" % (len(np.unique(train_question_combined))))

print('Train Shape: ', df_train.shape)
print('Test Shape: ', df_test.shape)

# df_train.drop_duplicates(inplace=True)
# df_train.dropna(inplace=True)
# eda(df_train)

test_q1_unique = np.unique(df_test['question1'])
test_q2_unique = np.unique(df_test['question2'])
test_question_combined = np.hstack([test_q1_unique, test_q2_unique])
# test_question_combined = test_question_combined.append(test_q2_unique, ignore_index=True)
print("Unique Test question text sentences = %s" % (len(np.unique(test_question_combined))))

print('***** execution completed *****')
