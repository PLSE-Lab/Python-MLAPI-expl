# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
def classi():
    trainn= [ ('Fee','HIGH'), ('Concession','HIGH'),('exam Schedule','HIGH'),('SEATING PLAN','HIGH'),('scholarship','HIGH'),('online exam','HIGH'),('offline exam','HIGH'),('Examination Form','HIGH'),('today','HIGH'),('tomorrow','HIGH'),('Important','HIGH'),('Urgent','HIGH'),('emergency','HIGH'),('viva','HIGH'),('oral','HIGH'),('fine','HIGH'),('return','HIGH'),('suspended','HIGH'),('Revaluation','HIGH'),('Photocopy','HIGH'),('Seating Arrangement','HIGH'),('Issue','HIGH'),('category','HIGH'),('compulsory','HIGH'),('mandatory','HIGH'),('detention','HIGH'),('Result','HIGH'),('holiday','HIGH'),('meeting','HIGH'),('minority','HIGH'),('Time table','HIGH'),('Fee Concession','HIGH'),('examination schedule','HIGH'),('Closed','LOW'),('finished','LOW'),('over','LOW'),('Invitation','LOW'),('feedback','LOW'),('celebrated','LOW'),('facility','LOW'),('party','LOW'),('parents','LOW'),('Submission','MID'),('session','MID'),('documents','MID'),('next week','MID'),('next month','MID'),('registration','MID'),('library','MID'),('books','MID'),('submit before','MID'),('submit after','MID'),('Recruitment','MID'),('campus Recruitment','MID'),('Attendance','MID'),('Short listed','MID'),('qualified','MID'),('form','MID'),('Admit Card','MID'),('hall ticket','MID'),('receipt','MID'),('test','MID'),('Collect','MID'),('university','MID'),('fellowship','MID'),('NSS','MID'),('guest','MID'),('competition','MID'),('list','MID'),('seminar','MID'),('presentation','MID'),('mock','MID'),('unit','MID'),('workshop','MID'),('annual','MID'),('gathering','MID'),('convocation','MID'),('auditions','MID'),('organising','MID'),('hostel','MID'),('camp','MID')]
    train=[(x.lower(), y) for x,y in trainn]
    msg="feedback closed"
    cl = NaiveBayesClassifier(train)
    prio=cl.classify(msg.lower())
    if prio=="HIGH":
    	return 0
    elif prio=="MID":
    	return 1
    else:
    	return 2
print(classi())