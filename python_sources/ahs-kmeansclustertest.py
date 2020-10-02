# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from matplotlib import style
import numpy as np
style.use('ggplot')
#from sklearn.cluster import KMeans
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC
#from sklearn.cluster import SpectralClustering
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.cluster import DBSCAN

from sklearn import preprocessing
#cross_validation deprecated
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

import os
import pickle

print(os.listdir("../input"))

#df = pd.read_csv('../input/ahs-woman-1/AHS_Woman_21_Odisha.csv', delimiter='|')
df = pd.read_csv('../input/ahstestonly/small.csv')
#print(df.head)
print(df.shape)
#columns to drop:



def scrubData(df):
    df.drop(['w_id','hl_id', 'client_w_id','state','district'], 1, inplace=True)
    df.drop(['psu_id', 'house_no','house_hold_no','year_of_intr', 'month_of_intr', 
             'date_of_intr'], 1, inplace=True)
        
    df.drop(['other_int_code','identifcation_code', 'w_expall_status','w_status','twsi_id', 'client_twsi_id'], 1, inplace=True)
    df.drop(['fid','hh_id', 'client_hh_id','member_identity','father_serial_no', 'mother_serial_no'], 1, inplace=True)
    df.drop(['client_hl_id','building_no', 'hl_expall_status','sn'], 1, inplace=True)
    df.drop(['headname','ever_born', 'wt'], 1, inplace=True)
    df.drop(['fidx','as', 'as_binned'], 1, inplace=True)
    df.drop(['fidh','cdoi', 'anym','catage1','respondentname', 'rtelephoneno'], 1, inplace=True)
    
    df.drop(['healthscheme_1','healthscheme_2'], 1, inplace=True)
    df.drop(['new_born_alive_female','new_born_alive_male', 'new_born_alive_total','new_surviving_female'], 1, inplace=True)
        
    df.drop(['new_surviving_male','new_surviving_total'], 1, inplace=True)
        
    df.drop(['isdeadmigrated'], 1, inplace=True)
    
    df.drop(['isnewrecord','recordupdatedcount', 'recordupdatedcount','schedule_id','year', 'id'], 1, inplace=True)
    
    df.drop(['date_of_marriage', 'month_of_marriage', 'year_of_marriage'],1, inplace=True)
    
    df.drop(['year_of_birth', 'month_of_birth', 'date_of_birth'],1, inplace=True)
    df.drop(['compensation_after_ster', 'received_compensation_after_ster', 'received_compensation_ster_rs'],1, inplace=True)
    
    df.drop(['is_tubectomy','delivered_any_baby','born_alive_female','born_alive_male','born_alive_total'],1, inplace=True)
    df.drop(['surviving_female','surviving_male','surviving_total'],1, inplace=True)
    df.drop(['last_preg_no','previous_last_preg_no','second_last_preg_no', 'third_last_preg_no'],1, inplace=True)
    
    df.drop(['edt','occupation','marital', 'modern', 'traditional'],1, inplace=True)
    df.convert_objects(convert_numeric=True)
    df.fillna(0, inplace=True)
    #print(df.head())
    #result_of_interview need to logic only = 1? drop all others?
    df = df[df.result_of_interview == 1]
    df.drop(['result_of_interview'],1, inplace=True)
    df = df[df.outcome_pregnancy != 0]
    df['outcome_pregnancy'].replace(2,0, inplace=True)
    
print("starting data pruning from csv list")
scrubData(df)

print(df.shape)
print("start analysis")

X = np.array(df.drop(['outcome_pregnancy'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['outcome_pregnancy'])

#clf = KMeans(n_clusters=2)
#clf.fit(X)
clf = LinearSVC()
clf.fit(X,y)
print("end analysis")
print("test")
correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
   
    if prediction[0] == y[i]:
        correct += 1

print("result: ", correct/len(X))
print("pckle model")
s = pickle.dumps(clf)
pickle.dump( s, open( "AHS_Woman_LinearSVC.p", "wb" ) )
