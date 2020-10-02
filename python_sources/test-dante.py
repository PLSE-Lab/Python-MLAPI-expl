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
import scipy
from operator import itemgetter
import matplotlib.pyplot as plt
import sys
import math

data=pd.read_csv('../input/train.csv',nrows=100000,dtype={'is_booking':bool,'srch_destination_id':np.int32, 'hotel_cluster':np.int32},parse_dates = ['date_time','srch_ci','srch_co'])#0000000

data_destnation=pd.read_csv('../input/destinations.csv',)

data=data.drop(['date_time','srch_ci','srch_co','is_mobile','user_id','is_package',
           'channel','orig_destination_distance'],axis=1)
#print(data.columns)

#data=data.groupby(['srch_destination_id'])#'user_id'

data_test =pd.read_csv('../input/test.csv',dtype={'is_booking':bool,'srch_destination_id':np.int32},
                       parse_dates = ['date_time','srch_ci','srch_co'],nrows=10000)

data_test=data_test.drop(['date_time','srch_ci','srch_co','is_mobile','user_id',
                'is_package','channel','orig_destination_distance'],axis=1)
#print(data_test.head())

print(data.columns)
i=0
submission_d=pd.DataFrame(columns=['hotel_cluster'])
while i<len(data_test):
    best5=[]
    j=0
    while j <len(data):
        if(data_test.ix[i,'srch_destination_id']==data.ix[j,'srch_destination_id']):
            dist = 0
            for k in data_test.columns[1:]:
                if ( not pd.isnull(data_test.ix[i,k]) and not pd.isnull(data.ix[i, k])):
                    N_dt=(data_test.ix[i,k]-data_test.ix[:,k].min())/(
                        data_test.ix[:,k].max() - data_test.ix[:, k].min())
                    N_d = (data.ix[i, k] - data.ix[:,k].min()) / (
                        data.ix[:, k].max() - data.ix[:,k].min())
                    dist =dist +math.pow (abs(float(N_dt) - float(N_d)) ,2)

            best5.append((data.ix[j,'hotel_cluster'],dist))
            same=0
            while(same<(len(best5)-1)):
                if(best5[len(best5)-1][0]==best5[same][0]):
                    if(best5[len(best5)-1][1]>=best5[same][1]):
                        del best5[len(best5) - 1]
                    else:
                        del best5[same]
                    break
                same=same+1
            sorted(best5, key=itemgetter(1))
            if (len(best5)>5):
                del best5[len(best5) - 1]
        j=j+1
    buffer=''
    for best in best5:
        buffer=buffer+str(best[0])+' '
    i = i + 1
    df = pd.DataFrame({'hotel_cluster': [buffer]}, index=[i])
    #print(df)
    #print(submission_d)
    #submission_d.concat(pd.DataFrame({'hotel_cluster':[buffer]}),ignore_index=True)
    #if(len(submission_d)==0):
    #   submission_d=df
    #else:
    submission_d=pd.concat([submission_d,df],axis=0)
    #print(submission_d)
    print("No. "+str(i))
print(submission_d.head(5))