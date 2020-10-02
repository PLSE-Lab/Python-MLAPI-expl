# -*- coding: utf-8 -*-
#This is test programm to predict day of the week person can come

from pandas import read_csv, DataFrame
from os.path import isfile
from math import sqrt
from numpy import float64
from sklearn.utils.extmath import weighted_mode

if isfile('../input/train.csv'):
    train_input = read_csv('../input/train.csv')
    
    test_mode = True 
    sample_num = 300000 #
    

    index  = [float64(
                    (i-1)%7
                    ) for i in range(0, 1100)]
                    
    #here the weight of the prev visit is calculated. It affects the accuracy a bit 
    powers = [float64(
                  ((i+6)//7) ** 2.7
                     )for i in range(0, 1100)]
                     
    string_represenation = [' '+str(res)
                                        for res in range(0,8)]
                                        
    
    def visit_wmode(row):
            visits = [int(s) for s in row.split(' ')[1:]]
            day_num = [index[i] for i in visits]
            weight  = [powers[i] for i in visits]
            
            #if last visit was long time ago expect that person will never come again
            if(1099 - visits[-1] > 200):
                result = -1
            else:
                result = int(weighted_mode(day_num, weight)[0][0])
            return string_represenation[result + 1]
            
    if test_mode:
        solution = DataFrame(columns = ['id', 'nextvisit'])
        solution['id'] = train_input['id']
        solution['nextvisit'] = train_input['visits'].apply(visit_wmode)
        solution.to_csv('solution.csv', index=False, sep =',')
        
else:
    print('Warning!')
 


# Any results you write to the current directory are saved as output.