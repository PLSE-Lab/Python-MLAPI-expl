# -*- coding: utf-8 -*-
from pandas import read_csv, DataFrame
from os.path import isfile
from math import sqrt
from numpy import float64
from sklearn.utils.extmath import weighted_mode

in_file = '../input/train.csv'

if isfile(in_file):
    train_input = read_csv(in_file)
    
    train_len = 1100

    index  = [int(
                    (i-1)%7
                    ) for i in range(0, train_len)]
                  
    powers = [float64(
                  pow((i+6)//7, 2.3)
                     )for i in range(0, train_len)]
                     
    string_represenation = [' '+str(res)
                                        for res in range(0,8)]
                     
    #total = len(train_input['id'])                   
    #cur = 0

    def TrainAndPredict(row):
        #global total
        #global cur

        #print(str(cur/total))
        #cur += 1;

        visits = [int(s) for s in row.split(' ')[1:]]
        day_num = [index[i] for i in visits]
        weights  = [powers[i] for i in visits]
        
        result = int(weighted_mode(day_num, weights)[0][0])
    
        return string_represenation[result + 1]
        
    solution = DataFrame(columns = ['id', 'nextvisit'])
    solution['id'] = train_input['id']
    solution['nextvisit'] = train_input['visits'].apply(TrainAndPredict)
    solution.to_csv('solution.csv', index=False, sep =',')
    
else:
    print('No file provided, give me data!')
 