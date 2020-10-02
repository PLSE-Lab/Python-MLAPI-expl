import numpy as np
import pandas as pd
from os.path import isfile
from sklearn.utils.extmath import weighted_mode

if isfile("./train.csv"):
    data = pd.read_csv("./train.csv")
    visits = data.get('visits')

    #create array of days
    array_test = []
    for i in range(len(visits)):
            array = []
            strings = visits[i].strip().split(' ')
            for num in strings:
                array.append((int(num)-1)%7+1)
            array_test.append(array)

    #create weight array
    vec_weight = []
    for i in array_test:
        array = []
        w = 1
        for j in range(len(i)):
            array.append(w)
            w = w + 0.1
        vec_weight.append(array)

    #form results
    vec_result = []
    for i in range(len(array_test)):
        a, b = weighted_mode(array_test[i], vec_weight[i])
        vec_result.append(int(a[0]))
        print("Done: ", i)

    pd.DataFrame({'id': np.arange(1, len(vec_result)+1), 'nextvisit': vec_result}).to_csv('./probs4.csv', index = False, sep = ",")