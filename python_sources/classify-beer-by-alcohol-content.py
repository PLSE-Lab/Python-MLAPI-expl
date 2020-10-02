# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data_set = pd.read_csv('../input/beers.csv')

abv_values = []
type_labels = []
types = []

for beer in data_set.itertuples():
    abv = float(beer[2])
    if not np.isnan([abv]):
        abv_values.append(abv)
        beer_type_label = beer[6]
        if beer_type_label in type_labels:
            type_num = int(type_labels.index(beer_type_label))
            if not np.isnan([type_num]): 
                types.append(type_num)
            else:
                abv_values.pop()
        else:
            type_labels.append(beer_type_label)
            type_num = int(type_labels.index(beer_type_label))
            if not np.isnan([type_num]): 
                types.append(type_num)
            else:
                abv_values.pop()

abv_values = np.array(abv_values)        
types = np.array(types)

split = int(abv_values.size/2)

gnb = GaussianNB()
abv_values = abv_values.reshape(-1,1)
prediction = gnb.fit(abv_values,types).predict(abv_values)
mean_accuracy = np.mean(prediction == types)
print('Accuracy =', mean_accuracy * 100, '%')
