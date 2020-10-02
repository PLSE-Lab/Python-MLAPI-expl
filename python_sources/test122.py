# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="aluado"
__date__ ="$Dec 5, 2016 2:09:39 PM$"

from sklearn.linear_model import LogisticRegression
from sklearn.qda import QDA
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing


import pandas as pd
import numpy as np

#from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print ("Hello World")

    all_data= pd.read_csv("../input/creditcard.csv")

    all_data = all_data.drop('Time', 1)
    all_data = all_data.drop('V8', 1)
    all_data = all_data.drop('V13', 1)
    all_data = all_data.drop('V15', 1)
    all_data = all_data.drop('V19', 1)
    all_data = all_data.drop('V20', 1)
    all_data = all_data.drop('V21', 1)
    all_data = all_data.drop('V22', 1)
    all_data = all_data.drop('V23', 1)
    all_data = all_data.drop('V24', 1)
    all_data = all_data.drop('V25', 1)
    all_data = all_data.drop('V26', 1)
    all_data = all_data.drop('V27', 1)
    all_data = all_data.drop('V28', 1)
    more_than_3000=[]
    for x in all_data['Amount']:
        if x >=3000:
            more_than_3000.append(999)
        else:
            more_than_3000.append(-999)
    all_data = all_data.drop('Amount', 1)

    all_data['isMore3000']=more_than_3000

    all_data['feature1']=all_data['V1']*all_data['V3']
    all_data['feature2']=all_data['V2']*all_data['V4']
    all_data['feature3']=all_data['V2']*all_data['V11']
    all_data['feature4']=all_data['V2']*all_data['V2']



    label =all_data["Class"].values
    all_data = all_data.drop('Class', 1)

    lr=LogisticRegression()
    scores = cross_val_score(lr, all_data.values,label, cv=5)
    print("Accuracy: %0.10f (+/- %0.10f)" % (scores.mean(), scores.std() * 2))
