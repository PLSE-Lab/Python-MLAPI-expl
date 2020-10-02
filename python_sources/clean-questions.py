# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataset='../input/train.csv'

try:
    train=pd.read_csv(dataset)
    print('Dataset %s successfully loaded'%dataset)
except Exception as k:
    print(k)
    raise
#see the first few rows of the data and general description of the dataset
print(train.head())
print(train.describe())

def clean(string):
    return re.sub('[!@#.,/$%^&*\(\)\{\}\[\]-_\<\>?\'\";:~`]',' ',str(string))

#replace Q1 and Q2 with new cleaned strings
train['question1']=train['question1'].apply(clean)
train['question2']=train['question2'].apply(clean)

print(train.head())