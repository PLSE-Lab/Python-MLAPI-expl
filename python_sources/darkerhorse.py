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
df_train = pd.read_csv('../input/train.csv')
print (df_train.head())
df1 = df_train[['id']].values.tolist()
#df1.apply(pd.to_numeric)
#print (type(float(df1[['id']])))
print (df_train[['question1']].values.tostring())
#
#for a in df1:
    #if df_train[['question1']][a].values.tostring() == df_train[['question2']][a]values.tostring():
        #print ('dude')