# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

"""
@script_author: dawn elza zachariah
@created on: 12 January 2020
@last modified: 13 January 2020
"""
#importing the required packages
import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier

irisdata=pd.read_csv('/kaggle/input/iris/Iris.csv')#loading the data
irisdata.isnull().sum()#checking for missing values
irisdata.info()#summary of the dataframe
cat=irisdata.select_dtypes(include='object').columns#columns with categorical values
for i in cat:
    irisdata[i]=pre.LabelEncoder().fit_transform(irisdata[i])#labelling the categorical values
    
y=irisdata['Species']
x=irisdata.drop(['Species','Id'],axis=1)

x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.3,random_state=2233)#Splitting into train data and test data

l1=[]
l2=[]

#finding euclidean distance and finding accuracy
for j in range(1,46):
    k=KNeighborsClassifier(n_neighbors=j,metric='euclidean')
    io=k.fit(x_train,y_train)
    pred=k.predict(x_test)
    l1.append(np.mean(pred !=y_test))
    l2.append(metrics.accuracy_score(y_test,pred))
print(l2)