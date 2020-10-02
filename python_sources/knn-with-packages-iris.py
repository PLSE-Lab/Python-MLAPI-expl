# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#importing packages
import sklearn.preprocessing as pre
import sklearn.model_selection as ms
import sklearn.metrics as metrics 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

       

from sklearn.neighbors import KNeighborsClassifier

#loading dataser
data=pd.read_csv('/kaggle/input/iris/Iris.csv')

#checking for missing values
data.isnull().sum()
data.info()

#checking for the categorical columns
col_obj=data.select_dtypes(include='object').columns

#Labeling the categorical variables 
for k in col_obj:
    data[k]=pre.LabelEncoder().fit_transform(data[k])
y=data['Species']
x=data.drop(['Species','Id'],axis=1)

#splitting the dataset into train and test
x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.3,random_state=2233)
f=[]
p=[]

#applying knn classification
for j in range(1,46):
    k=KNeighborsClassifier(n_neighbors=j,metric='euclidean')
    io=k.fit(x_train,y_train)
    pred=k.predict(x_test)
    f.append(np.mean(pred !=y_test))
    p.append(metrics.accuracy_score(y_test,pred))

#printing the accuracy
print(p)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



        

# Any results you write to the current directory are saved as output.