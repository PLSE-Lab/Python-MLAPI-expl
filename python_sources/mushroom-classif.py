# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn import svm
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier



print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_csv('../input/mushrooms.csv');
print(train.shape)
y_train=pd.DataFrame(columns=['class'])
y_train['class']=train['class']
y_train['class']=y_train['class'].map({'e':0,'p':1})
print(y_train.head())
train.fillna(value='-100',inplace=True)
train['cap-shape']=train['cap-shape'].map({'b':1,'c':2,'x':3,'f':4,'k':5,'s':6})
train['cap-surface']=train['cap-surface'].map({'f':1,'g':2,'y':3,'s':4})
train['cap-color']=train['cap-color'].map({'n':1,'b':2,'c':3,'g':4,'r':5,'p':6,'u':7,'e':8,'w':9,'y':10})

X_train=pd.DataFrame(columns=['cap','bruises','odor','gill','stalk','veil','ring','spore-print-color','population','habitat'])

X_train['cap']=(train['cap-shape']+train['cap-surface']+train['cap-color'])/3

X_train['bruises']=train['bruises'].map({'t':1,'f':2})

X_train['odor']=train['odor'].map({'a':1,'i':2,'c':3,'y':4,'f':5,'m':6,'n':7,'p':8,'s':9})

train['gill-attachment']=train['gill-attachment'].map({'a':1,'d':2,'f':3,'n':4})
train['gill-spacing']=train['gill-spacing'].map({'c':1,'w':2,'d':3})
train['gill-size']=train['gill-size'].map({'b':1,'n':2})
train['gill-color']=train['gill-color'].map({'k':1,'n':2,'b':3,'h':4,'g':5,'r':6,'o':7,'p':8,'u':9,'e':10,'w':11,'y':12})

X_train['gill']=(train['gill-attachment']+train['gill-spacing']+train['gill-size']+train['gill-color'])/4

train['stalk-shape']=train['stalk-shape'].map({'e':1,'t':2})
train['stalk-root']=train['stalk-root'].map({'b':1,'c':2,'u':3,'e':4,'z':5,'r':6,'oo':-1})
train['stalk-surface-above-ring']=train['stalk-surface-above-ring'].map({'f':1,'y':2,'k':3,'s':4,'-100':-1})
train['stalk-surface-below-ring']=train['stalk-surface-below-ring'].map({'f':1,'y':2,'k':3,'s':4,'-100':-1})
train['stalk-color-above-ring']=train['stalk-color-above-ring'].map({'n':1,'b':2,'c':3,'g':4,'o':5,'p':6,'e':7,'w':8,'y':9})
train['stalk-color-below-ring']=train['stalk-color-below-ring'].map({'n':1,'b':2,'c':3,'g':4,'o':5,'p':6,'e':7,'w':8,'y':9})

X_train['stalk']=(train['stalk-shape']+train['stalk-root']+train['stalk-surface-above-ring']+train['stalk-surface-below-ring']+train['stalk-color-above-ring']+train['stalk-color-below-ring'])/6

train['veil-type']=train['veil-type'].map({'p':1,'u':2})
train['veil-color']=train['veil-color'].map({'n':1,'o':2,'w':3,'y':4})

X_train['veil']=(train['veil-type']+train['veil-color'])/2

train['ring-number']=train['ring-number'].map({'n':1,'o':2,'t':3})
train['ring-type']=train['ring-type'].map({'c':1,'e':2,'f':3,'l':4,'n':5,'p':6,'s':7,'z':8})

X_train['ring']=(train['ring-number']+train['ring-type'])/2

X_train['spore-print-color']=train['spore-print-color'].map({'k':1,'n':2,'b':3,'h':4,'r':5,'o':6,'u':7,'w':8,'y':9})

X_train['population']=train['population'].map({'a':1,'c':2,'n':3,'s':4,'v':5,'y':6})

X_train['habitat']=train['habitat'].map({'g':1,'l':2,'m':3,'p':4,'u':5,'w':6,'d':7})


X_1,X_2,Y_1,Y_2=train_test_split(X_train,y_train['class'],test_size=0.20,random_state=20)
print(X_1.head())
X_1.fillna(value=-100,inplace=True)
X_2.fillna(value=-100,inplace=True)
classif=DecisionTreeClassifier(criterion="entropy")
print(classif.fit(X_1,Y_1))
predictions = classif.predict(X_2)
print(accuracy_score(Y_2, predictions))















# Any results you write to the current directory are saved as output.