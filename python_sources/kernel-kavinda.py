#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor'], value=[1, 2])
Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], value=[1, 2,3,4,5,6,7,8,9,10,11,12])
Trainset  = Trainset.replace(to_replace=[False, True], value=[1, 2])

xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor'], value=[1, 2])
xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], value=[1, 2,3,4,5,6,7,8,9,10,11,12])
xset  = xset.replace(to_replace=[False, True], value=[1, 2])

print(Trainset.shape)
print(list(Trainset.columns))
print(xset.shape)

Trainset
xArray = Trainset.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,12,13,14]].values
yArray = Trainset.iloc[:, 17].values

print(xArray)
print(len(yArray))
m = 10466
M =[]

xsetArray = xset.iloc[:, [0,1,2,3,4,5,6,7,8,9,11,12,13,14]].values

#print(xsetArray)

alpha = 0.05 

class LogisticRegression:    
    def gradient_decent(self, x, y):
        w = np.ones((np.shape(x)[1],1))
        print (type(x))
        wARR = np.mat(w)
        print (type(wARR))
        print (type(y))
        for i in range(m):
            w = w - alpha * x.transpose() * (self.logistic(x * w) - y)
        return w

    def classify(self, x, w):
        prob = self.logistic(sum(x * w))
        classification = 0
        if prob >  0.5: classification = 1
            #myData = 1
        #else:
          #  myData = 0
        #myData
        return classification
    
    def logistic(self,ws):
        return 1.0/(1 + np.exp(-ws))


logisticRegression = LogisticRegression()
wArray = logisticRegression.gradient_decent(np.mat(xArray), np.mat(yArray).transpose())
print ("close to window proceed")
#handle_command_line()
score = xsetArray
print(score)
for p in range(1850):
    #print(logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]), float(score[p][7])], wArray))
    M += [(p+1,logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13])], wArray))]

print(M)    
dfObj = pd.DataFrame(M, columns = ["ID","Revenue"]) 
#print(type(pr))

#dfObj.to_csv('csv_to_submit.csv', index = False)
#print("Writing complete")

# Any results you write to the current directory are saved as output.


# In[ ]:


def handle_command_line():
    flag = True
    while(flag):
        entry = input("input:")
        if (entry != "exit"):
            score = entry.split()
            #print(logisticRegression.classify([float(score[0]), float(score[1]),float(score[2]), float(score[3]),float(score[4]), float(score[5]),float(score[6]), float(score[7])], wArray))
            p = logisticRegression.classify([float(score[0]), float(score[1]),float(score[2]), float(score[3]),float(score[4]), float(score[5]),float(score[6]), float(score[7],float(score[8]), float(score[9])], wArray)
            
        else:
            flag = False


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

Trainset.values.tolist()

Trainset
Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor'], value=[1, 2])

#Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], value=[1, 2,3,4,5,6,7,8,9,10,11,12])

#Trainset  = Trainset.replace(to_replace=[False, True], value=[1, 2])
#temp_df2 = pd.DataFrame({'data': data.data.unique(), 'data_new':range(len(data.data.unique()))})# create a temporary dataframe 
#data = data.merge(temp_df2, on='data', how='left')# Now merge it by assigning different values to different strings.

print(Trainset)


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

print(Trainset.shape)
print(list(Trainset.columns))
print(xset.shape)

xArray = Trainset.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values
yArray = Trainset.iloc[:, 17].values

print(xArray)
print(type(xArray[0][9]))
print(len(yArray))
m = 10466
M =[]
count = 0
xsetArray = xset.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values

print(xsetArray)

alpha = 0.05 

class LogisticRegression:    
    def gradient_decent(self, x, y):
        w = np.ones((np.shape(x)[1],1))
        print (type(x))
        wARR = np.mat(w)
        print (type(wARR))
        print (type(y))
        for i in range(m):
            w = w - alpha * x.transpose() * (self.logistic(x * w) - y)
        return w

    def classify(self, x, w):
        prob = self.logistic(sum(x * w))
        classification = 0
        if prob >  0.5: 
            classification = 1
            #myData = 1
        #else:
          #  myData = 0
        #myData
        return classification
    def logistic(self,ws):
        return 1.0/(1 + np.exp(-ws))

logisticRegression = LogisticRegression()
wArray = logisticRegression.gradient_decent(np.mat(xArray), np.mat(yArray).transpose())
print ("close to window proceed")
#handle_command_line()
score = xsetArray

for p in range(1850):
    if logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13])], wArray) == 1:
        count+= 1                          
    #print(logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]), float(score[p][7])], wArray))
    M += [(p+1,logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13])], wArray))]

print("count=",count)
print(M)    
dfObj = pd.DataFrame(M, columns = ["ID","Revenue"]) 
#print(type(pr))

#dfObj.to_csv('csv_to_submit.csv', index = False)
#print("Writing complete")


# In[ ]:





# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

print(Trainset.shape)
print(list(Trainset.columns))
print(xset.shape)

xArray = Trainset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values
yArray = Trainset.iloc[:, 17].values

print(xArray)
print(type(xArray[0][9]))
print(len(yArray))
m = 10466
M =[]
count = 0
xsetArray = xset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values

print(xsetArray)

alpha = 0.05 

class LogisticRegression:    
    def gradient_decent(self, x, y):
        w = np.ones((np.shape(x)[1],1))
        print (type(x))
        wARR = np.mat(w)
        print (type(wARR))
        print (type(y))
        for i in range(m):
            w = w - alpha * x.transpose() * (self.logistic(x * w) - y)
        return w

    def classify(self, x, w):
        prob = self.logistic(sum(x * w))
        classification = 0
        if prob >  0.5: 
            classification = 1
            #myData = 1
        #else:
          #  myData = 0
        #myData
        return classification
    def logistic(self,ws):
        return 1.0/(1 + np.exp(-ws))

logisticRegression = LogisticRegression()
wArray = logisticRegression.gradient_decent(np.mat(xArray), np.mat(yArray).transpose())
print ("close to window proceed")
#handle_command_line()
score = xArray

for p in range(m):
    if logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13]),float(score[p][14]),float(score[p][15])], wArray) == yArray[p]:
        count+= 1                          
    #print(logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]), float(score[p][7])], wArray))
    M += [(p+1,logisticRegression.classify([float(score[p][0]), float(score[p][1]),float(score[p][2]), float(score[p][3]),float(score[p][4]), float(score[p][5]),float(score[p][6]),float(score[p][7]),float(score[p][8]),float(score[p][9]),float(score[p][10]), float(score[p][11]),float(score[p][12]),float(score[p][13]),float(score[p][14]),float(score[p][15])], wArray))]

print("count=",count)
print(M)    
dfObj = pd.DataFrame(M, columns = ["ID","Revenue"]) 
#print(type(pr))

#dfObj.to_csv('csv_to_submit.csv', index = False)
#print("Writing complete")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75
print(Trainset)

Train, Test = Trainset[Trainset['is_train'] == True], Trainset[Trainset['is_train'] == False]

print('Number of Training examples', len(Train))
print('Number of Testing examples', len(Test))

features = Trainset.columns[:16]
print(features)
y = Train['Revenue']
y

clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
clf.fit(Train[features],y)

clf.predict(Test[features])

clf.predict_proba(Test[features])

preds = clf.predict(Test[features])

pd.crosstab(Test['Revenue'], preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue'])
#Test['Revenue']
#X, y = Trainset.iloc[:, :16], Trainset.iloc[:, 17]

#Xset = xset.iloc[:, :16]
#print(X)
#print(y)

#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(X, y)
#clf

#clf = RandomForestClassifier(n_estimators=17, max_depth=None,min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y, cv=5)
#scores.mean() 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as ma

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Trainset = pd.read_csv("/kaggle/input/hackstat2k19/Trainset.csv", header = 0) 
Trainset = Trainset.dropna()

sampleset = pd.read_csv("/kaggle/input/hackstat2k19/sample_submisison.csv", header = 0) 
sampleset = sampleset.dropna()

Trainset  = Trainset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
Trainset  = Trainset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

xset = pd.read_csv("/kaggle/input/hackstat2k19/xtest.csv")
xset = xset.dropna()

xset  = xset.replace(to_replace=['Jan', 'Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'], value=[0,2,166,0,307,21,54,69,72,97,641,183])
xset  = xset.replace(to_replace=['Returning_Visitor', 'New_Visitor' , 'Other'], value=[1242, 366, 11])

#Trainset['is_train'] = np.random.uniform(0,1,len(Trainset)) <= 0.75
print(Trainset)

Train, Test = Trainset,xset

print('Number of Training examples', len(Train))
print('Number of Testing examples', len(Test))

features = Trainset.columns[:16]
print(features)
y = Train['Revenue']
y

id_ = sampleset['ID']
#new_ = id_.DataFarame(id_)
clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
clf.fit(Train[features],y)

clf.predict(Test[features])

clf.predict_proba(Test[features])

preds = clf.predict(Test[features])

#print(tpe(new_))
#print(type(preds))
YArray= id_.as_matrix(columns=None)
print (YArray)

df = pd.DataFrame({"ID" : YArray, "Revenue" : preds})
df.to_csv("submission.csv", index=False)
#dfObj = pd.DataFrame(YArray,preds,columns = ["ID","Revenue"]) 

#dfObj.to_csv('submit.csv', index = False)

#pd.crosstab(Test['Revenue'], preds, rownames = ['Actual Revenue'], colnames = ['Predicted Revenue'])
#Test['Revenue']
#X, y = Trainset.iloc[:, :16], Trainset.iloc[:, 17]

#Xset = xset.iloc[:, :16]
#print(X)
#print(y)

#clf = RandomForestClassifier(n_estimators=10)
#clf = clf.fit(X, y)
#clf

#clf = RandomForestClassifier(n_estimators=17, max_depth=None,min_samples_split=2, random_state=0)
#scores = cross_val_score(clf, X, y, cv=5)
#scores.mean() 


# In[ ]:


df


# In[ ]:





# In[ ]:





# In[ ]:




