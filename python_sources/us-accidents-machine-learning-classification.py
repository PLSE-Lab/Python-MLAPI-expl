#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
acc=pd.read_csv("../input/us-accidents/US_Accidents_May19.csv")
acc


# In[ ]:


a=acc['Source'].value_counts()
a
fig,ax=plt.subplots(figsize=(10,20))
plt.bar(a.index,a.values)

plt.show()


# In[ ]:


b=acc['Weather_Condition'].max()
b


# In[ ]:


acc.columns


# In[ ]:


x=acc.loc[:,['Distance(mi)','Source','Visibility(mi)','Wind_Speed(mph)','Traffic_Calming']]
y=acc.loc[:,[ 'Astronomical_Twilight']]
x


# In[ ]:


y=y.fillna(value='Astronomical_Twilight')
y


# In[ ]:


from sklearn.preprocessing import Imputer

missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', 
                               axis = 0)  
missingValueImputer = missingValueImputer.fit (x.loc[:,['Wind_Speed(mph)']])
x.loc[:,['Wind_Speed(mph)']] = missingValueImputer.transform(x.loc[:,['Wind_Speed(mph)']])
x


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x.iloc[:,0]=le.fit_transform(x.iloc[:,0])
x.iloc[:,1]=le.fit_transform(x.iloc[:,1])

x.iloc[:,4]=le.fit_transform(x.iloc[:,4])
x.iloc[:,2]=le.fit_transform(x.iloc[:,2])
x.iloc[:,3]=le.fit_transform(x.iloc[:,3])
x


# In[ ]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20,random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
pred=lr.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,pred))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(xtrain,ytrain)
pred1=gb.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,pred1))


# In[ ]:


from sklearn.svm import SVC
sc=SVC()
sc.fit(xtrain,ytrain)
pre=sc.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,pre))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)

dt.fit(xtrain,ytrain)
pred2=dt.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,pred2))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rc=RandomForestClassifier(n_estimators=10,max_depth=3,random_state=0,criterion='entropy')
rc.fit(xtrain,ytrain)
pred4=rc.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,pred4))


# In[ ]:



data_feature_names=['Distance(mi)','Source','Visibility(mi)','Wind_Speed(mph)','Traffic_Calming']
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(dt, out_file=None, feature_names=data_feature_names, filled = True,rounded=True))


display(SVG(graph.pipe(format='svg')))


# In[ ]:


estimators=rc.estimators_[5]
labels=['Distance(mi)','Source','Visibility(mi)','Wind_Speed(mph)','Traffic_Calming']
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display
graph = Source(tree.export_graphviz(estimators, out_file=None, feature_names=labels, filled = True))
display(SVG(graph.pipe(format='svg')))

