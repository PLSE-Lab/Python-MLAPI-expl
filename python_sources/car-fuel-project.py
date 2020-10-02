#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

consumption=pd.read_csv('../input/car-consume/measurements.csv')
consumption['distance']=consumption['distance'].str.replace(',','.').astype(float)
consumption['consume']=consumption['consume'].str.replace(',','.').astype(float)
consumption['temp_inside']=consumption['temp_inside'].str.replace(',','.').astype(float)
consumption


# In[ ]:


consumption.info()


# In[ ]:


data=consumption.drop(['specials','temp_outside','refill liters','refill gas'],axis=1)
data


# In[ ]:


data.info()


# In[ ]:


data['temp_inside']=data['temp_inside'].fillna(16.0)


# In[ ]:


data.describe()


# In[ ]:


data['speed'].plot.hist()
plt.show()


# In[ ]:


data['distance'].plot.hist()
plt.show()


# ## FUEL CONSUMPTION ACCORDING TO AC :-

# In[ ]:


sns.countplot(data['AC'])
plt.show()


# ## MIN. AND MAX. FUEL CONSUMPTION OF TRAVELLED DISTANCE :-

# In[ ]:


consmin=data.query('distance==1.300000')
consmin


# In[ ]:


consmax=data.query('distance==216.100000')
consmax    


# ## VISULIZATION OF MIN. AND MAX. FUEL CONSUMPTION OF TRAVELLED DISTANCE :-

# In[ ]:


sns.lmplot('distance','consume',consmin)
plt.show()


# In[ ]:


sns.lmplot('distance','consume',consmax)
plt.show()


# ## MIN. AND MAX. FUEL CONSUMPTION ACC TO SPEED :-

# In[ ]:


consmin=data.query('speed==14.000000')
consmin


# In[ ]:


consmax=data.query('speed==90.000000')
consmax


# ## VISULIZATION OF MAX. AND MAX. FUEL CONSUMPTION ACC TO SPEED :-

# In[ ]:


sns.lmplot('distance','speed',consmin)
plt.show()


# In[ ]:


sns.lmplot('distance','speed',consmax)
plt.show()


# ## MIN. FUEL CONSUMPTION IN RAINY DAYS:-

# In[ ]:


minrn=data.query('rain==0.000000')
minrn


# In[ ]:


sns.lmplot('distance','consume',data=minrn)
plt.show()


# ## MAXIMUM FUEL CONSUMPTION IN RAINY DAYS:-

# In[ ]:


maxrn=data.query('rain==1.000000')
maxrn


# In[ ]:


sns.lmplot('distance','consume',data=maxrn)
plt.show()


# ## MINIMUM FUEL CONSUMPTION IN SUNNY DAYS:-

# In[ ]:


minsun=data.query('sun==0.000000')
minsun


# In[ ]:


sns.lmplot('distance','consume',data=minsun)
plt.show()


# ## MAXIMUM FUEL CONSUMPTION IN SUNNY DAYS:-

# In[ ]:


maxsun=data.query('sun==1.000000')
maxsun


# In[ ]:


sns.lmplot('distance','speed',data=maxsun)
plt.show()


# ## MINIMUN FUEL CONSUME WITH AC:-

# In[ ]:


minac=data.query('AC==0.000000')
minac


# In[ ]:


sns.lmplot('distance','consume',data=minac)
plt.show()


# ## MAXIMUM FUEL CONSUME WITH AC:-

# In[ ]:


maxac=data.query('AC==1.000000')
maxac


# In[ ]:


sns.lmplot('distance','consume',data=maxac)
plt.show()


# ## THE DIFFERENT GAS TYPE USED BY USER:-

# In[ ]:


sns.countplot(data['gas_type'])
plt.show()


# In[ ]:


gas1=data[data['gas_type']=='E10']
gas1


# In[ ]:


gas1.describe()


# ## MINIMUM DISTANCE TRAVEL BY GAS1:-

# In[ ]:


mingas1=data.query('distance==1.700000')
mingas1


# In[ ]:


sns.lmplot('distance','consume',data=mingas1)
plt.show()


# ## MAXIMUM DISTANCE TRAVEL BY GAS1:-

# In[ ]:


maxgas1=data.query('distance==130.300000')
maxgas1


# In[ ]:


sns.lmplot('distance','consume',data=maxgas1)
plt.show()


# ## minimum speed consume by gas1:-

# In[ ]:


mingas1=data.query('speed==14.000000')
mingas1


# In[ ]:


sns.lmplot('distance','consume',data=mingas1)
plt.show()


# ## MAXIMUM SPEED CONSUME BY GAS1:-

# In[ ]:


maxgas1=data.query('speed==88.000000')
maxgas1


# In[ ]:


sns.lmplot('distance','consume',data=maxgas1)
plt.show()


# In[ ]:


gas2=data[data['gas_type']=='SP98']
gas2


# In[ ]:


gas2.describe()


# ## MINIMUM DISTANCE TRAVEL BY GAS2:-

# In[ ]:


mingas2=data.query('distance==1.300000')
mingas2


# In[ ]:


sns.lmplot('distance','consume',data=mingas2)
plt.show()


# ## MAXIMUM DISTANCE TRAVEL BY GAS2:-

# In[ ]:


maxgas2=data.query('distance==216.100000')
maxgas2


# In[ ]:


sns.lmplot('distance','consume',data=maxgas2)
plt.show()


# ## MINIMUM SPEED CONSUME BY GAS2 :-

# In[ ]:


mingas2=data.query('speed==16.000000')
mingas2


# In[ ]:


sns.lmplot('distance','consume',data=mingas2)
plt.show()


# ## MAXIMUM SPEED CONSUME BY GAS2:-

# In[ ]:


maxgas2=data.query('speed==90.000000')
maxgas2


# In[ ]:


sns.lmplot('distance','consume',data=maxgas2)
plt.show()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lr=LabelEncoder()
data['gas_type']=lr.fit_transform(data['gas_type'])


# In[ ]:


data['distance']=data['distance'].astype(int)
data['consume']=data['consume'].astype(int)
data['temp_inside']=data['temp_inside'].astype(int)
data


# ## DECISION MAKING:-

# In[ ]:


x=data.drop('consume',axis=1)
#print(x)

y=data.iloc[:,1]
y


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=0)
x_train


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
x_trainmn=mms.fit_transform(x_train)
x_testmn=mms.transform(x_test)


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_trainstd=ss.fit_transform(x_train)
x_teststd=ss.transform(x_test)


# In[ ]:


from sklearn.decomposition import PCA
pc=PCA(n_components=5)
x_trainpca=pc.fit_transform(x_train)
x_testpca=pc.transform(x_test)


# ## DECISION TREE ACC TO TRAIN TEST DATA:-

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt1=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=56)
dt1.fit(x_train,y_train)
pred1=dt1.predict(x_test)
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred1)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred1))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred1))


# In[ ]:


fet=['distance','speed','temp_inside','gas_type','AC','rain','sun']
#distance	speed	temp_inside	gas_type	AC	rain	sun
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(dt1, out_file=None, feature_names=fet, filled = True,rounded=True))


display(SVG(graph.pipe(format='svg')))


# In[ ]:


plt.scatter(y_test,pred1)
plt.xlabel('Values')
plt.ylabel('pred1')
plt.show()


# ## DECISION TREE ACC TO MINMAX SCALAR:-

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt2=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=56)
dt2.fit(x_trainmn,y_train)
pred2=dt2.predict(x_testmn)
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred2)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred2))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred2))


# In[ ]:


fet=['distance','speed','temp_inside','gas_type','AC','rain','sun']
#distance	speed	temp_inside	gas_type	AC	rain	sun
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(dt2, out_file=None, feature_names=fet, filled = True,rounded=True))


display(SVG(graph.pipe(format='svg')))


# In[ ]:


plt.scatter(y_test,pred2)
plt.xlabel(' Values')
plt.ylabel('pred2')
plt.show()


# ## DECISION TREE ACC TO STANDARD SCALAR:-

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt3=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=56)
dt3.fit(x_trainstd,y_train)
pred3=dt3.predict(x_teststd)
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred3)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred3))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred3))


# In[ ]:


fet=['distance','speed','temp_inside','gas_type','AC','rain','sun']
#distance	speed	temp_inside	gas_type	AC	rain	sun
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(dt3, out_file=None, feature_names=fet, filled = True,rounded=True))


display(SVG(graph.pipe(format='svg')))


# In[ ]:


plt.scatter(y_test,pred3)
plt.xlabel(' Values')
plt.ylabel('pred3')
plt.show()


# ## DECISION TREE ACC TO PCA:-

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt4=DecisionTreeRegressor(criterion='mse',splitter='best',max_depth=3,random_state=56)
dt4.fit(x_trainpca,y_train)
pred4=dt4.predict(x_testpca)
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pred4)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test,pred4))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pred4))


# In[ ]:


fetpca=['distance','speed','temp_inside','gas_type','AC']
#distance	speed	temp_inside	gas_type	AC	rain	sun
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(dt4, out_file=None, feature_names=fetpca, filled = True,rounded=True))


display(SVG(graph.pipe(format='svg')))


# In[ ]:


plt.scatter(y_test,pred4)
plt.xlabel(' Values')
plt.ylabel('pred4')
plt.show()


# ## RANDOM FOREST CLASSIFIER BY TRAIN TEST:-

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)

model1.fit(x_train, y_train)
pred1=model1.predict(x_test)
pred1


# In[ ]:


estimators=model1.estimators_[3]
labels=['distance','speed','temp_inside','gas_type','AC','rain','sun']
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(estimators, out_file=None
   , feature_names=labels
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:


sns.distplot((y_test-pred1),bins=50)
plt.show()


# ## RANDOM FOREST CLASSIFIER ACC TO MINMAX SCALAR:-

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model2= RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)

model2.fit(x_trainmn, y_train)
pred2=model2.predict(x_testmn)
pred2


# In[ ]:


estimators=model2.estimators_[3]
labels=['distance','speed','temp_inside','gas_type','AC','rain','sun']
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(estimators, out_file=None
   , feature_names=labels
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:


sns.distplot((y_test-pred2),bins=20)
plt.show()


# ## RANDOM FOREST ACC TO STANDARD SACALAR:-

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model3= RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)

model3.fit(x_trainstd, y_train)
pred3=model3.predict(x_teststd)
pred3


# In[ ]:


estimators=model3.estimators_[3]
labels=['distance','speed','temp_inside','gas_type','AC','rain','sun']
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(estimators, out_file=None
   , feature_names=labels
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:


sns.distplot((y_test-pred3),bins=50)
plt.show()


# ## RANDOM FOREST ACC TO PCA:-

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model4= RandomForestClassifier(n_estimators=5, random_state=50,criterion='entropy',max_depth=4)

model4.fit(x_trainpca, y_train)
pred4=model4.predict(x_testpca)
pred4


# In[ ]:


estimators=model4.estimators_[3]
labels=['distance','speed','temp_inside','gas_type','AC']
from sklearn import tree
from graphviz import Source
from IPython.display import SVG
from IPython.display import display

graph = Source(tree.export_graphviz(estimators, out_file=None
   , feature_names=labels
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:


sns.distplot((y_test-pred4),bins=50)
plt.show()


# ## LINEAR REGRESSION ACC TO TRAIN TEST DATA:-

# In[ ]:


from sklearn.linear_model import LinearRegression
lnr=LinearRegression() 
lnr.fit(x_train,y_train)
pd=lnr.predict(x_test)


# In[ ]:


print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pd)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, pd))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pd))


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(y_test,pd)
plt.xlabel('y_test(True Values)')
plt.ylabel('pd')
plt.show()


# In[ ]:


sns.distplot((y_test-pd),bins=50)
plt.show()


# ## LINEAR REGRESSION ACC TO MINMAX SCALAR:-

# In[ ]:


from sklearn.linear_model import LinearRegression
lnr1=LinearRegression()
lnr1.fit(x_trainmn,y_train)
pd1=lnr.predict(x_testmn)


# In[ ]:


print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pd1)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, pd1))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pd1))


# In[ ]:


plt.scatter(y_test,pd1)
plt.xlabel('Values')
plt.ylabel('pd1')
plt.show()


# In[ ]:


sns.distplot((y_test-pd1),bins=50)
plt.show()


# ## LINEAR REGRESSION ACC TO STANDARD SCALAR:-

# In[ ]:


from sklearn.linear_model import LinearRegression
lnr2=LinearRegression()
lnr2.fit(x_trainstd,y_train)
pd2=lnr2.predict(x_teststd)


# In[ ]:


print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pd2)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, pd2))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pd2))


# In[ ]:


plt.scatter(y_test,pd2)
plt.xlabel('Values')
plt.ylabel('pd2')
plt.show()


# In[ ]:


sns.distplot((y_test-pd2),bins=50)
plt.show()


# ## LINEAR REGRESSIION ACC TO PCA:-

# In[ ]:


from sklearn.linear_model import LinearRegression
lnr3=LinearRegression()
lnr3.fit(x_trainpca,y_train)
pd3=lnr3.predict(x_testpca)


# In[ ]:


print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test, pd3)))
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, pd3))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, pd3))


# In[ ]:


plt.scatter(y_test,pd3)
plt.xlabel('Values')
plt.ylabel('pd3')
plt.show()


# In[ ]:


sns.distplot((y_test-pd3),bins=50)
plt.show()

