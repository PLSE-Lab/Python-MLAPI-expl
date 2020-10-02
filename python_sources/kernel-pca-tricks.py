#!/usr/bin/env python
# coding: utf-8

# # Nonlinear dimensionality reduction

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("fivethirtyeight")
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


data,z=make_swiss_roll(n_samples=5000,noise=0.05)
#data[:,1] *=2
swissrolldata=pd.DataFrame(data,columns=['x1','x2','x3'])


# In[ ]:


ward=AgglomerativeClustering(n_clusters=6,linkage='ward').fit(swissrolldata)
labels=ward.labels_
swissrolldata['labels']=labels


# In[ ]:


swissrolldata.head()


# In[ ]:


swissrolldata.corr()


# In[ ]:



#%matplotlib qt
get_ipython().run_line_magic('matplotlib', 'inline')
xdim=swissrolldata['x1']
ydim=swissrolldata['x2']
zdim=swissrolldata['x3']
fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(xdim,ydim,zdim,c=swissrolldata['labels'],cmap="rainbow",s=70)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title("Swiss roll")
plt.show()


# In[ ]:


fig=px.scatter_3d(swissrolldata,x='x1',y='x2',z='x3',color='labels')
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()


# # Kernel functions and the kernel trick

# # Linear Kernel

# In[ ]:


from sklearn.decomposition import KernelPCA


# In[ ]:


kernel=KernelPCA(n_components=2,kernel='linear')
data=swissrolldata.drop('labels',axis=1)
newdata=kernel.fit_transform(data)
newdata=pd.DataFrame(newdata,columns=['z1','z2'])
newdata['labels']=labels


# In[ ]:


newdata.head()


# In[ ]:


fig=px.scatter(newdata,x='z1',y='z2',color='labels')
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()


# # Gaussian radial basis function (RBF) Kernel PCA

# In[ ]:


kernel=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
data=swissrolldata.drop('labels',axis=1)
newdata=kernel.fit_transform(data)
newdata=pd.DataFrame(newdata,columns=['z1','z2'])
newdata['labels']=labels


# In[ ]:


newdata.head()


# In[ ]:



fig=px.scatter(newdata,x='z2',y='z1',color='labels')
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()


# # Sigmoid kernel

# In[ ]:


kernel=KernelPCA(n_components=2,kernel='sigmoid')
data=swissrolldata.drop('labels',axis=1)
newdata=kernel.fit_transform(data)
newdata=pd.DataFrame(newdata,columns=['z1','z2'])
newdata['labels']=labels


# In[ ]:


newdata.head()


# In[ ]:


fig=px.scatter(newdata,x='z1',y='z2',color='labels')
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()


# # Polynomial Kernel

# In[ ]:


kernel=KernelPCA(n_components=2,kernel='poly')
data=swissrolldata.drop('labels',axis=1)
newdata=kernel.fit_transform(data)
newdata=pd.DataFrame(newdata,columns=['z1','z2'])
newdata['labels']=labels


# In[ ]:


newdata.head()


# In[ ]:


fig=px.scatter(newdata,x='z1',y='z2',color='labels')
fig.update_traces(marker=dict(size=5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[ ]:


clf=Pipeline([
    ("kcpa",KernelPCA(n_components=2)),
    ("logistic_regression",LogisticRegression())
            ])
param_grid=[{"kcpa__gamma":[0.02,0.04,0.05,1,10],
             "kcpa__kernel":["rbf","sigmoid","linear","poly"]
            }]
grid_search=GridSearchCV(clf,param_grid,verbose=3)
x=swissrolldata.drop('labels',axis=1)
y=swissrolldata['labels']
grid_search.fit(x,y)
print(grid_search.best_params_)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)
lg=LogisticRegression()
dc=DecisionTreeClassifier()
rnf=RandomForestClassifier(n_estimators=500)
sv=SVC()
ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=500,
                           algorithm="SAMME.R",learning_rate=0.5)
gbc=GradientBoostingClassifier()
xg=XGBClassifier()


# In[ ]:



dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=500),
    "adaboost":AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=500,
                           algorithm="SAMME.R",learning_rate=0.5),
    "xgboost":XGBClassifier(),
    "voting classifier":VotingClassifier(estimators=[('lr',lg),
                                        ('rf',rnf),('dt',dc),('kn',knn),
                                        ('svm',sv),('adaboost',ada),
                                        ('gbcb',gbc),('xgb',xg)],voting="hard")
}


# In[ ]:


no=len(dict_classifiers.keys())
def classifiers(kernel,data,labels):
    results = pd.DataFrame(data=np.zeros(shape=(no,4)),
                columns =['classifier','train_score','validation score','training_time'])
    count=0  
    for key,classifier in dict_classifiers.items():
        start = time.clock()
        ker=KernelPCA(n_components=2,kernel=kernel)
        x=ker.fit_transform(data)
        y=labels
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=32)
        classifier.fit(X_train, Y_train)
        y_pred=classifier.predict(X_test)
        acc=accuracy_score(y_pred,Y_test)
        train_score =classifier.score(X_train, Y_train)
        end = time.clock()
        diff = end-start
        results.loc[count,'classifier']=key
        results.loc[count,'train_score']=train_score
        results.loc[count,'validation score']=acc
        results.loc[count,'training_time']=diff
        count=count+1
    return results


# In[ ]:


d=swissrolldata.drop('labels',axis=1)
l=swissrolldata['labels']
kernels=['linear','rbf','sigmoid','poly']
for i in kernels:
    res=classifiers(i,d,l)
    print(f"kernel is {i}")
    print(res.sort_values(by='validation score',ascending=False))


# In[ ]:


d=swissrolldata.drop('labels',axis=1)
l=swissrolldata['labels']
knn=KNeighborsClassifier(n_neighbors=5)
ker=KernelPCA(kernel="sigmoid",gamma=10,n_components=2)
x=ker.fit_transform(d)
y=labels
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=32)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:



import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,yticklabels=False,fmt="d")
plt.title("Confusion matrix")


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


d=np.array(swissrolldata.drop('labels',axis=1))
l=np.array(swissrolldata['labels'])
encoding=OneHotEncoder()
y=encoding.fit_transform(l.reshape(-1,1))
ker=KernelPCA(kernel="sigmoid",gamma=10,n_components=2)
x=ker.fit_transform(d)
y=labels
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=32)


# In[ ]:


def modelcreating(optimizer="adam",loss="sparse_categorical_crossentropy"):
    model=Sequential()
    model.add(Dense(128,input_dim=2,activation="relu"))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(6,activation="softmax"))
    model.compile(loss=loss,optimizer=optimizer,metrics=["accuracy"])
    return model
model=modelcreating()


# In[ ]:


model.summary()


# In[ ]:


earlystopping=EarlyStopping(monitor="val_loss",mode="min",patience=100)

