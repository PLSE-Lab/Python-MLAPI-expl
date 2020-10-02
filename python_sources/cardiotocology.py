#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df=pd.read_csv("../input/fetalhr/CTG.csv")


# In[ ]:


df.head()


# ## FileName:	of CTG examination	
# ## Date:	of the examination	
# ## b:	start instant	
# ## e:	end instant	
# ## LBE:	baseline value (medical expert)	
# ## LB:	baseline value (SisPorto)	
# ## AC:	accelerations (SisPorto)	
# ## FM:	foetal movement (SisPorto)	
# ## UC:	uterine contractions (SisPorto)	
# ## ASTV:	percentage of time with abnormal short term variability  (SisPorto)	
# ## mSTV:	mean value of short term variability  (SisPorto)	
# ## ALTV:	percentage of time with abnormal long term variability  (SisPorto)	
# ## mLTV:	mean value of long term variability  (SisPorto)	
# ## DL:	light decelerations	
# ## DS:	severe decelerations	
# ## DP:	prolongued decelerations	
# ## DR:	repetitive decelerations	
# ## Width:	histogram width	
# ## Min:	low freq. of the histogram	
# ## Max:	high freq. of the histogram	
# ## Nmax:	number of histogram peaks	
# ## Nzeros:	number of histogram zeros	
# ## Mode:	histogram mode	
# ## Mean:	histogram mean	
# ## Median:	histogram median	
# ## Variance:	histogram variance	
# ## Tendency:	histogram tendency: -1=left assymetric; 0=symmetric; 1=right assymetric	
# ## A:	calm sleep	
# ## B:	REM sleep	
# ## C:	calm vigilance	
# ## D:	active vigilance	
# ## SH:	shift pattern (A or Susp with shifts)	
# ## AD:	accelerative/decelerative pattern (stress situation)	
# ## DE:	decelerative pattern (vagal stimulation)	
# ## LD:	largely decelerative pattern	
# ## FS:	flat-sinusoidal pattern (pathological state)	
# ## SUSP:	suspect pattern	
# ## CLASS:	Class code (1 to 10) for classes A to SUSP	
# ## NSP:	Normal=1; Suspect=2; Pathologic=3	
# 

# In[ ]:


df=df.drop(["FileName","Date","SegFile","b","e"],axis=1)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df=df.dropna()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dtypes


# In[ ]:


X=df[['LBE', 'LB', 'AC', 'FM', 'UC', 'DL',
       'DS', 'DP', 'DR']]
Y=df[["NSP"]]


# ## Peforming the scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
Scaler=StandardScaler()
X=Scaler.fit_transform(X)


# In[ ]:



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)


# ## Classifiying the NSP into, Normal=1; Suspect=2; Pathologic=3

# ### CHecking for appropriate values of gamma

# In[ ]:


from sklearn.svm import SVC

svm_clf=SVC(kernel="poly",degree=6,coef0=5,gamma=0.1)
svm_clf=svm_clf.fit(X_train,y_train)
y_pred=svm_clf.predict(X_test)


# ## Calculating different metrics

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


f1_score(y_test,y_pred,average='weighted')


# In[ ]:



accuracy_score(y_test,y_pred)


# In[ ]:


precision_score(y_test,y_pred,average='weighted')


# In[ ]:


recall_score(y_test,y_pred,average="weighted")


# # Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


tree_clf=DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=4, max_depth=6, )
tree_clf=tree_clf.fit(X_train,y_train)
y_pred=tree_clf.predict(X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


recall_score(y_test,y_pred,average="weighted")


# In[ ]:


precision_score(y_test,y_pred,average='weighted')


# In[ ]:


from sklearn.tree import export_graphviz
export_graphviz(
tree_clf, out_file="tree.dot",
feature_names=['LBE', 'LB', 'AC', 'FM', 'UC', 'DL',
       'DS', 'DP', 'DR'],
class_names="NSP",
rounded=True,
filled=True)


# In[ ]:


from subprocess import check_call
check_call(['dot','-Tpng','tree.dot','-o','tree.png'])


# In[ ]:





# # Using the ensemble technique

# In[ ]:


from sklearn.ensemble import VotingClassifier, RandomForestClassifier


# In[ ]:


svm_clf=SVC(kernel="poly",degree=6,coef0=5,gamma=0.1,probability=True)
decision_tree=DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=4, max_depth=6)
rnd_clf=RandomForestClassifier()
voting_clf=VotingClassifier(estimators=[("svm",svm_clf),('rf',rnd_clf),("decision_tree",decision_tree)],voting="hard")


# In[ ]:


voting_clf.fit(X_train,y_train)


# In[ ]:


for clf in (rnd_clf, svm_clf,decision_tree, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# ## As, we can observe that there is a slight increase in the overall acuracy while using the ensemble model.

# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,n_jobs=-1,max_samples=100, bootstrap=True)


# In[ ]:


bag_clf.fit(X_train,y_train)
y_pred=bag_clf.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


##This accuracy is better than our previous decision tree model
##Therefore, we will again call ensemble technique.


# In[ ]:


voting_clf=VotingClassifier(estimators=[("svm",svm_clf),('rf',rnd_clf),("bagging_clf",bag_clf)],voting="hard")


# In[ ]:


for clf in (rnd_clf, svm_clf,bag_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[ ]:


##Using trial and hit method, and performing out-of-box evaluation
##Since a predictor never sees the oob instances during training, it can be evaluated on these instances,
##without the need for a separate validation set or cross-validation. You can evaluate the ensemble itself by
##averaging out the oob evaluations of each predictor.


# In[ ]:


bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,n_jobs=-1,max_samples=100, bootstrap=True,oob_score=True)
bag_clf.fit(X_train,y_train)
y_pred=bag_clf.predict(X_test)
print(bag_clf.oob_score_)
print(accuracy_score(y_test,y_pred))


# # Applying the ADAboost 

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


ada_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=1000,learning_rate=0.1)


# In[ ]:


ada_clf.fit(X_train,y_train)
y_pred=ada_clf.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,y_pred))


# # Applying XGBoost

# In[ ]:


from xgboost import XGBClassifier
xgb_clf=XGBClassifier(learning_rate=0.001)
xgb_clf.fit(X_train,y_train)
y_pred=xgb_clf.predict(X_test)


# In[ ]:


print(accuracy_score(y_test,y_pred))


# In[ ]:


voting_clf=VotingClassifier(estimators=[("svm",svm_clf),("xgb_clf",xgb_clf)],voting="hard")


# In[ ]:


for clf in (svm_clf,xgb_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[ ]:




