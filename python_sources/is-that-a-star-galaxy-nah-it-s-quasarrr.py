#!/usr/bin/env python
# coding: utf-8

# # Karthik Narayanan 
# # Vishakaraj Shanmugavel

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv")
data = pd.DataFrame(data)
datanew = data


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
datanew['class'] = lab.fit_transform(datanew['class'])
 


# In[ ]:


datanew.head()


# In[ ]:


for col in datanew.columns:
    print("    {} \n ---------- \n".format(col),np.unique(np.asarray(datanew[col])),"\n")


# In[ ]:


from collections import Counter

count = Counter(datanew['class'])
count


# In[ ]:


Y = datanew['class']
Y = pd.DataFrame(Y)
Y.head()


# In[ ]:


X=datanew
X = X.drop(columns=['class','objid','rerun'])
X.head()


# # Preliminary Data Analysis

# In[ ]:


from sklearn import preprocessing

X_scaled = preprocessing.scale(X)

X_scaled = pd.DataFrame(data=X_scaled,columns=X.columns)
datanew_scaled = X_scaled.copy()
datanew_scaled['class']=Y
finaldata = datanew_scaled.copy()
datanew_scaled.head()


# In[ ]:


X_scaled.shape


# In[ ]:


import seaborn as sns
sns.pairplot(datanew_scaled,kind='scatter',hue='class',palette="Set2")


# In[ ]:


import scipy as sp
def corrfunc(x, y, **kws):
    r, _ = sp.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("{:.2f}".format(r), xy=(.1, .5), xycoords=ax.transAxes, size=50)


# In[ ]:


g = sns.PairGrid(datanew_scaled)
g = g.map_lower(plt.scatter)
g = g.map_diag(plt.hist, edgecolor="w")
g = g.map_upper(corrfunc)


# In[ ]:


datanew_scaled.corr()


# In[ ]:


plt.figure(figsize=(15,30), dpi=100)
plt.subplot(7,2,1)
plt.scatter('g','u', data=datanew_scaled)
plt.xlabel('g')
plt.ylabel('u')
plt.subplot(7,2,2)
plt.scatter('g','r',data=datanew_scaled)
plt.xlabel('g')
plt.ylabel('r')
plt.subplot(7,2,3)
plt.scatter('i','r',data=datanew_scaled)
plt.xlabel('i')
plt.ylabel('r')
plt.subplot(7,2,4)
plt.scatter('z','i',data=datanew_scaled)
plt.xlabel('z')
plt.ylabel('i')
plt.subplot(7,2,5)
plt.scatter('mjd','plate',data=datanew_scaled)
plt.xlabel('mjd')
plt.ylabel('plate')
plt.subplot(7,2,6)
plt.scatter('specobjid','plate',data=datanew_scaled)
plt.xlabel('specobjid')
plt.ylabel('plate')
plt.subplot(7,2,7)
plt.scatter('specobjid','mjd',data=datanew_scaled)
plt.xlabel('specobjid')
plt.ylabel('mjd')
plt.ylabel('plate')
plt.subplot(7,2,8)
plt.scatter('g','i',data=datanew_scaled)
plt.xlabel('g')
plt.ylabel('i')

plt.show()


# In[ ]:


fig,ax = plt.subplots()
fig.set_size_inches(15,8)
sns.boxplot(data=X_scaled)


# In[ ]:


datanew_scaled.describe()


# In[ ]:


Q1 = X_scaled.quantile(0.25)
Q3 = X_scaled.quantile(0.75)
IQR = Q3 - Q1


# In[ ]:


((X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))).sum()


# In[ ]:


mask = (X_scaled < (Q1 - 1.5 * IQR)) | (X_scaled > (Q3 + 1.5 * IQR))
datanew_scaled[mask] = np.nan


# In[ ]:


datanew_scaled = datanew_scaled.dropna()


# In[ ]:


datanew_scaled.shape


# In[ ]:


datanew_scaled = datanew_scaled.reset_index(drop=True)


# In[ ]:


from collections import Counter

count = Counter(datanew_scaled['class'])
count


# ## Since most of the datapoints deleted are w.r.t. quasars we need not delete them since they might be useful for classification of quasars

# In[ ]:


fig,ax = plt.subplots()
fig.set_size_inches(15,8)
sns.boxplot(data=datanew_scaled)


# In[ ]:


finaldata.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
X = finaldata.iloc[:,0:14]
Y = finaldata.iloc[:,15]

Y = np.asarray(Y)
Y = Y.astype('int')

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

for k in range(1,21,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X_train,y_train,cv=5,scoring="accuracy")
    print(scores.mean())   


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=5)

training_start = time.perf_counter()
knnfit = knn.fit(X_train,y_train)
training_end = time.perf_counter()
total_time = training_end-training_start
print("Training Accuracy:       ",knnfit.score(X_train,y_train))
scores = cross_val_score(knn,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy ", scores.mean())
print("\nTime consumed for training %5.4f" % (total_time))
a=[None]*6
a[0]=scores.mean()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
t = DecisionTreeClassifier(max_depth=5)

training_start = time.perf_counter()
tfit = t.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start
print("Training Accuracy:        ",tfit.score(X_train,y_train))
scores = cross_val_score(t,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean())   
print("\nTime consumed for training %5.4f" % (total_time))
a[1]=scores.mean()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
Gnb = GaussianNB(priors=None)


training_start = time.perf_counter()
Gnbfit = Gnb.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",Gnbfit.score(X_train,y_train))
scores = cross_val_score(Gnb,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %5.4f" % (total_time))
a[2]=scores.mean()


# In[ ]:


from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(hidden_layer_sizes = (1000,1000),max_iter = 1000)
training_start = time.perf_counter()
MLPfit = MLP.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start
print("Training Accuracy:        ",MLPfit.score(X_train,y_train))
scores = cross_val_score(MLP,X_train,y_train,cv=5,scoring="accuracy")

print("Cross Validation Accuracy:", scores.mean())
print("\nTime consumed for training %5.4f" % (total_time))
a[3]=scores.mean()


# In[ ]:


from sklearn.svm import LinearSVC
SVC = LinearSVC(penalty='l2',C=10.0,max_iter = 100000)
training_start = time.perf_counter()
SVCfit = SVC.fit(X_train,y_train)
training_end = time.perf_counter()


total_time = training_end-training_start

print("Training Accuracy:        ",SVCfit.score(X_train,y_train))
scores = cross_val_score(SVC,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %5.4f" % (total_time))
a[4]=scores.mean()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10,max_depth=10)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start
print("Training Accuracy:        ",RFCfit.score(X_train,y_train))
scores = cross_val_score(RFC,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %5.4f" % (total_time))
a[5]=scores.mean()


# In[ ]:


d1 = pd.DataFrame(a)
x=['KNN','Decision Tree','Naive Bayes','Neural Network','SVM','Random Forest']

fig,ax = plt.subplots()
fig.set_size_inches(15,8)
bottom, top = ax.set_ylim(0.85, 1)
plt.bar(x,a)


# In[ ]:


prediction_start = time.perf_counter()
knnpred = knnfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",knnfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime ))
b = [None]*6
b[0]=knnfit.score(X_test,y_test)
knnpred


# In[ ]:



prediction_start = time.perf_counter()
tpred = tfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",tfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
b[1]=tfit.score(X_test,y_test)
tpred


# In[ ]:


prediction_start = time.perf_counter()
Gnbpred = Gnbfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",Gnbfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
b[2]=Gnbfit.score(X_test,y_test)
Gnbpred


# In[ ]:


prediction_start = time.perf_counter()
MLPpred = MLPfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",MLPfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[3]=MLPfit.score(X_test,y_test)
MLPpred


# In[ ]:


prediction_start = time.perf_counter()
SVCpred = SVCfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",SVCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[4]=SVCfit.score(X_test,y_test)
SVCpred


# In[ ]:


prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[5]=RFCfit.score(X_test,y_test)
RFCpred


# In[ ]:



x=['KNN','Decision Tree','Naive Bayes','Neural Network','SVM','Random Forest']

fig,ax = plt.subplots()
fig.set_size_inches(15,8)
bottom, top = ax.set_ylim(0.85, 1)
plt.bar(x,b)


# In[ ]:


fig,ax = plt.subplots()
fig.set_size_inches(20,8)
plt.subplot(1,2,1)
bottom, top = ax.set_ylim(0.8, 1)
plt.bar(x,a)
plt.title("Training set")
plt.ylabel("Accuracy")

plt.subplot(1,2,2)
bottom, top = ax.set_ylim(0.8, 1)
plt.bar(x,b)
plt.title("Test set")
plt.ylabel("Accuracy")


# In[ ]:


import itertools
from sklearn.metrics import confusion_matrix

class_names =['GALAXY','QUASAR','STAR']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, RFCpred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV

grid_param = {
    'n_estimators' : [50,100,300,500,800,1000],
    'criterion' : ['gini','entropy'],
    'bootstrap' : [True, False],
    'max_depth' : [10,20,50,100]
}


# In[ ]:


RFCbest = GridSearchCV(RFC,param_grid=grid_param,scoring = "accuracy",cv=5,n_jobs=-1)


# In[ ]:


RFCbest.fit(X_train,y_train)


# In[ ]:


print(RFCbest.best_estimator_)


# In[ ]:


best_params = RFCbest.best_params_
print(best_params)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFCa = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCbestfit = RFCa.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCbestfit.score(X_train,y_train))
scores = cross_val_score(RFCa,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))

a[5]=scores.mean()


# In[ ]:


prediction_start = time.perf_counter()
RFCpred = RFCbestfit.predict(X_test)
prediction_end = time.perf_counter()

total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCbestfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))
b[5]=RFCbestfit.score(X_test,y_test)
RFCpred


# ### SMOTE with the best parameters (Random forest)

# In[ ]:


un = np.unique(np.asarray(finaldata['class']).astype('int'))


# In[ ]:


from collections import Counter
from imblearn.over_sampling import SMOTE

for i,k in enumerate(un):
    print("Before Oversampling", k,"  ",list(Counter(y_train).values())[i]) # counts the elements' frequency

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train_res,y_train_res)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCfit.score(X_train_res,y_train_res))
scores = cross_val_score(RFC,X_train_res,y_train_res,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
a[5]=scores.mean()


# In[ ]:



prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

b[5]=RFCfit.score(X_test,y_test)
RFCpred


# # PCA

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca_d = PCA()
pca_d.fit(X)
cumsum = np.cumsum(pca_d.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d


# In[ ]:


fig,ax = plt.subplots()
plt.plot(cumsum)
plt.grid()
plt.axvline(d,c='r',linestyle='--')


# In[ ]:


pca = PCA(n_components = 7)
d_reduced = pca.fit_transform(X)
d_reducedt = pca.inverse_transform(d_reduced)

print(d_reduced.shape,d_reducedt.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train,X_test,y_train,y_test = train_test_split(d_reduced,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
RFCa = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCbestfit = RFCa.fit(X_train,y_train)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCbestfit.score(X_train,y_train))
scores = cross_val_score(RFCa,X_train,y_train,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))


# In[ ]:



prediction_start = time.perf_counter()
RFCpred = RFCbestfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCbestfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))


# In[ ]:


from collections import Counter
from imblearn.over_sampling import SMOTE
for i,k in enumerate(un):
    print("Before Oversampling", k,"  ",list(Counter(y_train).values())[i]) # counts the elements' frequency

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train_res,y_train_res)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCfit.score(X_train_res,y_train_res))
scores = cross_val_score(RFC,X_train_res,y_train_res,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))


# In[ ]:


prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))


# # Feature subset selection

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))


# In[ ]:


# Fitting the model to get feature importances after training
model = RandomForestClassifier()
model.fit(X_train , y_train)

# Draw feature importances
imp = model.feature_importances_
f = X.columns
# Sort by importance descending
f_sorted = f[np.argsort(imp)[::-1]]
fig,ax = plt.subplots(figsize=(15,8))
sns.barplot(x=f,y = imp, order = f_sorted)


plt.title("Features importances")
plt.ylabel("Importance")
plt.show()


# In[ ]:


finaldatanew = finaldata[['redshift','specobjid','mjd','z','plate','i','r','g','u']]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

X = finaldatanew
Y = finaldata.iloc[:,15]

Y = np.asarray(Y)
Y = Y.astype('int')

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=66)
kf = KFold(n_splits=5)

for train, valid in kf.split(X_train):
	print('train: %s, valid: %s' % (train, valid))


# In[ ]:


from collections import Counter
from imblearn.over_sampling import SMOTE
for i,k in enumerate(un):
    print("Before Oversampling", k,"  ",list(Counter(y_train).values())[i]) # counts the elements' frequency

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(bootstrap=False,criterion='gini',max_depth=50,n_estimators=1000)

training_start = time.perf_counter()
RFCfit = RFC.fit(X_train_res,y_train_res)
training_end = time.perf_counter()

total_time = training_end-training_start

print("Training Accuracy:        ",RFCfit.score(X_train_res,y_train_res))
scores = cross_val_score(RFC,X_train_res,y_train_res,cv=5,scoring="accuracy")
print("Cross Validation Accuracy:", scores.mean()) 
print("\nTime consumed for training %6.5f" % (total_time))


# In[ ]:


prediction_start = time.perf_counter()
RFCpred = RFCfit.predict(X_test)
prediction_end = time.perf_counter()


total_testtime = prediction_end-prediction_start
print("Testing accuracy        ",RFCfit.score(X_test,y_test))
print("\nTime consumed for testing %6.5f" % (total_testtime))

