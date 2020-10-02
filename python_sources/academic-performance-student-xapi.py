import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ds = pd.read_csv("../input/xAPI-Edu-Data.csv")
ds.head()

X = ds.iloc[:,:-1].values
y = ds.iloc[:,16].values

sns.pairplot(ds)
sns.countplot(ds["gender"])
sns.set_context(context = 'paper')
plt.figure(figsize=(14,4))
sns.countplot(x = "NationalITy",hue = 'gender',data = ds)
sns.countplot('StageID',data = ds,hue = 'gender')
sns.countplot('GradeID',data = ds,hue = 'gender')
plt.figure(figsize = (12,3))
sns.barplot(x = 'Topic',y = 'raisedhands',data = ds,hue = 'gender')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X_2 = ds.apply(le.fit_transform)
X_2.head()

enc = OneHotEncoder()
enc.fit(X_2)

onehotlabels = enc.transform(X_2).toarray()
onehotlabels.shape
onehotlabels
X_2.head()
X_2.shape
X = X_2[['gender','NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation','raisedhands','VisITedResources','AnnouncementsView','Discussion','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']]
X.head()
y = X_2['Class']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_2, y, test_size=0.33, random_state=42)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

testf = SelectKBest(score_func = chi2, k = 4)
fit = testf.fit(X_2,y)
np.set_printoptions(precision=3)
print(fit.scores_)
print(" ")

features = fit.transform(X_2)
print(features[0:5,:])

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1.0)
ridge.fit(X_2,y)

def print_feat(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,key = lambda x:-np.abs(x[0]))
    return"+".join("%s * %s"%(round(coef,3), name) for coef,name in lst)
    
print("Ridge Model: ", print_feat(ridge.coef_))
    
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred)
#cm

#fig,axis = plt.subplots(ncols = 2)
#plt.figure(num = 2,figsize=(15,8))
#sns.countplot(y_test,ax = axis[0])
#sns.countplot(y_pred,ax = axis[1])

#sns.countplot(y_test)

#c_score = np.trace(cm)/np.sum(cm)
#c_score

#from sklearn.neighbors import KNeighborsClassifier
#K_classifier = KNeighborsClassifier(n_neighbors = 3,metric = 'minkowski',p = 2)
#K_classifier.fit(X_train,y_train)

#knn_y_pred = K_classifier.predict(X_test)
#from sklearn.metrics import confusion_matrix
#K_cm = confusion_matrix(y_test,knn_y_pred)
#K_cm

#knn_score = np.trace(K_cm)/np.sum(K_cm)
#knn_score

from sklearn.svm import SVC
svc = SVC(kernel = 'poly' ,random_state=0)
svc.fit(X_train,y_train)

svc_y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
svc_cm = confusion_matrix(y_test,svc_y_pred)
svc_cm

svc_score = np.trace(svc_cm)/np.sum(svc_cm)
svc_score

from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,y_train)

nb_y_pred = nb_classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
nb_cm = confusion_matrix(y_test,nb_y_pred)
nb_cm
nb_score = np.trace(nb_cm)/np.sum(nb_cm)
nb_score

from sklearn.tree import DecisionTreeClassifier
tree_classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)

tree_classifier.fit(X_train,y_train)

tree_y_pred = tree_classifier.predict(X_test)
tree_y_pred

from sklearn.metrics import confusion_matrix
tree_cm = confusion_matrix(y_test,tree_y_pred)
tree_cm

tree_score = np.trace(tree_cm)/np.sum(tree_cm)
tree_score

from sklearn.ensemble import RandomForestClassifier
rand_classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rand_classifier.fit(X_train,y_train)

rand_y_pred = rand_classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
rand_cm = confusion_matrix(y_test,rand_y_pred)
rand_cm

