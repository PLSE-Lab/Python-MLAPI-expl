import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Emap = {'S':1,'C':2,'Q':3}
train.Embarked.loc[pd.isnull(train.Embarked)] = 'S'

datasets = [train,test]
for dataset in datasets:
    dataset["EmbarkedMap"] = dataset.Embarked.map(Emap)

CabinMap = {'NaN':0,'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}
for dataset in datasets:
    dataset['HasCabin'] = 0
    dataset.HasCabin.loc[pd.notnull(dataset.Cabin)] = 1
    dataset["CMap"] = dataset.Cabin.str[0].map(CabinMap)
    dataset.CMap.fillna(0,inplace = True)


trainages = np.random.randint(train.Age.mean() - train.Age.std(),train.Age.mean() + train.Age.std(),size = sum(pd.isnull(train.Age)))
testages = np.random.randint(test.Age.mean() - test.Age.std(),test.Age.mean() + test.Age.std(),size = sum(pd.isnull(test.Age)))
train.Age.loc[pd.isnull(train.Age)] = trainages
test.Age.loc[pd.isnull(test.Age)] = testages

for dataset in datasets:
    dataset["Male"] = 0
    dataset.Male.loc[dataset.Sex == 'male'] = 1
    dataset["Title"] = dataset.Name.str.extract('( [A-Za-z]+)\.',expand = False)
    dataset["Agebins"] = pd.qcut(dataset.Age,4,labels = False)
    dataset.Title.loc[dataset.Title == ' Mlle'] = ' Miss'
    dataset.Title.loc[dataset.Title == ' Mme'] = ' Mrs'
    dataset.Title.loc[dataset.Title == ' Dona'] = ' Mrs'



d = dict(zip(np.unique(train.Title),np.arange(1,18)))

for dataset in datasets:
    dataset['TMap'] = dataset.Title.map(d)
    dataset.drop(['Name','Sex','Ticket','Cabin','Embarked'],1,inplace = True)
    dataset.TMap.loc[dataset.TMap == 1] = 0
    dataset.TMap.loc[dataset.TMap == 2] = 0
    dataset.TMap.loc[dataset.TMap == 3] = 0 
    dataset.TMap.loc[dataset.TMap == 4] = 0
    dataset.TMap.loc[dataset.TMap == 5] = 0
    dataset.TMap.loc[dataset.TMap == 6] = 0
    dataset.TMap.loc[dataset.TMap == 7] = 0
    dataset.TMap.loc[dataset.TMap == 8] = 0
    dataset.TMap.loc[dataset.TMap == 13] = 0
    dataset.TMap.loc[dataset.TMap == 14] = 0
    dataset.TMap.loc[dataset.TMap == 15] = 0
    
    
test.Fare.loc[pd.isnull(test.Fare)] = test.Fare.mean()


for dataset in datasets:
    dataset['Pclass1'] = 0
    dataset['Pclass2'] = 0
    dataset['Pclass3'] = 0
    dataset.Pclass1.loc[dataset.Pclass == 1] = 1
    dataset.Pclass2.loc[dataset.Pclass == 2] = 1
    dataset.Pclass3.loc[dataset.Pclass == 3] = 1
    dataset['EMap1'] = 0
    dataset['EMap2'] = 0
    dataset['EMap3'] = 0
    dataset.EMap1.loc[dataset.EmbarkedMap == 1] = 1
    dataset.EMap2.loc[dataset.EmbarkedMap == 2] = 1
    dataset.EMap3.loc[dataset.EmbarkedMap == 3] = 1
    dataset['TMap1'] = 0
    dataset['TMap2'] = 0
    dataset['TMap3'] = 0
    dataset['TMap4'] = 0
    dataset['TMap5'] = 0
    dataset.TMap1.loc[dataset.TMap == 0] = 1
    dataset.TMap2.loc[dataset.TMap == 9] = 1
    dataset.TMap3.loc[dataset.TMap == 10] = 1
    dataset.TMap4.loc[dataset.TMap == 11] = 1
    dataset.TMap5.loc[dataset.TMap == 12] = 1
    
#--------Feature Standardisation----------#

for dataset in datasets:
    dataset.Fare = (dataset.Fare - dataset.Fare.mean())/dataset.Fare.std()
    dataset.SibSp = (dataset.SibSp - dataset.SibSp.mean())/dataset.SibSp.std()
    dataset.Parch = (dataset.Parch - dataset.Parch.mean())/dataset.Parch.std()
    #dataset.EmbarkedMap=(dataset.EmbarkedMap - dataset.EmbarkedMap.mean())/dataset.EmbarkedMap.std()
    dataset.EMap1 = (dataset.EMap1 - dataset.EMap1.mean())/dataset.EMap1.std()
    dataset.EMap2 = (dataset.EMap2 - dataset.EMap2.mean())/dataset.EMap2.std()
    dataset.EMap3 = (dataset.EMap3 - dataset.EMap3.mean())/dataset.EMap3.std()
    #dataset.Pclass=(dataset.Pclass - dataset.Pclass.mean())/dataset.Pclass.std()
    dataset.Pclass1 = (dataset.Pclass1 - dataset.Pclass1.mean())/dataset.Pclass1.std()
    dataset.Pclass2 = (dataset.Pclass2 - dataset.Pclass2.mean())/dataset.Pclass2.std()
    dataset.Pclass3 = (dataset.Pclass3 - dataset.Pclass3.mean())/dataset.Pclass3.std()
    dataset.Male = (dataset.Male - dataset.Male.mean())/dataset.Male.std()
    dataset.Agebins = (dataset.Agebins - dataset.Agebins.mean())/dataset.Agebins.std()
    dataset.CMap = (dataset.CMap - dataset.CMap.mean())/dataset.CMap.std()
    dataset.Age = (dataset.Age - dataset.Age.mean())/dataset.Age.std()
    dataset.HasCabin = (dataset.HasCabin - dataset.HasCabin.mean())/dataset.HasCabin.std()
    #dataset.TMap=(dataset.TMap - dataset.TMap.mean())/dataset.TMap.std()
    dataset.TMap1 = (dataset.TMap1 - dataset.TMap1.mean())/dataset.TMap1.std()
    dataset.TMap2 = (dataset.TMap2 - dataset.TMap2.mean())/dataset.TMap2.std()
    dataset.TMap3 = (dataset.TMap3 - dataset.TMap3.mean())/dataset.TMap3.std()
    dataset.TMap4 = (dataset.TMap4 - dataset.TMap4.mean())/dataset.TMap4.std()
    dataset.TMap5 = (dataset.TMap5 - dataset.TMap5.mean())/dataset.TMap5.std()

#---------Training set splitting---------#
from sklearn.model_selection import train_test_split

train1,train2 = train_test_split(train,test_size = 0.20,random_state = 40)


#--------Applying Backward Elimination--------#
'''
import statsmodels.formula.api as sm

X = np.append(arr = np.ones((712,1)).astype(int), values = train1.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False), axis = 1)
X_opt = X[:, [0, 1, 2, 3, 6, 7, 9, 11, 12, 16]]
clf_new = sm.OLS(endog = train1.Survived, exog = X_opt).fit()
clf_new.summary()

X1_opt = train2[:,[]]
'''
#----------Applying kernel PCA---------------#
'''
from sklearn.decomposition import KernelPCA

pca = KernelPCA(n_components = 2, kernel = 'rbf')
train1_pca = pca.fit_transform(train1.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False))
train2_pca = pca.transform(train2.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False))
test_pca = pca.transform(test.drop(['PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace = False))
#explained_variance = pca.explained_variance_ratio_
'''
#----------Apply ML Algo now----------#

'''
from sklearn import svm
#clf = svm.SVC(gamma = 1e-8, C = 1e6)
clf = svm.SVC(C = 1000, kernel = 'rbf', gamma = 0.001)

#clf.fit(train1.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False), train1.Survived)

#clf.fit(train1_pca, train1.Survived)
#clf.score(train2_pca, train2.Survived)
clf.fit(train.drop(['Survived','PassengerId','Pclass','EmbarkedMap','TMap','Title','Pclass1','EMap1','TMap1'],1,inplace=False), train.Survived)
#clf.score(train2.drop(['Survived','PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace=False), train2.Survived)


#train2_pred = clf.predict(train2.drop(['Survived','PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace = False))


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(train2.Survived,train2_pred)
'''
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU

clf = Sequential()
#clf.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
clf.add(Dense(units = 20, kernel_initializer = 'uniform', input_dim = 19))
clf.add(LeakyReLU(alpha=0.05))
clf.add(Dense(units = 20, kernel_initializer = 'uniform', input_dim = 19))
clf.add(LeakyReLU(alpha=0.05))
clf.add(Dense(units = 20, kernel_initializer = 'uniform', input_dim = 19))
clf.add(LeakyReLU(alpha=0.05))
clf.add(Dense(units = 20, kernel_initializer = 'uniform', input_dim = 19))
clf.add(LeakyReLU(alpha=0.05))
#clf.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#clf.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
#clf.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

clf.fit(train.drop(['Survived','PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace=False), train.Survived, batch_size = 10, epochs = 100)

'''
#-------K-fold cross validation-----#
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = clf, X = train.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title','Pclass1','TMap1','EMap1'],1,inplace = False), y = train.Survived, cv = 10)
accuracies.mean()
'''
'''
#-----Applying grid search------#
from sklearn.model_selection import GridSearchCV
#parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']}, {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.5, 0.1, 0.01, 0.001, 0.0001]}]
parameters = [{'C' : [1000, 10000, 100000], 'kernel' : ['rbf'], 'gamma' : [0.001, 0.003, 0.005, 0.007, 0.009]}]
gs = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'accuracy', cv = 10)
gs = gs.fit(train.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False),train.Survived)

best_acc = gs.best_score_
best_params = gs.best_params_
'''

y_predd = (clf.predict(test.drop(['PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace = False)))
#y_predd = (y_predd > 0.5)
y_pred = np.where(y_predd[:,0] >0.5, 1, 0)

arr = pd.Series(y_pred, name = 'Survived')
df = pd.concat([test.PassengerId,arr],axis = 1)



'''

#arr = pd.Series(clf.predict(test_pca),name = 'Survived')
arr = pd.Series(clf.predict(test.drop(['PassengerId','Pclass','EmbarkedMap','TMap','Title','Pclass1','TMap1','EMap1'],1,inplace = False)),name = 'Survived')
df = pd.concat([test.PassengerId,arr],axis = 1)
'''
#np.savetxt("pred1.csv",df,delimiter=',',fmt='%d')

df.to_csv('pred1.csv',sep = ',',index = False)





















