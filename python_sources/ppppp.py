from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

# create the training & test sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
Y = dataset[[0]].values.ravel()
X = dataset.iloc[:,25:].values
datatest = pd.read_csv("../input/test.csv")
test = datatest.iloc[:,24:].values

dev_cutoff = len(Y) * 0.9
X_train = X[:dev_cutoff]
Y_train = Y[:dev_cutoff]
X_test = X[dev_cutoff:]
Y_test = Y[dev_cutoff:]

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

np.savetxt('Predicciones.csv', np.c_[Y_test], delimiter=',', header = 'Label', comments = '', fmt='%d')  

for i in range (15,31):
    rf = RandomForestClassifier(n_jobs=-1,n_estimators=100 ,criterion='gini',max_leaf_nodes=10000,random_state=i,oob_score=True)
    rf.fit(X_train,Y_train)
    pred = rf.predict(X_test)
    print(rf.score(X_test,Y_test))
    labelPredicted = 'LabelPred_RandState_'+ str(i)
    name = 'prediccion_train_'+str(i)+'.csv'
    np.savetxt(name, np.c_[pred], delimiter=',', header = labelPredicted, comments = '', fmt='%d')  
    labelPredictedTest = 'Id,LabelPredictedTest'+ str(i)
    rf.fit(X,Y)
    predtest = rf.predict(test)
    testname = 'prediccion_test_random_state_'+str(i)+'.csv'
    np.savetxt(testname, np.c_[range(1,len(test)+1),predtest], delimiter=',', header = labelPredictedTest, comments = '', fmt='%d')  
