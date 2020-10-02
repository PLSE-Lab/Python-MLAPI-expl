#importing files
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Reading preprocessed data CSV file
mas = pd.read_csv("../input/master.csv")
names = mas["Name"]
mas = mas.drop(columns= ['Name'])
mas.index = names

y = mas['CurSchol']
X = mas.drop(columns = ['CurSchol'])

#Splitting data randomly into training and test sets
InTr, InTs, LabTr, LabTs = train_test_split(X,y, test_size= 0.25)

#Network 1
InTrlvl1 = InTr.drop(columns= ['CurEnrol','CurGradYear','CurAssignScore'])
clf1 = MLPClassifier((100,100), max_iter= 500).fit(InTrlvl1,LabTr)

#Network 2
InTrlvl2 = InTr.iloc[:,9:12]
print(np.shape(InTrlvl2))
temp = pd.DataFrame(clf1.predict(InTrlvl1), columns= ['CurProb'])
temp.index = InTrlvl2.index
InTrlvl2 = pd.concat([InTrlvl2,temp], axis= 1)
LabTrlvl2 = LabTr
clf2 = MLPClassifier((5), max_iter= 500).fit(InTrlvl2,LabTr)

#Testing
InTslvl2 = InTs.iloc[:,9:12]
InTslvl1 = InTs.drop(columns= ['CurEnrol','CurGradYear','CurAssignScore'])
temp = pd.DataFrame(clf1.predict(InTslvl1), columns= ['CurProb'])
temp.index= InTslvl2.index
InTslvl2 = pd.concat([InTslvl2, temp], axis= 1)

#Printing Scores
score = clf2.score(InTslvl2,LabTs)
print("Test set accuracy : %f" %(score*100))
