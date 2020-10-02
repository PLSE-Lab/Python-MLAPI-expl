import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_rows', None)

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train['Survived'].head(8))

#print("\n\nSummary statistics of training data")
#print(train.describe())

yTrain = train['Survived'].values
#print(y_train)
#ActiveColsId = np.array(range(len(train.columns)))!=1
ActiveColsId = np.array([2,6,7,9])
ActiveColsNames = train.columns.values[ActiveColsId]

XTrain = train[ActiveColsNames].values
XTest = test[ActiveColsNames]
XTest['Fare'][152] = 40

#print(XTrain)
#print(XTest)
#print(yTrain)

clf = KNeighborsClassifier(5)
clf.fit(XTrain,yTrain)        
Z=clf.predict(XTest)

knn_output = np.zeros((418,2))
knn_output[:,0] = test['PassengerId'].values
knn_output[:,1] = Z

knn_output= pd.DataFrame(knn_output.astype(int))
knn_output.columns = ['PassengerId','Survived']

#print(knn_output)

knn_output.to_csv('knn_output.csv', index=False)

#np.savetxt('knn_output.csv',knn_output.astype(int))



#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)