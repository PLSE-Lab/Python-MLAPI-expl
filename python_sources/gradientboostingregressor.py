import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


#load training data
train_data=pd.read_csv("../input/train.csv")

for y in train_data.columns:
    if( train_data[y].dtype == np.object ):
        train_data=train_data.drop([y],axis=1)

for y in train_data.columns:
    train_data[y] = np.where(train_data[y].isnull() | (train_data[y]== 0), train_data[y].mean() , train_data[y])
    train_data[y]=train_data[y].astype(dtype=np.int64)
        

 #Divide all features from class
array = train_data.values
X_train = array[:,0:37]
Y_train = array[:,37]
       

#load testing data
test_data=pd.read_csv("../input/test.csv")

for y in test_data.columns:
    if( test_data[y].dtype == np.object ):
        test_data=test_data.drop([y],axis=1)

for y in test_data.columns:
     test_data[y] = np.where(test_data[y].isnull() | (test_data[y]==0) , test_data[y].mean() , test_data[y])
     test_data[y]=test_data[y].astype(dtype=np.int64)
        

#Divide all feature from class
array = test_data.values
X_test = array[:,0:37]


# Create GradientBoostingRegressor object 
model = GradientBoostingRegressor ( loss='huber', n_estimators=150) 
#fit training data into model
model.fit(X_train, Y_train)

#Predict Output
predicted= model.predict(X_test)


#For creating submission file
test_data=pd.read_csv("../input/test.csv")

#create submission file
submission = np.empty((1459,2),dtype=int)
submission[:,0] = test_data["Id"]
submission[:,1] = predicted
submission = pd.DataFrame(data=submission,columns=["id","SalePrice"])
submission.to_csv("submission.csv",index = False)