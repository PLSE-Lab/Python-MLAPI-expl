import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

dataset=pd.read_csv('../input/data.csv')
#print(dataset)

#print(dataset.columns)

#Drop the id column
dataset=dataset.drop("id",axis=1)
dataset=dataset.drop("Unnamed: 32",axis=1)
#print(dataset.columns)
#print(dataset.head(2))

#Change values in diagnosis column to numeric
#M = 0 and B = 1 
data={'M':1,'B':0}
dataset.diagnosis=[data[i] for i in dataset.diagnosis.astype(str)]

#print(dataset.head(2))
#print(dataset)

train,test=train_test_split(dataset,test_size=0.33,random_state=0)

train_features=train.ix[:,0:31]
train_label=train.ix[:,0]

test_features=test.ix[:,0:31]
test_label=test.ix[:,0]

#print(train_label.head(3))

gnb = GaussianNB()

gnb.fit(train_features,train_label)

prediction=gnb.predict(test_features)
print(prediction)