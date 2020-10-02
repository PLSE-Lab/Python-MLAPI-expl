import numpy as np 
import pandas as pd

def adjust(df,drop=True): 
    Df = df.copy() 
    Df = Df.drop(['Name', 'Ticket', 'Cabin'], axis=1) 

    age_median = Df.Age.median()    
    Df.Age = Df.Age.fillna(age_median) 

    fare_median = Df.Fare.median() 
    Df.Fare = Df.Fare.fillna(fare_median) 

    if drop==True:
      Df = Df.dropna()
      
    Df['Sex'] = Df['Sex'].map({'female': 0, 'male':1}).astype(int) 
    Df['Embarked'] = Df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int) 
    return Df

train_df = pd.read_csv("../input/train.csv") 
train_df = adjust(train_df)

train_data = train_df.values

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB

X = train_data[:,2:] 
y = train_data[:,1] 

n = 3
models = [0]*n
yp = np.zeros(len(y),dtype=int)

models[0] = LogisticRegression()
models[1] = RandomForestClassifier(n_estimators = 100)
models[2] = GaussianNB()

for i in range(n):
    model = models[i].fit(X,y)
    yp   += model.predict(X)
    
yp = np.round(yp/float(n))

test_df = pd.read_csv("../input/test.csv") 
test_df = adjust(test_df,drop=False)

X0 = test_df.ix[0:,1:] 
y0 = model.predict(X0)

passengers = test_df.ix[:,0] 
result = np.c_[passengers, y0.astype(int)] 
result_df = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived']) 
result_df.to_csv('titanic.csv', index=False)