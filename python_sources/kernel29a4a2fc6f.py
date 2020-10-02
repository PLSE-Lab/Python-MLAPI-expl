import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


train_data=pd.read_csv("../input/titanic/train.csv")
val = {'male':'0', 'female':'1'}
train_data['Sex']=train_data['Sex'].map(val)
features=['Pclass','Sex','SibSp','Parch']

X=train_data[features]
y=train_data.Survived
titan_model= LinearDiscriminantAnalysis()
titan_model.fit(X,y)

test_data = pd.read_csv("../input/titanic/test.csv")
test_data['Sex']=test_data['Sex'].map(val)
test_X = test_data[features]
test_preds=titan_model.predict(test_X)

print(test_preds)


