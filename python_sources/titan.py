import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

train.at[train['Sex']=='male','Sex']=1
train.at[train['Sex']=='female','Sex']=0

test.at[test['Sex']=='male','Sex']=1
test.at[test['Sex']=='female','Sex']=0

X=train[['Pclass', 'Fare','Age','Sex','Survived']]
X=X.dropna(axis=0)

XX=X[['Pclass', 'Fare','Age','Sex']]
Y=X['Survived'].as_matrix()
Z=XX.as_matrix()
clf=DecisionTreeClassifier(random_state=241)
clf.fit(Z,Y)

#X_predic=test[['Pclass', 'Fare','Age','Sex']].as_matrix()
#Y_predic=clf.predict(X_predic)
print(test.describe())


#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)