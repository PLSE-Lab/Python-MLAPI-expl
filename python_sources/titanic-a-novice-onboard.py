import numpy as np
import pandas as pd

#Data Wrangling Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

#Model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Model Selection and Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


#Print you can execute arbitrary python code
train_dta = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_dta = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Identify features and label
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
label = ['Survived']

class NumImputer(TransformerMixin):
    def fit(self, x, y=None):
        self.fill = pd.Series([x[c].value_counts().index[0]
            if x[c].dtype == np.dtype('O') else x[c].median() for c in x],
            index=x.columns)
        return self
    def transform(self, x, y=None):
        return x.fillna(self.fill)

train_imputed = NumImputer().fit_transform(train_dta[features]).apply(LabelEncoder().fit_transform)
test_imputed = NumImputer().fit_transform(test_dta[features]).apply(LabelEncoder().fit_transform)

#Split Data to Train and Test
x_train, x_test, y_train, y_test = train_test_split(train_imputed.values, train_dta[label].values.ravel(), test_size = 0.4, random_state=0)

#--------- Predicting the Test Data ---------

X_test = test_imputed.values

# Random Forest
rf = RandomForestClassifier(n_estimators=200)
rf.fit(x_train, y_train)
rf_predicted = rf.predict(X_test)

X_predicted = pd.DataFrame({"PassengerID":test_dta["PassengerId"].values,"Survived":(pd.Series(rf_predicted))})
X_predicted.groupby("Survived").count()

#generate the output file
X_predicted.to_csv('titanic_predicted.csv', index=False)