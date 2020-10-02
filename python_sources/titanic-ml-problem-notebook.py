# %% [code]

#titanic

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset_y = pd.read_csv('test.csv')
y = dataset.iloc[:, 1].values
#filling missing values
missing_value = dataset_y[(dataset_y.Pclass == 3) & 
                     (dataset_y.Embarked == "S") & 
                     (dataset_y.Sex == "male")].Fare.mean()
## replace the test.fare null values with test.fare mean
dataset_y.Fare.fillna(missing_value, inplace=True)
#feature engineering
dataset.loc[:,'Menbers_of_family'] = dataset.loc[:,'SibSp'].add(dataset.loc[:,'Parch'])
dataset_y.loc[:,'Menbers_of_family'] = dataset.loc[:,'SibSp'].add(dataset.loc[:,'Parch'])

#predicting missing ages
from sklearn.ensemble import RandomForestRegressor
age_df = dataset.loc[:,"Pclass":] 
age_df_y = dataset_y.loc[:,"Pclass":]   

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x1 = LabelEncoder()
age_df.iloc[:, 2] = labelencoder_x1.fit_transform(age_df.iloc[:, 2])

labelencoder_x2 = LabelEncoder()
age_df_y.iloc[:, 2] = labelencoder_x2.fit_transform(age_df_y.iloc[:, 2])
 
temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values
temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values
temp_train_y = age_df_y.loc[age_df_y.Age.notnull()] ## df with age values
temp_test_y = age_df_y.loc[age_df_y.Age.isnull()] ## df without age values


z = temp_train.Age.values ## setting target variables(age) in y 
x = temp_train.iloc[:, [0, 2, 7, 10]].values
z_y = temp_train_y.Age.values
x_y = temp_train_y.iloc[:, [0, 2, 7, 10]].values

rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
rfr.fit(x, z)
rfr.fit(x_y, z_y)

predicted_age = rfr.predict(temp_test.iloc[:, [0, 2, 7, 10]])   
predicted_age_y = rfr.predict(temp_test_y.iloc[:, [0, 2, 7, 10]])

dataset.loc[dataset.Age.isnull(), "Age"] = predicted_age
dataset_y.loc[dataset_y.Age.isnull(), "Age"] = predicted_age_y


X = dataset.iloc[:, [2, 4, 5, 12, 9]].values   
y_t = dataset_y.iloc[:, [1, 3, 4, 11, 8]].values


# Encoding categorical data

# Male/Female
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
labelencoder_y = LabelEncoder()
y_t[:, 1] = labelencoder_y.fit_transform(y_t[:, 1])

combine = [X, y_t]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)
y_t =  sc.transform(y_t)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(y_t)

#submission
submission = pd.DataFrame({
        "PassengerId": dataset_y["PassengerId"],
        "Survived": y_pred
    })

submission.to_csv('submission.csv', index=False)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
accuracies.mean()






