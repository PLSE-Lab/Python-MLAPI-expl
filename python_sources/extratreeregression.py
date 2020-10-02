import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# - - - Tomamos los datos
# - - - - - - - - - - - - - - - - - - - -
train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# - - - Preparamos datos
# - - - - - - - - - - - - - - - - - - - -
# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# - Preparamos los datos de Test
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# - - - Entrenar y probar
# - - - - - - - - - - - - - - - - - - - -
# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

X_train = train_data[100::,1::]
y_train = train_data[100::,0]
X_test = train_data[0:100,1::]
y_test = train_data[0:100,0]

print ('Training...')
estimator = ExtraTreesRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=1, min_samples_leaf=2, min_weight_fraction_leaf=0.49, max_features='auto', max_leaf_nodes=2, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
#estimator = ExtraTreesRegressor(n_estimators=2000)
#estimator = RandomForestClassifier(n_estimators=1000)
estimator = DecisionTreeRegressor(criterion="mse", max_leaf_nodes=None)
estimator = estimator.fit( X_train , y_train )

print ('Predicting...')
y_pred = estimator.predict(X_test).astype(int)

print ('Resultados...')
score = mean_squared_error(y_test, y_pred)
print (1-score)

# - - - Evaluamos datos finales y Escribimos los resultados
# - - - - - - - - - - - - - - - - - - - -
test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
y_pred = estimator.predict(test_data).astype(int)
f = open('forest.csv', 'w')
f.write('PassengerId,Survived'+"\n")
for index, row in test_df.iterrows():
    f.write (str(row['PassengerId']) + "," + str(y_pred[index]) +"\n")






