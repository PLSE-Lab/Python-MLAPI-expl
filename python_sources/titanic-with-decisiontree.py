import pandas as pd
from sklearn import tree


path = "../input/" #Change this to folder with your RAW-Data
data_train = path + "train.csv"
date_test = path + "test.csv"

matrix_train = pd.read_csv(data_train)


#unnötige Spalten entfernen
matrix_train = matrix_train.drop('Ticket', 1)
matrix_train = matrix_train.drop('Cabin', 1)
matrix_train = matrix_train.drop('Fare', 1)
matrix_train = matrix_train.drop('Name', 1)
matrix_train = matrix_train.drop('Embarked', 1)

#Zeilen mit fehlenden Daten löschen
matrix_train = matrix_train.dropna(axis=0, thresh = 7)

#Geschlecht als Integer und Alter als Altersgruppe ergänzen
matrix_train["sex_int"] = matrix_train["Sex"].apply(lambda x: 0 if x == "female" else 1)
matrix_train["age_goup"] = matrix_train["Age"].apply(lambda x: int(x%10))

#Geschlecht als Text und Alter als Zahl entfernen
matrix_train = matrix_train.drop('Sex', 1)
matrix_train = matrix_train.drop('Age', 1)

#DecisionTree bilden
features = list(matrix_train.columns[2:])
y = matrix_train["Survived"]
x = matrix_train[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

print(clf)

inputfile = path + "test.csv"
df = pd.read_csv(inputfile)

#unnötige Spalten entfernen
df = df.drop('Ticket', 1)
df = df.drop('Cabin', 1)
df = df.drop('Fare', 1)
df = df.drop('Name', 1)
df = df.drop('Embarked', 1)

#Geschlecht als Integer und Alter als Altersgruppe ergänzen
df["sex_int"] = df["Sex"].apply(lambda x: 0 if x == "female" else 1)
df["age_goup"] = df["Age"].apply(lambda x: int(x%10) if x>0 else 2)

#Geschlecht als Text und Alter als Zahl entfernen
df = df.drop('Sex', 1)
df = df.drop('Age', 1)

features = list(df.columns[1:])
z = df[features]

prediction = clf.predict(z)

df["Survived"] = prediction


df = df.drop('Pclass', 1)
df = df.drop('SibSp', 1)
df = df.drop('Parch', 1)
df = df.drop('sex_int', 1)
df = df.drop('age_goup', 1)


print(df)

#df.to_csv(path+"prediction.csv", index = False)