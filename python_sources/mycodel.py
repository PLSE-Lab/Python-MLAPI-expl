import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score

def dfready(df):
	df = df.drop('Name', axis=1) \
	.drop('Cabin', axis=1) \
	.drop('Ticket', axis=1)\
	.replace('male',0)\
	.replace('female',1)

	aux = np.array(df.Embarked)
	aux = np.where(aux=='Q', 0, aux)
	aux = np.where(aux=='C', 1, aux)
	aux = np.where(aux=='S', 2, aux)
	df["Embarked"] = aux

	def unnanize(df, col, strategy):
		aux = np.array(df[col]).reshape(-1, 1)
		imputer = Imputer(missing_values='NaN', strategy=strategy, axis = 0)
		imputer = imputer.fit(aux)
		df[col] = imputer.transform(aux).astype(int)
		return df

	df = unnanize(df, 'Age','mean')
	df = unnanize(df, 'Embarked', 'most_frequent')
	df = unnanize(df, 'Fare', 'mean')

	df.loc[df['Fare'] <= 10, 'Fare'] = 0
	df.loc[(df['Fare'] > 10) & (df['Fare'] <= 20), 'Fare'] = 1
	df.loc[(df['Fare'] > 20) & (df['Fare'] <= 50), 'Fare'] = 2
	df.loc[(df['Fare'] > 50) & (df['Fare'] <= 100), 'Fare'] = 3
	df.loc[df['Fare'] > 100, 'Fare'] = 4
	df['Fare'] = df['Fare'].astype(int)

	return df

df_train = dfready(pd.read_csv("data/train.csv"))[:418]
df_test = dfready(pd.read_csv("data/test.csv"))

# g = sns.FacetGrid(df_train, col='Survived')
# g.map(plt.hist,'Sex')
# plt.show()

x_train = df_train.drop("Survived", axis=1)
y_train = df_train["Survived"]
x_test = df_train.drop("PassengerId", axis=1).copy()
# X_test  = df_test.drop("PassengerId", axis=1).copy()

rf = RandomForestClassifier(n_estimators=300, min_samples_split=3)
rf.fit(x_train, y_train)

# x_train, x_test, y_train, y_text = train_test_split(X, y, test_size=0.2)
# knn = KNeighborsClassifier(1)
# rf = RandomForestClassifier(n_estimators=100, min_samples_split=3)
# dt = DecisionTreeClassifier(max_depth=30)
# lr = LogisticRegression()
# for i in knn,rf,dt,lr:
#     i.fit(x_train, y_train)
# for i in knn,rf,dt,lr:
#     print(str(type(i)).split('.')[-1][:-2])
#     print(cross_val_score(i, x_train, y_train, cv=20, scoring='f1'))
#     print()


Y_pred = rf.predict(x_test)

submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })

# g = sns.FacetGrid(submission, col='Survived')
# g.map(plt.hist,'Sex')
# plt.show()

submission.to_csv('submission.csv', index=False)