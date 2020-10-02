import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

titanic_train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64} )
titanic_test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64} )

df_target = titanic_train.Survived
df_combo = pd.concat((titanic_train.drop("Survived", axis = 1), titanic_test), axis = 0, ignore_index = True)

# Surnames
df_combo['Surname'] = df_combo['Name'].str.extract('(.*?),.*', expand = False)

# Titles
df_combo['Title'] = df_combo['Name'].str.extract('.*, (.*?)\. .*', expand = False)

Title_Dictionary = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Royalty",
"Don": "Royalty",
"Sir" : "Royalty",
"Dr": "Officer",
"Rev": "Rev",
"the Countess": "Royalty",
"Dona": "Royalty",
"Mme": "Mrs",
"Mlle": "Miss",
"Ms": "Mrs",
"Mr" : "Mr",
"Mrs" : "Mrs",
"Miss" : "Miss",
"Master" : "Master",
"Lady" : "Royalty"
}    
    
df_combo["Title"] = df_combo["Title"].apply(Title_Dictionary.get)

# Filling missing Embarked data
T_EmbarkedModes = df_combo.pivot_table('Embarked', index=["Pclass"], aggfunc = lambda x : x.mode()[0] )
df_combo['Embarked'] = df_combo.apply( (lambda x: T_EmbarkedModes[x.Pclass] if pd.isnull(x.Embarked) else x.Embarked), axis=1 )

# Filling missing Age data
T_AgeMedians = df_combo.pivot_table('Age', index=["Title", "Sex", "Pclass"], aggfunc='median')
df_combo['Age'] = df_combo.apply( (lambda x: T_AgeMedians[x.Title, x.Sex, x.Pclass] if pd.isnull(x.Age) else x.Age), axis=1 )

# Filling missing Fare data
df_combo['Fare'] = df_combo['Fare'].map( lambda x: np.nan if x==0 else x )
T_FareMedians = df_combo.pivot_table('Fare', index=["Embarked", "Pclass"], aggfunc='median')
df_combo['Fare'] = df_combo.apply( (lambda x: T_FareMedians[x.Embarked, x.Pclass] if pd.isnull(x.Fare) else x.Fare), axis=1 )

# Deck
df_combo['Cabin'].fillna("UNK", inplace = True)
df_combo['Deck'] = df_combo['Cabin'].str[0]

# FamilySizeily size
df_combo["FamilySize"] = df_combo.Parch + df_combo.SibSp + 1

#T_MaxFamily = df_combo.pivot_table('FamilySize', index=["Surname", "Ticket"], aggfunc='max')
#df_combo['FamilySize'] = df_combo.apply( (lambda x: T_MaxFamily[x.Surname, x.Ticket]), axis=1 )

# Number od survived neigbours (sharing the same ticket): +1 for each survived, -1 for each dead
T_SumSurvByTicket = df_combo.join(df_target).pivot_table('Survived',['Ticket'], aggfunc=('sum', 'count'))
df_combo['SurvNeighbours'] = df_combo.join(df_target).apply( (lambda x: 2*( T_SumSurvByTicket.at[x.Ticket,'sum'] - (x.Survived == 1) ) - ( T_SumSurvByTicket.at[x.Ticket,'count'] - pd.notnull(x.Survived) ) ), axis=1 )
df_combo['SurvNeighbours'].fillna(0, inplace = True)

# Surname + FamiliSize (if FamilySize > 1)
df_combo['SurnameEx'] = df_combo.apply((lambda x: x['Surname'] + str(x['FamilySize']) if x['FamilySize']>1 else None), axis=1)

# Persons per Ticket
Ticket_count = dict(df_combo.Ticket.value_counts())
df_combo["TicketPersons"] = df_combo.Ticket.apply(Ticket_count.get)

# Fare per person
df_combo['FarePerson'] = df_combo['Fare']/df_combo["TicketPersons"]
#df_combo.drop(['Fare'], axis=1, inplace = True)

df_combo_full = df_combo.copy()

# Drop unused
df_combo.drop(["PassengerId", "Name", "Ticket", "Surname", "SurnameEx", "Cabin", "Parch", "SibSp"], axis=1, inplace = True)

#mine
#df_combo.drop(["FamilySize"], axis=1, inplace = True)
df_combo.drop(['Title'], axis=1, inplace = True)

df_combo['Sex'] = df_combo['Sex'].map( {'female': -1, 'male': 1} )

# Encoding nominal categorical features
df_combo = pd.get_dummies(df_combo)

# Prediction
df_train = df_combo.loc[:len(titanic_train["Survived"])-1]
df_test = df_combo.loc[len(titanic_train["Survived"]):]

from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation, metrics

clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=5, max_depth = 4)

clf.fit(df_train, df_target)
predictions = clf.predict(df_train)
predict_proba = clf.predict_proba(df_train)[:,1]

# Plot Feature Importance
features = df_train.axes[1].tolist()
x = plt.xticks(range(len(features)), features, rotation='vertical')
plt.plot(clf.feature_importances_)
plt.show()

# Plot Tree
#from sklearn.externals.six import StringIO
#from sklearn.tree import export_graphviz
#from IPython.display import Image
#import pydot_ng as pydot
#dot_data = StringIO() 
#export_graphviz(clf, out_file=dot_data, feature_names=features, filled=True) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("titanic.pdf") 
#Image(graph.create_png())

cv_score = cross_validation.cross_val_score(clf, df_train, df_target, cv = 10)
print("Accuracy : %.7g" % metrics.accuracy_score(df_target, predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(df_target, predict_proba))
print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))


final_pred = clf.predict(df_test)
submission = pd.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": final_pred })
submission.to_csv("Output.csv", index=False)