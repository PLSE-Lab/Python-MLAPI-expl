import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
#print(train.head())

print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
# # Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
# PassengerId =np.array(test["PassengerId"]).astype(int)
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# print(my_solution)
# 
# # Check that your data frame has 418 entries
# print(my_solution.shape)
# 
# # Write your solution to a csv file with the name my_solution.csv
# my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])




train.head(5)
train.Survived.value_counts()
train.Pclass.value_counts()
train.Sex.value_counts()
train['Sex'].loc[train['Sex'] == 'male'] = 0
train['Sex'].loc[train['Sex'] == 'female'] = 1
train.columns
import numpy as np
age_for_na_values = int(np.mean(train.Age))
train['Age'] = train['Age'].fillna(age_for_na_values)
train.SibSp.value_counts()
train.columns
train.Parch.value_counts()
train.Embarked.value_counts()

train["Embarked"].loc[train["Embarked"] == "S"] = 0
train["Embarked"].loc[train["Embarked"] == "C"] = 1
train["Embarked"].loc[train["Embarked"] == "Q"] = 2
train.Embarked.value_counts()
train.columns
train.drop('Ticket', axis=1, inplace = True)
train.drop('Fare', axis=1, inplace = True)
train.drop('Cabin', axis=1, inplace = True)

train.drop('Name', axis=1, inplace = True)
train.columns
train.Embarked = train["Embarked"].fillna(0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
y_train = train['Survived'].values
lr.fit(X_train, y_train)
test.columns
test.Age = test.Age.fillna(np.mean(train.Age))
test["Embarked"].loc[test["Embarked"] == "S"] = 0
test["Embarked"].loc[test["Embarked"] == "C"] = 1
test["Embarked"].loc[test["Embarked"] == "Q"] = 2
test['Sex'].loc[test['Sex'] == 'male'] = 0
test['Sex'].loc[test['Sex'] == 'female'] = 1
X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
y_pred = lr.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

# Check that your data frame has 418 entries
#print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_svm.csv", index_label = ["PassengerId"])



from sklearn.cross_validation import train_test_split
X_train_train, X_train_valid, y_train_train, y_train_valid = train_test_split(X_train, y_train,
                                                                              test_size= 0.2,
                                                                              random_state =123)
#svm = SVC(kernel = 'rbf', gamma = 0.05, C= 5.0)
#svm.fit(X_train_train, y_train_train)
#accuracy_score(svm.predict(X_train_train), y_train_train)
#accuracy_score(svm.predict(X_train_valid), y_train_valid)
#svm.fit(X_train, y_train)
#y_pred = svm.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_svmC5g05.csv", index_label = ["PassengerId"])


from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier(n_estimators = 2000, criterion = 'entropy', random_state = 123)
#rfc.fit(X_train_train, y_train_train)
#rfc.feature_importances_
#accuracy_score(rfc.predict(X_train_train), y_train_train)
#accuracy_score(rfc.predict(X_train_valid), y_train_valid)
#rfc.fit(X_train, y_train)
#y_pred = rfc.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_rfc2000ent.csv", index_label = ["PassengerId"])


from sklearn.ensemble import AdaBoostClassifier
#abc = AdaBoostClassifier(n_estimators = 1500, learning_rate = 1.1)
#abc.fit(X_train_train, y_train_train)
#accuracy_score(abc.predict(X_train_train), y_train_train)
#accuracy_score(abc.predict(X_train_valid), y_train_valid)
#n_estimators = [100, 200, 500, 1000, 1500, 2000, 5000] #100
# for n in range(7, 20):
#     abc = AdaBoostClassifier(n_estimators = n)
#     abc.fit(X_train_train, y_train_train)
#     print(n)
#     print(accuracy_score(abc.predict(X_train_train), y_train_train))
#     print(accuracy_score(abc.predict(X_train_valid), y_train_valid))
#abc = AdaBoostClassifier(n_estimators = 7)
#abc.fit(X_train, y_train)
#y_pred = abc.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_adaboost7est.csv", index_label = ["PassengerId"])


rfc = RandomForestClassifier(max_depth = 12, min_samples_split = 3,
                             n_estimators = 250, random_state =1)
rfc.fit(X_train_train, y_train_train)
rfc.score(X_train_train, y_train_train)
rfc.score(X_train_valid, y_train_valid)
n_estimators = [240, 250, 251, 252, 253, 254]
# for n in n_estimators:
#     rfc = RandomForestClassifier(max_depth = 8, min_samples_split = 3,
#                              n_estimators = n, random_state =1)
#     rfc.fit(X_train_train, y_train_train)
#     print(n)
#     print(rfc.score(X_train_train, y_train_train))
#     print(rfc.score(X_train_valid, y_train_valid))

#rfc = RandomForestClassifier(max_depth = 8, min_samples_split = 3,
                             #n_estimators = 250, random_state =1)
#rfc.fit(X_train, y_train)
#y_pred = rfc.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_rfc250es8maxdepth.csv", index_label = ["PassengerId"])



#y_pred_reference = rfc.predict(X_test)
#from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
#etc = ExtraTreesClassifier(max_depth = 8, min_samples_split = 3,
#                            n_estimators = 260, random_state =1)
#etc.fit(X_train_train, y_train_train)
#etc.score(X_train_train, y_train_train)
#etc.score(X_train_valid, y_train_valid)
#n_estimators = [230, 240, 250, 260, 270, 280]
# for n in n_estimators:
#     etc = ExtraTreesClassifier(max_depth = 8, min_samples_split = 3,
#                              n_estimators = n, random_state =1)
#     etc.fit(X_train_train, y_train_train)
#     print(n)
#     print(etc.score(X_train_train, y_train_train))
#     print(etc.score(X_train_valid, y_train_valid))
#etc.fit(X_train, y_train)
#y_pred = etc.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_etc260est8depth.csv", index_label = ["PassengerId"])


#y_pred_reference

#gbc = GradientBoostingClassifier(max_depth = 12, min_samples_split = 3,
                                 #n_estimators = 200)
#gbc.fit(X_train_train, y_train_train)
#y_pred = gbc.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_gbc12md200est.csv", index_label = ["PassengerId"])

# rfc = RandomForestClassifier(max_depth = 2, n_estimators = 200,
#                              min_samples_split = 2, criterion = 'entropy')
# rfc.fit(X_train_train, y_train_train)
# rfc.feature_importances_
# features_subset = ['Pclass','Sex']
# X_train_subsetted = train[features_subset].values
# rfc.fit(X_train_subsetted, y_train)
# rfc.score(X_train_subsetted, y_train)
from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors = 5, algorithm = 'auto', p=1.2)
#knn.fit(X_train_train, y_train_train)
#print(knn.score(X_train_train, y_train_train))
#print(knn.score(X_train_valid, y_train_valid))
    
#knn.fit(X_train, y_train)
#y_pred = knn.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_knn5p12.csv", index_label = ["PassengerId"])







# rfc = RandomForestClassifier(max_depth = 8, min_samples_split = 3,
#                              n_estimators = 250, random_state =1)
# abc = AdaBoostClassifier(n_estimators = 1500, learning_rate = 1.1)
# svm = SVC(kernel = 'rbf', gamma = 0.05, C= 5.0)
# etc = ExtraTreesClassifier(max_depth = 8, min_samples_split = 3,
#                              n_estimators = 260, random_state =1)
# gbc = GradientBoostingClassifier(max_depth = 12, min_samples_split = 3,
#                                  n_estimators = 200)
# from scipy.stats import mode
# rfc.fit(X_train_train, y_train_train)
# abc.fit(X_train_train, y_train_train)
# svm.fit(X_train_train, y_train_train)
# etc.fit(X_train_train, y_train_train)
# gbc.fit(X_train_train, y_train_train)
# 
# y_valid_predict_rfc = rfc.predict(X_train_valid)
# y_valid_predict_abc = abc.predict(X_train_valid)
# y_valid_predict_svm = svm.predict(X_train_valid)
# y_valid_predict_etc = etc.predict(X_train_valid)
# y_valid_predict_gbc = gbc.predict(X_train_valid)
# y_valid_predict_majority = []
# a = []
# b = []
# for i in range(len(X_train_valid)):
#     a.append(mode([y_valid_predict_rfc[i], y_valid_predict_abc[i], y_valid_predict_svm[i],
#              y_valid_predict_etc[i]]))
#     b+= a[i][0].tolist()
# 
# b_array = np.asarray(b)
# b_int_array = b_array.astype(int)
# accuracy_score(b_int_array, y_train_valid)
# rfc.fit(X_train, y_train)
# abc.fit(X_train, y_train)
# svm.fit(X_train, y_train)
# etc.fit(X_train, y_train)
# 
# y_train_predict_rfc = rfc.predict(X_test)
# y_train_predict_abc = abc.predict(X_test)
# y_train_predict_svm = svm.predict(X_test)
# y_train_predict_etc = etc.predict(X_test)
# 
# a,b =[], []
# for i in range(0, len(X_test)):
#     a.append(mode([y_train_predict_rfc[i], y_train_predict_abc[i], y_train_predict_svm[i],
#              y_train_predict_etc[i]]))
#     b+= a[i][0].tolist()
# 
# b = np.asarray(b)
# y_valid_predict_majority = b.astype(int)
# y_pred = y_valid_predict_majority
# len(y_pred)
#rfc = RandomForestClassifier(max_depth = 8, min_samples_split = 3,
                             #n_estimators = 250, random_state =1)
# for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     rfc = RandomForestClassifier(max_depth = 5, min_samples_split = 4,
#                              n_estimators = 650, random_state =1)
#     rfc.fit(X_train_train, y_train_train)
#     print(n)
#     print(rfc.score(X_train_train, y_train_train))
#     print(rfc.score(X_train_valid, y_train_valid))
#rfc = RandomForestClassifier(max_depth = 5, min_samples_split = 4,
                             #n_estimators = 650, random_state =1)
#rfc.fit(X_train_train, y_train_train)
#rfc.fit(X_train, y_train)
#y_pred = rfc.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)

#Check that your data frame has 418 entries
#print(my_solution.shape)

#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_rfc620est5dep4split.csv", index_label = ["PassengerId"])


# y_pred_reference = y_pred
# rfc.feature_importances_
# X_train_subsetted = train[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Embarked']].values
# X_train_train_subsetted, X_train_valid_subsetted, y_train_train, y_train_valid = train_test_split(
#     X_train_subsetted, y_train, test_size = 0.2, random_state = 1)
# rfc.fit(X_train_train_subsetted, y_train_train)
# rfc.feature_importances_
# for n in [640, 650]:
#     rfc = RandomForestClassifier(max_depth = 5, min_samples_split = 4,
#                              n_estimators = n, random_state =1)
#     rfc.fit(X_train_train, y_train_train)
#     print(n)
#     print(rfc.score(X_train_train, y_train_train))
#     print(rfc.score(X_train_valid, y_train_valid))
# 
# 
# for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
#     rfc = RandomForestClassifier(max_depth = 6, min_samples_split = 18,
#                              n_estimators = 110, min_samples_leaf = n, random_state =1)
#     rfc.fit(X_train_train, y_train_train)
#     print(n)
#     print(rfc.score(X_train_train, y_train_train))
#     print(rfc.score(X_train_valid, y_train_valid))
# rfc.fit(X_train_train, y_train_train)
# rfc = RandomForestClassifier(max_depth = 5, min_samples_split = 4,
#                              n_estimators = 650, random_state =1, max_features = 2)
# rfc.fit(X_train_train, y_train_train)
# rfc.score(X_train_valid, y_train_valid)
#rfc2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2,
                             #n_estimators = 110, random_state =1)
#rfc2.fit(X_train_train, y_train_train)
# rfc2.score(X_train_valid, y_train_valid)
# rfc = RandomForestClassifier(max_depth = 5, min_samples_split = 4,
#                              n_estimators = 650, random_state =1, max_features = 2)
# rfc.fit(X_train_train, y_train_train)
# rfc.score(X_train_valid, y_train_valid)
#rfc2.fit(X_train, y_train)
#y_pred = rfc2.predict(X_test)




# for n in [110, 109, 107]:
#     rfc2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2,
#                              n_estimators = n, random_state =1)
#     rfc2.fit(X_train_train, y_train_train)
#     print(n)
#     print(rfc2.score(X_train_train, y_train_train))
#     print(rfc2.score(X_train_valid, y_train_valid))



#list_space = (np.linspace(10, 1000, 100).tolist())
#list_one = np.linspace(1, 100, 50)
#list_space
#train_error = []
#valid_error = []
#for n in list_space:
#    rfc2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2,
#                             n_estimators = int(n), random_state =1)
#    rfc2.fit(X_train_train, y_train_train)
#    train_error.append(rfc2.score(X_train_train, y_train_train))
#    valid_error.append(rfc2.score(X_train_valid, y_train_valid))
#valid_error


#rfc2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2, n_estimators = 6000, random_state =1)
#rfc2.fit(X_train, y_train)
#y_pred = rfc2.predict(X_test)


from sklearn.tree import DecisionTreeClassifier
list_one = np.linspace(1, 40, 40)
# train_error = []
# valid_error = []
# list_one
# for n in list_one: 
#     dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9,
#                              min_samples_split = 1, random_state = 11)
#     dtc.fit(X_train_train, y_train_train)
#     train_error.append(dtc.score(X_train_train, y_train_train))
#     valid_error.append(dtc.score(X_train_valid, y_train_valid))
#dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 9,
#                             min_samples_split = 1, random_state = 11)
#rfc2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2,
#                              n_estimators = 110, random_state =11)
#dtc.fit(X_train_train, y_train_train)
#rfc2.fit(X_train_train, y_train_train)
#dtc.fit(X_train, y_train)
#y_pred = dtc.predict(X_test)




from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
#clf1 = LogisticRegression(penalty = 'l2', C = 0.001, random_state = 0)
#clf2 = DecisionTreeClassifier(max_depth = 1, criterion = 'entropy', random_state =0)
#clf3 = KNeighborsClassifier(n_neighbors = 1, p=2, metric = 'minkowski')
from sklearn.preprocessing import StandardScaler
# pipe1 = Pipeline([['sc', StandardScaler()], ['clf1', clf1]])
# pipe3 = Pipeline([['sc', StandardScaler()], ['clf3', clf3]])
#clf_labels = ['Logisitic Regression', 'Decision Tree', 'KNN']

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
#clf1 = AdaBoostClassifier()
#clf2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2,                             
#                                      n_estimators = 110, random_state =1)
#clf3 = BaggingClassifier()
clf_labels = ['Adaboost', 'Random Forests', 'Bagging']




from sklearn.cross_validation import cross_val_score, train_test_split
#X_train_train, X_train_valid, y_train_train, y_train_valid = train_test_split(X_train, 
#                                                                              y_train, test_size = 0.5,
#                                                                              random_state = 1)

#print('10-fold cross validation: \n')
#for clf, label in zip([clf1, clf2, clf3], clf_labels):
#    scores = cross_val_score(estimator= clf, X = X_train_train, y= y_train_train, cv=10, scoring = 'roc_auc')
#    print('ROC AUC: %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))

#mv_clf = VotingClassifier(estimators= [('ada',clf1), ('rfc', clf2), ('bag', clf3)], voting = 'soft')
#clf_labels+= ['Majority Voting']
#all_clf = [clf1, clf2, clf3, mv_clf]
#for clf, label in zip(all_clf, clf_labels):
#    scores = cross_val_score(estimator=clf, X = X_train_train, y = y_train_train, cv = 10, scoring = 'roc_auc')
#    print('Accuracy : %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))

#mv_clf.fit(X_train_train, y_train_train)
#print(mv_clf.score(X_train_valid, y_train_valid))


#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)
## 
#Check that your data frame has 418 entries
#print(my_solution.shape)
# 
#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_mvc_lr_dtc_knn.csv", index_label = ["PassengerId"])
# 

#mv_clf.get_params()
from sklearn.grid_search import GridSearchCV
#params = {'ada__n_estimators' : [10, 20, 50, 100], 'ada__learning_rate' : [1.0, 1.5],
#         'bag__n_estimators': [10, 20, 50, 100], 'rfc__max_depth': [6, 7, 8]}
#grid = GridSearchCV(estimator= mv_clf, param_grid= params, cv = 10, scoring= 'roc_auc')
#grid.fit(X_train_train, y_train_train)
#for params, mean_score, scores in grid.grid_scores_:
#    print('%0.3f +/- %0.2f %r' % (mean_score, scores.std() /2, params))
#print ('Best parameters: %s' % grid.best_params_)
#print ('Accuracy: %s' % grid.best_score_)
from sklearn.cross_validation import cross_val_score, train_test_split
X_train_train, X_train_valid, y_train_train, y_train_valid = train_test_split(X_train, 
                                                                              y_train, test_size = 0.2,
                                                                              random_state = 1)
#mv_clf.fit(X_train_train, y_train_train)
#clf1 = AdaBoostClassifier(learning_rate = 1.5, n_estimators = 10)
#clf1 = AdaBoostClassifier(learning_rate = 1.46, n_estimators = 53, random_state =1)
#clf2 = RandomForestClassifier(max_depth = 7, min_samples_split = 18, max_features = 2,                             
#                                     n_estimators = 110, random_state =1)
#clf3 = BaggingClassifier(n_estimators = 9, max_features = 2)
#clf3 = BaggingClassifier(n_estimators = 53, max_features = 5, random_state = 16)
#print('10-fold cross validation: \n')
#for clf, label in zip([clf1, clf2, clf3], clf_labels):
#    scores = cross_val_score(estimator= clf, X = X_train_train, y= y_train_train, cv=10, scoring = 'accuracy')
#    print('ROC AUC: %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))

##mv_clf = VotingClassifier(estimators= [('ada',clf1), ('rfc', clf2), ('bag', clf3)], voting = 'soft')
#clf_labels+= ['Majority Voting']
#all_clf = [clf1, clf2, clf3, mv_clf]
#for clf, label in zip(all_clf, clf_labels):
#    scores = cross_val_score(estimator=clf, X = X_train_train, y = y_train_train, cv = 10, scoring = 'accuracy')
#    print('Accuracy : %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))
#train_score, valid_score = [], []
#for n in np.arange(1.01, 2, 0.01):
#    bag_1 = AdaBoostClassifier(learning_rate = 1.46, n_estimators = 53)
#    bag_1.fit(X_train_train, y_train_train)
#    train_score.append(bag_1.score(X_train_train, y_train_train))
#    valid_score.append(bag_1.score(X_train_valid, y_train_valid))
    #print (n)
    #print(bag_1.score(X_train_valid, y_train_valid))
#max(valid_score)
#bag_1 = AdaBoostClassifier(learning_rate = 1.46, n_estimators = 53)
#bag_1.fit(X_train_train, y_train_train)
#train_score.append(bag_1.score(X_train_train, y_train_train))
#valid_score.append(bag_1.score(X_train_valid, y_train_valid))
#print(bag_1.score(X_train_valid, y_train_valid))
#for i in range(len(valid_score)):
#    print("%d %s" % (i, valid_score[i]))
#np.arange(0.01, 1, 0.01)
#mv_clf.fit(X_train, y_train)
#y_pred = mv_clf.predict(X_test)
#PassengerId =np.array(test["PassengerId"]).astype(int)
#my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
#print(my_solution)
# 
#Check that your data frame has 418 entries
#print(my_solution.shape)
# 
#Write your solution to a csv file with the name my_solution.csv
#my_solution.to_csv("my_solution_mvc_ada_rfc_bag.csv", index_label = ["PassengerId"])
# 


#clf1 = AdaBoostClassifier(learning_rate = 0.31, n_estimators = 10)
#clf1 = RandomForestClassifier(max_depth = 5, min_samples_split = 4,
#                              n_estimators = 650, random_state =1, max_features = 2)
#clf2 = RandomForestClassifier(max_depth = 7, min_samples_split = 18, max_features = 2,                             
#                                      n_estimators = 110, random_state =1)
#clf3 = BaggingClassifier(n_estimators = 10, max_features = 2)
#clf3 = BaggingClassifier(n_estimators = 53, max_features = 5, random_state = 16)
#clf3 = KNeighborsClassifier(n_neighbors=5, p=2)
#clf_labels = ['Random Forest 1', 'Random Forest 2', 'KNN']
#print('10-fold cross validation: \n')
#for clf, label in zip([clf1, clf2, clf3], clf_labels):
#    scores = cross_val_score(estimator= clf, X = X_train_train, y= y_train_train, cv=40, scoring = 'accuracy')
#    print('ROC AUC: %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))
#mv_clf = VotingClassifier(estimators= [('rfc1',clf1), ('rfc2', clf2), ('knn', clf3)], voting = 'soft')
#clf_labels+= ['Majority Voting']
#all_clf = [clf1, clf2, clf3, mv_clf]
#for clf, label in zip(all_clf, clf_labels):
#    scores = cross_val_score(estimator=clf, X = X_train_train, y = y_train_train, cv = 30, scoring = 'accuracy')
#    print('Accuracy : %0.2f (+/- %0.2f) [%s]' %(scores.mean(), scores.std(), label))
#mv_clf.fit(X_train_train, y_train_train)
#mv_clf.score(X_train_valid, y_train_valid)
from sklearn.neural_network import MLPClassifier

rfc2 = RandomForestClassifier(max_depth = 6, min_samples_split = 18, max_features = 2,
                              n_estimators = 110, random_state =1)
print((y_train == 1).sum(), (y_train == 0).sum()) 
rfc2.fit(X_train_train, y_train_train)
valid_zero_list= list((np.zeros(len(y_train_valid))).astype(int))
rfc2.score(X_train_train, y_train_train)
rfc2.score(X_train_valid, y_train_valid)
pred_prob = rfc2.predict_proba(X_train_valid)
y_pred = []
for i in range(0, len(y_train_valid)):
    if (pred_prob[i][0]  > 0.6):
        y_pred.append(0)
    else:
        y_pred.append(1)
           
rfc2.score(X_train_valid, y_pred)
rfc2.fit(X_train, y_train)
y_pred_prob = rfc2.predict_proba(X_test)
y_pred = []
for i in range(0, len(X_test)):
    if (y_pred_prob[i][0] >= 0.5):
        y_pred.append(0)
    else:
        y_pred.append(1)
        


y_pred_one = []
for i in range(0, len(X_test)):
    if (y_pred_prob[i][0] >= 0.51):
        y_pred_one.append(0)
    else:
        y_pred_one.append(1)
count = 0
for i in range(0, len(X_test)):
    if(y_pred[i] != y_pred_one[i]):
        count = count + 1
       
print(count)








#tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2)

#tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = None)
#bag = BaggingClassifier(base_estimator = rfc2, n_estimators= 20,
#                       max_samples = 1.0, max_features = 1.0,
#                       bootstrap = True, bootstrap_features = False,
#                       n_jobs = 1, random_state = 1)
#from sklearn.metrics import accuracy_score
#tree = tree.fit(X_train_train, y_train_train)
#y_train_pred = tree.predict(X_train_train)
#y_valid_pred = tree.predict(X_train_valid)
#tree_train = accuracy_score(y_train_train, y_train_pred)
#tree_valid = accuracy_score(y_train_valid, y_valid_pred)
#print('Decision Tree Classifier/ Test accuracies: %0.3f / %0.3f' % (tree_train, tree_valid))
#bag = bag.fit(X_train_train, y_train_train)
#y_train_pred = bag.predict(X_train_train)
#y_valid_pred = bag.predict(X_train_valid)
#bag_train = accuracy_score(y_train_train, y_train_pred)
#bag_valid = accuracy_score(y_train_valid, y_valid_pred)
#print('Bagging Classifier/ Test accuracies: %0.3f / %0.3f' % (bag_train, bag_valid))

#tree = DecisionTreeClassifier(criterion= 'entropy', max_depth =1)
#ada = AdaBoostClassifier(base_estimator= bag, n_estimators= 20,
#                        learning_rate = 0.1, random_state = 0)
#tree = tree.fit(X_train_train, y_train_train)
#y_train_pred = tree.predict(X_train_train)
#y_valid_pred = tree.predict(X_train_valid)
#tree_train = accuracy_score(y_train_train, y_train_pred)
#tree_valid = accuracy_score(y_train_valid, y_valid_pred)
#print('Decision Tree Classifier/ Test accuracies: %0.3f / %0.3f' % (tree_train, tree_valid))

#ada = ada.fit(X_train_train, y_train_train)
#y_train_pred = ada.predict(X_train_train)
#y_valid_pred = ada.predict(X_train_valid)
#ada_train = accuracy_score(y_train_train, y_train_pred)
#ada_valid = accuracy_score(y_train_valid, y_valid_pred)
#print('Ada Classifier/ Test accuracies: %0.3f / %0.3f' % (ada_train, ada_valid))


#ada.fit(X_train, y_train)

#y_pred = ada.predict(X_test)

MLP = MLPClassifier(hidden_layer_sizes= (100, 33), activation = 'tanh')
##MLP.fit(X_train_train, y_train_train)
#print(MLP.score(X_train_train, y_train_train))
#print(MLP.score(X_train_valid, y_train_valid))

MLP.fit(X_train, y_train)
y_pred_nn = MLP.predict(X_test)


#X_train_subsetted = train[['Pclass', 'Sex', 'Age']].values
#X_test_subsetted = test[['Pclass', 'Sex', 'Age']].values
#import xgboost 
#xgb = xgboost.XGBClassifier()
#xgb.fit(X_train_train, y_train_train)
#xgb_pred = xgb.predict(X_train_valid)

#xgb.fit(X_train_subsetted, y_train)
#y_pred = xgb.predict(X_test_subsetted)


PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(y_pred_nn, PassengerId, columns = ["Survived"])
print(my_solution)
# 
#Check that your data frame has 418 entries
print(my_solution.shape)
# 
#Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_mlpnn.csv", index_label = ["PassengerId"])
# 

#rfc2.fit(X_train_train, y_train_train)
#print(rfc2.score(X_train_valid, y_train_valid))

#import xgboost 
#for n in range(0, 2):
#    xgb = xgboost.XGBClassifier(max_depth= 10, gamma = 1)
#    xgb.fit(X_train_train, y_train_train)
#    xgb_pred = xgb.predict(X_train_valid)
#    print (n)
#    print(accuracy_score(xgb_pred, y_train_valid))
#max(acc)
#np.arange(0, 10, 1)

