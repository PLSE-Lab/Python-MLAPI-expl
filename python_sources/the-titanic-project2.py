import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.cross_validation import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import feature_selection

# #Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

# #Any files you save will be available in the output tab below
# train.to_csv('copy_of_the_training_data.csv', index=False)

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        # Remove the 'pass' statement below 
        # and write your prediction conditions here
        if passenger['Sex']=='female':
            if passenger['Pclass']<3:
                predictions.append(1)
            else:
                if passenger['Age']<30:
                    predictions.append(1)
                else:
                    predictions.append(0)
        else:
            if passenger['Age']<10:
                if passenger['Pclass']<3:
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                    predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)




def predictions_4(data):
    return

# Make the predictions

#################################
# this is predictions_3 : 
# which is my simple decision tree solution

predictions = predictions_3(test)
test['Survived'] = predictions

# display(test_data.head())
# display(test[['PassengerId','Survived']])

test[['PassengerId','Survived']].to_csv('my_simple_DT.csv',index = False)

#################################


# #################################
# # this is predictions_4 : 
# # which is refine decision tree solution

# # print the general information
# print(train.info())
# print(test.info())


# selected_feature = [
#         'Pclass','Sex','Age','Embarked','SibSp','Parch','Fare'
#     ]
    
# X_train = train[selected_feature]
# X_test = test[selected_feature]

# y_train = train['Survived']

# # get the information of missing value columns
# # Embarked column
# print( X_train['Embarked'].value_counts()  )
# print( X_test['Embarked'].value_counts()   )

# # As there are missing value in Embarked
# # then use the most frequence type S to fill the null
# X_train['Embarked'].fillna('S',inplace = True)
# X_test['Embarked'].fillna('S', inplace = True)

# # also fill the age na with the mean value
# ###  Mark point, but this can be refined we can predict the age to fill it 
# X_train['Age'].fillna(X_train['Age'].mean(),inplace = True)
# X_test['Age'].fillna(X_test['Age'].mean(),inplace = True)
# X_test['Fare'].fillna(X_test['Fare'].mean(),inplace = True)

# # Check the data again!
# X_train.info()
# X_test.info()


# # use the import DictVectorizer 
# dict_vec = DictVectorizer(sparse=False)
# X_train = dict_vec.fit_transform(X_train.to_dict(orient = 'record'))
# X_test = dict_vec.transform(X_test.to_dict(orient = 'record'))
# print(dict_vec.feature_names_)
# print(len(dict_vec.feature_names_))


# # use the import RandomForestClassifier
# rfc = RandomForestClassifier()

# # use the import XGBClassifier 
# xgbc = XGBClassifier()

# # use the import DecisionTree
# dt = DecisionTreeClassifier(criterion='entropy')


# # use the import cross_val_score for get the score result
# print('Default RandomForestClassifier cross_val_score: ')
# rfc_score = cross_val_score(rfc, X_train, y_train, cv=5).mean()
# print(rfc_score)


# print('Default XGBClassifier cross_val_score: ')
# xgbc_score = cross_val_score(xgbc, X_train, y_train, cv=5).mean()
# print(xgbc_score)


# print('Default DecisionTreeClassifier cross_val_score: ')
# dt_score = cross_val_score(dt, X_train, y_train, cv=5).mean()
# print(dt_score)



# # Fit and compute the prediction result of test set
# # then output the submission
# rfc.fit(X_train,y_train)
# rfc_y_predict = rfc.predict(X_test)
# rfc_submission = pd.DataFrame(
#         {
#             'PassengerId':test['PassengerId'],
#             'Survived':rfc_y_predict
#         }
#     )

# rfc_submission.to_csv('rfc_submission.csv', index=False)


# xgbc.fit(X_train, y_train)
# xgbc_y_predict = xgbc.predict(X_test)
# xgbc_submission = pd.DataFrame(
#         {
#             'PassengerId':test['PassengerId'],
#             'Survived':xgbc_y_predict
#         }
#     )

# xgbc_submission.to_csv('xgbc_submission.csv', index=False)



# dt.fit(X_train, y_train)
# dt_y_predict = dt.predict(X_test)
# dt_submission = pd.DataFrame(
#         {
#             'PassengerId':test['PassengerId'],
#             'Survived':dt_y_predict
#         }
#     )

# dt_submission.to_csv('dt_submission.csv', index=False)



# fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=7)
# X_train_fs = fs.fit_transform(X_train, y_train)
# dt.fit(X_train_fs, y_train)
# X_test_fs = fs.transform(X_test)
# dt_fs_y_predict = dt.predict(X_test_fs)
# dt_fs_submission = pd.DataFrame(
#         {
#             'PassengerId':test['PassengerId'],
#             'Survived':dt_fs_y_predict
#         }
#     )

# dt_fs_submission.to_csv('dt_fs_submission.csv', index=False)








#################################