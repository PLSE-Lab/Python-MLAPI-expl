import os
import re
import numpy as np
import pandas as pd
import warnings

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, model_selection, svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from xgboost import XGBClassifier

input_io_dir="../input/"
output_io_dir="./"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Helper functions
#############################################################################

# Column selection
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# Assign names to columns
class ColumnLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X=pd.DataFrame(X,columns=self.column_names)
        return X
        
# One hot encoding using pandas's getdummies
class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dummy_na):
        self.dummy_na=dummy_na
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame()
        for col in sorted(X.columns):
            dummies=pd.get_dummies(X[col],prefix=col, drop_first=True,dummy_na=self.dummy_na)
            df[dummies.columns]=dummies
        X=df
        X=X.astype('category')
        return X

# Support functions for adding new features
# Replace texts based on a dictionary
def multipleReplace(text, wordDic):
    for key in wordDic:
        if text.lower()==key.lower():
            text=wordDic[key]
            break
    return text

# Normalise title names by grouping them
def normaliseTitle(title):
    wordDic = {
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mrs':'Mrs',
    'Master':'Master',
    'Mme': 'Mrs',
    'Lady': 'Nobility',
    'Countess': 'Nobility',
    'Capt': 'Army',
    'Col': 'Army',
    'Dona': 'Other',
    'Don': 'Other',
    'Dr': 'Other',
    'Major': 'Army',
    'Rev': 'Other',
    'Sir': 'Other',
    'Jonkheer': 'Other',
    }    
    title=multipleReplace(title,wordDic)
    return title

# Extract Title feature from name
def extractTitleFromName(name):
    pos_point=name.find('.')
    if pos_point == -1: return ""
    wordList=name[0:pos_point].split(" ")
    if len(wordList)<=0: return ""
    title=wordList[len(wordList)-1]
    normalisedTitle=normaliseTitle(title)
    return normalisedTitle

# Extract TicketType feature from name
def getTicketType(name, normalise=True):
    item=name.split(' ')
    itemLength=len(item)
    if itemLength>1:
        ticketType=""
        for i in range(0,itemLength-1):
            ticketType+=item[i].upper()
    else:
        ticketType="NORMAL"
    if normalise==True:
        ticketType= ticketType.translate(str.maketrans('','','./'))
    return ticketType
    
# Add new features
class CustomFeatureExtender(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['Title']=X['Name'].apply(lambda x: extractTitleFromName(x)).astype('category')
        X['NoCabin']=X['Cabin'].isnull().apply(lambda x: 1 if x is True else 0).astype('category')
        X['TicketType']=X['Ticket'].apply(lambda x: getTicketType(x)).astype('category')
        X['IsAlone']=(X["SibSp"]+X["Parch"]).apply(lambda x: 0 if x>0 else 1).astype('category')
        X['FamilySize']=X["SibSp"]+X["Parch"]+1
        return X

# Transform features - continuous to discrete
class CustomRangeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        Xdf=pd.DataFrame(X)
        Xdf.loc[Xdf['Age'] <= 16, 'Age'] = 0
        Xdf.loc[(Xdf['Age'] > 16) & (Xdf['Age'] <= 32), 'Age'] = 1
        Xdf.loc[(Xdf['Age'] > 32) & (Xdf['Age'] <= 48), 'Age'] = 2
        Xdf.loc[(Xdf['Age'] > 48) & (Xdf['Age'] <= 64), 'Age'] = 3
        Xdf.loc[ Xdf['Age'] > 64, 'Age'] = 4
        Xdf.loc[Xdf['Fare'] <= 7.91, 'Fare'] = 0
        Xdf.loc[(Xdf['Fare'] > 7.91) & (Xdf['Fare'] <= 14.454), 'Fare'] = 1
        Xdf.loc[(Xdf['Fare'] > 14.454) & (Xdf['Fare'] <= 31), 'Fare']   = 2
        Xdf.loc[ Xdf['Fare'] > 31, 'Fare'] = 3
        return Xdf

# Fill missing data
class CustomFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['Embarked'].fillna('S')
        return X

# Helper to normalise columns between training and test set
def NormaliseColumns(dataframeA,dataframeB):
    for testCol in dataframeB.columns:
        if testCol not in dataframeA.columns:
            dataframeA[testCol]=0
    for trainCol in dataframeA.columns:
        if trainCol not in dataframeB.columns:
            dataframeB[trainCol]=0
    return dataframeA,dataframeB

# Compare different models
def ModelSelection(clf_list,name_list,train_features,train_labels,scoring='accuracy'):
    best_score=0
    for clf, name in zip(clf_list,name_list) :
        scores = model_selection.cross_val_score(clf, train_features.values.astype(float), train_labels.values.ravel().astype(float), cv=10, scoring=scoring)  
        print("ModelSelection: Scoring  %0.2f +/- %0.2f (%s 95%% CI)" % (scores.mean(), scores.std()*2, name))
        reference_score=scores.mean()+scores.std()
        if (reference_score>best_score):
            best_clf=name
            best_score=reference_score
            learning_model=clf
    print("ModelSelection: Best model - "+best_clf)
    return learning_model

# Fine tune a model given a param_grid
def FineTuneLearningModel(learning_model, param_grid, train_features,train_labels,scoring='accuracy'):
    grid_search = GridSearchCV(learning_model, param_grid, scoring,cv=10)
    grid_search.fit(train_features.values.astype(float),train_labels.values.astype(float))
    print(grid_search.best_params_)
    cvres = grid_search.cv_results_
    for mean_score,std_score, params in zip(cvres["mean_test_score"], cvres["std_test_score"],cvres["params"]):
        print('FineTuneLearningModel',mean_score,'+-',std_score, params)
    return grid_search.best_estimator_

# Evaluate model performance - preview
def EvaluatePerformancePreview(learning_model,train_features,train_labels,scoring='accuracy'):
    scores=cross_val_score(learning_model,train_features.values.astype(float),train_labels.values.astype(float),scoring=scoring,cv=10)
    print('EvaluatePerformancePreview:Score',scores)
    print('EvaluatePerformancePreview:Mean',scores.mean())
    print('EvaluatePerformancePreview:Std',scores.std())

# Evaluate model performance - detailed
def EvaluatePerformanceDetailed(learning_model,train_features,train_labels,scoring='accuracy'):
    train_X, test_X, train_y, test_y = train_test_split(train_features.values.astype(float),train_labels.values.astype(float), test_size=0.25,random_state=42)
    learning_model.fit(train_X,train_y)
    y_probas = learning_model.predict_proba(test_X)
    y_scores=y_probas[:,1]
    predictions_proba=(y_scores>0.6).astype(float)
    predictions = learning_model.predict(test_X)
    print('EvaluatePerformanceDetailed:Predictions',predictions)
    print('EvaluatePerformanceDetailed:Predictions proba',predictions_proba)
    score=precision_score(test_y,predictions)
    score_proba=precision_score(test_y,predictions_proba)
    print('EvaluatePerformanceDetailed:Score',score)
    print('EvaluatePerformanceDetailed:Score Proba',score_proba)
    matrix=confusion_matrix(test_y,predictions)
    print('EvaluatePerformanceDetailed:Confusion matrix',matrix)
    matrix_proba=confusion_matrix(test_y,predictions_proba)
    print('EvaluatePerformanceDetailed:Confusion matrix proba',matrix_proba)
    
    matrix=confusion_matrix(test_y,predictions)
    print('EvaluatePerformanceDetailed:Confusion matrix',matrix)
    print('EvaluatePerformanceDetailed:Classification report')
    print(classification_report(test_y, predictions))

# Train and generate predictions
def TrainModelAndGeneratePredictionsOnTestSet(learning_model,train_features,train_labels,test_features, threshold=-1):
    learning_model.fit(train_features.values.astype(float),train_labels.values.astype(float))
    if threshold==-1:
        predictions = learning_model.predict(test_features.values.astype(float))
    else:
        if hasattr(learning_model,"decision_function"):
            y_scores=learning_model.decision_function(test_features.values.astype(float))
        else:
            y_proba=learning_model.predict_proba(test_features.values.astype(float))
            y_scores=y_proba[:,1]
        predictions=(y_scores>threshold).astype(float)
    pred=pd.Series(predictions)
    # Ensure no floats go out
    return pred.apply(lambda x: 1 if x>0 else 0)
    


# Preparation of data sets
def PrepareDataSets():
    # Load data
    original_train_data=pd.read_csv(input_io_dir+"train.csv")
    original_test_data=pd.read_csv(input_io_dir+"test.csv")
    print('PrepareDataSets:original_train_data',original_train_data.shape)
    print('PrepareDataSets:original_test_data',original_test_data.shape)

    passengerId = original_test_data['PassengerId']
    survived=original_train_data['Survived']

    original_train_data=original_train_data.drop('Survived',axis=1)
    original_all_data=original_train_data.append(original_test_data)
    print('PrepareDataSets:original_alldata',original_all_data.shape)

    preprocessor=Pipeline(steps=[
        ('extender', CustomFeatureExtender()),
    ])
    print("PrepareDataSets:Features extended")
    enriched_train_data=preprocessor.fit_transform(original_train_data)
    enriched_test_data=preprocessor.fit_transform(original_test_data)
    print('PrepareDataSets:enriched_train_data',enriched_train_data.shape)
    print('PrepareDataSets:enriched_test_data',enriched_test_data.shape)

    exclude_features=['Name','SibSp','Parch','Ticket','Cabin']
    filtered_train_data=enriched_train_data.drop(exclude_features,axis=1)
    filtered_test_data=enriched_test_data.drop(exclude_features[1:],axis=1)
    print('PrepareDataSets:filtered_train_data',filtered_train_data.shape)
    print('PrepareDataSets:filtered_test_data',filtered_test_data.shape)

    print("PrepareDataSets:Features filtered")

    print("PrepareDataSets:Encoding features")
    numeric_features=['Age','Fare','FamilySize']

    numeric_pipeline = Pipeline(steps=[
        ('selector', DataFrameSelector(numeric_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('labeler',ColumnLabeler(numeric_features)),
        ('range_transformer',CustomRangeTransformer()),
        ('scaler', StandardScaler()),
        ('labeler2',ColumnLabeler(numeric_features)),
    ])

    categorical_features = ['Embarked', 'Sex','Pclass','Title','NoCabin','IsAlone']
    categorical_pipeline = Pipeline(steps=[
        ('selector',DataFrameSelector(categorical_features)),
        ('filler', CustomFiller()),
        ('dummy', DummyTransformer(dummy_na=False)),
    ])

    num_encoded_train_data=pd.DataFrame(numeric_pipeline.fit_transform(filtered_train_data))
    print('PrepareDataSets:num_encoded_train_data',num_encoded_train_data.shape)
    cat_encoded_train_data=pd.DataFrame(categorical_pipeline.fit_transform(filtered_train_data))
    print('PrepareDataSets:cat_encoded_train_data',cat_encoded_train_data.shape)
    encoded_train_data=pd.concat([num_encoded_train_data,cat_encoded_train_data],axis=1)
    print('PrepareDataSets:encoded_train_data',encoded_train_data.shape)
    
    num_encoded_test_data=pd.DataFrame(numeric_pipeline.fit_transform(filtered_test_data))
    print('PrepareDataSets:num_encoded_test_data',num_encoded_test_data.shape)
    cat_encoded_test_data=pd.DataFrame(categorical_pipeline.fit_transform(filtered_test_data))
    print('PrepareDataSets:cat_encoded_test_data',cat_encoded_test_data.shape)
    encoded_test_data=pd.concat([num_encoded_test_data,cat_encoded_test_data],axis=1)
    print('PrepareDataSets:encoded_test_data',encoded_test_data.shape)


    print("PrepareDataSets:Features encoded")
    encoded_train_data,encoded_test_data=NormaliseColumns(encoded_train_data,encoded_test_data)
    print('PrepareDataSets:Adjusted encoded_train_data',encoded_train_data.shape)
    print('PrepareDataSets:Adjusted encoded_test_data',encoded_test_data.shape)
    print("PrepareDataSets:Train/test feature normalisation finished")
    encoded_train_data=encoded_train_data.reindex(sorted(encoded_train_data.columns), axis=1)
    encoded_test_data=encoded_test_data.reindex(sorted(encoded_test_data.columns), axis=1)
    train_features=encoded_train_data
    test_features=encoded_test_data
    train_labels=survived
    return passengerId,train_features,train_labels, test_features


# Generate results
def GenerateOutputFile(passengerId,predictions):
    output = pd.DataFrame({ 'PassengerId': passengerId,
                            'Survived': predictions })
    output.to_csv(output_io_dir+"output.csv", index=False)

def ConfigureLearningModelsForBinaryClassification():
    xgb_clf = XGBClassifier(n_estimators=100,max_depth=40, random_state=42)
    dt_clf = DecisionTreeClassifier(random_state=42)
    rf_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
    et_clf = ensemble.ExtraTreesClassifier(n_estimators=100, random_state=42)
    gb_clf = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=42)
    ada_clf = ensemble.AdaBoostClassifier(n_estimators=100, random_state=42)
    svm_clf = svm.LinearSVC(C=0.1,random_state=42)
    lg_clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=400,random_state=42)
    e_clf = ensemble.VotingClassifier(estimators=[('xgb', xgb_clf), ('dt', dt_clf),('rf',rf_clf), ('et',et_clf), ('gbc',gb_clf), ('ada',ada_clf), ('svm',svm_clf), ('lg',lg_clf)])
    clf_list = [xgb_clf, dt_clf, rf_clf, et_clf, gb_clf, ada_clf, svm_clf,lg_clf,e_clf]
    name_list = ['XGBoost', 'Decision Trees','Random Forest', 'Extra Trees', 'Gradient Boosted', 'AdaBoost', 'Support Vector Machine', 'LogisticRegression','Ensemble']
    return clf_list,name_list

# Main section
#############################################################################

print('TitanicMain:Starting')
passengerId,train_features,train_labels, test_features=PrepareDataSets()
print('TitanicMain:Evaluating different models performance')
clf_list,name_list=ConfigureLearningModelsForBinaryClassification()
learning_model=ModelSelection(clf_list,name_list,train_features,train_labels)
print('TitanicMain:Fine-tuning selected model')
learning_model = XGBClassifier(objective='binary:logistic')
param_grid = [
    {'booster': ['gbtree'],'gamma':[0.4], 'max_depth': [9], 'min_child_weight': [3],
    'colsample_bytree': [1], 'subsample': [1],'reg_alpha': [1], 'reg_lambda': [2],
    'learning_rate': [0.01], 'n_estimators': [100]}
]
learning_model=FineTuneLearningModel(learning_model, param_grid,train_features,train_labels)
print('TitanicMain:Evaluating performance - preview')
EvaluatePerformancePreview(learning_model,train_features,train_labels)
#print('TitanicMain:Evaluating performance - detailed')
#EvaluatePerformanceDetailed(learning_model,train_features,train_labels)
print('TitanicMain:Train model and generate predictions on test set')
predictions=TrainModelAndGeneratePredictionsOnTestSet(learning_model,train_features,train_labels,test_features)
print("TitanicMain:Predictions ready")
print("TitanicMain:Generating output file")
GenerateOutputFile(passengerId,predictions)
print("TitanicMain:Finished")
