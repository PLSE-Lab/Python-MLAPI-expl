# Data loading / visualization
import numpy as np
import pandas as pd

# Data processing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# Data modeling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def fitFullPipeline(data, num_attribs, cat_attribs):
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                             ('std_scaler', StandardScaler())])
    cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                             ('one_hot_encoding', OneHotEncoder())])
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),
                                      ("cat", cat_pipeline, cat_attribs)])
    return full_pipeline.fit(data)

# The following code will be used to train multiple models across multiple hyperparameters for model selection
# this code was taken from http://www.davidsbatista.net/blog/2018/02/23/model_optimization/

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
    
    def get_best_model(self):
        best_score = 0 
        for key in helper.grid_searches.keys():
            score = helper.grid_searches[key].best_score_
            if score > best_score:
                best_score = score
                model = helper.grid_searches[key].best_estimator_
        return model
            
            
    
        
    

if __name__ == '__main__':
    
    
    train_df = pd.read_csv('../input/titanic/train.csv')
    test_df = pd.read_csv('../input/titanic/test.csv')
    
    # for now we choose to drop these variables
    # 'PassengerID' and 'Name' likely have little to no predictive value in determining 
    # survival, 'Cabin' is almost all null values, and 'Ticket' is not standardized
    # enough to make sense of at this point
    
    y_train = train_df['Survived']
    
    train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)
    
    num_attribs = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    cat_attribs = ['Sex', 'Embarked']
    
    pipeline = fitFullPipeline(train_df, num_attribs, cat_attribs)
    
    X_train = pipeline.transform(train_df)
    
    
    models = {
        'LogisticRegressionClassifier': LogisticRegression(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'SVC': SVC()
    }

    params = {
        'LogisticRegressionClassifier': {'penalty' : ['l1', 'l2']},
        'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
        'SVC': [
            {'kernel': ['linear'], 'C': [1, 10]},
            {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
        ]
    }
    
    
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(X_train,y_train, scoring='accuracy', n_jobs=2, refit=True)
    
    # print(helper.score_summary(sort_by='max_score'))
    
    test_df_ids = test_df['PassengerId']
    
    test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    X_test = pipeline.transform(test_df)
    
    # get the best model from the above grid search / model selection
    
    predictor = helper.get_best_model()
    
    preds = predictor.predict(X_test)
    
    submission_df = pd.concat([test_df_ids, pd.Series(preds, name='Survived')], axis=1)
    submission_df.to_csv('titanic_submission_v1.csv', index=False)
    

    
