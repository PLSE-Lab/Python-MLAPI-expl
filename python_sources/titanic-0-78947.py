# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import lightgbm as lgb #pip install lightgbm 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def code():
    df = pd.read_csv('../input/train.csv',)
    meanage = df['Age'].mean() #all 29.69911764705882
    print(meanage)
    meanage_male = df.loc[df.Sex=='male']['Age'].mean() #male 30.72664459161148
    print(meanage_male)
    meanage_female = df.loc[df.Sex=='female']['Age'].mean() #female 27.915708812260537
    print(meanage_female)  
    
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df = df.replace(np.nan,0)
    
    for ind in df.index:
        if df['Age'][ind] <1 and df['Sex'][ind] >0:
            df['Age'][ind] = meanage_male
        if df['Age'][ind] <1 and df['Sex'][ind] <1:
            df['Age'][ind] = meanage_female
    
    print(df.head(6))
    
    X_df, y = df.drop(['Survived','PassengerId','Name','Ticket','Cabin','Embarked'], axis=1), df['Survived']
    X_df["Fare"] = X_df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    X_df["Age"] = X_df["Age"].map(lambda i: np.log(i) if i > 0 else 0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)       
    """
    rfc = RandomForestClassifier(random_state=42, n_jobs=4)    
    parameters = {"criterion": ['gini', 'entropy'],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth':[3,5,10,15,20],
                  "n_estimators" :[100, 200, 300,500], 
                  'min_samples_leaf': [3, 5, 7]}
    gcv = GridSearchCV(rfc, param_grid = parameters, scoring='roc_auc', n_jobs=4, cv=skf, verbose=1)
    gcv.fit(X_df, y)    
    RFC_best = gcv.best_estimator_
    best_param = gcv.best_params_
    print('The best parameter {}'.format( best_param))
    print(gcv.best_score_)     
    results = cross_val_score(RFC_best, X_df, y, cv=skf)
    print("RF accuracy score: {:.2f}%".format(results.mean()*100))
    
    The best parameter {'criterion': 'gini', 'max_depth': 15, '
    max_features': 'auto', 'min_samples_leaf': 3, 'n_estimators': 200}
    0.8670189149952645
    RF accuracy score: 83.05%
    """
    RFC_best = RandomForestClassifier(random_state=42,
                                      n_jobs=4,
                                      max_depth=15,
                                      max_features='auto',
                                      min_samples_leaf=3,
                                      n_estimators=200,
                                      criterion='gini',) 
    results = cross_val_score(RFC_best, X_df, y, cv=skf)
    print("RF accuracy score: {:.2f}%".format(results.mean()*100))
    
    """
    gbc = GradientBoostingClassifier(random_state=42)    
    gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300,500],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [3,5,10,15,20],
              'max_features': [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
              }
    gsGBC = GridSearchCV(gbc,param_grid = gb_param_grid, cv=skf, scoring="roc_auc", n_jobs= 4, verbose = 1)
    gsGBC.fit(X_df, y)
    GBC_best = gsGBC.best_estimator_
    best_param = gsGBC.best_params_
    print('The best parameter {}'.format( best_param))
    print(gsGBC.best_score_)     
    results = cross_val_score(GBC_best, X_df, y, cv=skf)
    print("GB accuracy score: {:.2f}%".format(results.mean()*100))
    The best parameter {'learning_rate': 0.01, 'loss': 'deviance', 
    'max_depth': 5, 'max_features': 0.5, 'n_estimators': 100}
    0.8706331268329449
    GB accuracy score: 82.83%
    """
    GBC_best = GradientBoostingClassifier(learning_rate=0.01,
                                          loss='deviance',
                                          max_depth=5,
                                          max_features=0.5,
                                          n_estimators=100,
                                          random_state=42)
    results = cross_val_score(GBC_best, X_df, y, cv=skf)
    print("GB accuracy score: {:.2f}%".format(results.mean()*100))
    """
    lg_param_grid={
            'n_estimators' : [100,200,300,500],
            'learning_rate': [0.1, 0.05, 0.01]
            }
    lg = lgb.LGBMClassifier(random_state=42,n_jobs=4)
    gsLG = GridSearchCV(lg,param_grid = lg_param_grid, cv=skf, scoring="roc_auc", n_jobs= 4, verbose = 1)
    gsLG.fit(X_df, y)
    LG_best = gsLG.best_estimator_
    best_param = gsLG.best_params_
    print('The best parameter {}'.format( best_param))   
    print(gsLG.best_score_) 
    results = cross_val_score(LG_best, X_df, y, cv=skf)
    print("LG accuracy score: {:.2f}%".format(results.mean()*100))   
    The best parameter {'learning_rate': 0.01, 'n_estimators': 500}
    0.8571065434978682
    LG accuracy score: 82.38%
    """
    LG_best =lgb.LGBMClassifier(random_state=42,
                                n_jobs=4,
                                learning_rate=0.01,n_estimators=500)    
    results = cross_val_score(LG_best, X_df, y, cv=skf)
    print("LG accuracy score: {:.2f}%".format(results.mean()*100))  
    """
    et = ExtraTreesClassifier(random_state=42,n_jobs=4)
    et_param_grid={
            'max_features': [0.2,0.3,0.4,0.5,0.6,0.7,0.8],
            'n_estimators': [100, 200, 300,500], 
            'max_depth': [3,5,10,15,20],
            }  
    gsET = GridSearchCV(et,param_grid = et_param_grid, cv=skf, scoring="roc_auc", n_jobs= 4, verbose = 1)
    gsET.fit(X_df, y)
    ET_best = gsET.best_estimator_
    best_param = gsET.best_params_
    print('The best parameter {}'.format( best_param))
    print(gsET.best_score_)    
    results = cross_val_score(ET_best, X_df, y, cv=skf)
    print("ET accuracy score: {:.2f}%".format(results.mean()*100))
    The best parameter {'max_depth': 5, 'max_features': 0.2, 'n_estimators': 200}
    0.8626722200990128
    ET accuracy score: 79.57%    
    """
    ET_best =ExtraTreesClassifier(random_state=42,
                                  max_depth=5,
                                  max_features=0.2, 
                                  n_estimators=200,
                                  n_jobs=4)
    
    results = cross_val_score(ET_best, X_df, y, cv=skf)
    print("ET accuracy score: {:.2f}%".format(results.mean()*100))
    
    """
    dec_param = {'max_depth':[3,5,10,15,20],'min_samples_leaf': [3, 5, 7]}
    dt = DecisionTreeClassifier(random_state=42)
    
    gsDT = GridSearchCV(dt,param_grid = dec_param, cv=skf, scoring="roc_auc", n_jobs= 4, verbose = 1)
    gsDT.fit(X_df, y)
    DT_best = gsDT.best_estimator_
    best_param = gsDT.best_params_
    print('The best parameter {}'.format( best_param))
    print(gsDT.best_score_) 
    results = cross_val_score(DT_best, X_df, y, cv=skf)
    print("DT accuracy score: {:.2f}%".format(results.mean()*100))
    The best parameter {'max_depth': 3, 'min_samples_leaf': 7}
    0.8474363880181008
    DT accuracy score: 81.26%
    """
    DT_best =DecisionTreeClassifier(random_state=42,max_depth=3,
                                    min_samples_leaf=7)
    results = cross_val_score(DT_best, X_df, y, cv=skf)
    print("DT accuracy score: {:.2f}%".format(results.mean()*100))
    
    """
    lr = [0.5,0.1, 0.2, 0.05, 0.01,0.02,0.03]
    for r in lr:
        Ada_best = AdaBoostClassifier(DT_best,random_state=42,learning_rate=r)    
        results = cross_val_score(Ada_best, X_df, y, cv=skf)
        print("Ada accuracy score: {:.2f}%".format(results.mean()*100), r)
    
    Ada accuracy score: 79.35% 0.5
    Ada accuracy score: 80.81% 0.1
    Ada accuracy score: 81.03% 0.2
    Ada accuracy score: 81.04% 0.05
    Ada accuracy score: 80.47% 0.01
    Ada accuracy score: 81.60% 0.02
    Ada accuracy score: 81.60% 0.03
    Ada accuracy score: 81.04% 0.05
    """
    r = 0.02
    Ada_best = AdaBoostClassifier(DT_best,random_state=42,learning_rate=r)    
    results = cross_val_score(Ada_best, X_df, y, cv=skf)
    print("Ada accuracy score: {:.2f}%".format(results.mean()*100), r)

    svc = SVC(random_state=42,verbose=0,
              kernel='linear',
              C=0.025,probability=True)
    results = cross_val_score(svc, X_df, y, cv=skf)
    print("SVC accuracy score: {:.2f}%".format(results.mean()*100))
    #SVC accuracy score: 78.91%    
    votingC = VotingClassifier(
            estimators=[('rfc', RFC_best),
                        ('gbc',GBC_best),
                        ('lg',LG_best), 
                        ('et',ET_best),
                        ('dt',DT_best),
                        ('ada',Ada_best),
                        ('svc',svc)
                        ], voting='soft', n_jobs=4)
    
    results = cross_val_score(votingC, X_df, y, cv=skf)
    print("Vote accuracy score: {:.2f}%".format(results.mean()*100))
    votingC = votingC.fit(X_df, y)   
    """
    RF accuracy score: 83.05%
    GB accuracy score: 82.83%
    LG accuracy score: 82.38%
    ET accuracy score: 79.57%
    DT accuracy score: 81.26%
    Ada accuracy score: 81.60% 0.02
    SVC accuracy score: 78.91%
    Vote accuracy score: 82.16%
    PB - 0.78947
    """
    test_df = pd.read_csv('../input/test.csv')
    meanage_male = test_df.loc[test_df.Sex=='male']['Age'].mean() #male 30.72664459161148
    meanage_female = test_df.loc[test_df.Sex=='female']['Age'].mean() #female 27.915708812260537
    
    
    label_encoder = LabelEncoder()
    test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])
    test_df = test_df.replace(np.nan,0)
    
    for ind in test_df.index:
        if test_df['Age'][ind] <1 and test_df['Sex'][ind] >0:
            test_df['Age'][ind] = meanage_male
        if test_df['Age'][ind] <1 and test_df['Sex'][ind] <1:
            test_df['Age'][ind] = meanage_female
            
    X_test  = test_df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1).copy()
    X_test["Fare"] = X_test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    X_test["Age"] = X_test["Age"].map(lambda i: np.log(i) if i > 0 else 0)
    
    Y_pred = votingC.predict(X_test)
    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": Y_pred
        })
    
    submission.to_csv('submission.csv', index=False)
    print('ok')
if __name__ == '__main__': 
    code()