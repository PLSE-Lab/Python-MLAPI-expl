
#IMPORT NECESSARY LIBRARIES

import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC



from sklearn.model_selection import cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.pipeline import Pipeline





def eda():
    df=pd.read_csv('../input/heart-disease-uci/heart.csv')
    print(df.head())
    print(df.shape)
    print(df.info())

    cols=df.columns

    print(df.isnull().sum())

    sns.countplot(df['target'],hue=df['target'])

    #sns.catplot("target","chol",data=df)
    plt.show()
    imp_features=feature_selection(df) # 10 FEATURES ARE FOUND TO BE OF MORE USE ---> USING FEATURE SELECTION
    print(imp_features)

    for f in imp_features:
        sns.catplot("target","chol",data=df) #CATEGORICAL PLOT WITH IMP FEATURES 

    X=df[imp_features] 
    Y=df['target']

    model_it(X,Y)






def feature_selection(df):
    X=df.drop('target',axis=1)
    Y=df['target']


    fs=SelectKBest(score_func=f_classif,k=10).fit(X,Y)

    scorer=pd.DataFrame({'column names':X.columns,'score':fs.scores_}).sort_values(by='score',ascending=False)

    return scorer['column names'][:9]



def model_it(X,Y):
    X=X
    Y=Y
    mms=MinMaxScaler()
    X=mms.fit_transform(X)


    log_clf=Pipeline([('log_reg',LogisticRegression(penalty='l2'))])
    rf_clf=Pipeline([('Random Forest',RandomForestClassifier(criterion= 'entropy', max_depth=9, n_estimators=1200))])
    gb_clf=Pipeline([('Gradient Boost',GradientBoostingClassifier())])
    sv_clf=Pipeline([('svc',SVC())])
    pipelines = [log_clf, rf_clf, gb_clf,sv_clf]

    pipe_dict = {0: 'Logistic Regression', 1: 'RF', 2: 'GB',3:'SVC'}



    for i,model in enumerate(pipelines):
        print(pipe_dict[i],' :',cross_val_score(model,X,Y,cv=10).mean())










if __name__ == '__main__':
    eda()