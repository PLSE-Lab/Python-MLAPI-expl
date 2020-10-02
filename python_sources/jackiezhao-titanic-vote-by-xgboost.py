import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.feature_selection import *
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import xgboost as xgb
def fillNameData(name_data):
    if name_data.find("Mr")>=0:
        return 0
    elif name_data.find("Mrs")>=0:
        return 1
    elif name_data.find("Miss")>=0:
        return 2
    elif name_data.find("Master")>=0:
        return 3
    else:
        return 4
def cutFeature(cut_lst,data):
    cut_group=0
    for i in range(len(cut_lst)):
        if data <cut_lst[i]:
            cut_group=i
            break
        if i == len(cut_lst)-1:
            if data >=cut_lst[i]:
                cut_group=i+1
    return cut_group


def readData(train_path,test_path):
    data = pd.DataFrame(pd.read_csv(train_path))
    data_test = pd.DataFrame(pd.read_csv(test_path))
    data_test = data_test[["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
    x = data[["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
    y = data[["Survived"]]
    data_full = [x,data_test]
    for data_set in data_full:
        print (data_set.tail())
        print (data_set.tail())
        print ("Embarked")
        print (data_set["Embarked"].unique())
        data_set["name_len"] = data_set["Name"].apply(len)
        data_set["name_class"] = data_set["Name"].apply(fillNameData)
        
        data_set["Embarked"]=data_set["Embarked"].fillna('S')
        data_set["Embarked"]=data_set["Embarked"].map({'S':1,'C':2,'Q':3}).astype(int)
        print ("cabin not nan num %d" %(x["Cabin"].count()))
        data_set["has_cabin"] = data_set["Cabin"].apply(lambda x : 0 if type(x)==float else 1)
        print ("Fare desicrib")
        #print data_set["Fare"].describe()
        data_set["Fare"]=data_set["Fare"].fillna(x["Fare"].mean())
        data_set["Fare"] = data_set["Fare"].apply(lambda x:cutFeature([7.91,14.45,31.0],x))
        print ("delect Ticket")
        print ("ticket not nan num %d"%(data_set["Ticket"].count()))
        print ("Parch")
        print (data_set["Parch"].unique())
        data_set["Parch"]=data_set["Parch"].fillna(0)
        print ("SibSp")
        print (data_set["SibSp"].unique())
        data_set["SibSp"]=data_set["SibSp"].fillna(0)
        data_set["family_num"] = data_set["Parch"] + data_set["SibSp"] +1
        print ("Age desicrib")
        print (data_set["Age"].describe())
        data_set["Age"]=data_set["Age"].fillna(x["Age"].mean())
        data_set["Age"] = data_set["Age"].apply(lambda x:cutFeature([16,32,48,64],x))
        print ("Sex")
        print (data_set["Sex"].unique())
        data_set["Sex"] = data_set["Sex"].apply(lambda x : 0 if x=='male' else 1)
        print ("ticket not nan num %d" % (data_set["Name"].count()))
        print ("Pclass")
        print (data_set["Pclass"].unique())
        print (data_set.tail())
    [x, data_test]=data_full
    x = x.drop(["Ticket", "Cabin", "Name"], axis=1)
    data_test = data_test.drop(["Ticket", "Cabin", "Name"], axis=1)
    return [x,y,data_test]

def modelScore(x_train,labels_train,x_test,model_name,et_params):
    print("--------{0}------------".format(model_name))
    model = model_name(**et_params)

    model.fit(x_train, labels_train)
    if "feature_importances_" in dir(model):
        print (model.feature_importances_)

    print (classification_report(
        labels_train,
        model.predict(x_train)))
    return model

def buildModel(train_path,test_path):
    x_train,y_train,x_test = readData(train_path,test_path)
    print (x_train.tail())
    print (x_test.tail())
    y_passengerID = np.array(pd.DataFrame(pd.read_csv("../input/test.csv"))["PassengerId"])
    

    labels_train = LabelBinarizer().fit_transform(y_train)
    
    rf_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        'warm_start': True,
        # 'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'verbose': 0
    }
    # Extra Trees Parameters
    et_params = {
        'n_jobs': -1,
        'n_estimators': 500,
        # 'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # AdaBoost parameters
    ada_params = {
        'n_estimators': 500,
        'learning_rate': 0.75
    }

    # Gradient Boosting parameters
    gb_params = {
        'n_estimators': 500,
        # 'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    }

    # Support Vector Classifier parameters
    svc_params = {
        'kernel': 'linear',
        'C': 0.025
    }
    x_train = x_train.drop(["SibSp","Parch"], axis=1)
    x_test = x_test.drop(["SibSp","Parch"], axis=1)

    rf_model =modelScore(x_train, labels_train, x_test, RandomForestClassifier, rf_params)
    et_model =modelScore(x_train, labels_train, x_test, ExtraTreesClassifier, et_params)
    ada_model=modelScore(x_train, labels_train, x_test, AdaBoostClassifier, ada_params)
    gb_model =modelScore(x_train, labels_train, x_test, GradientBoostingClassifier, gb_params)
    svc_model =modelScore(x_train, labels_train, x_test, SVC, svc_params)
    x_train_df = pd.DataFrame([])
    x_test_df = pd.DataFrame([])
    i=0
    for model in [rf_model,et_model,ada_model,gb_model,svc_model]:
        i+=1
        x_train_df.insert(0,i ,model.predict(x_train))
        x_test_df.insert(0,i , model.predict(x_test))
    print(x_train_df.tail())
    gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(x_train_df, y_train)
    predictions = gbm.predict(x_test_df)
    print (classification_report(
        y_train,
        gbm.predict(x_train_df)))
    StackingSubmission = pd.DataFrame({ 'PassengerId': y_passengerID,
                            'Survived': predictions })
    StackingSubmission.to_csv("StackingSubmission.csv", index=False)

if __name__ == '__main__':
    train_path = "../input/train.csv"
    test_path = "../input/test.csv"
    buildModel(train_path,test_path)


