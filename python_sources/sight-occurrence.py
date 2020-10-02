import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.feature_selection as fs
import matplotlib.pyplot as plt

import xgboost
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import neural_network

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# get data
df = pd.read_csv("../input/sight.csv")
df_X = df.iloc[:,3:17]

# get targets
df_T = df[['total']].values
df_T_binary = df['total'].apply(lambda x: 1 if x > 0 else x).values
df_L = df[['lancha']].values
df_L_binary = df['lancha'].apply(lambda x: 1 if x > 0 else x).values
df_G = df[['gear']].values
df_G_binary = df['gear'].apply(lambda x: 1 if x > 0 else x).values

# scaling
scalar = MinMaxScaler(feature_range=(0, 1))
df_X_scaled = scalar.fit_transform(df_X)

# feature selection -> tree based approach suggests 8 predictors, thus k=8 choosen below
# feature selection doesn't increase the accuracy of the model (tested separately), thus we omit this here!
    # selector = fs.SelectKBest(score_func=fs.f_classif, k=8).fit(df_X, df_T)
    # mask = selector.get_support() 
    # new_features = [] 
    # for bool, feature in zip(mask, list(df_X.columns)):
    #     if bool:
    #         new_features.append(feature)
    # df_X_reduced = df_X[new_features]

# our base performance measures
def accuracy_score_regressor(actual, prediction):
    count_positive_cases = 0
    count_positive_prediction = 0
    count_negative_cases = 0
    count_negative_prediction = 0
    for a, p in zip(actual, prediction):
        if a > 0:
            count_positive_cases += 1
            if p >= 0.5:
                count_positive_prediction += 1
        if a == 0:
            count_negative_cases += 1
            if p < 0.5:
                count_negative_prediction += 1
    negative_case_accuracy = count_negative_prediction / count_negative_cases
    positive_case_accuracy = count_positive_prediction / count_positive_cases
    return [negative_case_accuracy, positive_case_accuracy]  

def accuracy_score_classifier(actual, prediction):
    count_positive_cases = 0
    count_positive_prediction = 0
    count_negative_cases = 0
    count_negative_prediction = 0
    for a, p in zip(actual, prediction):
        if a == 1:
            count_positive_cases += 1
            if p == 1:
                count_positive_prediction += 1
        if a == 0:
            count_negative_cases += 1
            if p == 0:
                count_negative_prediction += 1
    negative_case_accuracy = count_negative_prediction / count_negative_cases
    positive_case_accuracy = count_positive_prediction / count_positive_cases
    return [negative_case_accuracy, positive_case_accuracy]  

# formatting output print
def format_print(result):
    print("Average negative accuracy is {0:.2f}%".format(result[0]),
                    "and average positive accuracy is {0:.2f}%".format(result[1]))    

# simulation
def simulate(X, Y, number_of_run, algorithm, accuracy_score):
    average_negative_accuracy = 0.0
    average_positive_accuracy = 0.0
    for i in range(number_of_run):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=i)
        model = algorithm          
        model.fit(x_train, y_train.ravel())
        prediction = model.predict(x_test)
        average_negative_accuracy += accuracy_score(y_test, prediction)[0]
        average_positive_accuracy += accuracy_score(y_test, prediction)[1]
        #print("Results of run {} {}".format(i, accuracy_score(y_test, prediction)))
    average_negative_accuracy /= number_of_run   
    average_positive_accuracy /= number_of_run  
    return [average_negative_accuracy * 100, average_positive_accuracy * 100]

# run
RUN_TIME = 25
REGRESSORS = [linear_model.LinearRegression(),
            linear_model.Ridge(),
            linear_model.Lasso(),
            linear_model.SGDRegressor(),
            tree.DecisionTreeRegressor(),
            tree.ExtraTreeRegressor(),
            ensemble.RandomForestRegressor(),
            ensemble.GradientBoostingRegressor(),
            xgboost.XGBRegressor(),
            svm.SVR()]
CLASSIFIERS = [linear_model.LogisticRegression(),
            neighbors.KNeighborsClassifier(),
            tree.DecisionTreeClassifier(),
            tree.ExtraTreeClassifier(),
            ensemble.RandomForestClassifier(),
            ensemble.GradientBoostingClassifier(),
            xgboost.XGBClassifier(),
            naive_bayes.BernoulliNB(),
            svm.SVC()]

dash_number = 75

print("-" * dash_number, "\nRegression models:")
for alg in REGRESSORS:
    print("-" * dash_number)
    print("Simulaiton run time {}, Algorithm used {}:".format(RUN_TIME, alg.__class__.__name__))
    print("-> Target is 'total':", format_print(simulate(df_X_scaled, df_T, RUN_TIME, alg, accuracy_score_regressor)))
    print("-> Target is 'lancha':", format_print(simulate(df_X_scaled, df_L, RUN_TIME, alg, accuracy_score_regressor)))
    print("-> Target is 'gear':", format_print(simulate(df_X_scaled, df_G, RUN_TIME, alg, accuracy_score_regressor))) 
    
print("-" * dash_number, "\nClassification models:")
for alg in CLASSIFIERS:
    print("-" * dash_number)
    print("Simulaiton run time {}, Algorithm used {}:".format(RUN_TIME, alg.__class__.__name__))
    print("-> Target is 'total':", format_print(simulate(df_X_scaled, df_T_binary, RUN_TIME, alg, accuracy_score_classifier)))
    print("-> Target is 'lancha':", format_print(simulate(df_X_scaled, df_L_binary, RUN_TIME, alg, accuracy_score_classifier)))
    print("-> Target is 'gear':", format_print(simulate(df_X_scaled, df_G_binary, RUN_TIME, alg, accuracy_score_classifier)))


    
        
    
    
