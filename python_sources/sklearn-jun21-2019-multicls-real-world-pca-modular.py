'''
June21-2019
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for netmeds company logistics turn-around-time prediction real world problem
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

Results:
train accuracy: 0.994684398811569
test accuracy: 0.9563300242872975

'''
import os
import pandas
import numpy as np
from sklearn import preprocessing
import time
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from xgboost.sklearn import XGBClassifier  

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import warnings
import pickle

def read_traindata():
    '''
    Read the training dataset
    '''
    warnings.filterwarnings('ignore')
    dataFrame = pandas.read_csv('../input/Actual_TAT_Train_18_06_2019_Version7.csv')
    dataframe = dataFrame.dropna(axis=0).copy()
    predictors = dataFrame.iloc[:, dataFrame.columns != 'actual_TAT']
    target = dataFrame.iloc[:, dataFrame.columns == 'actual_TAT']
    # here we do label encoding to transform categorical columns into numbers...but this isn't enough, we need to do one-hot encoding or get_dummies after this.
    # ...else, simply one-hot all input data manually before running this program.
    X = pandas.DataFrame()
    for column in predictors.columns:
        if predictors[column].dtype == 'object':
            X[column] = predictors[column].copy()
    
    objects = []
    for each in range(len(X.columns)):
        objects.append(preprocessing.LabelEncoder())
    
    for column, obj in zip(X.columns, objects):
        X[column] = obj.fit_transform(X[column])
    
    for column in X.columns:
        predictors[column] = X[column]
    
    X_train, X_test, y_train, y_test = train_test_split(predictors, target.values.ravel(), test_size=0.3, random_state=42) #use most of the data for training
    print('len(X_train):', len(X_train))
    print('len(X_test):', len(X_test))
    
    return X_train, X_test, y_train, y_test
    
def normalize_data(X_train, X_test):
    '''
    Let's normalize this data to 0 mean and unit std
    '''
    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler
    # Define the scaler 
    scaler = StandardScaler().fit(X_train)
    # Scale the train set
    X_train = scaler.transform(X_train)
    # Scale the test set
    X_test = scaler.transform(X_test)
    return scaler, X_train, X_test
    
    ##########################################################

def pca_transform(X_train, X_test):
    '''
    reduce dimensions using PCA
    '''
    from sklearn.decomposition import PCA
    pca = PCA(.98, whiten=True)			#select components/columns which explain 98% of the variance in the data
    #pca = PCA(n_components=13)	#select k components/columns  which contribute the most to variance
    pca.fit(X_train)
    print("pca.n_components_:", pca.n_components_)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return pca, X_train, X_test

def initialize_and_fit_classifiers(X_train, y_train):
    '''
    initialize the classifiers and fit the data
    '''
    
    neuralNet = MLPClassifier()
    rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, class_weight='balanced')
    sgd = SGDClassifier(loss="hinge", penalty="l2", max_iter=5, class_weight='balanced')
    neigh = KNeighborsClassifier(n_neighbors=3)
    svmc = svm.SVC(gamma='scale', class_weight='balanced')
    xgb = XGBClassifier(class_weight='balanced')  

    model = neuralNet.fit(X_train, y_train)
    # model = rf.fit(X_train, y_train)
    # model = sgd.fit(X_train, y_train)
    # model = neigh.fit(X_train, y_train)
    # model = svmc.fit(X_train, y_train)
    # model = xgb.fit(X_train, y_train)
    
    pickle.dump(model, open('nn-Actual_TAT_Train_18_06_2019_Version7.pkl', 'wb'))
    

def make_inference(scaler, pca, X_train, y_train, X_test, y_test):
    '''
    make inferences using the trained model
    '''
    model = pickle.load(open('nn-Actual_TAT_Train_18_06_2019_Version7.pkl', 'rb'))
    train_prediction = model.predict(X_train)
    precision, recall, fbeta, support = precision_recall_fscore_support(y_train, train_prediction)
    print("train accuracy:", accuracy_score(y_train, train_prediction))
    print("train f1score:", f1_score(y_train, train_prediction, average='weighted'))
    
    prediction = model.predict(X_test)
    # print("type(prediction):", type(prediction))
    # print("X_test[:10]:", X_test[:10])
    # print("prediction[:10]:", prediction[:10])
    
    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, prediction)
    
    print("test accuracy:", accuracy_score(y_test, prediction))
    print("test f1score:", f1_score(y_test,prediction,average='weighted'))
    
    xTest_runtime = pandas.read_csv('../input/Actual_TAT_Test_18_06_2019_Version7.csv')
    # xTest_runtime = dataset_runtime.iloc[:, dataset_runtime.columns != 'SalaryType']
    
    # Scale the runtime set
    xTest_runtime = scaler.transform(xTest_runtime)
    
    #pca
    xTest_runtime = pca.transform(xTest_runtime)

    yPrediction_runtime = model.predict(xTest_runtime)
    print('yPrediction_runtime:', yPrediction_runtime)
    print('len(yPrediction_runtime):', len(yPrediction_runtime))
    print("yPrediction_runtime[:10]:", yPrediction_runtime[:10])
    np.savetxt('Actual_TAT_Test_18_06_2019_Version7_nn_pca_out.csv', yPrediction_runtime, delimiter=',')

if __name__ == "__main__":
    
    startTime = time.time()
    
    X_train, X_test, y_train, y_test = read_traindata()
    scaler, X_train, X_test = normalize_data(X_train, X_test)
    pca, X_train, X_test = pca_transform(X_train, X_test)
    initialize_and_fit_classifiers(X_train, y_train)
    make_inference(scaler, pca, X_train, y_train, X_test, y_test)
    # make_inference(X_train, y_train, X_test, y_test)
    endTime = time.time()
    
    #print elapsed time in hh:mm:ss format
    hours, rem = divmod(endTime-startTime, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))