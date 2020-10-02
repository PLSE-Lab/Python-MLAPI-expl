import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier as XGB
from sklearn.ensemble import RandomForestClassifier as RF
from lightgbm import LGBMClassifier as LGBM
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import re
import pickle

class Model:
    
    def __init__(self,train):
        self.train = train.copy()
        self.select_features()
        
    def select_features(self):
        
        #----------Balancing out data to contain equal number of defaulter classes as it is heavily skewed---------------------
        tr_0 = self.train[self.train['defaulter'] == 0]
        tr_1 = self.train[self.train['defaulter'] == 1]
        tr_0 = tr_0[:len(tr_1)]
        self.train = pd.concat([tr_0, tr_1], axis=0)
        
        #------------------------------drop_features are those which seemed unimportant, but can be edited to test results-----------------------
        
        self.drop_features = ['Unnamed: 0','collections_12_mths_ex_med', 'funded_amnt', 'policy_code', 'pub_rec', 'pymnt_plan', 'revol_util', 'application_type', 'delinq_2yrs', 'revol_bal', 'loan_status']
        self.null_value_features = ['emp_title','open_acc', 'acc_now_delinq', 'tot_coll_amt' , 'emp_length', 'tot_cur_bal' ,'total_rev_hi_lim', 'zip_code','addr_state']
        #self.forward_searching = ['id', 'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'annual_inc', 'verification_status', 'purpose', 'dti','total_acc', 'initial_list_status', 'out_prncp', 'home_ownership']
        
        self.train.drop(columns = self.drop_features, inplace = True)
        self.train.drop(columns = self.null_value_features, inplace = True)
        #self.train.drop(columns = self.forward_searching, inplace = True)

        #----------Tranforming non-numerical features----------------------------------- 
        
        self.LE_grade = preprocessing.LabelEncoder()
        self.LE_sub_grade = preprocessing.LabelEncoder()
        self.LE_verification_status = preprocessing.LabelEncoder()
        self.LE_purpose = preprocessing.LabelEncoder()
        self.LE_initial_list_status = preprocessing.LabelEncoder()
        
        
        self.train.dropna(inplace = True)        #dropping rows which have any null value
        
        self.LE_grade.fit(self.train['grade'])
        self.train['grade'] = self.LE_grade.transform(self.train['grade'])

        self.LE_sub_grade.fit(self.train['sub_grade'])
        self.train['sub_grade'] = self.LE_sub_grade.transform(self.train['sub_grade'])

        self.oe = preprocessing.LabelBinarizer()
        self.oe.fit(sorted(self.train.home_ownership.unique()))

        onehot = pd.DataFrame(self.oe.transform(self.train.home_ownership))
        onehot.columns = ["HO_"+i for i in self.oe.classes_]
        onehot.index = self.train.index
        self.train = self.train.join(onehot)
        
        self.train.drop('home_ownership',axis = 1, inplace = True)

        self.LE_verification_status.fit(self.train['verification_status'])
        self.train['verification_status'] = self.LE_verification_status.transform(self.train['verification_status'])

        self.LE_purpose.fit(self.train['purpose'])
        self.train['purpose'] = self.LE_purpose.transform(self.train['purpose'])

        self.LE_initial_list_status.fit(self.train['initial_list_status'])
        self.train['initial_list_status'] = self.LE_initial_list_status.transform(self.train['initial_list_status'])

        self.label = self.train['defaulter']
        self.train.drop('defaulter', axis = 1, inplace = True)

        '''
        oe.fit(sorted(self.train.addr_state.unique()))

        onehot = pd.DataFrame(oe.transform(self.train.addr_state))
        onehot.columns = ["HO_"+i for i in oe.classes_]
        onehot.index = self.train.index
        self.train = self.train.join(onehot)
        onehot = pd.DataFrame(oe.transform(self.test.addr_state))
        onehot.columns = ["HO_"+i for i in oe.classes_]
        onehot.index = self.test.index
        self.test = self.test.join(onehot)
        
        self.train.drop('addr_state',axis = 1, inplace = True)
        self.test.drop('addr_state',axis = 1, inplace = True)
        '''
    
        print(self.train.columns)
        
    def select_test_features(self,test):
        
        test.drop(columns = self.drop_features, inplace = True)
        test.drop(columns = self.null_value_features, inplace = True)
        
        print("in function ", test.shape)
        #test.drop(columns = self.forward_searching, inplace = True)
        
        test['grade'] = self.LE_grade.transform(test['grade'])
        
        test['sub_grade'] = self.LE_sub_grade.transform(test['sub_grade'])
        
        onehot = pd.DataFrame(self.oe.transform(test.home_ownership))
        onehot.columns = ["HO_"+i for i in self.oe.classes_]
        onehot.index = test.index
        test = test.join(onehot)
        
        test.drop('home_ownership', axis = 1, inplace = True)
        
        test['verification_status'] = self.LE_verification_status.transform(test['verification_status'])

        test['purpose'] = self.LE_purpose.transform(test['purpose'])

        test['initial_list_status'] = self.LE_initial_list_status.transform(test['initial_list_status'])

        test.drop('defaulter', axis = 1, inplace = True)
        
        return test
    
#-------------------------For training LGBM model----------------------------------------
    
    def fit_LGBM(self):
        self.model = LGBM(random_state=200, num_iterations=1000)
        self.model.fit(self.train,self.label)
        
#-------------------------For training XG Boost Model-------------------------------------
        
    def fit_XGB(self):
        self.model = XGB(learning_rate=0.06,
                    n_estimators= 100, 
                    max_depth=6,
                    random_state=100)
        self.model.fit(self.train,self.label)
        
#------------------------For training Gaussian Naive Bayes Model--------------------------
        
    def fit_GNB(self):
        self.model = GaussianNB()
        self.model.fit(self.train, self.label)
        
#-------------------------Funtion to make predictions from test dataa----------------------

    def predict(self):
        return self.model.predict(self.test)
        
#---------------------Generating Confusion Matrix and analysing sensitivity,preciion,specificity and accuracy--------------------------

def check_result(test, pred):
    cm = (confusion_matrix(y_true = test['defaulter'], y_pred = pred, labels = [0,1]))
    print(cm)
    print("sensitivity", cm[1][1]/(cm[1][1]+cm[1][0]))
    print("precision", cm[1][1]/(cm[1][1]+cm[0][1]))
    print("Specificity", cm[0][0]/(cm[0][0]+cm[0][1]))
    print("Accuracy", (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]))

def train_model():
    
    print ("training\n")
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv("../input/test.csv")
    
#---------------------------Converting defaulter attribute values to 1 if defaulting and 0 if not------------------------
    
    train['defaulter'] = train.defaulter.apply(lambda x: 0 if x==1 else 1)
    test['defaulter'] = test.defaulter.apply(lambda x: 0 if x==1 else 1)
    
    print(train.shape)
    print(test.shape)
    
    #  for validation purpose validation 
    # train, val = train_test_split(train, shuffle=True, test_size=0.3, random_state=200)
    
    md = Model(train)
    md.fit_LGBM()

    # validation
    # model_test = md.select_test_features(val)
    # pred = md.model.predict(model_test)
    # check_result(val, pred)
    
    # test
    model_test = md.select_test_features(test)
    pred = md.model.predict(model_test)
    check_result(test, pred)
    print(pred)
    
#--------------------------pickling the model---------------------------------

    pickle_out = open("model.pickle","wb")
    pickle.dump(md, pickle_out)
    pickle_out.close()
    
train_model()