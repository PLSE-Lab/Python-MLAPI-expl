
#Predective Model on Website Hits
#Kagglers advise required on this model
#I have tried Logical regression, Decision tree and random forest but got max accuracy of 37%
#However with Ensemble model finally got 42%. I feel this is much less. Let me know how can I improve accuracy on this model.
#Here is the code
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read data file with separator ;
data=pd.read_csv("../input/WebsiteHits.csv", sep=';')

#Query the first few lines of data
data.head(0)
#Replace character \N found in dataset
data=data.replace('\\N', np.nan)

#Impute missing values for number fields
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
#Identify columns with missing values
#data.isna().any()
#path_id_set, session_durantion and hits have nan or missing values
#data.info()
#si_num, hour_of_day, user_id, page, traffic_type are integers
#locale, day_of_week, path, duration, hits are strings
df_num_col=['si_num', 'hour_of_day', 'user_id', 'page', 'traffic_type', 'duration', 'hits']
data_num = data[df_num_col]
imputer=imputer.fit(data_num)
data[df_num_col]=imputer.transform(data_num)

#Encode Categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
df_cat_col =['locale', 'day_of_week', 'path']

#not supported between instances of 'str' and 'float'
data_cat=data[df_cat_col].fillna('NA')

for i in range(len(data_cat.columns)):
  data_cat.iloc[:,i] = labelencoder.fit_transform(data_cat.iloc[:,i])

#Removed onehotencoder due to low accuracy_score
#onehotencoder=OneHotEncoder(categorical_features=[0])
#data_cat=onehotencoder.fit_transform(data_cat).toarray()  
#onehotencoder=OneHotEncoder()
#data_cat=onehotencoder.fit_transform(data_cat).toarray()
  
data[df_cat_col]=data_cat

#Remove outliers
from scipy import stats
data=data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

#Independent variables
x=data.iloc[:,1:9].values
#Dependent variables
y=data.iloc[:,9].values

#Split train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#Encoding dependent variable to get rid of continous label error
y_train_encoded = labelencoder.fit_transform(y_train)
y_test_encoded = labelencoder.fit_transform(y_test)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

#Ensemble Model
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
#Classifier
def get_ensemble_models():
    rf =RandomForestClassifier(n_estimators=51,min_samples_leaf=5,min_samples_split=3)
    bagg = BaggingClassifier(n_estimators=51,random_state=42)
    extra = ExtraTreesClassifier(n_estimators=51,random_state=42)
    ada = AdaBoostClassifier(n_estimators=51,random_state=42)
    grad = GradientBoostingClassifier(n_estimators=51,random_state=42)
    classifier_list = [rf,bagg,extra,ada,grad]
    classifier_name_list = ['Random Forests','Bagging','Extra Trees','AdaBoost','Gradient Boost']
    return classifier_list,classifier_name_list
#Evaluation   
def print_evaluation_metrics(trained_model,trained_model_name,x_test,y_test_encoded):
    print('--------- Model : ', trained_model_name, ' ---------------\n')
    predicted_values = trained_model.predict(x_test)
    #print(metrics.classification_report(y_test_encoded,predicted_values))
    print("Accuracy Score : ",accuracy_score(y_test_encoded,predicted_values))
    print("---------------------------------------\n")
#Accuracy   
classifier_list, classifier_name_list = get_ensemble_models()
for classifier,classifier_name in zip(classifier_list,classifier_name_list):
    classifier.fit(x_train,y_train_encoded)
    print_evaluation_metrics(classifier,classifier_name,x_test,y_test_encoded)
