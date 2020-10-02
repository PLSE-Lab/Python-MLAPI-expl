# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

def load_attriion_data():
    return pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")  

attrition_rates=load_attriion_data()


# create separate dataframe for numeric attributes
attrition_rates_num=pd.DataFrame(attrition_rates.select_dtypes(include=['int64']))
#Dropping irrelevant columns
attrition_rates_num=attrition_rates_num.drop("EmployeeNumber",axis=1)
attrition_rates_num=attrition_rates_num.drop("EmployeeCount",axis=1)
attrition_rates_num=attrition_rates_num.drop("StandardHours",axis=1)

#attrition_rates_num.info()

import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
yearsCurrentRole_idx=20,
yearsAtCompany_idx=19,
yearsSinceLastPromotion_idx=21,
totalWorkingYears_idx=16,
numCompaniesWorked_idx=11,
jobInvolvement_idx=6
jobLevel_idx=7
envSatisfaction_idx=4
relationSatisfaction_idx=14
jobSatisfaction_idx=8

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        YearsCurrentRole_per_yearsAtCompany=X[:,yearsCurrentRole_idx]/X[:,yearsAtCompany_idx]
        YearsSinceLastPromotion_per_yearsAtCompany=X[:,yearsSinceLastPromotion_idx]/X[:,yearsAtCompany_idx]
        YearsAtCompany_per_TotalWorkingYears=X[:,yearsAtCompany_idx]/X[:,totalWorkingYears_idx]
        NumCompaniesWorked_per_totalWorkingYears=X[:,numCompaniesWorked_idx]/X[:,totalWorkingYears_idx]
        JobInvolvement_per_JobLevel=X[:,jobInvolvement_idx]/X[:,jobLevel_idx]
        OverAllSatisfaction=X[:,envSatisfaction_idx]+X[:,relationSatisfaction_idx]+X[:,jobSatisfaction_idx]
        return np.c_[X,YearsCurrentRole_per_yearsAtCompany,YearsSinceLastPromotion_per_yearsAtCompany,YearsAtCompany_per_TotalWorkingYears,NumCompaniesWorked_per_totalWorkingYears,JobInvolvement_per_JobLevel,OverAllSatisfaction]
    
attr_adder=CombinedAttributesAdder()
attrition_rates_num_extra_attribs=attr_adder.transform(attrition_rates_num.values)
#creating dataframe out of extra attribs
attrition_rates_num_extra_attribs_df=pd.DataFrame(attrition_rates_num_extra_attribs)

#Extract separate dataframe for categorical attribs
attrition_rates_cat=pd.DataFrame(attrition_rates.select_dtypes(include=['object']))
  
#Encode Categorical Attribs to get Attrition attribute
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# apply "le.fit_transform"
attrition_rates_cat_encoded = attrition_rates_cat.apply(le.fit_transform)

#attrition_rates_num_extra_attribs_df.info()


#add Attrition and extra attribs to existing numerical dataframe 
attrition_rates_num["Attrition"]=attrition_rates_cat_encoded["Attrition"]
attrition_rates_num["YearsCurrentRole_per_yearsAtCompany"]=attrition_rates_num_extra_attribs_df[23]
attrition_rates_num["YearsSinceLastPromotion_per_yearsAtCompany"]=attrition_rates_num_extra_attribs_df[24]
attrition_rates_num["YearsAtCompany_per_TotalWorkingYears"]=attrition_rates_num_extra_attribs_df[25]
attrition_rates_num["NumCompaniesWorked_per_totalWorkingYears"]=attrition_rates_num_extra_attribs_df[26]
attrition_rates_num["JobInvolvement_per_JobLevel"]=attrition_rates_num_extra_attribs_df[27]
attrition_rates_num["OverAllSatisfaction"]=attrition_rates_num_extra_attribs_df[28]

# fill blank values 
from sklearn.preprocessing import Imputer
imputer=Imputer(strategy="median")
imputer.fit(attrition_rates_num)
imputer.statistics_
X=imputer.transform(attrition_rates_num)

attrition_rates_num_tr=pd.DataFrame(X,columns=attrition_rates_num.columns)


#Applying corr_matrix to numerical attributes
corr_matrix=attrition_rates_num_tr.corr()
corr_matrix["Attrition"].sort_values(ascending=False)

attrition_rates_num_tr=attrition_rates_num_tr.drop(["NumCompaniesWorked","JobInvolvement_per_JobLevel","YearsAtCompany_per_TotalWorkingYears","HourlyRate","PerformanceRating","EnvironmentSatisfaction","RelationshipSatisfaction","JobSatisfaction"],axis=1)
#attrition_rates_num_tr.info()

attrition_rates_cat=attrition_rates_cat.drop(["Attrition","Over18"],axis=1)

attrition_rates_cat=pd.get_dummies(attrition_rates_cat)
#attrition_rates_cat.info()

attrition_rates_num_tr=pd.concat([attrition_rates_num_tr,attrition_rates_cat],axis=1)
#attrition_rates_num_tr.info()
attrition_rates_num_tr.drop("JobRole_Research Scientist",axis=1)  

#replacing any infinite values with median
attrition_rates_num_tr=attrition_rates_num_tr.replace([np.inf, -np.inf], np.nan)
attrition_rates_num_tr=attrition_rates_num_tr.fillna(attrition_rates_num_tr.median())

#Applying standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(attrition_rates_num_tr)
attrition_rates_num_tr_arr=scaler.transform(attrition_rates_num_tr)

corr_matrix=pd.DataFrame(attrition_rates_num_tr_arr,columns=attrition_rates_num_tr.columns).corr()
corr_matrix["Attrition"].sort_values(ascending=False)    

#Splitting train and test data
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(attrition_rates_num_tr,test_size=0.2,random_state=42)
X_train=train_set.drop(["Attrition"],axis=1).values
y_train=train_set["Attrition"].values

X_test=test_set.drop(["Attrition"],axis=1).values
y_test=test_set["Attrition"].values

#Applying SGD classifier
from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
print("Accuracy is",sum(y_pred==y_test)/len(y_pred))

#Accuracy cannot be considered as the only criteria for judging a  model
from sklearn.model_selection import cross_val_predict
y_train_pred=cross_val_predict(sgd_clf,X_train,y_train,cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train,y_train_pred)

from sklearn.preprocessing import PolynomialFeatures
poly_features=PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly_features.fit_transform(X_train)

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_poly,y_train)
lin_reg.intercept_, lin_reg.coef_


train_errors, val_errors = [], []
from sklearn.metrics import mean_squared_error
def plot_learning_curves(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for m in range(1, len(X_train)):
        lin_reg.fit(X_train[:m], y_train[:m])
        y_train_predict = lin_reg.predict(X_train[:m])
        y_test_predict = lin_reg.predict(X_test)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_test_predict, y_test))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()
    
    
X=attrition_rates_num_tr.drop(["Attrition"],axis=1).values
y=attrition_rates_num_tr["Attrition"].values
from sklearn.pipeline import Pipeline
polynomial_regression = Pipeline(
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("sgd_reg", LinearRegression()),
    )
plot_learning_curves(polynomial_regression, X, y)