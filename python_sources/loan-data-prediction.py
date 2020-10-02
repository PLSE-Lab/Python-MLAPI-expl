import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Loading Raw Data
loan_data=pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

## identifying Null Values in each column and removing it
loan_data.isnull().sum()
loan_data.dropna(inplace=True)

## removing unwanted column
loan_data_filtered=loan_data.drop(['Loan_ID'],axis=1)

## Identifying Likely Categorical columns,
## You can also look at the data with column data type object and consider it a categorical.
## This approach gives a good chances of identifying categorical column even if the dataset is numerical.
likely_cat = {}
for var in loan_data_filtered.columns:
    likely_cat[var] = 1.*loan_data_filtered[var].nunique()/loan_data_filtered[var].count() < 0.05

## Mapping the data and converting to numerical structure
gender_mapper = {'Male': 0, 'Female': 1}
married_mapper = {'No': 0, 'Yes': 1}
dependents_mapper={'0': 0, '1': 1, '2': 2, '3+':3}
education_mapper={'Not Graduate': 0, 'Graduate': 1}
self_employed_mapper = {'No': 0, 'Yes': 1}
property_area_mapper={'Semiurban': 0, 'Urban': 1, 'Rural': 2}
loan_status_mapper = {'N': 0, 'Y': 1}
credit_history_mapper={0.0: 0, 1.0: 1}

loan_data_filtered['Gender'].replace(gender_mapper, inplace=True)
loan_data_filtered['Married'].replace(married_mapper, inplace=True)
loan_data_filtered['Dependents'].replace(dependents_mapper, inplace=True)
loan_data_filtered['Education'].replace(education_mapper, inplace=True)
loan_data_filtered['Self_Employed'].replace(self_employed_mapper, inplace=True)
loan_data_filtered['Property_Area'].replace(property_area_mapper, inplace=True)
loan_data_filtered['Credit_History'].replace(credit_history_mapper, inplace=True)
loan_data_filtered['Loan_Status'].replace(loan_status_mapper, inplace=True)

## Removing the outliers in the dataset, which can impact the model
zscores=np.abs(stats.zscore(loan_data_filtered))
loan_data_df=loan_data_filtered[(zscores<3).all(axis=1)]

## Perform analysis on Continuous Features and finding its correlation for model building
loan_continuous_var_data=loan_data_df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term','Loan_Status']]
X1=loan_continuous_var_data.iloc[:,0:4]
Y1=loan_continuous_var_data.iloc[:,4]
corrmat=loan_continuous_var_data.corr()
top_corr_features=corrmat.index
sns.heatmap(corrmat,annot=True)
plt.show()
## No Variables got selected as couldn't found a good correlated feature

## Performing analysis on Categorical Features
loan_categorical_var_data=loan_data_df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed','Credit_History', 'Property_Area', 'Loan_Status']]
X2=loan_categorical_var_data.iloc[:,0:7]
Y2=loan_categorical_var_data.iloc[:,7]
test=SelectKBest(score_func=chi2,k=4)
fit=test.fit(X2,Y2)
np.set_printoptions(precision=3)
print(fit.scores_)
## We get some result and select top 4 features with highest scores

## Next we build the model once we get the features
loan_features_selected_df=loan_data_df[['Gender', 'Married','Self_Employed','Credit_History', 'Property_Area', 'Loan_Status']]
X=loan_features_selected_df.iloc[:,0:5]
Y=loan_features_selected_df.iloc[:,5]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
loan_Predicit = LogisticRegression()
loan_Predicit.fit(x_train, y_train)

accuracy=loan_Predicit.score(x_test,y_test)
print("accuracy = ", accuracy * 100, "%")
pred=loan_Predicit.predict(x_test)
accuracy_score(y_test,pred)