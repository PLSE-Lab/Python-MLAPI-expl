import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
data_set = pd.read_csv("../input/HR_comma_sep.csv",",")
#print(data_set.info())
# print(data_set.head(10))

# data_set.pivot_table('last_evaluation', 'number_project','left').plot(kind='bar', stacked=True)
# print(data_set.pivot_table('last_evaluation', 'number_project', 'left'))
# plt.show()



#F.E.:
# data_set['Hours|Proj'] = data_set["average_montly_hours"]/data_set["number_project"]
#print(data_set["number_project"].unique())

#proj:
# data_set["Have7proj"] = data_set["number_project"].apply(lambda x: 1 if x == 7 else 0 ) #24%
data_set["Have7or6proj"] = data_set["number_project"].apply(lambda x: 1 if x == 7 or x ==6 else 0 ) #30%
# data_set["Have6proj"] = data_set["number_project"].apply(lambda x: 1 if x == 6 else 0 ) #22%
# data_set["Have4proj"] = data_set["number_project"].apply(lambda x: 1 if x == 4 else 0 ) #22%
data_set["Have4or3proj"] = data_set["number_project"].apply(lambda x: 1 if x == 4 or x ==3 else 0 ) #48% !!!
# data_set["Have3proj"] = data_set["number_project"].apply(lambda x: 1 if x == 3 else 0 ) # 31%
data_set["Have2proj"] = data_set["number_project"].apply(lambda x: 1 if x == 2 else 0 ) #!!! 43%
data_set.drop('number_project', axis=1, inplace=True)

#average_montly_hours:
# print(data_set["average_montly_hours"].max())
# print(data_set["average_montly_hours"].min())
data_set["hour>273"] = data_set["average_montly_hours"].apply(lambda x: 1 if x>273 else 0)
data_set.drop('average_montly_hours', axis=1, inplace=True)

#time_spend_company:
# print(data_set["time_spend_company"].unique())
data_set["time_sp_comp2"] = data_set["time_spend_company"].apply(lambda x: 1 if x==2 else 0)#27%
# data_set["time_sp_comp3"] = data_set["time_spend_company"].apply(lambda x: 1 if x==3 else 0)
# data_set["time_sp_comp4"] = data_set["time_spend_company"].apply(lambda x: 1 if x==4 else 0) 
data_set["time_sp_comp5"] = data_set["time_spend_company"].apply(lambda x: 1 if x==5 else 0)#25%
# data_set["time_sp_comp6"] = data_set["time_spend_company"].apply(lambda x: 1 if x==6 else 0)
# data_set["time_sp_comp7"] = data_set["time_spend_company"].apply(lambda x: 1 if x==7 else 0)
# data_set["time_sp_comp8"] = data_set["time_spend_company"].apply(lambda x: 1 if x==8 else 0)
# data_set["time_sp_comp10"] = data_set["time_spend_company"].apply(lambda x: 1 if x==10 else 0)
data_set.drop('time_spend_company', axis=1, inplace=True)


#last_evaluation:
data_set["le>07"] = data_set["last_evaluation"].apply(lambda x: 1 if x>0.68 and x<0.80 else 0)
data_set.drop('last_evaluation', axis=1, inplace=True)
data_set.drop('Work_accident', axis=1, inplace=True)
data_set.drop('promotion_last_5years', axis=1, inplace=True)



#sales:
# print(data_set["sales"].unique())
data_set['salesID'] = pd.factorize(data_set.sales)[0]
data_set.drop('sales', axis=1, inplace=True)

print(data_set["salary"].unique())
# for i in set(data_set["salary"]):
#     g ="salary:"+str(i)
#     data_set.insert(data_set.columns.size-1,g,data_set['salary']== i)
# data_set = data_set.drop(["salary"], axis=1)
# data_set["SalaryNoLow"] = data_set["salary"].apply(lambda x: 1 if x=='high' or x=='medium' else 0)
data_set['saleryID'] = pd.factorize(data_set.salary)[0]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#correlation matrix:
correlation = data_set.corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(correlation, vmax=1, square=True,  annot=True ) 
plt.show()

# Important fields:
fields =  (np.extract(abs((correlation.left))>0.15, (correlation.columns)))
fields = np.delete(fields,1)
print("Important fields: ",fields)

# Formation of test sets:
train = data_set[fields]
target = data_set["left"]
(x_train,x_test,y_train,y_test) = train_test_split(train,target,test_size = 0.15, random_state=9)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

#create model:
model_gb = GaussianNB()
model_gb.fit(x_train,y_train)

rF = RandomForestRegressor(n_estimators=70,max_features = 9,max_depth=11)
rF.fit(x_train,y_train)

model_knc = KNeighborsClassifier(n_neighbors = 12)
model_knc.fit(x_train,y_train)

model_lr = LogisticRegression(penalty='l1', tol=0.01)
model_lr.fit(x_train,y_train)

model_svc = svm.SVC()
model_svc.fit(x_train,y_train)

#results:
print(model_gb.score(x_test, y_test)) 
print(rF.score(x_test,y_test))
print(model_knc.score(x_test,y_test))
print(model_lr.score(x_test,y_test))
print(model_svc.score(x_test,y_test))