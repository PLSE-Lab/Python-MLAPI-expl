#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_orig1 = pd.read_csv("/input/train.csv", sep=',')
df = data_orig1

data_orig2 = pd.read_csv("/input/test_1.csv", sep=',')
df_test = data_orig2


# In[ ]:


#replacing '?' with nan

x = float('nan')
df.replace('?', x, inplace = True)

df_test.replace('?', x, inplace = True)


# In[ ]:


df.info()


# In[ ]:


df = df.drop('ID', axis=1)
df_test = df_test.drop('ID', axis=1)


# In[ ]:


#In the below cells all categorical attributes values have been obtained so as to know the skewness of data and 
#remove the attributes accordingly


# In[ ]:


df['Worker Class'].value_counts()


# In[ ]:


df_test['Worker Class'].value_counts()


# In[ ]:


df['Enrolled'].value_counts()


# In[ ]:


df['Married_Life'].value_counts()


# In[ ]:


df_test['Married_Life'].value_counts()


# In[ ]:


df['Schooling'].value_counts()


# In[ ]:


df['MIC'].value_counts()


# In[ ]:


df['MOC'].value_counts()


# In[ ]:


df['Cast'].value_counts()


# In[ ]:


df['Hispanic'].value_counts()


# In[ ]:


df_test['Hispanic'].value_counts()


# In[ ]:


df['Sex'].value_counts()


# In[ ]:


df['MLU'].value_counts()


# In[ ]:


df_test['MLU'].value_counts()


# In[ ]:


df_test['Reason'].value_counts()


# In[ ]:


df['Reason'].value_counts()


# In[ ]:


df['Full/Part'].value_counts()


# In[ ]:


df['Tax Status'].value_counts()


# In[ ]:


df['Area'].value_counts()


# In[ ]:


df_test['Area'].value_counts()


# In[ ]:


df['State'].value_counts()


# In[ ]:


df_test['State'].value_counts()


# In[ ]:


df['Detailed'].value_counts()


# In[ ]:


df_test['Detailed'].value_counts()


# In[ ]:


df['Summary'].value_counts()


# In[ ]:


df['MSA'].value_counts()


# In[ ]:


df_test['MSA'].value_counts()


# In[ ]:


df['REG'].value_counts()


# In[ ]:


df_test['REG'].value_counts()


# In[ ]:


df['MOVE'].value_counts()


# In[ ]:


df_test['MOVE'].value_counts()


# In[ ]:


df['Live'].value_counts()


# In[ ]:


df_test['Live'].value_counts()


# In[ ]:


df['PREV'].value_counts()


# In[ ]:


df['Teen'].value_counts()


# In[ ]:


df_test['Teen'].value_counts()


# In[ ]:


df['COB FATHER'].value_counts()


# In[ ]:


df_test['COB FATHER'].value_counts()


# In[ ]:


df['COB MOTHER'].value_counts()


# In[ ]:


df_test['COB MOTHER'].value_counts()


# In[ ]:


df['COB SELF'].value_counts()


# In[ ]:


df_test['COB SELF'].value_counts()


# In[ ]:


df['Citizen'].value_counts()


# In[ ]:


df_test['Citizen'].value_counts()


# In[ ]:


df['Fill'].value_counts()


# In[ ]:


df_test['Fill'].value_counts()


# In[ ]:


df['Class'].value_counts()


# In[ ]:


#In the next few cells those attributes are dropped which are either very skewed or have large number 
#of rows with nan values


# In[ ]:


df = df.drop('COB SELF', axis=1)


# In[ ]:


df_test = df_test.drop('COB SELF', axis=1)


# In[ ]:


df = df.drop('COB MOTHER', axis=1)


# In[ ]:


df_test = df_test.drop('COB MOTHER', axis=1)


# In[ ]:


df = df.drop('COB FATHER', axis=1)


# In[ ]:


df_test = df_test.drop('COB FATHER', axis=1)


# In[ ]:


df = df.drop('Detailed', axis=1) #in best


# In[ ]:


df_test = df_test.drop('Detailed', axis=1) #in best


# In[ ]:


df = df.drop('State', axis=1)


# In[ ]:


df_test = df_test.drop('State', axis=1)


# In[ ]:


df.fillna(df.mean(), inplace = True)


# In[ ]:


df_test.fillna(df_test.mean(), inplace = True)


# In[ ]:


df = df.drop('Enrolled', axis=1)
df = df.drop('MLU', axis=1)
df = df.drop('Fill', axis=1)
df = df.drop('Reason', axis=1)


# In[ ]:


df_test = df_test.drop('Enrolled', axis=1)
df_test = df_test.drop('MLU', axis=1)
df_test = df_test.drop('Fill', axis=1)
df_test = df_test.drop('Reason', axis=1)


# In[ ]:





# In[ ]:


#Replacing nan with mode for categorical variables in both training and test data


# In[ ]:


for column in ['Worker Class', 'Married_Life', 'Schooling', 'MIC', 'MOC', 'Sex', 'Area', 'Summary', 'MSA', 
               'REG', 'MOVE', 'PREV', 'Full/Part', 'Tax Status', 'Teen', 'Hispanic', 'Citizen', 'Cast', 'Live']:
    df[column].fillna(df[column].mode()[0], inplace=True) #after best


# In[ ]:


for column in ['Worker Class', 'Married_Life', 'Schooling', 'MIC', 'MOC', 'Sex', 'Area', 'Summary', 'MSA', 
               'REG', 'MOVE', 'PREV', 'Full/Part', 'Tax Status', 'Teen', 'Hispanic', 'Citizen', 'Cast', 'Live']:
    df_test[column].fillna(df_test[column].mode()[0], inplace=True) #after best


# In[ ]:





# In[ ]:


#Label encoding for categorical variables


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['Worker Class'] = labelencoder.fit_transform(df['Worker Class'])


# In[ ]:


df_test['Worker Class'] = labelencoder.fit_transform(df_test['Worker Class'])


# In[ ]:


df['Married_Life'] = labelencoder.fit_transform(df['Married_Life'])


# In[ ]:


df_test['Married_Life'] = labelencoder.fit_transform(df_test['Married_Life'])


# In[ ]:


df['Schooling'] = labelencoder.fit_transform(df['Schooling'])


# In[ ]:


df_test['Schooling'] = labelencoder.fit_transform(df_test['Schooling'])


# In[ ]:


df['MIC'] = labelencoder.fit_transform(df['MIC'])


# In[ ]:


df_test['MIC'] = labelencoder.fit_transform(df_test['MIC'])


# In[ ]:


df['MOC'] = labelencoder.fit_transform(df['MOC'])


# In[ ]:


df_test['MOC'] = labelencoder.fit_transform(df_test['MOC'])


# In[ ]:


df['Cast'] = labelencoder.fit_transform(df['Cast'])


# In[ ]:


df_test['Cast'] = labelencoder.fit_transform(df_test['Cast'])


# In[ ]:


df['Sex'] = labelencoder.fit_transform(df['Sex'])


# In[ ]:


df_test['Sex'] = labelencoder.fit_transform(df_test['Sex'])


# In[ ]:


df['Area'] = labelencoder.fit_transform(df['Area'])


# In[ ]:


df_test['Area'] = labelencoder.fit_transform(df_test['Area'])


# In[ ]:


df['Summary'] = labelencoder.fit_transform(df['Summary'])


# In[ ]:


df_test['Summary'] = labelencoder.fit_transform(df_test['Summary'])


# In[ ]:


df['MSA'] = labelencoder.fit_transform(df['MSA'])


# In[ ]:


df_test['MSA'] = labelencoder.fit_transform(df_test['MSA'])


# In[ ]:


df['REG'] = labelencoder.fit_transform(df['REG'])


# In[ ]:


df_test['REG'] = labelencoder.fit_transform(df_test['REG'])


# In[ ]:


df['MOVE'] = labelencoder.fit_transform(df['MOVE'])


# In[ ]:


df_test['MOVE'] = labelencoder.fit_transform(df_test['MOVE'])


# In[ ]:


df['PREV'] = labelencoder.fit_transform(df['PREV'])


# In[ ]:


df_test['PREV'] = labelencoder.fit_transform(df_test['PREV'])


# In[ ]:


df['Full/Part'] = labelencoder.fit_transform(df['Full/Part'])


# In[ ]:


df_test['Full/Part'] = labelencoder.fit_transform(df_test['Full/Part'])


# In[ ]:


df['Tax Status'] = labelencoder.fit_transform(df['Tax Status'])


# In[ ]:


df_test['Tax Status'] = labelencoder.fit_transform(df_test['Tax Status'])


# In[ ]:


df['Teen'] = labelencoder.fit_transform(df['Teen'])


# In[ ]:


df_test['Teen'] = labelencoder.fit_transform(df_test['Teen'])


# In[ ]:


df['Hispanic'] = labelencoder.fit_transform(df['Hispanic']) #after best


# In[ ]:


df_test['Hispanic'] = labelencoder.fit_transform(df_test['Hispanic']) #after best


# In[ ]:


df['Citizen'] = labelencoder.fit_transform(df['Citizen']) #after best


# In[ ]:


df_test['Citizen'] = labelencoder.fit_transform(df_test['Citizen']) #after best


# In[ ]:





# In[ ]:


df.info()


# In[ ]:


df['Live'] = labelencoder.fit_transform(df['Live'])


# In[ ]:


df_test['Live'] = labelencoder.fit_transform(df_test['Live'])


# In[ ]:





# In[ ]:


#In the next few cells heat maps have been obtained and attributes have been dropped accordingly


# In[ ]:



import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


df = df.drop('NOP', axis=1)


# In[ ]:


df_test = df_test.drop('NOP', axis=1)


# In[ ]:


df = df.drop('Vet_Benefits', axis=1)


# In[ ]:


df_test = df_test.drop('Vet_Benefits', axis=1)


# In[ ]:


df = df.drop('MSA', axis=1)


# In[ ]:


df_test = df_test.drop('MSA', axis=1)


# In[ ]:





# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


df = df.drop('Weaks', axis=1) #remove


# In[ ]:


df_test = df_test.drop('Weaks', axis=1) #remove


# In[ ]:


df = df.drop('REG', axis=1)


# In[ ]:


df_test = df_test.drop('REG', axis=1)


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


df = df.drop('IC', axis = 1)


# In[ ]:


df_test = df_test.drop('IC', axis = 1)


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:





# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(30, 25))
corr = df_test.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), annot = True);


# In[ ]:


df = df.drop('Live', axis=1) #after best


# In[ ]:


df_test = df_test.drop('Live', axis=1) #after best


# In[ ]:


df =  df.drop('Tax Status', axis=1)


# In[ ]:


df_test =  df_test.drop('Tax Status', axis=1)


# In[ ]:


y=df['Class']
X=df.drop(['Class'],axis=1)
X.head()

#X_test = df_test


# In[ ]:


df_test.info()


# In[ ]:





# In[ ]:


#splitting of training data for training the models

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_test)
df_test = pd.DataFrame(np_scaled)

df_test.head()


# In[ ]:


#Training and running Naive Bayes on the test data. This method is chosen as it gives the highest AUC ROC score.

np.random.seed(42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB


# In[ ]:


nb = NB()
nb.fit(X_train,y_train)
nb.score(X_val,y_val)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[ ]:


print(classification_report(y_val, y_pred_NB))


# In[ ]:


y_pred_NB


# In[ ]:


y_ans = nb.predict(df_test)


# In[ ]:


print(y_ans.tolist())


# In[ ]:





# In[ ]:


#Training and testing Logistic Regression model for classification. Only used for analysis

from sklearn.linear_model import LogisticRegression


# In[ ]:


lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_val,y_val)


# In[ ]:


y_pred_LR = lg.predict(X_val)
print(confusion_matrix(y_val, y_pred_LR))


# In[ ]:


print(classification_report(y_val, y_pred_LR))


# In[ ]:





# In[ ]:


#Training and testing Decision Tree model for classification. Only used for analysis

from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_acc = []
test_acc = []
for i in range(1,15):
    dTree = DecisionTreeClassifier(max_depth=i)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(max_depth = 6, min_samples_split=i, random_state = 42)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[ ]:


dTree = DecisionTreeClassifier(max_depth=6, random_state = 42)
dTree.fit(X_train,y_train)
dTree.score(X_val,y_val)


# In[ ]:


y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(y_val, y_pred_DT))


# In[ ]:


print(classification_report(y_val, y_pred_DT))


# In[ ]:





# In[ ]:


#Training and testing Random Forest model for classification. Only used for analysis

from sklearn.ensemble import RandomForestClassifier


score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[ ]:


rf = RandomForestClassifier(n_estimators=13, random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)


# In[ ]:


y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)


# In[ ]:


print(classification_report(y_val, y_pred_RF))


# In[ ]:





# In[ ]:


#Obtaining AUC ROC value for each classification model to know which model gives the best prediction


# In[ ]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


roc_curve(y_val, y_pred_NB)


# In[ ]:


roc_auc_score(y_val, y_pred_NB)


# In[ ]:


roc_auc_score(y_val, y_pred_RF)


# In[ ]:


roc_auc_score(y_val, y_pred_DT)


# In[ ]:


roc_auc_score(y_val, y_pred_LR)


# In[ ]:





# In[ ]:


#Training and testing Logistic Regression model for classification. Only used for analysis

rom sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

rf_temp = RandomForestClassifier(n_estimators = 13)        #Initialize the classifier object

parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)


# In[ ]:


rf_best = RandomForestClassifier(n_estimators = 13, max_depth = 5, min_samples_split = 2)
rf_best.fit(X_train, y_train)
rf_best.score(X_val,y_val)


# In[ ]:


y_pred_RF_best = rf_best.predict(X_val)
confusion_matrix(y_val, y_pred_RF_best)


# In[ ]:


print(classification_report(y_val, y_pred_RF_best))


# In[ ]:


roc_auc_score(y_val, y_pred_RF_best)


# In[ ]:





# In[ ]:


res1 = []
for i in range(len(y_ans)):
    if y_ans[i] == 0:
        res1.append(0)
    elif y_ans[i] == 1:
        res1.append(1)


# In[ ]:


#Obtaining the final result

res2 = pd.DataFrame(res1)
final = pd.concat([data_orig2["ID"], res2], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final


# In[ ]:


final.to_csv('2015B4A70317G.csv', index = False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode())
payload = b64.decode()
html
=
'<a
download="{filename}"
href="data:text/csv;base64,{payload}"
target="_blank">{title}</a>'
html = html.format(payload=payload,title=title,filename=filename)
return HTML(html)
create_download_link(final)


# In[ ]:


#FINISH

