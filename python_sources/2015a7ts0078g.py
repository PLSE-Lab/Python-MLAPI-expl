#!/usr/bin/env python
# coding: utf-8

# # Data Mining Assignment 2
# 
# Author : Sourabh Majumdar, Id 2015A7TS0078G
# 
# This assignment is submitted in accordance with the course Data Mining (Semester-2) of Birla Institute of Technology and Science, Pilani,KK Birla Goa Campus

# **Importing all relevant libraries**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import preprocessing


# **load the data in the data frame**

# In[ ]:


# load the data frame to see at a glance it's relevant variables
#data_frame = pd.read_csv('train.csv') # load the data frame
data_frame = pd.read_csv('../input/dataset.csv')
data_frame.head() # see it's initial columns

# load the testinng data simulatneously
data_frame_test = pd.read_csv('test.csv')


# In[ ]:


data_frame['Class'].unique().tolist() # find the number of classes


# In[ ]:


data_frame['Class'].value_counts()


# **Handling the class imbalance issue**

# In[ ]:


data_frame_majority = data_frame[data_frame.Class==0]
data_frame_minority = data_frame[data_frame.Class==1]


# In[ ]:


data_frame_minority_unsampled = resample(data_frame_minority,replace=True,n_samples=93708,random_state=123)
data_frame_sampled = pd.concat([data_frame_majority,data_frame_minority_unsampled])


# In[ ]:


data_frame_sampled.Class.value_counts()


# In[ ]:


raw_train_data = data_frame_sampled.iloc[:,1:-1] # extract all the relevant training variables from the data frame
train_y = data_frame_sampled.iloc[:,-1] # extract the class labels from the data frame

raw_test_data = data_frame_test.iloc[:,1:]


# In[ ]:


# replaceing every '?' value in the data frame with the NaN value
list_of_cols = raw_train_data.columns.tolist()
for col in list_of_cols :
    if str(raw_train_data[col].dtype) == 'object' :
        #print('col {}'.format(col))
        raw_train_data[col].replace({'?' : np.nan},inplace=True)
    if str(raw_test_data[col].dtype) == 'object' :
        raw_test_data[col].replace({'?' : np.nan},inplace=True)


# In[ ]:


# now we need to remove the NaN values
raw_train_data.fillna(method='ffill',inplace=True)
raw_train_data.fillna(method='bfill',inplace=True)

# doing the same for test data
raw_test_data.fillna(method='ffill',inplace=True)
raw_test_data.fillna(method='bfill',inplace=True)


# In[ ]:


# we need to replace every categorical attribute with a numerical value
def get_replace_dict(column,data_frame) :
    column_list = data_frame[column].unique().tolist()
    column_dict = dict()
    for index, attribute in enumerate(column_list) :
        column_dict[attribute] = index
    return column_dict


# In[ ]:


list_of_cols = raw_train_data.columns.tolist()
for col in list_of_cols :
    if str(raw_train_data[col].dtype) == 'object' :
        replace_dict = get_replace_dict(col,raw_train_data)
        raw_train_data.replace(replace_dict,inplace=True)
        raw_test_data.replace(replace_dict,inplace=True)


# **Plotting the correlation matrix**

# In[ ]:


f, ax = plt.subplots(figsize=(15, 16))
corr = raw_train_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# **Removing all the correlated columns**

# In[ ]:


cols_to_remove_set = list()
cols_list_to_check = corr.columns.tolist()
# finding cols that are co-related
for i in range(len(cols_list_to_check)-1) :
    for j in range(i+1,len(cols_list_to_check)) :
        corr_score = corr[cols_list_to_check[i]][cols_list_to_check[j]]
        if corr_score > 0.7 and (cols_list_to_check[i] not in cols_to_remove_set):
            cols_to_remove_set.append(cols_list_to_check[i])
cols_to_remove_list = list(cols_to_remove_set)
raw_train_data.drop(cols_to_remove_list,1,inplace=True)
raw_test_data.drop(cols_to_remove_list,1,inplace=True)


# In[ ]:


raw_test_data.replace({'D36' : 0},inplace=True)


# **Preparing the train, validation and test data**

# In[ ]:


X = raw_train_data
X_test = raw_test_data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, train_y, test_size=0.20, random_state=42)


# In[ ]:


#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
print(X_test.shape)
np_scaled_test = min_max_scaler.transform(X_test)
X_test = pd.DataFrame(np_scaled_test)
print(X_test.shape)


# ## Testing with respect to all models
# 
# To Test we will be using 4 Machine Learning Models

# **#1 Using Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB
nb = NB()
nb.fit(X_train,y_train)
nb.score(X_val,y_val)
preds = nb.predict(X_val)
preds.shape
fbeta_score(preds,y_val,beta=1) # fbeta score for Naive Bayes Classifier


# In[ ]:


y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[ ]:


print(classification_report(y_val, y_pred_NB))


# **#2 Using Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_val,y_val)
preds = lg.predict(X_val)
preds.shape
fbeta_score(preds,y_val,beta=1) # fbeta score for Logistic Regression Classifier


# In[ ]:


y_pred_LG = lg.predict(X_val)
print(confusion_matrix(y_val, y_pred_LG))


# In[ ]:


print(classification_report(y_val, y_pred_LG))


# **#3 Using Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
# Random Forest Classifier


# In[ ]:


print(X_test.shape)


# In[ ]:


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


rf = RandomForestClassifier(n_estimators=15,random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)
preds = rf.predict(X_val)
preds.shape
fbeta_score(preds,y_val,beta=1) # fbeta score for Random Forest Classifier


# In[ ]:


y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)


# In[ ]:


print(classification_report(y_val, y_pred_RF))


# In[ ]:


test_preds_rf = rf.predict(X_test)
submit_data_frame = pd.DataFrame({'ID' : data_frame_test['ID'].values.tolist(),'Class' : test_preds_rf})
submit_data_frame.to_csv('submition.csv',index=False)


# **#4 Using Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

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


dTree = DecisionTreeClassifier(max_depth=12, random_state = 42)
dTree.fit(X_train,y_train)
dTree.score(X_val,y_val)
preds = dTree.predict(X_val)
preds.shape
fbeta_score(preds,y_val,beta=1) # fbeta score for Decision Tree Classifier


# In[ ]:


y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(y_val, y_pred_DT))


# In[ ]:


print(classification_report(y_val, y_pred_DT))


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(data_frame)

