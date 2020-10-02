#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is my first ever Machine learning model, So I've tried to use some basic and advanced algorithms to get used to them.
# 
# The notebook is trying to analyse the Titanic disaster and predict if the provided passengers in the test dataset are survived or not.

# # Import the necessary libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
from functools import reduce
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# # Exploring the training dataset 

# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# Some featues have NULL values like **Age, Cabin** and **Embarked**
# 
# As **Embarked** missing values are 2 records we will fill them with the most major values

# In[ ]:


train_df.Embarked.value_counts()


# # Create a function to fill the Null values and drop the unneeded columns

# In[ ]:


def fix_nas(dataset): 
        dataset["Embarked"].fillna("S",inplace=True)
        dataset.drop(["Cabin"],axis=1,inplace=True)
        #mean_age = round(train_df.Age.mean())
        #dataset.Age.fillna(mean_age,inplace=True)    
        dataset.drop(["Ticket"],axis=1,inplace=True)
        
        ages_mean= dataset[["Pclass","SibSp","Parch","Sex","Age"]].groupby(["Pclass","SibSp","Parch","Sex"]).agg('median').reset_index()
        
        for i,data in ages_mean.iterrows():
                dataset.loc[:,"Age"][(dataset["Pclass"]==int(data["Pclass"])) & (dataset["SibSp"]==int(data["SibSp"])) & 
                                     (dataset["Parch"]==int(data["Parch"]))& (dataset["Sex"]==data["Sex"]) & 
                                     (dataset.Age.isnull())] =data["Age"]
                
        dataset.loc[:,"Age"][(dataset.Age.isnull())] =dataset["Age"].median()
        dataset.loc[:,"Age"]=dataset.loc[:,"Age"].astype(int) 


# # Get the Titles from the passenger name

# In[ ]:


def get_title(data):
    title = data.split(".")[0].split(",")[1].replace(" ","")
    if (["Master","Mr"].count(title)!=0):
        title = "Mr"
    elif (["Major","Col","Mlle","Don","Lady","Mme","Jonkheer","theCountess","Capt","Sir"].count(title)!=0):
        title = "NA"  
    elif (["Miss","Ms"].count(title)!=0):
        title = "Ms"    
    return title


# # Get the number of family members

# In[ ]:


def get_family_members(dataset):
    fam_num = dataset["SibSp"]+dataset["Parch"]+1
    if fam_num >7:
        fam_num=8
    return fam_num


# # Check if the passenger is single or not

# In[ ]:


def is_single(dataset):
    if (dataset["Family"] >1):
        return 0
    else:
        return 1


# # Get the category of the age as following
# 1. Less than 10 years
# 2. From 10 to 20 years
# 3. From 20 to 40 years
# 4. From 40 to 60 years
# 5. More than 60 years

# In[ ]:


def get_age_category(dataset):
    #["Less than 20","20 to 40","40 to 60","More than 60"]
    if (dataset["Age"]<=10):
        return 0
    elif((dataset["Age"]>10)&(dataset["Age"]<20)):
        return 1
    elif((dataset["Age"]>=20)&(dataset["Age"]<40)):
         return 2
    elif((dataset["Age"]>=40)&(dataset["Age"]<60)):
         return 3
    elif((dataset["Age"]>=60)):
         return 4     


# # Data cleaning
# Cleaning the data using the created functions and dropping the unneeded columns

# In[ ]:


train_df.drop(["PassengerId"],axis=1,inplace=True)


fix_nas(train_df)    
fix_nas(test_df)  

train_df["Title"]=train_df.Name.apply(get_title)
train_df["Family"] = train_df.apply(get_family_members,axis=1)
train_df["Is_Single"] = train_df.apply(is_single,axis=1)
#train_df["Ticket"] = train_df.apply(get_ticket_number,axis=1)

test_df["Title"]=train_df.Name.apply(get_title)
test_df["Family"] = train_df.apply(get_family_members,axis=1)
test_df["Is_Single"] = train_df.apply(is_single,axis=1)
#test_df["Ticket"] = test_df.apply(get_ticket_number,axis=1)

gender_le = LabelEncoder()
train_df.Sex = gender_le.fit_transform(train_df["Sex"])
test_df.Sex = gender_le.transform(test_df["Sex"])
gender_le_classes = gender_le.classes_.tolist()

embarked_le = LabelEncoder()
train_df.Embarked = embarked_le.fit_transform(train_df["Embarked"])
test_df.Embarked = embarked_le.transform(test_df["Embarked"])
embarked_le_classes = embarked_le.classes_.tolist()


title_le = LabelEncoder()
train_df.Title = title_le.fit_transform(train_df["Title"])
test_df.Title = title_le.transform(test_df["Title"])
title_le_classes = title_le.classes_.tolist()

train_df["Age_Cat"]= train_df.apply(get_age_category,axis=1)
test_df["Age_Cat"]= test_df.apply(get_age_category,axis=1)
train_df["Fare_Cat"]= (pd.cut(train_df.Fare,bins=5,labels=['1','2','3','4','5'])).astype(int)
test_df["Fare_Cat"]= (pd.cut(test_df.Fare,bins=5,labels=['1','2','3','4','5']))##
train_df.drop(['Age','Fare','SibSp','Parch'],axis=1,inplace=True)
test_df.drop(['Age','Fare','SibSp','Parch'],axis=1,inplace=True)
test_df.Fare_Cat.fillna("1",inplace=True,axis=0) 
train_df.drop(["Name"],axis=1,inplace=True)


# # Data correlation
# Plotting the correlation between features in order to take a deeper look to our data and how are the features are correlated to the target value

# In[ ]:


fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(train_df[["Survived","Pclass","Sex","Age_Cat","Fare_Cat","Embarked","Is_Single","Family"]].corr(),annot=True)


# # Features Exploration
# Exploring some features in a visual way to know more about the distrubution of the key features 

# In[ ]:


survived_df = train_df[train_df.Survived==1]
non_survived_df = train_df[train_df.Survived==0]
plot_features = ["Pclass","Sex","Age_Cat","Fare_Cat","Embarked","Title","Is_Single","Family"]


# In[ ]:


#### Plot (Survived)

lab = train_df["Survived"].value_counts().keys().tolist()
#values
val = train_df["Survived"].value_counts().values.tolist()
desc = ["Survived" if i==1 else "Not Survived" for i in lab]
trace = go.Pie(labels = desc ,
               values = val ,
               marker = dict(colors =  [ 'red' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Survived / Not Survived Passengers",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)


#### Plot (Pclass)
pclass_df =pd.DataFrame(train_df["Pclass"].value_counts(sort=False)).reset_index()
pclass_df.columns=['label','val']
#lab = train_df["Pclass"].value_counts(sort=False).keys().tolist()
lab=pclass_df.label
#values
#val = train_df["Pclass"].value_counts(sort=False).values.tolist()
val=pclass_df.val
desc = ["Class "+str(i) for i in lab]
trace = go.Pie(labels = desc ,
               values = val ,
             
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Passengers Classes",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)

#### Plot (Age)
age_df =pd.DataFrame(train_df["Age_Cat"].value_counts(sort=False)).reset_index()
age_df.index=["Less than 10","10 to 20","20 to 40","40 to 60","More than 60"]
age_df.reset_index(inplace=True)
age_df.columns=['desc','label','val']
#lab = train_df["Pclass"].value_counts(sort=False).keys().tolist()
lab=age_df.desc
#values
#val = train_df["Pclass"].value_counts(sort=False).values.tolist()
val=age_df.val
desc = [str(i) for i in lab]
trace = go.Pie(labels = desc ,
               values = val ,
             
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Age Categories",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)


# # Exploration results
# * Unfortunately 62% of the passengers didn't survive.
# * Around 55% of passengers were in Class 3.
# * The passengers majority were between 20 and 40 years old.

# In[ ]:


def histogram(column) :
    trace1 = go.Histogram(x  = survived_df[column],
                        #  histnorm= "percent",
                          name = "Survived",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 ,marker_color='green'
                         ) 
    
    trace2 = go.Histogram(x  = non_survived_df[column],
                        #  histnorm = "percent",
                          name = "Not Survived",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9,marker_color='red'
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in passangers survival ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "Number of Cases",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
   
    fig  = go.Figure(data=data,layout=layout)
    if(column=='Sex'):
        fig.update_layout(
        xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1],
        ticktext = ['Female','Male']
        )
        )
    if(column=='Age_Cat'):
        fig.update_layout(
        xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1,2,3,4],
        ticktext = ["Less than 10","10 to 20","20 to 40","40 to 60","More than 60"]
        )
        ) 
    if(column=='Title'):
        fig.update_layout(
        xaxis = dict(
        tickmode = 'array',
        tickvals = [0,1,2,3,4,5],
        ticktext = title_le_classes
        )
        )      
    py.iplot(fig)
  


# In[ ]:


for idx ,feature in enumerate (plot_features, start=1) :
    histogram(feature)


# # Conclusion
# * Class 3 surviving chance was too low comparing to Class 1
# * Most of the survivals are females 
# * Surviving chance for kids less than 10 years was slightly higher than other age ranges
# * Singles were the biggest portion of the non survivals

# # Preraring Test and Submission Dataframes

# In[ ]:


gender_submission_df = test_df[["PassengerId"]]
test_df.drop("Name",axis=1,inplace=True)
test_df.drop("PassengerId",axis=1,inplace=True)


# In[ ]:


test_df.Fare_Cat = test_df.Fare_Cat.astype(int)
test_df.info()


# # Extracting Features and Target and splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , roc_auc_score,roc_curve , recall_score , precision_score ,accuracy_score,f1_score
X = train_df.iloc[:,1:]
y = train_df.iloc[:,0]
X_train, X_valid , y_train, y_valid = train_test_split(X,y,random_state=123,test_size=.3)
scores_df = pd.DataFrame(columns=["model_name","model","precision","accuracy","recall","f1score","rocauc","fpr","tpr"])


# In[ ]:


def model_scores(model_name,model,y_test,y_pred,fpr,tpr):
    precision = precision_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1score = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_pred)
    scores = {"model_name":[model_name],
              "model":[model],
              "precision":[precision],
              "accuracy":[accuracy],
              "recall":[recall],
              "f1score":[f1score],
              "rocauc":[roc_auc],
              "fpr":[fpr] ,
              "tpr":[tpr]}
    scores = (pd.DataFrame(scores))
    global scores_df
    scores_df=pd.concat([scores_df,scores],axis=0)


# # **Modeling using KNN **

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier as KNN
knn_params= {"n_neighbors":[3,4,5,6]}
knn_grid = GridSearchCV(estimator=KNN(),param_grid=knn_params,cv=4)
knn_grid.fit(X_train,y_train)
y_pred = knn_grid.predict(X_valid)
y_proba =knn_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("KNN",knn_grid.best_estimator_ ,y_valid,y_pred,fpr,tpr)


# In[ ]:



print(classification_report(y_valid,y_pred))


# # **Modeling using LogReg **

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg_param ={"C":[0.01,0.01,1,10]}
logreg_grid = GridSearchCV(estimator=LogisticRegression(),param_grid=logreg_param,cv=4)
logreg_grid.fit(X_train,y_train)
y_pred = logreg_grid.predict(X_valid)
y_proba = logreg_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("LogReg",logreg_grid.best_estimator_ ,y_valid,y_pred,fpr,tpr)

print(classification_report(y_valid,y_pred))


# # **Modeling using SVM**

# In[ ]:


from sklearn.svm import SVC
svm_param ={"C":[0.01,0.01,1,2,5,10,50,100],"gamma":[.001,.01,1,10],"kernel":["rbf"],"probability":[True]}
svm_grid = GridSearchCV(estimator=SVC(),param_grid=svm_param,cv=4)
svm_grid.fit(X_train,y_train)
y_pred = svm_grid.predict(X_valid)
y_proba = svm_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("SVM",svm_grid.best_estimator_ ,y_valid,y_pred,fpr,tpr)

print(classification_report(y_valid,y_pred))


# # **Modeling using LinearSVM**

# In[ ]:


from sklearn.svm import SVC
lin_svm_param ={"C":[0.001,0.01,0.01,1],"kernel":["linear"],"probability":[True]}
lin_svm_grid = GridSearchCV(estimator=SVC(),param_grid=lin_svm_param,cv=4)
lin_svm_grid.fit(X_train,y_train)
y_pred = lin_svm_grid.predict(X_valid)
y_proba = lin_svm_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("LinSVM",lin_svm_grid.best_estimator_ ,y_valid,y_pred,fpr,tpr)


# # **Modeling using Decision Tree**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt_param={"max_depth":[3,4,5]}
dt_grid = GridSearchCV(DecisionTreeClassifier(),param_grid=dt_param,cv=4)
dt_grid.fit(X_train,y_train)
y_pred = dt_grid.predict(X_valid)
y_proba = dt_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("DescisionTree",dt_grid.best_estimator_ ,y_valid,y_pred,fpr,tpr)
print(dt_grid.best_estimator_)


# In[ ]:


dtree.feature_importances_


# In[ ]:


fig = plt.figure(figsize=(20,10))
dtree= dt_grid.best_estimator_
dtree.feature_importances_
sns.set_context("notebook")
sns.set_style("whitegrid")
ax=sns.barplot(y=X_train.columns , x=dtree.feature_importances_)
ax.set(Title="Decision Tree Features Importance")
plt.show()


# # **Modeling using Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_param={"max_depth":[3,4,5],"n_estimators":[100,150,200],"min_samples_split":[2,3,4,5]}
rf_grid = GridSearchCV(RandomForestClassifier(),param_grid=rf_param,cv=4)
rf_grid.fit(X_train,y_train)
y_pred = rf_grid.predict(X_valid)
y_proba = rf_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("RandomForest",rf_grid.best_estimator_ ,y_valid,y_pred,fpr,tpr)


# # **Modeling using XGBoost Classifier**

# In[ ]:


import xgboost as xgb
xg_param={"objective":['binary:logistic'],"n_estimators":[10,20,50]}
xg_grid = GridSearchCV(xgb.XGBClassifier(seed=123),param_grid=xg_param,cv=4)


xg_grid.fit(X_train,y_train)
y_pred = xg_grid.predict(X_valid)
y_proba = xg_grid.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("XGBClassifierGrid",xg_grid.best_estimator_  ,y_valid,y_pred,fpr,tpr)


# In[ ]:


def plot_roc_auc(model_name,fpr,tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                    mode='lines',
                    name=model_name))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1],
                    mode='lines',
                    line = dict(color='red', width=4, dash='dash'), showlegend=False))

    fig.show()


# In[ ]:


scores_df[["model_name","precision","accuracy","recall","f1score","rocauc"]]


# # **Voting Classifier**
# Picking the top classifier and use the voting to get the best predictions

# In[ ]:


from sklearn.ensemble import VotingClassifier
voting_classifiers_list = [("XGBClassifierGrid",xg_grid.best_estimator_ ),("RandomForest",rf_grid.best_estimator_),("SVM",svm_grid.best_estimator_)]
vote_clf = VotingClassifier(estimators=voting_classifiers_list,voting="soft")
vote_clf.fit(X_train,y_train)
y_pred = vote_clf.predict(X_valid)
y_proba = vote_clf.predict_proba(X_valid)[:,1]
fpr,tpr,thr = roc_curve(y_valid,y_proba)
model_scores("Voting",vote_clf  ,y_valid,y_pred,fpr,tpr)


# # **Plotting ROC AUC **

# In[ ]:


#plot_roc_auc(model_name,fpr,tpr)
for i,rec in scores_df.iterrows():
    #print(rec.fpr)
    plot_roc_auc(rec.model_name,rec.fpr,rec.tpr)


# # Results submission

# In[ ]:


gender_submission_df["Survived"] = vote_clf.predict(test_df)
gender_submission_df.set_index(["PassengerId"],inplace=True)
gender_submission_df.to_csv('/kaggle/working/my_submission.csv')


# In[ ]:


#gender_submission_df


# In[ ]:




