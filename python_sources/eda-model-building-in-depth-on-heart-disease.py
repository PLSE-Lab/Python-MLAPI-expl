#!/usr/bin/env python
# coding: utf-8

# # EDA + Model building in Depth On Heart Diseases

# ### Objective : 
# Main obejctive behind this notebook is to give an idea along with workflow of Machine Learning Processes.
# 
# Starting from **Getting data informaion to Exploratory Data Analysis, Data Manipulation, Building and then Validation of Model.**
# 
# I am trying to keep it as **simple** as i can so that newbie can also understand the workflow.
# 
# If you learn anything useful from this notebook then **Give Upvote :)
# 

# * ## Contents of the Notebook:
# 
# #### Part1: Exploratory Data Analysis(EDA)
# 1) Analysis of the features.
# 
# 2) Finding any relations or trends considering multiple features.
# #### Part2: Data Cleaning:
# 1) Adding any few features if any.
# 
# 2) Removing redundant features.
# 
# 3) Converting features into suitable form for modeling.
# #### Part3: Predictive Modeling
# 1) Running Basic Algorithms.
# 
# 2) Cross Validation.
# 
# 3) Important Features Extraction.
# 
# 4) Plotting ROC Curve, Precision/Recall Curve, AUC
# 
# 5) Model Comparison (Accuracy + F1 Score)
# 
# 6) Ensemble model

# ## Part1: Exploratory Data Analysis(EDA)

# ### Attribute Information:
#     1. age                                                  2. sex
#     3. chest pain type (4 values)                           4. resting blood pressure
#     5. serum cholestoral in mg/dl                           6. fasting blood sugar > 120 mg/dl
#     7. resting electrocardiographic results (values 0,1,2)  8. maximum heart rate achieved
#     9. exercise induced angina                              10. oldpeak = ST depression induced by exercise relative to rest
#     11. the slope of the peak exercise ST segment           12. number of major vessels (0-3) colored by flourosopy
#     13. thal: 3 = normal; 6 = fixed defect; 
#             7 = reversable defect                           14. target column

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
color=sns.color_palette()


# In[ ]:


data=pd.read_csv('../input/heart.csv')
data.head()


# ### How many people are suffering from Heart Disease ?

# In[ ]:


fig,ax=plt.subplots(1, 2, figsize = (14,5))
sns.countplot(data=data, x='target', ax=ax[0],palette='Set2')
ax[0].set_xlabel("Disease Count \n [0]->No [1]->Yes")
ax[0].set_ylabel("Count")
ax[0].set_title("Heart Disease Count")
data['target'].value_counts().plot.pie(explode=[0.1,0.0],autopct='%1.1f%%',ax=ax[1],shadow=True, cmap='Greens')
plt.title("Heart Disease")


# From above graph we can say that more than half of the population suffering from Heart Disease with parcentage of 54.5%. 

# Let's explore more feature to get more insight from dataset

# ### Feature analysis

# ### a)->SEX (Category)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='sex',data=data,hue='target',palette='Set1',ax=ax[0])
ax[0].set_xlabel("0 ->Female , 1 ->Male")
data.sex.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0], cmap='Reds')
ax[1].set_title("0 ->Female , 1 -> Male")


# **This is interesting**

# **Number of Women suffering from Heart Disease are more than Men** but **Men population is more than Women**. We will use these insight for our model developement.

# Let's explore other feature

# ### b)-> fasting blood sugar (Fbs) (Category)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='fbs',data=data,hue='target',palette='Set3',ax=ax[0])
ax[0].set_xlabel("0-> fps <120 , 1-> fps>120",size=12)
data.fbs.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0],cmap='Oranges')
ax[1].set_title("0 -> fps <120 , 1 -> fps>120",size=12)


# This is weird **People having fps < 120 have more chance of having Heart Disease than people havnig fps >120**

# This insight will also be useful for our model

# Let's explore more

# ### c) resting electrocardiographic results (values 0,1,2) (Category)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='restecg',data=data,hue='target',palette='Set3',ax=ax[0])
ax[0].set_xlabel("resting electrocardiographic",size=12)
data.restecg.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,
                                     explode=[0.005,0.05,0.05],cmap='Blues')
ax[1].set_title("resting electrocardiographic",size=12)


# An electrocardiogram (ECG) is a test which measures the electrical activity of your heart to show whether or not it is working normally. An ECG records the heart's rhythm and activity on a moving strip of paper or a line on a screen. -> **THANKS GOOGLE**

# With above graph as a refrence we can **if resting electrocardiographic is 1 then person have more chances of suffering from Heart Disease**

# ### d) the slope of the peak exercise ST segment (slope)(Category)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='slope',data=data,hue='target',palette='Set1',ax=ax[0])
ax[0].set_xlabel("peak exercise ST segment",size=12)
data.slope.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,explode=[0.005,0.05,0.05],cmap='Blues')

ax[1].set_title("peak exercise ST segment ",size=12)


# Feature (the peak exercise ST segment slope) has three symbolic values (flat, up sloping, downsloping)
# 

# Therefore **People having up sloping are more prone to Heart Disease than flat and downsloping**. This is useful for our model

# ### e)  number of major vessels colored by flourosopy (category)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='ca',data=data,hue='target',palette='Set2',ax=ax[0])
ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)
data.ca.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Oranges')
ax[1].set_title("number of major vessels colored by flourosopy",size=12)


# Since **Fluoroscopy** use to  produce x-ray which will makes possible to see internal organs in motion. Fluoroscopy uses x-ray to produce real-time video images.

# **THIS seems to be important info from data** 

# ### f) thal 3 = normal, 6 = fixed defect, 7 = reversable defect (category feature)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='thal',data=data,hue='target',palette='Set2',ax=ax[0])
ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)
data.thal.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Greens')
ax[1].set_title("number of major vessels colored by flourosopy",size=12)


# ### g) Chest Pain (category)

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(14,5))
sns.countplot(x='cp',data=data,hue='target',palette='Set3',ax=ax[0])
ax[0].set_xlabel("Chest Pain")
data.cp.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0.01,0.01,0.01,0.01],shadow=True, cmap='Blues')
ax[1].set_title("Chest pain")


# 4-Levels of chest pain given in data where 3 is highest

# **People who are on 3rd level of chest pain are very less as compared to people who are on 2nd level of chest pain**. 
# I guess **Most people died after 2nd level of chest pain**

# This insight will be very usefull for our model

# ### Let's explore Continuous data now with categorical and ordinal data

# ### h) Trestbps (continuous feature)

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(14,10))
sns.boxplot(y='trestbps',data=data,x='sex',hue='target',palette='Set2',ax=ax[0,0])
ax[0,0].set_title("Trestbps V/S Sex")
sns.factorplot(y='trestbps',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set2')
ax[0,1].set_title("Trestbps V/S Chest Pain")
sns.violinplot(y='trestbps',data=data,x='exang',hue='target',ax=ax[1,0],palette='Set2')
ax[1,0].set_title("Trestbps V/S Exang")
sns.swarmplot(y='trestbps',data=data,x='ca',hue='target',ax=ax[1,1],palette='Set2')
ax[1,1].set_title("Trestbps V/S CA (Major Vessel Coloured)")


# Based on above analysis we can say that Gender plays minor role with respect to Blood Pressure (trestbps). But **Chest Pain play's Vital Role** . As Chest pain increases Blood Pressure will also increases along with chances of Heart Diseases.

# Lets Move to other features

# ### i) Cholestrol (continuous feature)

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(14,10))
sns.boxplot(y='chol',data=data,x='sex',hue='target',palette='Set3',ax=ax[0,0])
ax[0,0].set_title("Cholestrol V/S Sex")
sns.boxplot(y='chol',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set3')
ax[0,1].set_title("Cholestrol V/S Chest Pain")
sns.swarmplot(y='chol',data=data,x='thal',hue='target',ax=ax[1,0],palette='Set3')
ax[1,0].set_title("Cholestrol V/S Thal")


# Female have **higher cholestrol level** than Men. Chances of **Heart Diseases** decreases with decrease in **Cholestrol level**.

# With 2nd Graph (Cholestrol V/S Chest Pain) we can say that if **cholestrol is less than 240 approx** and **Chest pain is at level 3~4 then chances of having heart diseases are higher**

# ### j) Oldpeak (continuous feature)

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(14,10))
sns.boxplot(y='oldpeak',data=data,x='sex',hue='target',palette='Set1',ax=ax[0,0])
ax[0,0].set_title("oldpeak V/S Sex")
sns.boxplot(y='oldpeak',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set1')
ax[0,1].set_title("oldpeak V/S Chest Pain")
sns.swarmplot(y='oldpeak',data=data,x='thal',hue='target',ax=ax[1,0],palette='Set1')
ax[1,0].set_title("oldpeak V/S Thal")
sns.factorplot(y='oldpeak',data=data,x='ca',hue='target',ax=ax[1,1],palette='Set1')
ax[1,1].set_title("oldpeak V/S CA")


# Based on above plots we can comclude that if Old peak is less then people will have more chances of **having heart diseases**

# ### Let's do some Advanced EDA now 

# ### Distribution of each features

# In[ ]:


fig,ax=plt.subplots(4,3,figsize=(15,15))
for i in range(12):
    plt.subplot(4,3,i+1)
    sns.distplot(data.iloc[:,i],kde=True, color='blue')


# ### PLOT WITH RESPECT TO MEAN OF EACH ROW

# In[ ]:


fig,ax=plt.subplots(1,1,figsize=(15,5))
features = data.columns
sns.distplot(data[features].mean(axis=1),kde=True,bins=30,color='red')


# ### Plot with respect to Standard Deviation per Row

# In[ ]:


fig,ax=plt.subplots(1,1,figsize=(15,5))
features = data.columns
sns.distplot(data[features].std(axis=1),kde=True,bins=30,color='green')


# ## Part2: Data Cleaning

# ### Handling Missing Data

# In[ ]:


fig,ax=plt.subplots(figsize=(15,5))
sns.heatmap(data.isnull(), annot=True)


# THAT'S Good. **No Missing Values**

# Let's see **correlation between different features**

# In[ ]:


fig=plt.figure(figsize=(18,18))
sns.heatmap(data.corr(), annot= True, cmap='Blues')


# Based on above heatmap we can say **most of the features are in high correlation with each other**.

# ### Now let's work on each feature conversion

# In[ ]:


data.sex=data.sex.astype('category')
data.cp=data.cp.astype('category')
data.fbs=data.fbs.astype('category')
data.restecg=data.restecg.astype('category')
data.exang=data.exang.astype('category')
data.ca=data.ca.astype('category')
data.slope=data.slope.astype('category')
data.thal=data.thal.astype('category')


# In[ ]:


data_label=data['target']
del data['target']
data_label=pd.DataFrame(data_label)


# ### Creating dummies variables

# In[ ]:


data=pd.get_dummies(data,drop_first=True)
data.head(),data_label.head()


# ### Normalization (To get value b/w 0 and 1)

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
data_scaled=MinMaxScaler().fit_transform(data)
data_scaled=pd.DataFrame(data=data_scaled, columns=data.columns)


# In[ ]:


data_scaled.head()


# ## Part3: Predictive Modeling

# ### Splittting data into test and train set

# In[ ]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.20,
                                             stratify=data_label,random_state=9154)


# ### Importing ML libraries

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import StackingClassifier Need to update sklearn to use inbuilt stacking classifier
from sklearn.ensemble import VotingClassifier


# ### Evalutation metrics to check model performance

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


# ### Cross validation helper function

# In[ ]:


def CrossVal(dataX,dataY,mode,cv=3):
    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='accuracy')
    return(np.mean(score))


# ### Function to plot ROC and Precision Recall Curve 

# In[ ]:


def plotting(true,pred):
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    precision,recall,threshold = precision_recall_curve(true,pred[:,1])
    ax[0].plot(recall,precision,'g--')
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))
    fpr,tpr,threshold = roc_curve(true,pred[:,1])
    ax[1].plot(fpr,tpr)
    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))
    ax[1].plot([0,1],[0,1],'k--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')


# ### a)Stochastic Gradient Descent 

# In[ ]:


sgd=SGDClassifier(tol=1e-10, random_state=23,loss='log', penalty= "l2", alpha=0.2)
score_sgd=CrossVal(Xtrain,Ytrain,sgd)
print("Accuracy is : ",score_sgd)
sgd.fit(Xtrain,Ytrain)
plotting(Ytest,sgd.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,sgd.predict(Xtest)), annot= True, cmap='Oranges')
sgd_f1=f1_score(Ytest,sgd.predict(Xtest))
plt.title('F1 Score = {}'.format(sgd_f1))


# ### b) K-Nearest Neighbors

# In[ ]:


k=KNeighborsClassifier(algorithm='auto',n_neighbors= 19)
score_k=CrossVal(Xtrain,Ytrain,k)
print("Accuracy is : ",score_k)
k.fit(Xtrain,Ytrain)
plotting(Ytest,k.predict_proba(Xtest))


fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,k.predict(Xtest)), annot= True, cmap='Reds')
k_f1=f1_score(Ytest,k.predict(Xtest))
plt.title('F1 Score = {}'.format(k_f1))


# ### c) Logistic Regression

# In[ ]:


lr=LogisticRegression(class_weight='balanced', tol=1e-10)
score_lr=CrossVal(Xtrain,Ytrain,lr)
print("Accuracy is : ",score_lr)
lr.fit(Xtrain,Ytrain)
plotting(Ytest,lr.predict_proba(Xtest))


fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,lr.predict(Xtest)), annot= True, cmap='Greens')
lr_f1=f1_score(Ytest,lr.predict(Xtest))
plt.title('F1 Score = {}'.format(lr_f1))


# ### d) Decision Tree Classifier

# In[ ]:


dtc=DecisionTreeClassifier(max_depth=6)
score_dtc=CrossVal(Xtrain,Ytrain,dtc)
print("Accuracy is : ",score_dtc)
dtc.fit(Xtrain,Ytrain)
plotting(Ytest,dtc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,dtc.predict(Xtest)), annot= True, cmap='Blues')

dtc_f1=f1_score(Ytest,dtc.predict(Xtest))
plt.title('F1 Score = {}'.format(dtc_f1))


# ### e) Support vector machine

# In[ ]:


svc=SVC(C=0.2,probability=True,kernel='rbf',gamma=0.1)
score_svc=CrossVal(Xtrain,Ytrain,svc)
print("Accuracy is : ",score_svc)
svc.fit(Xtrain,Ytrain)
plotting(Ytest,svc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,svc.predict(Xtest)), annot= True, cmap='Greys')
svc_f1=f1_score(Ytest,svc.predict(Xtest))
plt.title('F1 Score = {}'.format(svc_f1))


# ### f) Random Forest Classifier

# In[ ]:


rf=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=97)
score_rf= CrossVal(Xtrain,Ytrain,rf)
print('Accuracy is:',score_rf)
rf.fit(Xtrain,Ytrain)
plotting(Ytest,rf.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,rf.predict(Xtest)), annot= True, cmap='Oranges')

rf_f1=f1_score(Ytest,rf.predict(Xtest))
plt.title('F1 Score = {}'.format(rf_f1))


# ### g) Extra Trees Classifier

# In[ ]:


etc=ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=2)
score_etc= CrossVal(Xtrain,Ytrain,etc)
print('Accuracy is:',score_etc)
etc.fit(Xtrain,Ytrain)
plotting(Ytest,etc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,etc.predict(Xtest)), annot= True, cmap='Greens')

etc_f1=f1_score(Ytest,etc.predict(Xtest))
plt.title('F1 Score = {}'.format(etc_f1))


# ### h) Ada Boost Classifier

# In[ ]:


abc=AdaBoostClassifier(sgd,n_estimators=100, random_state=343, learning_rate=0.012)
score_ada= CrossVal(Xtrain,Ytrain,abc)
print('Accuracy is:',score_ada)
abc.fit(Xtrain,Ytrain)
plotting(Ytest,abc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,abc.predict(Xtest)), annot= True, cmap='Reds')

abc_f1=f1_score(Ytest,abc.predict(Xtest))
plt.title('F1 Score = {}'.format(abc_f1))


# ### i) Gradient Boosting Classifier 

# In[ ]:


gbc=GradientBoostingClassifier(n_estimators=100, random_state=43, learning_rate = 0.01)
score_gbc= CrossVal(Xtrain,Ytrain,gbc)
print('Accuracy is:',score_gbc)
gbc.fit(Xtrain,Ytrain)
plotting(Ytest,gbc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,gbc.predict(Xtest)), annot= True, cmap='Blues')

gbc_f1=f1_score(Ytest,gbc.predict(Xtest))
plt.title('F1 Score = {}'.format(gbc_f1))


# ### j) Bagging Classifier 

# In[ ]:


bc=BaggingClassifier(lr,max_samples=23, bootstrap=True, n_jobs= -1)
score_bc= CrossVal(Xtrain,Ytrain,gbc)
print('Accuracy is:',score_bc)
bc.fit(Xtrain,Ytrain)
plotting(Ytest,bc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,bc.predict(Xtest)), annot= True, cmap='Greys')

bc_f1=f1_score(Ytest,bc.predict(Xtest))
plt.title('F1 Score = {}'.format(bc_f1))


# ### IMOPORTANT FEATURE
# 

# In[ ]:


fig= plt.figure(figsize=(10,10))
important=pd.Series(rf.feature_importances_, index=Xtrain.columns)
sns.set_style('whitegrid')
important.sort_values().plot.barh()
plt.title('Feature Importance')


# So after modeling we can say that **thalach, thal_2 and oldeak** are most important feature in prediction

# ### Model accuracy plot

# In[ ]:


model_accuracy = pd.Series(data=[score_sgd, score_k, score_lr, score_dtc, score_svc, score_rf, score_etc, 
                           score_ada, score_gbc, score_bc], 
                           index=['Stochastic GD','KNN','logistic Regression','decision tree', 'SVM', 'Random Forest',
                            'Extra Tree', 'Ada Boost' , 'Gradient Boost','Bagging Classfier'])
fig= plt.figure(figsize=(8,8))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accracy')


# If you consider accuracy then Stochastic, Logistic Regression and K-Nearest Neighbours are doing better than other ML algorithms. 

# But for Classification task **ACCURACY is not important**.  Instead of accuracy model should be judged on basis of **AUC (Area under curve), ROC CURVE, High Precision and High Recall values**. **F1 score** also play imporant role which is equals to **2/(1/precision + 1/Recall) score**

# #### AS FURTHER COMPARISON BETWEEN MODEL PLOTTING F1 SCORE

# In[ ]:


model_f1_score = pd.Series(data=[sgd_f1, k_f1, lr_f1, dtc_f1, svc_f1, rf_f1, etc_f1, 
                           abc_f1, gbc_f1, bc_f1], 
                           index=['Stochastic GD','KNN','logistic Regression','decision tree', 'SVM', 'Random Forest',
                                'Extra Tree', 'Ada Boost' , 'Gradient Boost', 'Bagging Classfier'])
fig= plt.figure(figsize=(8,8))
model_f1_score.sort_values().plot.barh()
plt.title('Model F1 Score Comparison')


# ### Now Let's combine best classifier from above results and feed into Voting Classifier

# ## K) Voting Classifier 

# In[ ]:


vc=VotingClassifier(estimators=[('knn',k),('SGD',sgd),('lr',lr)],
                    voting='soft')
score_vc= CrossVal(Xtrain,Ytrain,vc)
print('Accuracy is:',score_vc)
vc.fit(Xtrain,Ytrain)
plotting(Ytest,vc.predict_proba(Xtest))

fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,vc.predict(Xtest)), annot= True, cmap='Greys')

vc_f1=f1_score(Ytest,vc.predict(Xtest))
plt.title('F1 Score = {}'.format(vc_f1))


# ### By combining KNN + SGD + Logistic Regression we got 81.0 F1 score with 87 AUC

# This is the power of ensembling. In this case Voting classifier is classifing instance based on Votes. For eg. Out of 3 classifier suppose if 2 classifier voting for postive class and other one is for negative class then Voting classifier will choose positive class for that instance.   

# #### MOST IMPORTANT NOTE HERE IS :Voting Classifier will perform better if all of the classifier which are choosen for voting are making different mistakes. So Voting Classifier will not make that mistake by choosing most voted class.

# # Stacking models

# ![](https://cdn-images-1.medium.com/max/800/0*GHYCJIjkkrP5ZgPh.png)

# Now we will stack our best models (base models) then after that we will use one meta model which will uses predictions made by base models and try to improve evaluation metrics.

# ### Now Sklearn also provide StackingClassifier() as well as StackingRegressor() under ensemble module. Update Sklearn to use those libraries

# In[ ]:


from sklearn.model_selection import StratifiedKFold
k=StratifiedKFold(n_splits= 5, shuffle=False, random_state=6)


# In[ ]:


def stacking(model, Xtrain, Ytrain, Xtest, name):
    prediction_train = np.zeros(len(Xtrain))
    prediction_test = np.zeros((len(Xtest)))
    for train_index, test_index in k.split(Xtrain,Ytrain):
        trainset, trainset_label =  Xtrain.iloc[train_index,:], Ytrain.iloc[train_index]
        cv_set, cv_label =  Xtrain.iloc[test_index,:], Ytrain.iloc[test_index]
        
        model.fit(trainset, trainset_label)
        prediction_train[test_index] = model.predict(cv_set)
        
    prediction_test = model.predict(Xtest)
    return (pd.DataFrame({name:prediction_train}),pd.DataFrame({name:prediction_test}))                               


# In[ ]:


# stacking SGD , Logistic regression, voting classifier
sgd_train, sgd_test = stacking(sgd, Xtrain, Ytrain, Xtest, 'sgd')
lr_train, lr_test = stacking(lr, Xtrain, Ytrain, Xtest, 'logistic')
vc_train, vc_test = stacking(vc, Xtrain, Ytrain, Xtest, 'voting') 


# Now we have to combine the prediction made by our best classifier which will use to feed to meta classifier

# In[ ]:


# Combining prediction made by all the three classifiers
trainset = pd.concat([sgd_train,lr_train,vc_train],axis=1)
testset = pd.concat([sgd_test,lr_test,vc_test],axis=1)

# checking correlation 
sns.heatmap(trainset.corr(), annot =True, cmap='Greens')


# Seems pretty good !!!! Now we have to feed this data to our meta classifier. For this i am going to use Logistic Regression. Let's see what it does

# In[ ]:


# meta classifeir
lr=LogisticRegression(class_weight='balanced', tol=1e-20)
score_lr=CrossVal(trainset,Ytrain,lr)
print("Accuracy is : ",score_lr)
lr.fit(trainset,Ytrain)
plotting(Ytest,lr.predict_proba(testset))


fig=plt.figure()
sns.heatmap(confusion_matrix(Ytest,lr.predict(testset)), annot= True, cmap='Greens')
lr_f1=f1_score(Ytest,lr.predict(testset))
plt.title('F1 Score = {}'.format(lr_f1))


# #### Finally after stacking model we got 80 and 80 F1 and AUC score it is still very less than Voting Classifier. Therefore it is no guranteed that you will get best result after stacking sometimes simple model can outperform complex models.

# ### Stay tuned for more updates. And don't forget to give an upvote if you like it 

# ### Feel free to ask any doubt/question/ or to give any suggestion :)

# In[ ]:




