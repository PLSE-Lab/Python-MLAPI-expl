#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This Kernel run on Google colab. First you need create a folder in your Google Drive and install datasets and kernels. Please check my blog and learn how to set up Kaggle with Colab.
# 
# Run Kaggle kernels on Google Colab:
# https://medium.com/@erdemalpkaya/run-kaggle-kernel-on-google-colab-1a71803460a9
# 
# 
# ### Improvement
# 
# I used **pd.get_dummies()** for some of converting categorical columns to numerical columm. When you use the function an example for sex it creates male(0,1) and female(0,1) 
# ![get_dummies example](https://cdn-images-1.medium.com/max/1600/1*HfhgywtwXtxVcUmQuyu-_w.png)
# 
# So we do not need to keep both features and sometimes reducing the that features increase your accuracy.
# 
# Previously my submission was in top 16% after removing the dummy features my submission is in -> top 8%
# 

# 

# In[1]:


#!pip install kaggle


# In[ ]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[ ]:


#from google.colab import files
#files.upload() 


# In[ ]:


#!pip install -q kaggle
#!mkdir -p ~/.kaggle
#!cp kaggle.json ~/.kaggle/
#!ls ~/.kaggle
## we need to set permissions 
#!chmod 600 /root/.kaggle/kaggle.json


# In[ ]:





# In[ ]:


# Google Colab directory setting. Comment out after run this line

#import os
#os.chdir('/content/gdrive/My Drive/Competitions/kaggle/Kaggle-Titanic/nbs')  #change dir


# In[ ]:


#!pwd


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plotly
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.style as style
style.available


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ## Introduction
# 
# This notebook included some EDA and predictions of survived Machine learning techniques with Scikit-learn

# In[ ]:


df = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})


# In[ ]:


df.head()


# In[ ]:


test.head()


# ### Titanic Dataset

# We are going to predict travels who survived or not with features below

# <b>Target</b>: Our target is find which travels survied (Survived==1) or not (Survived==0)
# 

# <b>Pclass:</b> Ticket class[](http://)
# - A proxy for socio-economic status

# In[ ]:


df.groupby(by=['Pclass'])['Survived'].agg(['mean','count'])


# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/2bc37b51-c9e4-402e-938e-70d3145815f2/d787jna-1b3767d2-f297-4b73-a874-7cfa6d1e8a69.png/v1/fill/w_1600,h_460,q_80,strp/r_m_s__titanic_class_system_by_monroegerman_d787jna-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NDYwIiwicGF0aCI6IlwvZlwvMmJjMzdiNTEtYzllNC00MDJlLTkzOGUtNzBkMzE0NTgxNWYyXC9kNzg3am5hLTFiMzc2N2QyLWYyOTctNGI3My1hODc0LTdjZmE2ZDFlOGE2OS5wbmciLCJ3aWR0aCI6Ijw9MTYwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.6krQcPQvsfcQ_ZJ_CGvufi9MT-PJkkg1I8-grLy7Hiw")


# <b>Sex:</b>

# In[ ]:


sex_survived= df.groupby(by=['Sex','Survived'])['Survived'].agg(['count']).reset_index()
sex_survived


# In[ ]:


plt.figure(figsize=(10, 5))
style.use('seaborn-notebook')
sns.barplot(data=sex_survived, x='Sex',y='count', hue='Survived');


# In[ ]:


# Plotly configuration function for Google Colab. We need to run this function for showing plotly graph in the Google colab
def configure_plotly_browser_state():
    
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))


# In[ ]:


male_survived=pd.DataFrame(df['Age'][(df['Sex']=='male')& (df['Survived']==1)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})
female_survived=pd.DataFrame(df['Age'][(df['Sex']=='female')& (df['Survived']==1)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})
male_not_survived=pd.DataFrame(df['Age'][(df['Sex']=='male') & (df['Survived']==0)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})
female_not_survived=pd.DataFrame(df['Age'][(df['Sex']=='female') & (df['Survived']==0)].value_counts().sort_index(ascending=False)).reset_index().rename(columns={'index':'Age','Age':'Number'})


# In[ ]:


from plotly import tools

#Add function here
configure_plotly_browser_state()
init_notebook_mode(connected=False)

trace1 = go.Scatter(
    x = male_survived['Age'].sort_values(ascending=False),
    y = male_survived['Number'],
    name='Survived Male',
    fill='tozeroy',
    #connectgaps=True

)
trace2 = go.Scatter(
    x = female_survived['Age'].sort_values(ascending=False),
    y = female_survived['Number'],
    name='Survived Female',
    fill='tozeroy',
    #connectgaps=True

)
trace3 = go.Scatter(
    x = male_not_survived['Age'].sort_values(ascending=False),
    y = male_not_survived['Number'],
    fill='tozeroy',
    name='Not Survived Male',
    #connectgaps=True

)
trace4 = go.Scatter(
    x = female_not_survived['Age'].sort_values(ascending=False),
    y = female_not_survived['Number'],
    fill='tozeroy',
    name = 'Not Survived Female',
    

)

fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Male', 'Female'))

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace3, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace4, 2, 1)

fig['layout']['xaxis2'].update(title='Age')
fig['layout'].update(height=700, width=1200,
                     title='Age Gender Survive')



iplot(fig)


# <b>Family:</b> 
# > SibSpof: siblings-spouses aboard the Titanic  
# > Parchof: parents-children aboard the Titanic
# 
# 
# 
# Step: Create a new features with _SibSpof_ and _Parchof_

# In[ ]:


# We need to make some data wrangling with both train and test data
df_all = [df,test]


# In[ ]:





# In[ ]:


for data in df_all:
    print(f"\n -------- {data.index } ------- \n")
    print(data.isnull().sum())


# We are going to drop columns that we will not use in Machine learning process

# In[ ]:





# In[ ]:



for data in df_all:
    data['isAlone']=1

    data['Family_No'] = data['Parch'] + data['SibSp'] + 1
        
    data['isAlone'].loc[data['Family_No']>1]=0
    
    data['Age'].fillna(round(data['Age'].mean()), inplace=True)
    
    #``df.fillna(df.mode().iloc[0])`` If you want to impute missing values with the mode in a dataframe 
    data['Embarked'].fillna(data['Embarked'].mode().iloc[0], inplace=True)
    
    # mean of each Pclass
    #data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.mean()))
    
    


# In[ ]:


df.head()


# In[ ]:


test.isAlone.value_counts()


# In[ ]:


# Drop features that will not process
for data in df_all:
    data.drop(columns=['PassengerId','Name','Cabin','Ticket','SibSp','Parch'],inplace=True,axis=1)


# In[ ]:


for data in df_all:
    print(f"\n -------- {data.index } ------- \n")
    print(data.isnull().sum())


# #### Convert categorical data to Numerical data for process

# In[ ]:


#get_dummies() function allows us to make a column for each categorical variable in features
test = pd.get_dummies(test,columns=['Sex','Embarked'])
df = pd.get_dummies(df,columns=['Sex','Embarked'])


# In[ ]:


df.head()


# In[ ]:


y=df['Survived']
X=df.drop(columns=['Survived'],axis=1)


# In[ ]:


#Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[ ]:


# Original pararamaters
DT= DecisionTreeClassifier()
DT.fit(X_train, y_train)
DT.score(X_test,y_test)


# In[ ]:


# Checking the hyperparamates of decision tree classifier
from IPython.display import HTML, IFrame
IFrame("https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier", width=1100, height=500)


# In[ ]:





# In[ ]:


# Grid CV
parameters1 = [{'max_depth':np.linspace(1, 15, 15),'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True)}]


# In[ ]:


# Grid Search for Decision Treee
Grid1 = GridSearchCV(DT, parameters1, cv=4,return_train_score=True,iid=True)

Grid1.fit(X_train,y_train)


# In[ ]:


scores = Grid1.cv_results_


# In[ ]:


for param, mean_train in zip(scores['params'],scores['mean_train_score']):
    print(f"{param} accuracy on training data is {mean_train}")


# In[ ]:


# best estimator for in Decision tree paramaters that we define. 
Grid1.best_estimator_


# In[ ]:


#Max score for above parameters
max(scores['mean_train_score'])


# We are going to try one XGBoost with Grid Search and check the result

# In[ ]:


XGB = XGBClassifier()


# In[ ]:


#parameters2 = [{'max_depth':np.linspace(1, 15, 15),'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True),'n_estimators':[100]}]

parameters3 =[{"learning_rate": [0.05, 0.10, 0.15, 0.20] ,"max_depth": [ 3, 4, 5, 6], "min_child_weight": [3,5,7],"gamma": [ 0.0, 0.1, 0.2 ,0.3],"colsample_bytree" : [ 0.4, 0.5]}]


# In[ ]:


Grid1 = GridSearchCV(XGB, parameters3, cv=2,return_train_score=True)

Grid1.fit(X_train,y_train)


# In[ ]:


scores = Grid1.cv_results_


# In[ ]:


# best estimator for in Decision tree paramaters that we define. 
Grid1.best_estimator_


# In[ ]:


max(scores['mean_train_score'])


# In[ ]:


XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=5, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[ ]:


XGB.fit(X_train, y_train)


# In[ ]:


XGB.score(X_test,y_test)


# ### Checking the result on XGB
# We are going to submit only XGBoost result

# In[ ]:


#pred = XGB.predict(test)


# In[ ]:


#result = pd.DataFrame(pred,columns=['Survived'])


# In[ ]:


#test1  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})


# In[ ]:


#submission = result.join(test1['PassengerId']).iloc[:,::-1]


# In[ ]:


#submission.to_csv('submission.csv', index=False)


# The accuracy of the test data is <span style="color:red">77.9</span> We need to try add feature or try ensemle models 

# ### Feature Engineering
# Now we are going to try add more feature and try to change 
# 
# 1. We are going to create some groups in with age column.
# 
# 2. Add a new features with Name titles like Mrs,Miss etc.. 
# 
# 3. For some machine learning techniques we need to create features for each variables first we will try without and we will compare accuracy on the test data.
# 
# 4. Look close ensemble models and we will add multiple scikit learns models.
# 

# In[ ]:


df = pd.read_csv('../input/train.csv' , header = 0,dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0,dtype={'Age': np.float64})


# In[ ]:


df.info()


# In[ ]:





# In[ ]:



for data in [df,test]:
    data['isAlone']=1

    data['Family_No'] = data['Parch'] + data['SibSp'] + 1
        
    data['isAlone'].loc[data['Family_No']>1]=0
    
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    
    #``df.fillna(df.mode().iloc[0])`` If you want to impute missing values with the mode in a dataframe 
    data['Embarked'].fillna(data['Embarked'].mode().iloc[0], inplace=True)
    
    # mean of each Pclass
    #data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    data['Fare'] = df.groupby('Pclass')['Fare'].apply(lambda x: x.fillna(x.mean()))


# <b>Ticket</b>

# In[ ]:





# In[ ]:


#import re
# We have two types tickets first only number and the second one letter and number. We are going to have letters and create a feature.
#trial_addFeature['Ticket_name'] =[]
#test_addFeature['Ticket_name'] =[]
# for data in df_all:
#     for i,k in enumerate(data['Ticket']):
#         try:
#             x=k.split(" ")[1]
#             data['Ticket'].replace(data['Ticket'][i],k.split(" ")[0],inplace=True)
#         except IndexError:
#             data['Ticket'].replace(data['Ticket'][i],"No_letter",inplace=True)


#     data['Ticket'] =data['Ticket'].map(lambda x: re.sub('[./]', '', x))
#     data['Ticket'] =data['Ticket'].map(lambda x: x.upper())


#df['Ticket_name'] =df['Ticket_name'].map(lambda x: re.sub('[./]', '', x))
#df['Ticket_name'] =df['Ticket_name'].map(lambda x: x.upper())

#set(Ticket_name)            


# In[ ]:


#set(data['Ticket'])


# In[ ]:


#test['Ticket'] = test_addFeature['Ticket']
#df['Ticket'] = df_addFeature['Ticket']


# In[ ]:





# ### Age
# As we imagine group of age is important than a single age (We think children women and the elderly had a chance for safe boats)

# In[ ]:


test.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()
#Upper limit is 100 but the oldest person is 80 years old
for data in [test,df]:
    bins = [-1,0,5,10, 15, 25, 50,100]
    labels = ['Unknown','Baby','Child','Young','Teen','Adult','Old']
    data['Age'] = pd.cut(data['Age'], bins=bins,labels=labels)
    data['Age'] = data['Age'].astype(str)
test['Age'] = LE.fit_transform(test['Age'])  
df['Age'] = LE.fit_transform(df['Age'])  

#data['Age'] = data['Age'].astype(int)


# ### Name Title
# We take the name and create a new features with Title of person

# In[ ]:


for data in [test,df]:
    for i,k in enumerate(data['Name']):
        x=k.split(",")[1]
        data['Name'].replace(data['Name'][i],x.split(" ")[1],inplace=True)
        


# In[ ]:





# In[ ]:


df['Name'].value_counts()


# <b>Mlle: </b>The term Mademoiselle is a French familiar title, abbreviated Mlle, traditionally given to an unmarried woman.
# 
# <b>Mme: </b>French abbreviation for Madame

# In[ ]:





# In[ ]:


all_data = [df,test]
Known = ['Mr.','Miss.','Mrs.','Master.','Ms.','Mlle.','Mme.']
for k in (all_data):
    for i,data in enumerate(k['Name']):
        if (data) in Known:
            if(data=='Mlle.'):
                k['Name'] = k['Name'].replace('Mlle.','Miss.')
            elif(data=='Ms.'):
                k['Name'] = k['Name'].replace('Ms.','Miss.')
            elif(data=='Mme.'):
                k['Name'] = k['Name'].replace('Mme.','Mrs.')
            else:
                continue
        else:
            k['Name'] = k['Name'].replace(data,'not_known')
        
            
            
        
        
    


# In[ ]:


# Survived difference between people who had different title
df['Name'][df['Survived']==1].value_counts()/df['Name'].value_counts()


# In[ ]:


df.info()


# In[ ]:





# In[ ]:


#columns = ['Embarked','Age','Sex','Name']
#
# for data in [df,test]:
#     for i in columns:
#         data[i] = data[i].astype(str)
#         data[i] = LE.fit_transform(data[i])

# Create feature for each categories
test=pd.get_dummies(test,columns=['Embarked','Name'])
df=pd.get_dummies(df,columns=['Embarked','Name'])
test['Sex'] = LE.fit_transform(test['Sex'])
df['Sex'] = LE.fit_transform(df['Sex'])


# In[ ]:


for data in [df,test]:
    data.drop(columns=['Ticket','Cabin','SibSp','Parch','PassengerId'], inplace=True, axis=1)


# In[ ]:


df.drop(columns=['Embarked_Q'],axis=1,inplace=True)
test.drop(columns=['Embarked_Q'],axis=1,inplace=True)


# #### Fare

# In[ ]:


for data in [df,test]:

    scale = StandardScaler().fit(data[['Fare']])
    data[['Fare']] = scale.transform(data[['Fare']])


# In[ ]:


y=df['Survived']
X=df.drop(columns=['Survived'],axis=1)


# In[ ]:


#Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[ ]:





# In[ ]:



df.head()


# ### Ensemble Model

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,  VotingClassifier


# In[ ]:


parameters_DC = [{'max_depth':[50,100],'min_samples_split': [0.1,0.2,0.5,0.8,0.9]}]

paramaters_RF = [{'max_depth':[2,5,10,15,20,50],'min_samples_split': [0.1,0.2,0.5,0.8],'n_estimators':[100]}]

parameters_XGB =[{"learning_rate": [0.2,0.5,0.8,0.9] ,"max_depth": [1, 3,5, 10], "min_child_weight": [3,5,7,10,20],"gamma": [0.1, 0.2 ,0.4,0.7],'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],'n_estimator':[100,1000,2000,4000]}]

parameters_GBC =[{"learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01] ,"max_depth": [ 3, 4, 5, 6], "min_samples_leaf" :[50,100,150],"n_estimators" : [16, 32, 64, 128]}]

parameters_ADA =[{'algorithm':['SAMME'],"base_estimator__criterion" : ["gini"],"base_estimator__splitter" :   ["best", "random"],"n_estimators": [500,1000],"learning_rate":  [ 0.01, 0.1, 1.0]}] 


# In[ ]:





# In[ ]:





# #### DecisionTree

# In[ ]:


DC = DecisionTreeClassifier()



Grid_DC = GridSearchCV(DC, parameters_DC, cv=4,scoring="accuracy", n_jobs= 4,return_train_score=True, verbose = 1)
#Fit the model
Grid_DC.fit(X_train,y_train)

# Best estimator parameters
DC_best = Grid_DC.best_estimator_

# Best score for the model with the paramaters
Grid_DC.best_score_


# #### RandomForest

# In[ ]:


RF = RandomForestClassifier()

Grid_RF = GridSearchCV(RF, paramaters_RF, cv=4,scoring="accuracy", n_jobs= 4,return_train_score=True, verbose = 1)
#Fit the model
Grid_RF.fit(X_train,y_train)

# Best estimator parameters
RF_best = Grid_RF.best_estimator_

# Best score for the model with the paramaters
Grid_RF.best_score_


# #### XGBoost

# In[ ]:


XGB = XGBClassifier()

Grid_XGB = GridSearchCV(XGB, parameters_XGB, cv=4,scoring="accuracy", n_jobs= 4,return_train_score=True, verbose = 1)
#Fit the model
Grid_XGB.fit(X_train,y_train)

# Best estimator parameters
XGB_best = Grid_XGB.best_estimator_

# Best score for the model with the paramaters
Grid_XGB.best_score_


# In[ ]:





# #### GradientBoosting

# In[ ]:


GBC = GradientBoostingClassifier()


Grid_GBC = GridSearchCV(GBC,parameters_GBC, cv=4, scoring="accuracy", n_jobs= 4, return_train_score=True,verbose = 1)

Grid_GBC.fit(X_train,y_train)

GBC_best = Grid_GBC.best_estimator_

# Best score
Grid_GBC.best_score_


# #### AdaBoostClassifier

# In[ ]:


ADA = AdaBoostClassifier(DC_best)


Grid_ADA = GridSearchCV(ADA,parameters_ADA, cv=4, scoring="accuracy", n_jobs= 4, return_train_score=True,verbose = 1)

Grid_ADA.fit(X_train,y_train)

ADA_best = Grid_ADA.best_estimator_

# Best score
Grid_ADA.best_score_


# #### SVM

# In[ ]:


parameters_SVM = {'C': [0.1, 1, 10,50,100], 'gamma' : [0.001, 0.01, 0.1, 1,10]}


# In[ ]:



from sklearn.svm import SVC
SVMC =SVC(probability=True)
Grid_SVC = GridSearchCV(SVMC, parameters_SVM, scoring="accuracy", return_train_score=True,verbose = 1,cv=2)

Grid_SVC.fit(X_train, y_train)

SVM_best = Grid_SVC.best_estimator_

# Best score
Grid_SVC.best_score_


# ### Voting Model

# In[ ]:


voting = VotingClassifier(estimators=[('ADA', ADA_best),('DC', DC_best),('RF', RF_best),('GBC',GBC_best),('XGB',XGB_best),('SVC',SVM_best)],weights=[3,0,0,1,3,3], voting='hard', n_jobs=4)

voting_result = voting.fit(X_train, y_train)


# In[ ]:


voting.score(X_test,y_test)


# In[ ]:





# In[ ]:


pred = voting.predict(test)


# In[ ]:


test_2 = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})


# In[ ]:


result = pd.DataFrame(pred,columns=['Survived'])
submission13 = result.join(test_2['PassengerId']).iloc[:,::-1]


# In[ ]:


submission13.to_csv('submission13.csv', index=False)


# In[ ]:


#!kaggle kernels push


# In[ ]:





# In[ ]:




