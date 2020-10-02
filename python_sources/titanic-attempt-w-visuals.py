#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivors Kaggle Submission
# 
# ## Abstract
# The purpose of my submission for the Kaggle ML competition is to determine which ML algo will most accurately predict whether passengers listed in a test.csv file survived the Titanic. The algo will be trained using data from a file called "Train.CSV" and tested using a "Train-Test-Split" method to measure the accuracy of the algo.
# 
# **Procedures**
# First Steps...
# - Import, clean and recategorize data
#     - Test for Null/Bad Values cross every column
#     - Name Column
#         - Use distinctions of Name to determine family orientation, relationship to port of departure/class
#     - Sex Column
#         - break into dummy variables 
#         - Use Female/Name cross-section to determine "Marriage" title
#     - Convert the Pclass, Survived and Sex Columns into Category Types
#         
#         
# If anyone comes across my first attempt, feel free to leave a comment! 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import time
import requests
import re
import seaborn as sns
from pandas.api.types import CategoricalDtype
import sys  
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode, plot
init_notebook_mode(connected=True)
from itertools import combinations

#Import the ML Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model as LM
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
sns.set(style="whitegrid", palette="muted")
pd.options.display.float_format = '{:,.2f}'.format

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#initialize data frames
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
dfset = [train, test]


# ### Data Clean up and Formatting
# 
# - [ ] We need to make uniform changes to the Train and Test sets so that the model we train can work on both sets
# - [ ] Our first task to clarify on the individual parameters
#     - Are people married?
#         - Find a way to connect married people
#     - Do they have special Titles?
#     - Do they have any children?
#     - Do they have any other type of family with them
# - [ ] Pclass and Cabin connection
#     - Also how do cabins play a role?
#     - Is there any pattern in the ticket prefixes

# In[ ]:


Married_Dict = {"Mrs": 1,"Mme": 1, "Lady": 1, "Mlle": 1, "th": 1, "Dr": 1, "Dona": 1,
                   "Miss":0, "Ms": 0}
femsub_list = []

#Functions to pull out middle names of males and females
def middle_name(lst):
    try:
        remaining = lst[2:]
    except:
        pass
    
    return " ".join(remaining)

def fmiddle_name(lst):
    try:
        remaining = lst[1:]
    except:
        pass
    
    return " ".join(remaining)

#Following are Regex patterns to be used later
pattern1 = r'\(\w+\s(\w+\s)?\w+\)' #Pattern to find Wife name
pattern2 = r'\w+\s' #Pattern to find Husband Name

#We try to make all changes in one for-loop pass
for df in dfset:
    #Rename the Name Column and separate Name components
    df.rename(columns={"Name":"Full_Name"}, inplace = True)
    df["Last_Name"] = df.Full_Name.str.split(',').apply(lambda x: x[0])
    df["First_Name"] = df.Full_Name.str.split(',').apply(lambda x: x[1]).str.slice(1)
    
    #Pull out titles
    df["Title"] = df.First_Name.str.split(" ").                                apply(lambda x: x[0]).str.slice(0,-1)
    
    #Use female titles to determine whether married
    
    df["Is_Married_Bin"] = df.Title.loc[df.Sex=="female"].map(Married_Dict).astype("int")
    
    #create a set of married femalse
    femsub = df.loc[df.Sex=="female"].Is_Married_Bin
    femsub_list.append(femsub)
    
    #Try to track and match the Married Male pairs
    #First create some dummy series to manipulate
    married_firstname, married_lastname = df.loc[(df.Is_Married_Bin == 1) & (df.Sex == 'female'), 'First_Name'] ,    df.loc[(df.Is_Married_Bin == 1) & (df.Sex == 'female'), 'Last_Name'] #We split the women's names into 2 series
    df['Spouse_First'] = np.nan #We Initialize empty series for the Spouse First name
    df['Middle_Name'] = np.nan #We Initialize emptys series for middle names
    df['Couple_Survived'] = np.nan
    df['Spouse_Present'] = 0
    df['Cabin_Rooms'] = 0 #To see how many rooms each person has
    df['Cabin_Group'] = np.nan
    df['Cabin_Number'] = np.nan
    try:
        df.loc[df.Sex=='male','Middle_Name'] = df.loc[df.Sex=='male','First_Name'].str.split(" ").apply(middle_name) #This pulls out the male middle names
    except:
        pass
    df.loc[df.Sex=='male','First_Name'] = df.loc[df.Sex=='male','First_Name'].str.split(" ").apply(lambda x: x[1]) #This pulls out male first name
    
    
    #Next create for loop to pull out the wife first name and husband names
    for e in married_firstname.index:
        try:
            wife_first = (re.search(pattern1,married_firstname[e])[0]) 
            wife_first = wife_first[1:-1] #Use regex pattern to strip out name and then remove parenthesis
            husband_first = (re.search(pattern2,married_firstname[e]))[0]
            df.loc[e,['First_Name','Spouse_First']] = wife_first, husband_first.replace(' ','')

        except:
            pass
    
    try:
        df.loc[df.Sex=='female', 'Middle_Name'] = df.loc[df.Sex=='female','First_Name'].str.split(" ").apply(fmiddle_name)
    except:
        pass
    df.loc[df.Sex=='female','First_Name'] = df.loc[df.Sex=='female','First_Name'].str.split(" ").apply(lambda x: x[0])
    
    #Now we should have the females matched to males...or atleast a clue
    
    #We will create a list of potential men who can be married
    married_men_candidate = {name: [] for name in married_lastname.unique()}
    
    #For Men women, we will create a dictionary of last names and number of married women
    married_women = {name: married_lastname.value_counts()[i] for i, name in enumerate(married_lastname.value_counts().index)}
    
    #Fill in the empty married_men_candidates with potential first names
    for e in married_lastname.unique():
        married_men_candidate[e] = list(df.loc[(df.Last_Name == e) & (df.Sex=='male'), 'First_Name'])
        if len(married_men_candidate[e]) == 0: #Remove keys with empty lists
            del married_men_candidate[e]
    
    #The following For Loop wil map the male candidates to their wives
    #If the male candidate and wife match, we must flip the Spouse_Present Parameter to "1", and update the male "Spouse_First"
    for lastname in married_men_candidate: #Pull out each last name
        for m_name in married_men_candidate[lastname]: #Loop through each first name of Last name
            for f_name in df.loc[(df.Last_Name == lastname) & (df.Sex == 'female'),'Spouse_First']: #Pull up Spouse candidates
                try:
                    if m_name == f_name: #If the spouse first name and male first name match
                        female_name = df.loc[(df.Last_Name==lastname)&(df.Sex =='female') & (df.Spouse_First==f_name),'First_Name']
                        df.loc[(df.Sex=='male') & (df.First_Name == m_name) & (df.Last_Name ==lastname),'Spouse_First'] = female_name[female_name.index[0]]
                        df.loc[(df.Sex=='male') & (df.First_Name == m_name) & (df.Last_Name ==lastname),'Spouse_Present'] = 1
                        df.loc[(df.Sex=='female') & (df.Spouse_First == f_name) & (df.Last_Name== lastname),'Spouse_Present'] = 1

                except:
                    pass

    
    #Convert Pclass to ordered Categories
    cat_type_class = CategoricalDtype(categories=[3,2,1], ordered=True)
    df["Pclass_cat"] = df.Pclass.astype(cat_type_class)
    
    #Round up Fares to clean up the data
    df['Fare'] = np.ceil(df['Fare'])
    
    df['Is_Married_Bin'].fillna(0, inplace=True)
    df.loc[df.Age.isnull(),'Age'] = np.random.randint(15,45)
    
    #Lets perform a For-Loop through the cabins to extract relevant features
    #Pull out number of rooms, Cabin group and Cabin room number
    for room in df.loc[df.Cabin.notnull(),'Cabin'].index:
        try:
            rooms = df.loc[room,'Cabin'].split(' ') #First split cabin string into list of different rooms
            df.loc[room, 'Cabin_Rooms'] = len(rooms)
            for room1 in rooms: #Split individual rooms into string units to pull out extra info
                df.loc[room, 'Cabin_Group'] = room1[0]
                df.loc[room, 'Cabin_Number'] = room1[1:]
        except:
            df.loc[room,'Cabin_Rooms'] = 0
            



# In[ ]:


#Implement Dummy variables for Sex that can't be done with For-Loop :(
sex_male = pd.get_dummies(train.Sex, prefix="Sex", drop_first=True)
embarking = pd.get_dummies(train.Embarked, prefix='Embarked', drop_first=True)
cabin_grp = pd.get_dummies(train.Cabin_Group, prefix='Cabin_Grp', drop_first=True)
train = pd.concat([train, sex_male, embarking, cabin_grp], axis=1)

sex_male = pd.get_dummies(test.Sex, prefix='Sex', drop_first=True)
embarking = pd.get_dummies(test.Embarked, prefix='Embarked', drop_first=True)
cabin_grp = pd.get_dummies(test.Cabin_Group, prefix='Cabin_Grp', drop_first=True)
test = pd.concat([test, sex_male, embarking, cabin_grp], axis=1)


# ### Visualization
# 
# Now that we have most of the formating out of the way (finally!) we can make some visualizations to start understanding the trends and key parameters

# In[ ]:


Null_Hypo = ['Not Survied', 'Survied']

print(Null_Hypo[np.argmax([1-train.Survived.mean(), train.Survived.mean()])], '{:.2f}%'.format(max(1-train.Survived.mean(), train.Survived.mean())*100))


# With this we have a Null rate of 61.5% - i.e I should expect to get ~61.5% of my predictions correct if I just pick "not survived" as my prediction

# In[ ]:


first = train.groupby('Cabin_Rooms')['Fare', 'Survived','Pclass'].mean()
second = train.groupby('Cabin_Rooms')['Full_Name'].count()
cabins = pd.concat([first, second], axis=1)
cabin_df = pd.DataFrame(cabins)
cabin_df.rename(columns={'Full_Name':'Num_People'}, inplace=True)
cabin_df['Approx_Dead'] = round((1-cabin_df['Survived'])*cabin_df['Num_People'])

bar_fig = {'data': [],
          'layout': {}}

bar_fig['layout'] = go.Layout(
    autosize = False, width = 750, height=500, 
    title = 'Cabin Perspective',
    xaxis = dict(title = 'Cabin Rooms',
                 showline = True,
                 dtick = 1,
                 showgrid = False,
                domain=[0,0.8]),
    yaxis = dict(title='Fares',
                 showline = True,
                 showgrid = False
                ),
    yaxis2 = dict(title = 'Number of People',
                 showline = True,
                 overlaying = 'y', #Allows for second axis
                 side = 'right',
                 anchor = 'x'),
    
    yaxis3 = dict(title = 'Average Pclass',
                  showline = False,
                  overlaying = 'y',
                  side = 'right',
                  anchor = 'free',
                  position = 1
                 ),
    legend = dict(x = 1.15, y = 1),
    hovermode = 'closest'

)

bar_data = go.Bar(x= cabin_df.index, 
                  y=cabin_df.Fare, 
                  name='Fare'
                 )
line_data = go.Scatter(x=cabin_df.index, 
                       y=cabin_df.Approx_Dead, 
                       yaxis = 'y2', 
                       name='Approx Dead',
                       mode='lines'
                      )
line2_data = go.Scatter(x=cabin_df.index,
                      y=cabin_df.Num_People,
                      name='Number of People',
                      mode='lines',
                      yaxis = 'y2',
                     )
dot_data = go.Scatter(x=cabin_df.index,
                     y=cabin_df.Pclass,
                     name='Avareage Pclass',
                     mode='markers',
                     yaxis = 'y3')

bar_fig['data'].append(bar_data)
bar_fig['data'].append(line_data)
bar_fig['data'].append(line2_data)
bar_fig['data'].append(dot_data)

iplot(bar_fig)


# ### What does this tell us?
# 
# From what we can see above, there is clearly a correlation between survival and the number of rooms that you have. Further investigation needed on the Pclass trend breakdown

# In[ ]:


# Lets visualize a little

surv_set = train.loc[train.Survived==1]

facplot = sns.FacetGrid(surv_set, col='Embarked', row='Pclass', col_order=['C','S','Q'])
facplot.map(sns.kdeplot, 'Sex_male')
facplot.fig.suptitle('Gender Survival Density Distribution', y=1.1)


# In[ ]:


# Lets visualize a little

death_set = train.loc[train.Survived==0]

facplot = sns.FacetGrid(death_set, col='Embarked', row='Pclass', col_order=['C','S','Q'])
facplot.map(sns.kdeplot, 'Sex_male')
facplot.fig.suptitle('Gender Death Density Distribution', y=1.1)


# ### What do we learn from the Facet Grids?
# 
# Nothing earth shattering, other than that people who left Queenstown clearly weren't very rich (as they were only in the 3rd class tickets...which isn't hugely helpful). Not the most helpful image...let's try out luck with a few other things

# In[ ]:


#Create some plots which help distinguish trends based on age and class based on sex
sns.lmplot(x="Pclass_cat", y="Survived", hue="Sex_male", height=7, palette = ['r','y'], data=train)
sns.lmplot(x='Age', y='Survived', data=train, hue="Sex_male", palette = ['r','y'], height = 7)


# In[ ]:


#Create a graph which plots Number of Female survival victims split by Pclass and age. 
femaleSet = train.loc[train.Sex=="female"]
f_pop_dist = femaleSet.Pclass.value_counts(normalize=True)
sns.lmplot(x="Is_Married_Bin", y = "Survived",  scatter=True, hue="Pclass",           palette = ('r','c','y'), data=femaleSet, logistic=True) 
#sns.lmplot(data=femaleSet, y = "Survived", x="Age",  palette = ('r','c','y'), hue='Pclass', logistic=True)


# ### What does this tell us?
# 
# We can see that there is a clear divergence to being a female and that the not only is the probability of survival vastly different, but its sensitivity to Pclass is also a lot higher than the male group. Also, interestingly, the male group and female group have divergent trends as a function of age - this hints that a simple logistic regression. We see a correlation between Pclass and surival quite clearly

# In[ ]:


#Lets get a better picture picture on fare vs Pclass

figure = {'data': [],
         'layout': {}}

figure['layout'] = go.Layout(
    autosize = False, width = 500, height=500, 
    title = 'Pclass vs Fares',
    xaxis = dict(title = 'Pclass',
                 showline = True,
                 dtick = 1,
                 showgrid = False),
    yaxis = dict(title='Fares',
                 showline = True,
                 showgrid = False
                ),
    hovermode = 'closest'
)

for cls in train.Pclass.unique():
    pset = train.loc[train.Pclass == cls]
    y = pset['Fare']
    x = [cls]*len(pset['Fare'])
    name = 'Pclass {}'.format(cls)
    text_list = []
    for ix in pset['Fare'].index:
        text = 'Person: {pr}<br>Age: {ag}<br>Port: {pt}'.format(pr = pset.loc[ix,'Full_Name'], 
                                                            ag = pset.loc[ix,'Age'], 
                                                            pt = pset.loc[ix,'Embarked'])
        text_list.append(text)
    data = go.Scatter(x = x, y = y, name = name, mode = 'markers', text = text_list)
    figure['data'].append(data)

iplot(figure)


# ### What does this tell us?
# 
# This is a simple check on whether "Fare" is a material parameter - and it seems it isn't. Given the presence of inconsistent values, we see Fare tends to be less useful than Pclass but sometimes useful when the value is very large (hinting at specialness). This characteristic hints that a tree method might be useful (or use of non-linearity)

# In[ ]:


# To kick off our ML side of the excercise, we're going to build a function to give us an accuracy breakdown

def c_matrix(y_true,y_pred):
    scoring = confusion_matrix(y_true,y_pred) #Following creates array for confusion matrix
    tn, fp, fn, tp = scoring.ravel() #Create individual parameters of confusino matrix
    
    test = pd.DataFrame(scoring, columns = [('Actual', 'Negative'), ('Actual', 'Positive')], 
                        index = [('Predicted', 'Negative'), ('Predicted', 'Positive')])
    
    #Accuracy variables
    accu_rate = accuracy_score(y,y_pred) 
    
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(tn+fp)
    npv = tn/(tn+fn)
    
    return test, accu_rate, recall, precision, specificity, npv



# In[ ]:


#Next we will create a group of features to later split into feature groups in the feature set
cols_set = ['Pclass', 'Is_Married_Bin','Spouse_Present','Cabin_Rooms', 
           'Sex_male', 'Embarked_Q', 'Embarked_S', 'Cabin_Grp_B', 'Cabin_Grp_C', 'Cabin_Grp_D', 
            'Cabin_Grp_E','Cabin_Grp_F', 'Cabin_Grp_G', 'Cabin_Grp_T']


#The fuction creates combinations...so it kinda becomes a bit like a random tree regressor
#We will use the logistic regression class to see if we can get some different error sets
def feature_create(feature_set):
    two_group = []
    three_group = []
    four_group = []
    five_group = []
    for item in combinations(feature_set, 2):
        two_group.append(list(item))
    for item in combinations(feature_set, 3):
        three_group.append(list(item))
    #for item in combinations(feature_set, 4):
       # four_group.append(list(item))
    #for item in combinations(feature_set, 5):
       # five_group.append(list(item))
    
    grp_list = [two_group, three_group]#, four_group, five_group]
    
    return grp_list

feature_set = feature_create(cols_set)

train[cols_set].isnull().sum()


# In[ ]:


#Start setting up your Machine Learning Parameters, begin by testing logistic regression


log_confus = []
log_accu = []
log_recall = []
log_precise = []
log_spec = []
log_npv = []
log_feat = []

y = train.Survived
log = LogisticRegression()
rand_state = range(20)
log_param_grid = dict(random_state=rand_state)
for feat_grp in feature_set:
    for feat_cols in feat_grp:
        X = train[feat_cols]
        gridLog = GridSearchCV(log, log_param_grid, cv=2, scoring = 'accuracy')
        gridLog.fit(X,y)
        best_log = gridLog.best_estimator_
        y_pred = best_log.predict(X)
        confus_matrix, accu, recall, precise, spec, npv = c_matrix(y,y_pred)
        log_confus.append(confus_matrix), log_accu.append(accu), log_recall.append(recall)
        log_precise.append(precise), log_spec.append(spec), log_npv.append(npv) 
        
        feat_text = ",".join(feat_cols)
        
        log_text = "Feature Cols: {}"
        feat_text = ",".join(feat_cols)
        log_text = "Feature Cols: {}".format(feat_text)
        log_feat.append(log_text)

        


# In[ ]:


def confus_fig(accu, recall, precise, feat, spec, npv, title):
    fig = {'data': [],
              'layout': {}}

    fig['data'].append(go.Scatter(x=accu, 
                                      y=precise, 
                                      text=feat, 
                                      mode='markers', 
                                      name = 'Precision vs Accuracy'
                                     ))
    fig['data'].append(go.Scatter(x=accu, 
                                      y=recall, 
                                      text=feat, 
                                      mode='markers', 
                                      name='Recall vs Accuracy'
                                     ))
    fig['data'].append(go.Scatter(x=accu,
                                      y=npv,
                                      text=feat,
                                      mode='markers',
                                      name='NPV vs Accuracy'))
    fig['data'].append(go.Scatter(x=accu,
                                      y=spec,
                                      text=feat,
                                      mode='markers',
                                      name='Specifity vs Accuracy'
                                     ))
    fig['layout']['title'] = '{} Training Error Results'.format(title)
    fig['layout']['hovermode'] = 'closest'
    fig['layout']['xaxis'] = dict(title='Accuracy',
                                  range=[0.5,.9],
                                     showline = True,
                                     showgrid = False)
    fig['layout']['yaxis'] = dict(title='Positive: Precision & Recall',
                                  range=[0.5,1],
                                     showline = True,
                                     showgrid = False)
    fig['layout']['yaxis2'] = dict(title='Negative: Specifity & NPV',
                                   range=[0.5,1],
                                      showline = True,
                                      showgrid = False)
    
    return fig

log_fig = confus_fig(log_accu, log_recall, log_precise, log_feat, log_spec, log_npv, "Logistic Regression")
iplot(log_fig)


# In[ ]:


top5_accu = log_accu.copy()
top5_accu = list(set(top5_accu))
top5_accu.sort(reverse=True)
top5_accu = top5_accu[0:5]
for i, a in enumerate(top5_accu):
    log_index = log_accu.index(a)
    print(i+1, log_feat[log_index], ", Accuracy {:,.2f}%".format(a*100))


# ### What does this tell us?
# 
# So far we have exceeded our estimate using the Null Hypothesis by using a healthy margin and that it is beneficial to mix the gender in all the predictions with mixed results when introducing whether spouse is present or cabin groups. 
# 
# Surprisingly, hovering over the dots shows that "embarked" performed quite poorly for accuracy prediction. The results of the confusion matrix show that narrowing down the features to presnet atleas tthe Spouse Present, Gender and evidence of a cabin group seem to present the highest probability of survival

# In[ ]:


col_set2 = ['Pclass', 'Is_Married_Bin','Spouse_Present','Cabin_Rooms', 'Fare',
           'Sex_male', 'Embarked_Q', 'Embarked_S', 'Cabin_Grp_B', 'Cabin_Grp_C', 'Cabin_Grp_D', 
            'Cabin_Grp_E','Cabin_Grp_F', 'Cabin_Grp_G', 'Cabin_Grp_T']
feature_set2 = feature_create(col_set2)


# In[ ]:


svc_confus = []
svc_accu = []
svc_recall = []
svc_precise = []
svc_spec = []
svc_npv = []
svc_feat = []

svc = LinearSVC()

svc_param_grid1 = {'penalty':['l1','l2'],
                 'random_state':[i for i in range(1,20)],
                   'dual': [False]
                 }
svc_param_grid2 = {'penalty': ['l2'],
                   'loss': ['hinge'],
                  'random_state': [i for i in range(1,20)]
                  }
for feat_grp in feature_set2:
    for feat_cols in feat_grp:
        X = train[feat_cols]
        grid_svc = GridSearchCV(svc,[svc_param_grid1, svc_param_grid2], cv=2, scoring='accuracy')
        grid_svc.fit(X,y)
        best_svc = grid_svc.best_estimator_
        ypred = best_svc.predict(X)
        confus_matrix, accu, recall, precise, spec, npv = c_matrix(y,ypred)
        svc_confus.append(confus_matrix), svc_accu.append(accu), svc_recall.append(recall)
        svc_precise.append(precise), svc_spec.append(spec), svc_npv.append(npv)
        
        feat_text = ','.join(feat_cols)
        svc_text = "Feature Cols: {}".format(feat_text)
        svc_feat.append(svc_text)

        
        
        
        


# In[ ]:


svc_fig = confus_fig(svc_accu, svc_recall, svc_precise, svc_feat, svc_spec, svc_npv, "Linear SVC Regression")
iplot(svc_fig)


# In[ ]:


print("The Linear SVC best Test Score: {:.2f}%".format(grid_svc.cv_results_['mean_test_score'].max()*100))
print('The Linear SVC best Train Score: {:.2f}%'.format(np.asarray(svc_accu).max()*100))

top5_accu = svc_accu.copy()
top5_accu = list(set(top5_accu))
top5_accu.sort(reverse=True)
top5_accu = top5_accu[0:5]
for i, a in enumerate(top5_accu):
    svc_index = svc_accu.index(a)
    print(i+1, svc_feat[svc_index], ", Accuracy {:,.2f}%".format(a*100))


# ### What does this tell us?
# 
# Our test using the Linear Support Vector Classifier from the Support Vector Machine Library did not yield substantially better results than the logistic regression. Overall, there isn't a substantialy difference in result thought it is curious that Fare is appearing as a relevant variable this time - let's try our luck using Fare in the test with a Random Forest Classifier

# In[ ]:


#Next we are gonna try out the Random Tree Classifier
tree = RFC()

tree_param_grid = dict(n_estimators= [i for i in range(1,10)], 
                   max_depth= [None]+[i for i in range(1,10)], 
                   max_features= ['auto']+[i for i in range(2,5)]
                      )


gridTree = GridSearchCV(tree, tree_param_grid, cv=10, scoring='accuracy' )


# In[ ]:


X = train[col_set2]
gridTree.fit(X,y)


# In[ ]:


#Now that we have a fitted tree, lets create some dictionaries to later produce charts
def grid_output(gridobj):
    rank_dict = {}
    top5_testerror = {}
    top5_param = {}
    top5_trainerror = {}
    rank = 1
    while len(rank_dict)<5:
        #Create dictionary of top 5 performing trees
        ranks = gridobj.cv_results_['rank_test_score']
        rank_dict[rank] = np.where(ranks == rank)
        top5_testerror[rank] = []
        top5_param[rank] = []
        top5_trainerror[rank] = []
        if len(rank_dict[rank][0]) == 0:
            del rank_dict[rank] 
            del top5_testerror[rank] 
            del top5_trainerror[rank]
            del top5_param[rank]
        rank +=1

    for rnk in rank_dict:
        for lst in rank_dict[rnk]:
            for ix in lst:
                top5_param[rnk].append(gridobj.cv_results_['params'][ix])
                top5_testerror[rnk].append(gridobj.cv_results_['mean_test_score'][ix])
                top5_trainerror[rnk].append(gridobj.cv_results_['mean_train_score'][ix])        
                
    return rank_dict, top5_testerror, top5_trainerror, top5_param

rank_dict, top5_testerror, top5_trainerror, top5_param = grid_output(gridTree)


# In[ ]:


#Create an output to show the test and training error of the

randtree_fig = {'data': [],
               'layout': {}}
randtree_fig['layout']['title'] = 'Random Tree Forest Training/Test Error Results'
randtree_fig['layout']['hovermode'] = 'closest'
randtree_fig['layout']['xaxis'] = dict(title='Max depth',
                                       dtick=1,
                                       showline = True,
                                       showgrid = False)
randtree_fig['layout']['yaxis'] = dict(title='Testing Accuracy Rate',
                                       range = [0.80,.90],
                                 showline = True,
                                 showgrid = False)
randtree_fig['layout']['yaxis2'] = dict(title='Training Accuracy Rate',
                                        range = [0.80,.90],
                                        showline = True,
                                        showgrid = False, 
                                        side = 'right',
                                       overlaying = 'y')
randtree_fig['layout']['legend'] = dict(x= 1.1, y = 1)
randtree_fig['layout']['updatemenus'] = list([
    dict(type='buttons',
         active=-1,
         buttons = list([
             {'args': [{'visible': [True, False]},
                       {'title': 'Random Forest Test Error Results'}],
              'label': 'Test Error',
              'method': 'update'},
             {'args': [{'visible': [False,True]},
                      {'title': 'Random Forest Train Error Results'}],
             'label': 'Train Error',
             'method':'update'},
             {'args': [{'visible':[True,True]},
                      {'title': 'Random Forest Train/Test Error Results'}],
             'label': 'Both',
             'method': 'update'}
             ]),
         direction = 'left',
         pad = {'r': 10, 't': 10},
         showactive = True,
         x = 0,
         xanchor = 'left',
         y = 1.05,
         yanchor = 'top'
        )
])







for rnk in rank_dict:
    train_ylist = []
    test_ylist =[]
    xlist = []
    slist = []
    tlist = []
    for lst in top5_testerror[rnk]:
        test_ylist.append(lst)
    for lst in top5_trainerror[rnk]:
        train_ylist.append(lst)
    for lst in top5_param[rnk]:
        xlist.append(lst['max_depth'])
        slist.append(lst['n_estimators']*5)
        tlist.append("Max Features: {}".format(lst['max_features']))
    test_data = go.Scatter(x = xlist,
                           y = test_ylist,
                           mode = 'markers',
                           text = tlist,
                           name = "Rank {} Test Data".format(rnk),
                           marker = dict(size = slist,
                                         opacity = .9
                                        )     
                          )
    train_data = go.Scatter(x = xlist,
                            y = train_ylist,
                            yaxis = 'y2',
                            mode = 'markers',
                            text = tlist,
                            name = "Rank {} Train Data".format(rnk),
                            marker = dict(size = slist,
                                         opacity = .9
                                        )
                           )
    
    randtree_fig['data'].append(test_data)
    randtree_fig['data'].append(train_data)
    
    #for err in top5_testerror[rnk]:

iplot(randtree_fig)                        


# In[ ]:


print("The RFC best Test Score: {:.2f}%".format(gridTree.cv_results_['mean_test_score'].max()*100))
print('The RFC best Train Score: {:.2f}%'.format(gridTree.cv_results_['mean_train_score'].max()*100))


# ### Summary
# 
# The graph shows the spectrum of accuracies accross depth. The size of the dot relates to the number of estimators. Hover over the dot to check the number of max features.
# 
# The above graph proves the adage that adding complexity can lead to overfitting. The graph shows the full spectrum and we can see the max test score and and max train score. From the charts we can see that adding to the max depth helps in improving the the training data but doesn't necessarily produce the best results with the test data. The current bench mark of the Random Tree Classifier beats the  Logistic Regression Train Error by a healthy margin. Even though the below best estmator attribute shows us that the gridsearchcv operation favors a max depth of 6, I think it's worth considering the max depth of 7 with smaller number of estimators as well.

# In[ ]:


gbmTree = GBC()

gbm_param_grid = {'learning_rate': [0.05,.1,.2,.5],
                  'n_estimators': [80,90,100,110,120,150],
                  'max_depth': [2,3,4,5]}

grid_gbmTree = GridSearchCV(gbmTree,gbm_param_grid, cv=10, scoring='accuracy')


# In[ ]:


grid_gbmTree.fit(X,y)


# In[ ]:


rank_dict, top5_testerror, top5_trainerror, top5_param = grid_output(grid_gbmTree)


# In[ ]:


#Create an output to show the test and training error of the

gbm_fig = {'data': [],
               'layout': {}}
gbm_fig['layout']['title'] = 'Gradient Boosting Training/Test Error Results'
gbm_fig['layout']['hovermode'] = 'closest'
gbm_fig['layout']['xaxis'] = dict(title='Number of Estimators',
                                       dtick=10,
                                       showline = True,
                                       showgrid = False)
gbm_fig['layout']['yaxis'] = dict(title='Testing Accuracy Rate',
                                       range = [0.80,.95],
                                 showline = True,
                                 showgrid = False)
gbm_fig['layout']['yaxis2'] = dict(title='Training Accuracy Rate',
                                        range = [0.80,.95],
                                        showline = True,
                                        showgrid = False, 
                                        side = 'right',
                                       overlaying = 'y')
gbm_fig['layout']['legend'] = dict(x= 1.1, y = 1)
gbm_fig['layout']['updatemenus'] = list([
    dict(type='buttons',
         active=-1,
         buttons = list([
             {'args': [{'visible': [True, False]},
                       {'title': 'Gradient Boosting Test Error Results'}],
              'label': 'Test Error',
              'method': 'update'},
             {'args': [{'visible': [False,True]},
                      {'title': 'Gradient Boosting Train Error Results'}],
             'label': 'Train Error',
             'method':'update'},
             {'args': [{'visible':[True,True]},
                      {'title': 'Gradient Boosting Train/Test Error Results'}],
             'label': 'Both',
             'method': 'update'}
             ]),
         direction = 'left',
         pad = {'r': 10, 't': 10},
         showactive = True,
         x = 0,
         xanchor = 'left',
         y = 1.05,
         yanchor = 'top'
        )
])







for rnk in rank_dict:
    train_ylist = []
    test_ylist =[]
    xlist = []
    slist = []
    tlist = []
    for lst in top5_testerror[rnk]:
        test_ylist.append(lst)
    for lst in top5_trainerror[rnk]:
        train_ylist.append(lst)
    for lst in top5_param[rnk]:
        xlist.append(lst['n_estimators'])
        slist.append(lst['max_depth']*10)
        tlist.append("Learning Rate: {}".format(lst['learning_rate']))
    test_data = go.Scatter(x = xlist,
                           y = test_ylist,
                           mode = 'markers',
                           text = tlist,
                           name = "Rank {} Test Data".format(rnk),
                           marker = dict(size = slist,
                                         opacity = .9
                                        )     
                          )
    train_data = go.Scatter(x = xlist,
                            y = train_ylist,
                            yaxis = 'y2',
                            mode = 'markers',
                            text = tlist,
                            name = "Rank {} Train Data".format(rnk),
                            marker = dict(size = slist,
                                         opacity = .9
                                        )
                           )
    
    gbm_fig['data'].append(test_data)
    gbm_fig['data'].append(train_data)
    
    #for err in top5_testerror[rnk]:

iplot(gbm_fig)   


# In[ ]:


grid_gbmTree.cv_results_['mean_test_score'].max()
print("The GBM best Test Score: {:.2f}%".format(grid_gbmTree.cv_results_['mean_test_score'].max()*100))
print('The GBM best Train Score: {:.2f}%'.format(grid_gbmTree.cv_results_['mean_train_score'].max()*100))



# ### Summary
# 
# The conclusion of this excercise (atleast for now) yields that the Gradient Boosting Method seems to be yielding the most accurate prediction. Though it is clear that we are at a higher risk of overfitting to the training data with the Gradient Boosting method, there doesn't seem to be material downside to using it vs the Random Forest Classifier.

# In[ ]:


best_gbmTree = grid_gbmTree.best_estimator_


# In[ ]:


test['Fare'] = test['Fare'].fillna(0)
test['Cabin_Grp_T'] = 0
test_X = test[col_set2]
y_test = best_gbmTree.predict(test_X)


# In[ ]:


kaggle_submit = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_test})

#kaggle_submit.to_csv('kaggle_submission.csv')

