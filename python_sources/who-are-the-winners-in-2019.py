#!/usr/bin/env python
# coding: utf-8

# <h1>Indian Candidates for General Election 2019 Analysis & Prediction of winning</h1>
# <img src='data:image/jpeg;base64,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'>
# <h2>Description</h2>
# <p>With over 600 Million voters voting for 8500+ candidates across 543 constituencies, the general elections in the world's largest democracy are a potential goldmine of data. While there are existing separate datasets about the votes each candidate received and the personal information of each candidate, there was no comprehensive dataset that included both these information. Thus, this dataset will provide more usability than most existing datasets in this domain.</p>
# <h2>Inspiration of Dataset Author</h2>
# <p>There are 2 main tasks that can be performed on this dataset: Exploratory Data Analytics to visualize the impact of each feature of the candidate and the use of machine learning to predict the chances of winning of a candidate.</p>

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
import os
plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


election_df=pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')
election_df.head()


# In[ ]:


for i in election_df.columns:
    x=i.lower().replace(' ','_').replace('\n','_').replace('__','_')
    election_df=election_df.rename(columns={i:x})


# In[ ]:


election_df.info()


# <h1>EDA</h1>

# <ul>
#     <li>number of states?,number of constituencies in each state?</li>
#     <li>number of candidates in each state</li>
#     <li>percentage of winners and losers</li>
#     <li>Age distribution of candidates</li>
#     <li>number of parties participate in the election</li>
#     <li>The percentage of participation of each party in each state</li>
#     <li>Percaentage of males and females in 2019 election</li>
#     <li>winner women in 2019 election</li>
#     <li>states that have women winners</li>
#     <li>Percentage of each edcational level of the participants in 2019 election</li>
#     <li>How many participant have a certain edcational level win, this question could give us an indicator if people in india choose participants who have higher educational background or not.</li>  
#     <li>Compare between assets and liabilities of the winners</li>

# In[ ]:


print('number of indian states: ',len(election_df['state'].unique()))


# <p>India is a federal union comprising 28 states and 8 union territories, for a total of 36 entities. The states and union territories are further subdivided into districts and smaller administrative divisions.</p>
# <a href='https://en.wikipedia.org/wiki/States_and_union_territories_of_India'>wikipedia</a>

# In[ ]:


states=[]
num_constituency=[]
for i in election_df['state'].unique():
    states.append(i)
    num_constituency.append(len(election_df[election_df['state']==i]['constituency'].unique()))
plt.figure(figsize=(20,8))
plt.bar(x=states,height=num_constituency)
plt.xlabel('states')
plt.ylabel('number of constituencies')
plt.title('number of constituencies in indian states in 2019')
plt.xticks(rotation=90)
plt.show()


# <p>We can predict that as the number of constituencies increases, the Area and the density of citizens increase.</p>

# In[ ]:


states=[]
num_candidates=[]
for i in election_df['state'].unique():
    states.append(i)
    num_candidates.append(len(election_df[election_df['state']==i]['name'].unique()))
plt.figure(figsize=(20,8))
plt.bar(x=states,height=num_candidates)
plt.xlabel('states')
plt.ylabel('number of candidates')
plt.title('number of candidates in indian states in 2019')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
election_df['winner'].value_counts().plot.pie(autopct='%.2f%%')
plt.title('percentage of winners and losers')
plt.ylabel('')
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
sns.distplot(election_df['age'])
plt.axvline(election_df['age'].mean(),color='red',label='mean')
plt.axvline(election_df['age'].median(),color='blue',label='median')
plt.axvline(election_df['age'].std(),color='green',label='std')
plt.legend()
plt.show()


# <p>distribtion of ages approximately normal distribution</p>

# In[ ]:


print('number of parties: ',len(election_df['party'].unique()))


# <p>the total number of parties registered was 2599, with 8 national parties, 53 state parties and 2538 unrecognised parties.</p>
# <a href='https://en.wikipedia.org/wiki/List_of_political_parties_in_India'>Wikipedia</a>

# In[ ]:


election_df['party'].value_counts()


# <p>biggest 5 parties in 2019 election is BJP, INC, NOTA, IND, BSP</p>

# In[ ]:


plt.figure(figsize=(15,8))
election_df['gender'].value_counts().plot.pie(autopct='%.2f%%')
plt.title('percentage of males and females')
plt.ylabel('')
plt.show()


# In[ ]:


women_only=election_df[election_df['gender']=='FEMALE']
women_only['winner'].value_counts().plot.bar()
plt.show()


# In[ ]:


print('75 women succeded from ',str(len(women_only)))


# In[ ]:


for i in women_only['state'].unique():
    print('state is : ',i)
    c=women_only[women_only['state']==i]
    for j,k,z in zip(c['name'],c['winner'],c['constituency']):
        if k==1:
            print('winner woman: ',j,' constituency: ',z)


# In[ ]:


election_df['education']=election_df['education'].str.replace('\n','')


# In[ ]:


plt.figure(figsize=(15,8))
election_df['education'].value_counts().plot.pie(autopct='%.2f%%')
plt.title('Percentage of each edcational level of the participants in 2019 election')
plt.ylabel('')
plt.show()


# In[ ]:


for i in election_df['education'].unique():
    print('edcational level: ',i)
    c=election_df[election_df['education']==i]
    total=len(c)
    winners=0
    for j in c['winner']:
        if j==1:
            winners+=1
    if total>0:
        print(winners/total)


# <p>We can say that most of winners have a higher edcational background, people loves those people.</p>  

# In[ ]:


def change_val(x):
    try:
        c = (x.split('Rs')[1].split('\n')[0].strip())
        c_2 = ''
        for i in c.split(","):
            c_2 = i+c_2
        return c_2
    except:
        x = 0
        return x
election_df['assets'] = election_df['assets'].apply(change_val).astype('int')
election_df['liabilities'] = election_df['liabilities'].apply(change_val).astype('int')


# In[ ]:


winner_only=election_df[election_df['winner']==1]
comp_dict={}
for i,j,k in zip(winner_only['name'],winner_only['assets'],winner_only['liabilities']):
    comp_dict[i]=j-k
print(comp_dict)


# <h1>Data Cleaning</h1>

# In[ ]:


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
print("We're using TF", tf.__version__)
import keras
print("We are using Keras", keras.__version__)
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import backend as K


# In[ ]:


x=election_df.drop('winner',axis=1)
y=election_df['winner']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)


# In[ ]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[ ]:


cols_to_remove=['name','general_votes','postal_votes','over_total_electors_in_constituency','over_total_votes_polled_in_constituency','total_electors']
x_train=x_train.drop(cols_to_remove,axis=1)
x_test=x_test.drop(cols_to_remove,axis=1)


# In[ ]:


def replacing(x):
    if x=='Not Available':
        x=x.replace('Not Available','0')
        x=int(x)
    else:
        return x
    return x
def convert_nan(x):
    if x==0:
        return np.nan
    else:
        return x 
    
x_train['criminal_cases']=x_train['criminal_cases'].apply(replacing).apply(convert_nan)
x_test['criminal_cases']=x_test['criminal_cases'].apply(replacing).apply(convert_nan)
x_train['criminal_cases']=x_train['criminal_cases'].astype('float')
x_test['criminal_cases']=x_test['criminal_cases'].astype('float')


# In[ ]:


for i,j in zip(x_train.columns,x_test.columns):
    if x_train[i].dtype=='object':
        x_train[i]=x_train[i].str.lower()
    if x_test[j].dtype=='object':
        x_test[j]=x_test[j].str.lower()


# In[ ]:


x_train['age']=x_train['age'].fillna(x_train['age'].median())
x_test['age']=x_test['age'].fillna(x_test['age'].median())
x_train['symbol']=x_train['symbol'].fillna('Unknown')
x_test['symbol']=x_test['symbol'].fillna('Unknown')


# In[ ]:


x_train['category']=x_train['category'].fillna('Unknown')
x_test['category']=x_test['category'].fillna('Unknown')
x_train['education']=x_train['education'].fillna('Not Available')
x_test['education']=x_test['education'].fillna('Not Available')


# In[ ]:


x_train['gender']=x_train['gender'].fillna('Unknown')
x_test['gender']=x_test['gender'].fillna('Unknown')
x_train['assets']=x_train['assets'].fillna(0)
x_test['assets']=x_test['assets'].fillna(0)


# In[ ]:


x_train['liabilities']=x_train['liabilities'].fillna(0)
x_test['liabilities']=x_test['liabilities'].fillna(0)
x_train['criminal_cases']=x_train['criminal_cases'].fillna(0)
x_test['criminal_cases']=x_test['criminal_cases'].fillna(0)


# <h1>Preprocessing & Feature Engineering</h1>

# <h2>Categorical Variables</h2>

# In[ ]:


#Check categorical columns for cardinality
for i in x_train.columns:
    if x_train[i].dtype=='object':
        print(i,': ',str(len(x_train[i].unique())))


# In[ ]:


object_cols = [col for col in x_train.columns if x_train[col].dtype == "object"]
numeric_cols=[col for col in x_train.columns if x_train[col].dtype !='object']


# In[ ]:


low_card_cols = [col for col in object_cols if x_train[col].nunique() < 130]
high_card_cols = list(set(object_cols)-set(low_card_cols))


# In[ ]:


OH_encoder=OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(x_train[low_card_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(x_test[low_card_cols]))
OH_cols_train.index = x_train.index
OH_cols_test.index = x_test.index
x_train = x_train.drop(low_card_cols, axis=1)
x_test = x_test.drop(low_card_cols, axis=1)

# Add one-hot encoded columns to numerical features
x_train = pd.concat([x_train, OH_cols_train], axis=1)
x_test = pd.concat([x_test, OH_cols_test], axis=1)


# In[ ]:


good_label_cols=[i for i in high_card_cols if set(x_train[i])==set(x_test[i])]
bad_label_cols=list(set(high_card_cols)-set(good_label_cols))


# In[ ]:


bad_label_cols


# In[ ]:


good_label_cols


# In[ ]:


x_train=x_train.drop('constituency',axis=1)
x_test=x_test.drop('constituency',axis=1)


# In[ ]:


for i in numeric_cols:
    x_train[i]=((x_train[i]-x_train[i].min())/(x_train[i].max()-x_train[i].min()))
    x_test[i]=((x_test[i]-x_test[i].min())/(x_test[i].max()-x_test[i].min()))


# <h1>Model Selection</h1>

# In[ ]:


def select_model():
    models=[{
        'name':'LogisticRegression',
        'estimator':LogisticRegression(),
        'hyperparameters':{
            'solver':["newton-cg", "lbfgs", "liblinear"]
        }
    },
    {
        'name':'KNeighborsClassifier',
        'estimator':KNeighborsClassifier(),
        'hyperparameters':{
            "n_neighbors": range(1,20,2),
            "weights": ["distance", "uniform"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "p": [1,2]
        }
    },
    {
        'name':'RandomForestClassifier',
        'estimator':RandomForestClassifier(),
        'hyperparameters':{
            "n_estimators": [4, 6, 9],
            "criterion": ["entropy", "gini"],
            "max_depth": [2, 5, 10],
            "max_features": ["log2", "sqrt"],
            "min_samples_leaf": [1, 5, 8],
            "min_samples_split": [2, 3, 5]
        }
    }
        
    ]
    for i in models:
        print(i['name'])
        grid=GridSearchCV(i['estimator'],
                          param_grid=i['hyperparameters'],
                          cv=10,
                          scoring='roc_auc')
        grid.fit(x_train,y_train)
        i["best_params"] = grid.best_params_
        i["best_score"] = grid.best_score_
        i["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(i["best_score"]))
        print("Best Parameters: {}\n".format(i["best_params"]))

    return models

select_model()


# <h1>Train and Predict</h1>

# In[ ]:


lr=LogisticRegression(solver='newton-cg')
lr.fit(x_train,y_train)
pred=lr.predict(x_test)


# In[ ]:


print('roc_auc_score is ',roc_auc_score(y_test,pred))

