#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy as sc
from sklearn import preprocessing

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
pd.options.mode.chained_assignment = None

#Clean Data
def clean(tr,ts):
    le = preprocessing.LabelEncoder()
    #Save ID
    Id_tr=tr['AnimalID']
    tr.drop('AnimalID',axis=1,inplace=True)
    tr.drop('OutcomeSubtype',axis=1,inplace=True)
    Id_ts=ts['ID']
    ts.drop('ID',axis=1,inplace=True)
    
    #Name Binarizer
    tr['Name_bool']=pd.isnull(tr['Name']).astype(int)
    ts['Name_bool']=pd.isnull(ts['Name']).astype(int)
    
    #OutcomeType
    tr.ix[tr.OutcomeType == 'Return_to_owner', 'Target'] = 0
    tr.ix[tr.OutcomeType == 'Euthanasia', 'Target'] = 1
    tr.ix[tr.OutcomeType == 'Adoption', 'Target'] = 2
    tr.ix[tr.OutcomeType == 'Transfer', 'Target'] = 3
    tr.ix[tr.OutcomeType == 'Died', 'Target'] = 4
    
    #Animal Binarizer
    tr.ix[tr.AnimalType == 'Dog', 'AnimalType_cat'] = 0
    tr.ix[tr.AnimalType == 'Cat', 'AnimalType_cat'] = 1
    
    ts.ix[ts.AnimalType == 'Dog', 'AnimalType_cat'] = 0
    ts.ix[ts.AnimalType == 'Cat', 'AnimalType_cat'] = 1
    
    #SexuponOutcome
    tr.ix[tr.SexuponOutcome == 'Intact Female', 'SexuponOutcome_cat'] = 0
    tr.ix[tr.SexuponOutcome == 'Spayed Female', 'SexuponOutcome_cat'] = 1
    tr.ix[tr.SexuponOutcome == 'Neutered Male', 'SexuponOutcome_cat'] = 2
    tr.ix[tr.SexuponOutcome == 'Intact Male', 'SexuponOutcome_cat'] = 3
    tr.ix[tr.SexuponOutcome == 'Unknown', 'SexuponOutcome_cat'] = 4
    
    ts.ix[ts.SexuponOutcome == 'Intact Female', 'SexuponOutcome_cat'] = 0
    ts.ix[ts.SexuponOutcome == 'Spayed Female', 'SexuponOutcome_cat'] = 1
    ts.ix[ts.SexuponOutcome == 'Neutered Male', 'SexuponOutcome_cat'] = 2
    ts.ix[ts.SexuponOutcome == 'Intact Male', 'SexuponOutcome_cat'] = 3
    ts.ix[ts.SexuponOutcome == 'Unknown', 'SexuponOutcome_cat'] = 4
    
    #AgeuponOutcome split
    tr['AgeuponOutcome_num']=0
    tr['AgeuponOutcome_lad']=''
    
    ts['AgeuponOutcome_num']=0
    ts['AgeuponOutcome_lad']=''
    
    for i in range(0,tr.shape[0]-1):
        try:
            tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome'][i].split()[0]
        except AttributeError:
            tr['AgeuponOutcome_num'][i]=0
        try:
            tr['AgeuponOutcome_lad'][i]=tr['AgeuponOutcome'][i].split()[1]
        except AttributeError:
            tr['AgeuponOutcome_lad'][i]='day'
        
    for i in range(0,ts.shape[0]-1):
        try:
            ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome'][i].split()[0]
        except AttributeError:
            ts['AgeuponOutcome_num'][i]=0
        try:
            ts['AgeuponOutcome_lad'][i]=ts['AgeuponOutcome'][i].split()[1]
        except AttributeError:
            ts['AgeuponOutcome_lad'][i]='day'
    
    tr.replace('year','years',inplace=True)
    tr.replace('day','days',inplace=True)
    tr.replace('month','months',inplace=True)
    tr.replace('week','weeks',inplace=True)
    ts.replace('year','years',inplace=True)
    ts.replace('day','days',inplace=True)
    ts.replace('month','months',inplace=True)
    ts.replace('week','weeks',inplace=True)
    
    for i in range(0,tr.shape[0]-1):
            if tr['AgeuponOutcome_lad'][i]=='years':
                tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome_num'][i]*365
            if tr['AgeuponOutcome_lad'][i]=='months':
                tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome_num'][i]*30
            if tr['AgeuponOutcome_lad'][i]=='weeks':
                tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome_num'][i]*7
    for i in range(0,ts.shape[0]-1):
            if ts['AgeuponOutcome_lad'][i]=='years':
                ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome_num'][i]*365
            if ts['AgeuponOutcome_lad'][i]=='months':
                ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome_num'][i]*30
            if ts['AgeuponOutcome_lad'][i]=='weeks':
                ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome_num'][i]*7
    
    #Breed split and bin
    #Breed 1 et Breed 2
    tr['Breed1']=''
    tr['Breed2']=''
    ts['Breed1']=''
    ts['Breed2']=''

    
    for i in range(0,ts.shape[0]-1):
        try:
            ts['Breed1'][i]=ts['Breed'][i].split('/')[0]
        except IndexError:
            ts['Breed1'][i]=ts['Breed'][i]
        try:
            ts['Breed2'][i]=ts['Breed'][i].split('/')[1]
        except IndexError:
            ts['Breed2'][i]='One_Breed'
    for i in range(0,tr.shape[0]-1):
        try:
            tr['Breed1'][i]=tr['Breed'][i].split('/')[0]
        except IndexError:
            tr['Breed1'][i]=tr['Breed'][i]
        try:
            tr['Breed2'][i]=tr['Breed'][i].split('/')[1]
        except IndexError:
            tr['Breed2'][i]='One_Breed'
            
    #Color split
    tr['Color1']=''
    tr['Color2']=''
    tr['Color_prin1']=''
    tr['Shade1']=''
    tr['Color_prin2']=''
    tr['Shade2']=''
    
    ts['Color1']=''
    ts['Color2']=''
    ts['Color_prin1']=''
    ts['Shade1']=''
    ts['Color_prin2']=''
    ts['Shade2']=''
    
    for i in range(0,tr.shape[0]-1):
        tr['Color1'][i]=tr['Color'][i].split('/')[0]
        try:
            tr['Color2'][i]=tr['Color'][i].split('/')[1]
        except IndexError:
            tr['Color2'][i]=''
        tr['Color_prin1'][i]=tr['Color1'][i].split(' ')[0]
        try:
            tr['Shade1'][i]=tr['Color1'][i].split(' ')[1]
        except IndexError:
            tr['Shade1'][i]=''        
        tr['Color_prin2'][i]=tr['Color2'][i].split(' ')[0]
        try:    
            tr['Shade2'][i]=tr['Color2'][i].split(' ')[1]
        except IndexError:
            tr['Shade2'][i]=''
            
            
    for i in range(0,ts.shape[0]-1):        
        ts['Color1'][i]=ts['Color'][i].split('/')[0]
        try:
            ts['Color2'][i]=ts['Color'][i].split('/')[1]
        except IndexError:
            ts['Color2'][i]=''
        ts['Color_prin1'][i]=ts['Color1'][i].split(' ')[0]
        try:
            ts['Shade1'][i]=ts['Color1'][i].split(' ')[1]
        except IndexError:
            ts['Shade1'][i]=''        
        ts['Color_prin2'][i]=ts['Color2'][i].split(' ')[0]
        try: 
            ts['Shade2'][i]=ts['Color2'][i].split(' ')[1]
        except IndexError:
            ts['Shade2'][i]=''  

    #UniColor
    tr['Unicolor']=0
    ts['Unicolor']=0
    
    for i in range(0,tr.shape[0]-1):
         if tr['Color2'][i]=='':
            tr['Unicolor'][i]=1

    for i in range(0,ts.shape[0]-1):
         if ts['Color2'][i]=='':
            ts['Unicolor'][i]=1
            
    #Datetime
    tr['year'] = pd.DatetimeIndex(tr['DateTime']).year
    tr['month'] = pd.DatetimeIndex(tr['DateTime']).month
    tr['day'] = pd.DatetimeIndex(tr['DateTime']).day
    tr['hour'] = pd.DatetimeIndex(tr['DateTime']).hour
    tr['minute'] = pd.DatetimeIndex(tr['DateTime']).minute
    
    ts['year'] = pd.DatetimeIndex(ts['DateTime']).year
    ts['month'] = pd.DatetimeIndex(ts['DateTime']).month
    ts['day'] = pd.DatetimeIndex(ts['DateTime']).day
    ts['hour'] = pd.DatetimeIndex(ts['DateTime']).hour
    ts['minute'] = pd.DatetimeIndex(ts['DateTime']).minute
    
    #Season
    tr['Season']=0
    ts['Season']=0
    
    for i in range(0,ts.shape[0]-1):
            if 4<=ts['month'][i]<=6:
                ts['Season'][i]=1
            if 7<=ts['month'][i]<=9:
                ts['Season'][i]=2
            if 10<=ts['month'][i]<=12:
                ts['Season'][i]=3
    for i in range(0,tr.shape[0]-1):
            if 4<=tr['month'][i]<=6:
                tr['Season'][i]=1
            if 7<=tr['month'][i]<=9:
                tr['Season'][i]=2
            if 10<=tr['month'][i]<=12:
                tr['Season'][i]=3
                
    #Holidays
    tr['Holidays']=0
    ts['Holidays']=0
    
    for i in range(0,tr.shape[0]-1):
            #confederate Heroes
            if tr['month'][i]==1 and tr['day'][i]==19:
                 tr['Holidays']=1
            #Texas Independance
            if tr['month'][i]==3 and tr['day'][i]==2:
                 tr['Holidays']=2
            #San Jacinto Day        
            if tr['month'][i]==4 and tr['day'][i]==21:
                 tr['Holidays']=3                       
            #Mothers Day        
            if tr['month'][i]==5 and tr['day'][i]==10:
                 tr['Holidays']=4                       
            #Emancipation Day        
            if tr['month'][i]==6 and tr['day'][i]==19:
                 tr['Holidays']=5
            #Fathers Day        
            if tr['month'][i]==6 and 18<=tr['day'][i]<=22:
                 tr['Holidays']=6
            #Lyndon day       
            if tr['month'][i]==8 and tr['day'][i]==27:
                 tr['Holidays']=7                       
            #Veterans day       
            if tr['month'][i]==11 and tr['day'][i]==11:
                 tr['Holidays']=8 
            #ThanksGiving                    
            if tr['month'][i]==11 and 25<=tr['day'][i]<=27:
                 tr['Holidays']=9
            #Christmas                    
            if tr['month'][i]==12 and 23<=tr['day'][i]<=27:
                 tr['Holidays']=10                        
                    
                    
    for i in range(0,ts.shape[0]-1):
            #confederate Heroes
            if ts['month'][i]==1 and ts['day'][i]==19:
                 ts['Holidays']=1
            #Texas Independance
            if ts['month'][i]==3 and ts['day'][i]==2:
                 ts['Holidays']=2
            #San Jacinto Day        
            if ts['month'][i]==4 and ts['day'][i]==21:
                 ts['Holidays']=3                       
            #Mothers Day        
            if ts['month'][i]==5 and ts['day'][i]==10:
                 ts['Holidays']=4                       
            #Emancipation Day        
            if ts['month'][i]==6 and ts['day'][i]==19:
                 ts['Holidays']=5
            #Fathers Day        
            if ts['month'][i]==6 and 18<=ts['day'][i]<=22:
                 ts['Holidays']=6
            #Lyndon day       
            if ts['month'][i]==8 and ts['day'][i]==27:
                 ts['Holidays']=7                       
            #Veterans day       
            if ts['month'][i]==11 and ts['day'][i]==11:
                 ts['Holidays']=8 
            #ThanksGiving                    
            if ts['month'][i]==11 and 25<=ts['day'][i]<=27:
                 ts['Holidays']=9
            #Christmas                    
            if ts['month'][i]==12 and 23<=ts['day'][i]<=27:
                 ts['Holidays']=10   
                    
    return print(tr.shape), print(ts.shape)

clean(train,test)


# In[ ]:


from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

train['Breed_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed']]
train['Color_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color']]

corpustr_br = train['Breed_vec']
corpustr_col = train['Color_vec']

vectorizertr = CountVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             binary=True , token_pattern=r'\w+' )

tfidftr_br=vectorizertr.fit_transform(corpustr_br).todense()
tfidftr_col=vectorizertr.fit_transform(corpustr_col).todense()

train2=train.join(pd.DataFrame(tfidftr_br,dtype=float)).join(pd.DataFrame(tfidftr_col,dtype=float),rsuffix='_col')


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier

train_select=pd.DataFrame(train2.select_dtypes(include=['float64','int64','int32','float32']))
target=pd.DataFrame(train_select['Target'])

train_select.drop('Target',axis=1,inplace=True)
train_select=train_select.fillna(0)

x_train, x_test, y_train, y_test = train_test_split(train_select,target,test_size=0.30, 
                                                    random_state=30,stratify=target)


# In[ ]:


#gbm
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

clf_gbm=GradientBoostingClassifier(n_estimators=100,random_state=10).fit(x_train,y_train)
clf_probs = clf_gbm.predict_proba(x_test)
print(log_loss(y_test, clf_probs))


# In[ ]:


clf_xgbm=xgb.XGBClassifier(n_estimators=10).fit(x_train,y_train)
clf_probs_x = clf_xgbm.predict_proba(x_test)
print(log_loss(y_test, clf_probs_x))


# In[ ]:


import matplotlib.pyplot as plt
feature_importance = clf_gbm.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos[260:], feature_importance[sorted_idx][260:], align='center')
plt.yticks(pos[260:], x_train.columns.values[sorted_idx][260:])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

print(x_train.columns.values[sorted_idx][0:])

