#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

input_dir="../input/titanic"
print(os.listdir(input_dir))

# Any results you write to the current directory are saved as output.


# <font size="5">**Read data**</font>

# In[ ]:


data=pd.read_csv(input_dir+'/train.csv',header=0)
#data=data.dropna()
data.head()


# <font size="5">**Check how many nans in each column**</font>

# In[ ]:


data.isna().sum()


# <font size=5>**Try to replace nans, use medians. 
#     For Age nan, Guess from name Initial; 
#     For Fare nan, Guess from Pclass
#     **</font>

# #Master--officers/young boy
# #Don--Mr
# #Rev--Reverend, Christian clergy and ministers
# #Dr--...
# #Mme--madam,only one, 24.0
# #Ms--only one,28.0
# #Lady--only one,48.0, probably married
# #Sir
# #Mlle--miss
# #Major,Col,Capt--military related
# #Countess--noble class wife
# #Jonkheer--male, lowest noble
# 

# In[ ]:


def fix_nan(data_df):
    #Age
    for name_string in data_df['Name']:
        data_df['Title']=data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)
    
    #replacing the rare title with more common one.
    mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Rev':'Mr','Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
              'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
    data_df.replace({'Title': mapping}, inplace=True)

    titles=['Mr','Miss','Mrs','Master','Dr']
    for title in titles:
        age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
        #print(age_to_impute)
        data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
    
    #fix any remaining nan Age with median
    data_df.loc[(data_df['Age'].isnull()) , 'Age']= data_df['Age'].median()
    
    #Fare, if inneed
    Pclasss=data_df['Pclass'].unique()
    Pclasss=Pclasss.tolist();
    for pc in Pclasss:
        Fare_to_impute = data_df.groupby('Pclass')['Fare'].median()[pc]
        #print(age_to_impute)
        data_df.loc[(data_df['Fare'].isnull()) & (data_df['Pclass'] == pc), 'Fare'] = Fare_to_impute
    
    #Embark, with "S"
    data.loc[(data['Embarked'].isnull()) , 'Embarked']= "S"
    
    
    return data_df


# In[ ]:


data=fix_nan(data)


# <font size=5>**Replace sex with number**

# In[ ]:


sexMap={'female':0,'male':1}
data=data.replace({'Sex':sexMap})


# <font size=5>**Replace embark with number**

# In[ ]:


data['Embarked'].unique()


# In[ ]:


embarkMap={'S':1,'C':2,'Q':3}
data=data.replace({'Embarked':embarkMap})


# <font size=5>**explore cross correlation**

# In[ ]:


# from: A Data Science Framework: To Achieve 99% Accuracy
def correlationMapPlot(data):
    
    import seaborn as sns

    plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    sns.heatmap(
        data.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)


# In[ ]:


correlationMapPlot(data)


# <Font size=5>**Some other explorations**

# In[ ]:


plt.hist(x=data.loc[(data['Survived']==1) , 'Age'], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)


# <Font size=5>**Feature engineering**

# In[ ]:


#collection of some common feature engineering practices founded
def featureEngineering(data):    
    data['isInfant'] = data ['Age']<5;
    
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    data['isAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'isAlone'] = 1
    #IndividualFare can be dependent to FamilySize
    data['IndividualFare']=(data['Fare'])/(data ['SibSp'] +data['Parch']+1);

    
    return data


# In[ ]:


data=featureEngineering(data)


# In[ ]:


correlationMapPlot(data)


# <font size=5>**Training and test set**

# In[ ]:


X = np.asarray(data[['Sex','Pclass','Fare','SibSp','Parch','Embarked','isInfant','isAlone','FamilySize','IndividualFare']])
y = np.asarray(data[['Survived']])

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)


# <font size=5>**DFF network**

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


model = keras.Sequential([
  keras.layers.Dense(16, activation='relu', input_shape=(10,)),
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(16, activation='relu'),
  keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, epochs=5)


# <font size=5>**Evalution**

# In[ ]:


yhat=model.predict(X_test)


# In[ ]:


yhat=model.predict(X_test)
for i in range(0,len(yhat)):
    if yhat[i]>=0.6:
        yhat[i]=1;
    else:
        yhat[i]=0;   
        


# In[ ]:



from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# <font size=5>**Processing testing set**

# In[ ]:


data_fin.isna().sum()


# In[ ]:


data_fin=pd.read_csv(input_dir+'/test.csv',header=0)
data_fin_1=data_fin.replace({'Sex':sexMap})
data_fin_1=data_fin_1.replace({'Embarked':embarkMap})
data_fin_1=fix_nan(data_fin_1);
data_fin_1=featureEngineering(data_fin_1);
X_fin = np.asarray(data_fin_1[['Sex','Pclass','Fare','SibSp','Parch','Embarked','isInfant','isAlone','FamilySize','IndividualFare']])
X_fin = preprocessing.StandardScaler().fit(X_fin).transform(X_fin)


# In[ ]:


data_fin_1


# In[ ]:


yhat_fin = model.predict(X_fin)
for i in range(0,len(yhat_fin)):
    if yhat_fin[i]>=0.6:
        yhat_fin[i]=1;
    else:
        yhat_fin[i]=0;  


# In[ ]:


yhat_fin=yhat_fin.astype(int)


# <font size=5>**generate gender_submission.csv**

# In[ ]:


yhat_fin = yhat_fin.flatten()


# In[ ]:


yhat_fin.shape


# In[ ]:


submission_dat=pd.DataFrame()
submission_dat.loc[:,'PassengerId']=data_fin['PassengerId']
submission_dat.loc[:,'Survived']=pd.Series(yhat_fin, index=submission_dat.index)


# In[ ]:


submission_dat['Survived'].unique()


# In[ ]:


submission_dat.to_csv('sampleSubmission.csv',index=False)

