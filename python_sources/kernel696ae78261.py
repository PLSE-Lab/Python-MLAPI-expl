#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 1 Load Libraries

# In[ ]:


import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
#from google.colab import files
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# ## 2 Get data, including EDA

# In[ ]:


# load the datasets
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")

print("Train size: {0}\nTest size: {1}".format(train.shape,test.shape))


# In[ ]:


train.info()


# Cabin might be a relevant feature, since its location could be determinant to the survival rate of their occupant. However, more than 70% of the data is missing. We may try to retrive indirectly this information using fare feature.

# In[ ]:


aux = train.copy()
print(aux.Fare.isnull().sum())


# In[ ]:


plt.hist(train['Fare'],bins=20)


# In[ ]:


# divide fare column into a range of values
    cut_points = [0,50,100,150,200]
    label_names = ["F1","F2","F3","F4"]
    aux["Fare_categories"] = pd.cut(aux["Fare"],
                                 cut_points,
                                 labels=label_names)


# In[ ]:


# Create dimensions
embarked_dim = go.parcats.Dimension(values=aux.Embarked, label="Embarked")

gender_dim = go.parcats.Dimension(values=aux.Sex, label="Gender")

class_dim = go.parcats.Dimension(
    values=aux.Pclass,
    categoryorder='category ascending', label="Class"
)

fare_dim = go.parcats.Dimension(values=aux.Fare_categories, label="Fare_categories")

survival_dim = go.parcats.Dimension(
    values=aux.Survived, label="Outcome", categoryarray=[0, 1], 
    ticktext=['perished', 'survived']
)

# Create parcats trace
color = aux.Survived
colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]

fig = go.Figure(data = [go.Parcats(dimensions=[gender_dim, class_dim, fare_dim, survival_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='all',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},bundlecolors=True, 
        arrangement='freeform')])
fig.update_layout(width=800,height=500)

fig.show()


# Age have 20% of missing values, we can imput with a good guess this data. (tip from source: https://www.kaggle.com/goldens/titanic-on-the-top-with-a-simple-model)

# In[ ]:


aux['Age'].isnull().sum()/aux.shape[0]


# In[ ]:


aux['title'] = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

aux['title'].unique()


# In[ ]:


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "Countess":   "Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
    }

aux['title'] = aux.title.map(newtitles)


# In[ ]:


# Create dimensions
embarked_dim = go.parcats.Dimension(values=aux.Embarked, label="Embarked")

gender_dim = go.parcats.Dimension(values=aux.Sex, label="Gender")

class_dim = go.parcats.Dimension(
    values=aux.Pclass,
    categoryorder='category ascending', label="Class"
)

title_dim = go.parcats.Dimension(values=aux.title, label="Title")

survival_dim = go.parcats.Dimension(
    values=aux.Survived, label="Outcome", categoryarray=[0, 1], 
    ticktext=['perished', 'survived']
)

# Create parcats trace
color = aux.Survived
colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]

fig = go.Figure(data = [go.Parcats(dimensions=[gender_dim, class_dim, title_dim, survival_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='all',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},bundlecolors=True, 
        arrangement='freeform')])
fig.update_layout(width=800,height=500)

fig.show()


# In[ ]:


aux.groupby(['title','Sex']).Age.mean()


# In[ ]:


def newage (cols):
    title=cols[0]
    Sex=cols[1]
    Age=cols[2]
    if pd.isnull(Age):
        if title=='Master' and Sex=="male":
            return 4.57
        elif title=='Miss' and Sex=='female':
            return 21.8
        elif title=='Mr' and Sex=='male': 
            return 32.37
        elif title=='Mrs' and Sex=='female':
            return 35.72
        elif title=='Officer' and Sex=='female':
            return 49
        elif title=='Officer' and Sex=='male':
            return 46.56
        elif title=='Royalty' and Sex=='female':
            return 40.50
        else:
            return 42.33
    else:
        return Age


# In[ ]:


aux['Age'] = aux[['title','Sex','Age']].apply(newage, axis=1)
np.dtype(aux['Age'])


# Checking if we can extract some correlation with tickets so maybe the cabin information can be retrieved. (Tip from: https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)

# In[ ]:


aux['Ticket'].isnull().sum()


# In[ ]:


aux["Ticket"].unique()


# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 
Ticket = []
for i in list(aux.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) # Take prefix
    else:
        Ticket.append("X")
        
aux["Ticket1"] = Ticket
aux["Ticket1"].unique()
#aux["Ticket1"].unique().shape


# In[ ]:


aux2 = test.copy()

## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 
Ticket = []
for i in list(aux2.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) # Take prefix
    else:
        Ticket.append("X")
        
aux2["Ticket1"] = Ticket
aux2["Ticket1"].unique()
#aux2["Ticket1"].unique().shape


# In[ ]:


# Create dimensions
embarked_dim = go.parcats.Dimension(values=aux.Embarked, label="Embarked")

gender_dim = go.parcats.Dimension(values=aux.Sex, label="Gender")

class_dim = go.parcats.Dimension(
    values=aux.Pclass,
    categoryorder='category ascending', label="Class"
)

ticket_dim = go.parcats.Dimension(values=aux.Ticket1, label="Ticked1")

survival_dim = go.parcats.Dimension(
    values=aux.Survived, label="Outcome", categoryarray=[0, 1], 
    ticktext=['perished', 'survived']
)

# Create parcats trace
color = aux.Survived
colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]

fig = go.Figure(data = [go.Parcats(dimensions=[gender_dim, class_dim, ticket_dim, survival_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='all',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},bundlecolors=True, 
        arrangement='freeform')])
fig.update_layout(width=800,height=500)

fig.show()


# ## 3 Clean, prepare and manipulate Data (feature engineering)

# Useful functions.

# In[ ]:


def create_dummies(df,column_name):
    # drop_first = True to avoid colinearity
    dummies = pd.get_dummies(df[column_name],
                             prefix=column_name,
                             drop_first=True)
    df = pd.concat([df,dummies],axis=1)
    return df


# In[ ]:


#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector(BaseEstimator, TransformerMixin ):
  #Class Constructor 
  def __init__( self, feature_names ):
    self.feature_names = feature_names 
    
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
    return self 
    
  #Method that describes what we need this transformer to do
  def transform(self, X, y = None):
    return X[self.feature_names]


# # 3.1 Pipelines

# ### 3.1.1 Categorical Pipeline

# In[ ]:


#converts certain features to categorical
class CategoricalTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, model=0):
    """Class constructor method that take: 
    model: 
      - 0: Sex column (categorized), Pclass (raw)
      - 1: Sex column (get_dummies(drop_first=True)), Pclass (raw)
      - 2: Sex column (get_dummies(drop_first=True)), Pclass (get_dummies(drop_first=False))
      - 3: Sex column (get_dummies(drop_first=True)), Pclass (raw), Age (get_dummies(drop_first=False))
      - 4: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (categorized)
      - 5: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 6: New Sex column (get_dummies(drop_first=False)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 7: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 8: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size
      - 9: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size, Embarked (get_dummies(drop_first=False))
      - 10: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Fare (get_dummies(drop_first=False))
      - 11: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), Fare (get_dummies(drop_first=False))
      - 12: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), Fare2 (scaled)
      - 13: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), ticket(encoded) 
      - 14: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), ticket(get_dummies(drop_first=True)) 
      - 15: genderModel(get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
    """
    self.model = model

  #Return self nothing else to do here    
  def fit( self, X, y = None ):
    return self 

  def create_dummies(self, df, column_name, drop_first_col):
    """Create Dummy Columns from a single Column
    """
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=drop_first_col)
    return dummies
    
  def apply_one_hot_encoder(self, df_fit, df_transform, col):
      
    labeler = LabelEncoder()
    labeler.fit( df_fit[col] )    
  
    npa_encoded_fit = labeler.transform( df_fit[col] )
    npa_encoded = labeler.transform( df_transform[col] )
      
    encoder = OneHotEncoder( categories='auto', drop='first', dtype=np.uint8 )   
    encoder.fit( npa_encoded_fit.reshape(-1,1) )
      
    npa_hot = encoder.transform( npa_encoded.reshape(-1,1) )   
    df_hot = pd.DataFrame( npa_hot.toarray(), index=df_transform.index )
      
    cols = [ col + '_' + str(num) for num in df_hot.columns.tolist() ]
    df_hot.columns = cols
      
    return df_hot
  
  def apply_encoder(self, df_fit, df_transform, col):
    labeler = LabelEncoder()
    labeler.fit( df_fit[col] )    
  
    npa_encoded = labeler.transform( df_transform[col] )
      
    df_encoded = pd.DataFrame( npa_encoded, columns=[col], index=df_transform.index)
        
    return df_encoded

  def process_family(self, df):
         
    # create sex column
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    titles = {
        "Mr" :         "man",
        "Mme":         "woman",
        "Ms":          "woman",
        "Mrs" :        "woman",
        "Master" :     "boy",
        "Mlle":        "woman",
        "Miss" :       "woman",
        "Capt":        "man",
        "Col":         "man",
        "Major":       "man",
        "Dr":          "man",
        "Rev":         "man",
        "Jonkheer":    "man",
        "Don":         "man",
        "Sir" :        "man",
        "Countess":    "woman",
        "Dona":        "woman",
        "Lady" :       "woman"
    }
    
    df["Sex"]    = df["Title"].map(titles)

    # get surname
    df["Surname"]    = df["Name"].str.extract('([A-Za-z]*),.',expand=False)

    # if man: no group
    df.loc[df["Sex"]    == 'man', 'Surname'] = 'NoGroup'

    # if alone: no group
    df['SurnameFreq'] = df['Surname'].map(df['Surname'].value_counts().to_dict())
    df.loc[df["SurnameFreq"] <= 1, 'Surname'] = 'NoGroup'
    
    df.loc[ df["Surname"] != 'NoGroup', 'Surname'] = 'woman-child-groups'

    if self.model == 15:
      # encode
      df_encoded = self.create_dummies(df, "Surname", True)
      return df_encoded
    else:
      return None

  # need column survived
  def process_survivalRate(self, df): 
    ## survival family rate
    # man and loners dies
    df['SurvivalRate'] = 0
  
    # survival family rate = mean of survived status in each worman-boy-family group
    surnames_list = df.Surname.unique().tolist()
    surnames_list.remove('NoGroup')
    for s in surnames_list:
      df.loc[ df['Surname'] == s, 'SurvivalRate'] = df[ df['Surname'] == s].Survived.mean()

    # adjust survival rates for use on training set 
    # discount yourself
    #df['SurvivalRateAjusted'] = (df['SurvivalRate']*df['SurnameFreq'] - df['Survived']) / (df['SurnameFreq'] - 1)

    if self.model == 15:
      return df
    else:
      return None

  def process_family_size(self, df):
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    titles = {
        "Mr" :         "man",
        "Mme":         "woman",
        "Ms":          "woman",
        "Mrs" :        "woman",
        "Master" :     "boy",
        "Mlle":        "woman",
        "Miss" :       "woman",
        "Capt":        "man",
        "Col":         "man",
        "Major":       "man",
        "Dr":          "man",
        "Rev":         "man",
        "Jonkheer":    "man",
        "Don":         "man",
        "Sir" :        "man",
        "Countess":    "woman",
        "Dona":        "woman",
        "Lady" :       "woman"
    } 

    # new gender: man, woman, boy
    df["Gender"] = df["Title"].map(titles)

    # family surname
    df["family"] = df["Name"].str.extract('([A-Za-z]+)\,',expand=False)

    # count the number of boy and women by family
    boy_women = df[df["Gender"] != "man"].groupby(by=["family"])["Name"].agg("count")

    # fill with zero that passengers are traveling alone or with family without boy and women
    df["family_size"] = df["family"].map(boy_women).fillna(0.0)

    if self.model in [8,9]:
      return pd.DataFrame(df["family_size"],columns=["family_size"])
    else:
      return None

  def process_sex(self, df):
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    titles = {
        "Mr" :         "man",
        "Mme":         "woman",
        "Ms":          "woman",
        "Mrs" :        "woman",
        "Master" :     "boy",
        "Mlle":        "woman",
        "Miss" :       "woman",
        "Capt":        "man",
        "Col":         "man",
        "Major":       "man",
        "Dr":          "man",
        "Rev":         "man",
        "Jonkheer":    "man",
        "Don":         "man",
        "Sir" :        "man",
        "Countess":    "woman",
        "Dona":        "woman",
        "Lady" :       "woman"
    }
    
    if self.model == 0:
      df["Sex"] = pd.Categorical(df.Sex).codes
      return pd.DataFrame(df["Sex"],columns=["Sex"])
    elif self.model in [1,2,3,4,5]:  
      sex_dummies = self.create_dummies(df,"Sex",True)
      return sex_dummies
    elif self.model == 6:
      df["Sex"] = df["Title"].map(titles)
      sex_dummies = self.create_dummies(df,"Sex",False)
      return sex_dummies
    elif self.model in [7,8,9,10,11,12,13,14]:
      df["Sex"] = df["Title"].map(titles)
      sex_dummies = self.create_dummies(df,"Sex",False)
      sex_dummies.drop(labels="Sex_woman",axis=1,inplace=True)
      return sex_dummies
    else:
      return None

  def process_embarked(self, df):
    if self.model in [0,1,2,3,8,10]:
      return None
    elif self.model == 4:
      # fill null values using the mode
      df["Embarked"].fillna("S",inplace=True)
      df["Embarked"] = pd.Categorical(df.Embarked).codes
      return pd.DataFrame(df["Embarked"],columns=["Embarked"])
    elif self.model in [5,6,7,9,11,12,13,14,15]:
      df["Embarked"].fillna("S",inplace=True)
      embarked_dummies = self.create_dummies(df,"Embarked",False)
      return embarked_dummies

  def process_ticket(self, df):
    ## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

    # extracting prefic from fit df
    Ticket = []
    for i in list(train.Ticket):
      if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) # Take prefix
      else:
        Ticket.append("X")
        
    train["Ticket"] = Ticket

    Ticket = []
    for i in list(test.Ticket):
      if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) # Take prefix
      else:
        Ticket.append("X")
        
    test["Ticket"] = Ticket

    #  extracting prefic from transform df
    Ticket = []
    for i in list(df.Ticket):
      if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) # Take prefix
      else:
        Ticket.append("X")
        
    df["Ticket"] = Ticket

    if self.model == 13:
      ticket_encoded = self.apply_encoder(pd.concat([train, test], sort=False), df, "Ticket")
      return ticket_encoded
    elif self.model == 14:
      ticket_dummyfied = self.apply_one_hot_encoder(pd.concat([train, test], sort=False), df, "Ticket")
      return ticket_dummyfied
    else:
      return None

  #Transformer method we wrote for this transformer 
  def transform(self, X , y = None ):
    df = X.copy()
    sex = self.process_sex(df)
    embarked = self.process_embarked(df)
    family_size = self.process_family_size(df)
    ticket = self.process_ticket(df)
    genderModel = self.process_family(df)

    if self.model in [0,1,2,3]:
      return sex
    elif self.model in [4,5,6,7,11,12]:
      return pd.concat([sex,embarked],axis=1)
    elif self.model == 8:
      return pd.concat([sex,family_size],axis=1)
    elif self.model == 9:
      return pd.concat([sex,family_size,embarked],axis=1)
    elif self.model == 10:
      return pd.concat([sex],axis=1)
    elif self.model in [13, 14]:
      return pd.concat([sex,embarked,ticket],axis=1) 
    elif self.model == 15:
      return pd.concat([genderModel,embarked],axis=1)   
    else:
      return None


# In[ ]:


# for validation purposes only
select = FeatureSelector(train.select_dtypes(include=["object"]).columns).transform(train)

# change the value of model 0,1,2,3,....9
model = CategoricalTransformer(model=15)
df_cat = model.transform(select)
cat_cols_final = df_cat.columns
df_cat.head()


# In[ ]:


cat_cols_final


# ### 3.1.2 Numerical Pipeline

# In[ ]:


# converts certain features to numerical 
class NumericalTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, model=0):
    """Class constructor method that take: 
    model: 
      - 0: Sex column (categorized), Pclass (raw)
      - 1: Sex column (get_dummies(drop_first=True)), Pclass (raw)
      - 2: Sex column (get_dummies(drop_first=True)), Pclass (get_dummies(drop_first=False))
      - 3: Sex column (get_dummies(drop_first=True)), Pclass (raw), Age (get_dummies(drop_first=False))
      - 4: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (categorized)
      - 5: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 6: New Sex column (get_dummies(drop_first=False)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 7: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 8: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size
      - 9: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size, Embarked (get_dummies(drop_first=False))
      - 10: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Fare (get_dummies(drop_first=False))
      - 11: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), Fare (get_dummies(drop_first=False))
      - 12: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), Fare2 (scaled)
      - 13: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), ticket(encoded) 
      - 14: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), ticket(get_dummies(drop_first=True)) 
      - 15: genderModel(get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
    """
    self.model = model

  #Return self nothing else to do here    
  def fit( self, X, y = None ):
    return self 

  def create_dummies(self, df, column_name, drop_first_col):
    """Create Dummy Columns from a single Column
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name, drop_first=drop_first_col)
    return dummies

  # manipulate column "Age"
  def process_age(self, df):
    # fill missing values with -0.5
    #df["Age"] = df["Age"].fillna(-0.5)

    # impute missing value using title information
    df['Title2'] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "Countess":   "Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
    }

    df['Title2'] = df.Title2.map(newtitles)

    df['Age'] = df[['Title2','Sex','Age']].apply(newage, axis=1)

    # divide age column into a range of values
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],
                                 cut_points,
                                 labels=label_names)
          
    if self.model == 3:
      return self.create_dummies(df,"Age_categories",False)
    else:
      return None
   
  def process_fare(self, df):
    # divide fare column into a range of values
    cut_points = [0,50,100,150,200]
    label_names = ["F1","F2","F3","F4"]
    df["Fare_categories"] = pd.cut(df["Fare"],
                                 cut_points,
                                 labels=label_names)
         
    if self.model in [10,11]:
      return self.create_dummies(df,"Fare_categories",True)
    else:
      return None

  def process_fare2(self, df):
    # Apply log to Fare to reduce skewness distribution
    df["Fare_norm"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    return pd.DataFrame(df["Fare_norm"],columns=["Fare_norm"])

  def process_pclass(self, df):
    if self.model in [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15]:
      return pd.DataFrame(df["Pclass"],columns=["Pclass"])
    elif self.model == 2:
      return self.create_dummies(df,"Pclass",False) 
    else:
      return None
        
  #Transformer method we wrote for this transformer 
  def transform(self, X , y = None ):
    df = X.copy()

    age = self.process_age(df)  
    pclass = self.process_pclass(df)
    fare = self.process_fare(df)
    fare2 = self.process_fare2(df)

    
    if self.model in [0,1,2,4,5,6,7,8,9,13,14,15]:
      return pclass
    elif self.model == 3:
      return pd.concat([pclass,age],axis=1)
    elif self.model in [10,11]:
      return pd.concat([pclass,fare],axis=1)
    elif self.model in [12]:
      return pd.concat([pclass,fare2],axis=1)  
    else:
      return None


# In[ ]:


# for validation purposes only
select = FeatureSelector(pd.concat([train.drop(labels=["Survived"],axis=1).select_dtypes(include=["int64","float64"])
                                   ,train[['Name','Sex']]],axis=1).columns).transform(train)

# change model to 0,1,2,3, ..., 15
model = NumericalTransformer(model=15)
df = model.transform(select)
num_cols_final = df.columns
df.head()


# In[ ]:


num_cols_final


# In[ ]:


plt.figure(figsize = (10,10))
sns.heatmap(pd.concat([df.select_dtypes(include=["int64","uint8","float64"]),
                       df_cat.select_dtypes(include=["int64","uint8"]),
                       train.Survived],axis=1).corr(),cmap=("RdBu_r"),annot=True,fmt='.2f')
plt.xticks(rotation=90) 
plt.show()


# In[ ]:


pd.concat([df.select_dtypes(include=["int64","uint8","float64"]),
                       df_cat.select_dtypes(include=["int64","uint8"]),
                       train.Survived],axis=1).corr()["Survived"].abs().sort_values()


# ## 4 Modeling

# In[ ]:


# global varibles
seed = 42
num_folds = 10
scoring = {'Accuracy': make_scorer(accuracy_score)}


# In[ ]:


# load the datasets
train = pd.read_csv("../input/titanic/train.csv")

# split-out train/validation and test dataset
X_train, X_test, y_train, y_test = train_test_split(train.drop(labels="Survived",axis=1),
                                                    train["Survived"],
                                                    test_size=0.20,
                                                    random_state=seed,
                                                    shuffle=True,
                                                    stratify=train["Survived"])


# In[ ]:


# Categrical features to pass down the categorical pipeline 
categorical_features = X_train.select_dtypes(include=["object"]).columns

# Numerical features to pass down the numerical pipeline 
numerical_features = pd.concat([X_train.select_dtypes(include=["int64","float64"]),
                                X_train[['Name','Sex']] ], axis=1).columns


# Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),
                                         ('cat_transformer', CategoricalTransformer(model=15)), 
                                         ]
                                )
# Defining the steps in the numerical pipeline     
numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),
                                       ('num_transformer', NumericalTransformer(model=15)) 
                                       ]
                              )

# Combining numerical and categorical piepline into one full big pipeline horizontally 
# using FeatureUnion
full_pipeline_preprocessing = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_pipeline),
                                                               ('numerical_pipeline', numerical_pipeline)
                                                               ]
                                           )


# In[ ]:


# for validate purposes
new_data = full_pipeline_preprocessing.fit_transform(X_train)
new_data_df = pd.DataFrame(new_data,)#columns=cat_cols_final.tolist() + num_cols_final.tolist())
new_data_df.head()


# ## 5 Algorithm Tuning

# In[ ]:


"""
    model: 
      - 0: Sex column (categorized), Pclass (raw)
      - 1: Sex column (get_dummies(drop_first=True)), Pclass (raw)
      - 2: Sex column (get_dummies(drop_first=True)), Pclass (get_dummies(drop_first=False))
      - 3: Sex column (get_dummies(drop_first=True)), Pclass (raw), Age (get_dummies(drop_first=False))
      - 4: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (categorized)
      - 5: Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 6: New Sex column (get_dummies(drop_first=False)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 7: New Sex column (get_dummies(drop_first=False)+drop(Sex_woman)), Pclass (raw), Embarked (get_dummies(drop_first=False))
      - 8: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size
      - 9: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Family_Size, Embarked (get_dummies(drop_first=False))
      - 10: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Fare (get_dummies(drop_first=False))
      - 11: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), Fare (get_dummies(drop_first=False))
      - 12: New Sex column (get_dummies(drop_first=True)+drop(Sex_woman)), Pclass (raw), Embarked (get_dummies(drop_first=False)), Fare2 (scaled)
      - 13: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), ticket(encoded) 
      - 14: New Sex column (get_dummies(drop_first=True)), Pclass (raw), Embarked (get_dummies(drop_first=False)), ticket(get_dummies(drop_first=True)) 
"""

# The full pipeline as a step in another pipeline with an estimator as the final step
pipe = Pipeline(steps = [('full_pipeline', full_pipeline_preprocessing),
                         #("fs",SelectKBest()),
                         ("clf",XGBClassifier())])

# create a dictionary with the hyperparameters
search_space = [
                {"clf":[RandomForestClassifier()],
                 "clf__n_estimators": [100],
                 "clf__criterion": ["entropy"],
                 "clf__max_leaf_nodes": [64],
                 "clf__random_state": [seed],
                 "full_pipeline__numerical_pipeline__num_transformer__model":[14],
                 "full_pipeline__categorical_pipeline__cat_transformer__model":[14]                
                 },
                {"clf":[LogisticRegression()],
                 "clf__solver": ["liblinear"],
                 "full_pipeline__numerical_pipeline__num_transformer__model":[14],
                 "full_pipeline__categorical_pipeline__cat_transformer__model":[14]
                 },
                {"clf":[GradientBoostingClassifier()],
                 "clf__max_depth": [2],
                 "clf__n_estimators": [3],
                 "clf__learning_rate": [1.0],
                 "clf__random_state": [seed],
                 "full_pipeline__numerical_pipeline__num_transformer__model":[14],
                 "full_pipeline__categorical_pipeline__cat_transformer__model":[14]                 
                 },
                {"clf":[XGBClassifier()],
                 "clf__n_estimators": [50],
                 "clf__max_depth": [3],
                 'clf__min_child_weight': [1],
                 "clf__learning_rate": [0.01],
                 "clf__random_state": [seed],
                 "clf__subsample": [0.7],
                 "clf__colsample_bytree": [1.0],
                 "full_pipeline__numerical_pipeline__num_transformer__model":[14],
                 "full_pipeline__categorical_pipeline__cat_transformer__model":[14]
                 },
                {"clf":[AdaBoostClassifier()],
                 "clf__base_estimator": [DecisionTreeClassifier(max_depth=2)],
                 "clf__algorithm": ["SAMME.R"],
                 "clf__n_estimators": [200],
                 "clf__learning_rate": [1.0],
                 "clf__random_state": [seed],
                 "full_pipeline__numerical_pipeline__num_transformer__model":[14],
                 "full_pipeline__categorical_pipeline__cat_transformer__model":[14]
                 }
                ]

# create grid search
kfold = StratifiedKFold(n_splits=num_folds,random_state=seed)

# return_train_score=True
# official documentation: "computing the scores on the training set can be
# computationally expensive and is not strictly required to
# select the parameters that yield the best generalization performance".
grid = GridSearchCV(estimator=pipe, 
                    param_grid=search_space,
                    cv=kfold,
                    scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1,
                    refit="Accuracy")

tmp = time.time()

# fit grid search
best_model = grid.fit(X_train,y_train)

print("CPU Training Time: %s seconds" % (str(time.time() - tmp)))


# In[ ]:


print("Best: %f using %s" % (best_model.best_score_,best_model.best_params_))


# In[ ]:


result = pd.DataFrame(best_model.cv_results_)
result.head()


# In[ ]:


result_acc = result[['mean_train_Accuracy', 'std_train_Accuracy','mean_test_Accuracy', 'std_test_Accuracy','rank_test_Accuracy']].copy()
result_acc["std_ratio"] = result_acc.std_test_Accuracy/result_acc.std_train_Accuracy
result_acc.sort_values(by="rank_test_Accuracy",ascending=True)


# In[ ]:


# best model
predict_first = best_model.best_estimator_.predict(X_test)
print(accuracy_score(y_test, predict_first))


# ## 6 Submission File

# In[ ]:


predict_final = best_model.best_estimator_.predict(test)


# In[ ]:


holdout_ids = test["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": predict_final}
submission = pd.DataFrame(submission_df)

submission.to_csv("submission.csv",index=False)


# ## 7 Non-ML Models

# Gender model, following: https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818.
# 
# Hypothesis: woman and children were prioritized in rescue + woman and children from the same family survived or perished together.

# In[ ]:


# load the datasets
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")


# In[ ]:


def process_sex(df):
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    
    titles = {
        "Mr" :         "man",
        "Mme":         "woman",
        "Ms":          "woman",
        "Mrs" :        "woman",
        "Master" :     "boy",
        "Mlle":        "woman",
        "Miss" :       "woman",
        "Capt":        "man",
        "Col":         "man",
        "Major":       "man",
        "Dr":          "man",
        "Rev":         "man",
        "Jonkheer":    "man",
        "Don":         "man",
        "Sir" :        "man",
        "Countess":    "woman",
        "Dona":        "woman",
        "Lady" :       "woman"
    }
    
    df["Sex"] = df["Title"].map(titles)
    
    return df


# In[ ]:


# create sex and surname column
def process_family(df):
  ## create sex column
  df = process_sex(df)

  ## get surname
  df["Surname"] = df["Name"].str.extract('([A-Za-z]*),.',expand=False)
  
  ## creating worman-boy-family-groups
  # if man: no group
  df.loc[df["Sex"] == 'man', 'Surname'] = 'NoGroup'
  
  # if alone: no group
  df['SurnameFreq'] = df['Surname'].map(df['Surname'].value_counts().to_dict())
  df.loc[df["SurnameFreq"] <= 1, 'Surname'] = 'NoGroup'

  return df

def process_survivalRate(df):
  ## survival family rate
  # man and loners dies
  df['SurvivalRate'] = 0
  
  # survival family rate = mean of survived status in each worman-boy-family group
  surnames_list = df.Surname.unique().tolist()
  surnames_list.remove('NoGroup')
  for s in surnames_list:
    df.loc[ df['Surname'] == s, 'SurvivalRate'] = df[ df['Surname'] == s].Survived.mean()

  # adjust survival rates for use on training set 
  # discount yourself
  df['SurvivalRateAjusted'] = (df['SurvivalRate']*df['SurnameFreq'] - df['Survived']) / (df['SurnameFreq'] - 1)

  return df


# In[ ]:


aux = process_family(train)

surnames_list = aux.Surname.unique().tolist()
surnames_list.remove('NoGroup')

for s in surnames_list:
  aux.loc[ aux['Surname'] == s, 'SurvivalRate'] = aux[ aux['Surname'] == s].Survived.mean()


# Gender model vs 2 (without ticket correction, but doing SurvivalRate correction)
# 
# Score: 0.83253

# In[ ]:


# load the datasets
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")

# create  worman-boy-family-groups for all dataset
aux = process_family( pd.concat([train,test], sort=False, keys=['train','test']) )
new_train = aux.loc['train'].copy()
new_test = aux.loc['test'].copy()

## survival family rate
new_train = process_survivalRate(new_train)


# survival family rate = mean of survived status in each worman-boy-family group
new_test['SurvivalRate'] = 0

for i in new_test.index:
  if not new_train[new_test.iloc[i-1,:].Surname == new_train['Surname']].empty:
    new_test.iloc[i-1, 15] = new_train[new_test.iloc[i-1,:].Surname == new_train['Surname']].SurvivalRateAjusted.mean()

# apply gender model to test dataset
new_test['Survived'] = 0
new_test.loc[ (new_test['Sex'] == 'boy') & (new_test['SurvivalRate'] == 1), 'Survived']  = 1
new_test.loc[ (new_test['Sex'] == 'woman') , 'Survived']  = 1
new_test.loc[ (new_test['Sex'] == 'woman') & (new_test['SurvivalRate'] == 0), 'Survived'] = 0

print(new_test.Survived.value_counts())


# In[ ]:


holdout_ids = new_test["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": new_test["Survived"]}
submission = pd.DataFrame(submission_df)

submission.to_csv("submission_genderModel_vs2.csv",index=False)

