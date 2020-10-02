#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1, color_codes=True)

#%reset -f
dataset = pd.read_csv('../input/pokemon.csv') #classfication


# In[ ]:


dataset.drop(['japanese_name','name','pokedex_number'], axis=1, inplace=True)
dataset['capture_rate'] = pd.to_numeric(dataset['capture_rate'], errors='coerce')

dataset.isnull().sum()/ len(dataset)
dataset.columns[dataset.isnull().any()]


# In[ ]:


c_df = dataset.select_dtypes(['object','category'])
c_df['type2'] = c_df['type2'].fillna('Other')
dataset.update(c_df)


# In[ ]:


n_df = dataset.select_dtypes(include=['int64','float64'])
for col in n_df.columns:
    n_df[col] = n_df[col].fillna(n_df[col].median())
dataset.update(n_df)    


# In[ ]:


#Duplicate check
print(sum(dataset.duplicated(dataset.columns)))
dataset = dataset.drop_duplicates(dataset.columns, keep='last')    


# In[ ]:


#import phik
#from phik import resources, report
#
#corr = dataset.phik_matrix()
#corr = corr['is_legendary'].abs()
#print(corr.sort_values())
#
#to_drop_1 = [col for col in corr.index if corr[col]<0.2]
#dataset.drop(to_drop_1, axis=1, inplace=True)
#
#corr = dataset.phik_matrix()
#col = corr.index
#for i in range(len(col)):
#    for j in range(i+1, len(col)):
#        if corr.iloc[i,j] >= 0.8:
#            print(f"{col[i]} -{col[j]}")


# In[ ]:


dataset.drop(['against_dark'
,'against_dragon'
,'against_flying'
,'base_happiness'
,'experience_growth'
,'height_m'
,'hp'
,'sp_defense'
,'speed'
,'type1'],axis=1, inplace=True)   

dataset.drop(['weight_kg'
,'sp_attack'
,'capture_rate'],axis=1, inplace=True)   


# In[ ]:



dataset['abilities'] = dataset['abilities'].apply(lambda x: x.replace('[',''))
dataset['abilities'] = dataset['abilities'].apply(lambda x: x.replace(']',''))


# In[ ]:


mask = (dataset['abilities'] != 'Levitate')         & (dataset['abilities'] != 'Beast Boost')         & (dataset['abilities'] != 'Shed Skin')        & (dataset['abilities'] != 'Justified')        & (dataset['abilities'] != "'Keen Eye', 'Tangled Feet', 'Big Pecks'")        & (dataset['abilities'] != "'Poison Point', 'Rivalry', 'Hustle'")        & (dataset['abilities'] != "'Clear Body', 'Light Metal'")        & (dataset['abilities'] != "'Overgrow', 'Long Reach'")        & (dataset['abilities'] != "'Flash Fire', 'Flame Body', 'Infiltrator'")        & (dataset['abilities'] != "'Rivalry', 'Intimidate', 'Guts'")
dataset.loc[mask, 'abilities'] = 'Other'


# In[ ]:


dataset['abilities'].nunique()
dataset['classfication'].nunique()
dataset.drop(['classfication'],axis=1, inplace=True)   


# In[ ]:


dataset.columns


# In[ ]:



dataset = pd.get_dummies(dataset)


# In[ ]:


X = dataset

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)

y = dataset['is_legendary']


# In[ ]:



#Splitting the Dataset into Training set and Test Set
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=7)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Classification
models =[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))



# In[ ]:



for name,model in models:
    cv_res = model_selection.cross_val_score(model,X_train,y_train,cv=10,scoring='accuracy')
    cv_predict = model_selection.cross_val_predict(model,X_train,y_train,cv=10)
    print(f"{name}: {confusion_matrix(y_train, cv_predict)}")
    print(f"{accuracy_score(y_train,cv_predict)}")


# In[ ]:


for name,model in models:
    model.fit(X_train,y_train)
    pres= model.predict(X_test)
    print(accuracy_score(pres,y_test))
    print(confusion_matrix(y_test,pres))
    print(classification_report(y_test,pres))


# In[ ]:




