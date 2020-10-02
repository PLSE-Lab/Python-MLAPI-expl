#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install chart_studio


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from IPython.display import display, HTML

# Using plotly + cufflinks in offline mode

import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.set_config_file(offline=True)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
import plotly_express as px
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/train.csv.zip')
test = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/test.csv.zip')
sam_submit = pd.read_csv('/kaggle/input/forest-cover-type-kernels-only/sample_submission.csv.zip')
df = train.copy()


# In[ ]:


train.describe().T


# In[ ]:


train.shape,test.shape


# In[ ]:


df.drop(['Id','Soil_Type7','Soil_Type15'],axis = 1,inplace = True)
test.drop(['Id','Soil_Type7','Soil_Type15'],axis = 1,inplace = True)


# In[ ]:


df.Elevation.iplot(kind = 'hist',bins = 40,linecolor = 'black',xTitle = 'Elevation',yTitle = 'count',title = 'Elevation distributions')


# In[ ]:


df.Aspect.iplot(kind = 'hist',linecolor = 'black',title = 'Aspect Distribution')


# In[ ]:


df.Slope.iplot(kind = 'hist',linecolor = 'black',title = 'Slope distribution')


# In[ ]:


df.Vertical_Distance_To_Hydrology.iplot(kind = 'hist',linecolor = 'black',xTitle='Vertical Distance to Hydrology',yTitle='Count')


# In[ ]:


df.Horizontal_Distance_To_Hydrology.iplot(kind = 'hist',linecolor = 'black',xTitle='Horizontal_Distance_To_Hydrology',yTitle='count')


# In[ ]:


df.Horizontal_Distance_To_Fire_Points.iplot(kind = 'hist',linecolor = 'black',xTitle='Horizontal_Distance_To_Fire_Points',yTitle='count')


# In[ ]:


df.Horizontal_Distance_To_Roadways.iplot(kind = 'hist',linecolor = 'black',xTitle='Horizontal_Distance_To_Roadways',yTitle='count')


# In[ ]:


fig = go.Figure().add_trace(go.Histogram(x = df.Hillshade_3pm,name = 'Hillshade_3pm'))
fig.add_trace(go.Histogram(x = df.Hillshade_Noon,name = 'Hillshade_noon'))
fig.add_trace(go.Histogram(x = df.Hillshade_9am,name = 'Hillshade_9am'))


# In[ ]:


sns.countplot(df.Wilderness_Area1)
plt.show()


# In[ ]:


sns.countplot(df.Wilderness_Area2)
plt.show()


# In[ ]:


sns.countplot(df.Wilderness_Area3)
plt.show()


# In[ ]:


sns.countplot(df.Wilderness_Area4)
plt.show()


# #### ***Wilderness Area and Soil Type are One Hot Encoded Features because of it the dimensionality of our dataset has becoome large, so I will now first decode Soil Type and Wilderness Area Features and after that i will try some other technique for encoding them***

# In[ ]:


w1 = df.Wilderness_Area1.value_counts().to_dict()


# In[ ]:


w1[1] = 'Wild_area1'


# In[ ]:


w1[0] = '0'


# In[ ]:


df.Wilderness_Area1 = df.Wilderness_Area1.map(w1)


# In[ ]:


df.Wilderness_Area1.value_counts()


# In[ ]:


w2 = df.Wilderness_Area2.value_counts().to_dict()
w2[1] = 'Wild_area2'
w2[0] = '0'
df.Wilderness_Area2 = df.Wilderness_Area2.map(w2)
df.Wilderness_Area2.value_counts()


# In[ ]:


w3 = df.Wilderness_Area3.value_counts().to_dict()
w3[1] = 'Wild_area3'
w3[0] = '0'
df.Wilderness_Area3 = df.Wilderness_Area3.map(w3)
df.Wilderness_Area3.value_counts()


# In[ ]:


w4 = df.Wilderness_Area4.value_counts().to_dict()
w4[1] = 'Wild_area4'
w4[0] = '0'
df.Wilderness_Area4 = df.Wilderness_Area4.map(w4)
df.Wilderness_Area4.value_counts()


# In[ ]:


df['Wild_area'] = df[['Wilderness_Area1', 'Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']].apply(lambda x: ''.join(x), axis = 1) 


# In[ ]:


df.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'],axis = 1,inplace = True)


# In[ ]:


def decode(df,column):
    s = df[column].value_counts().to_dict()
    s[1] = column
    s[0] = '0'
    df[column] = df[column].map(s)
    return df


# In[ ]:


columns = ['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type8'
            ,'Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type16'
           ,'Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24'
          ,'Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33'
           ,'Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']


# In[ ]:


for i in columns:
    df = decode(df,i)


# In[ ]:


df['soil_type'] = df[columns].apply(lambda x: ''.join(x), axis = 1) 


# In[ ]:


df_frequency_map = df.Wild_area.value_counts().to_dict()
df.Wild_area = df.Wild_area.map(df_frequency_map)


# In[ ]:


df.Wild_area = df.Wild_area/15120*100


# In[ ]:


df_frequency_map = df.soil_type.value_counts().to_dict()
df.soil_type = df.soil_type.map(df_frequency_map)
df.soil_type = df.soil_type/15120*100


# In[ ]:


df.drop(columns,axis = 1,inplace = True)


# In[ ]:


df.head().T


# In[ ]:


df.info()


# In[ ]:


corrs = df.corr()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(corrs,annot = True,linewidths=0.25,linecolor='white',cmap='terrain')
plt.show()


# In[ ]:


y = df.Cover_Type
df.drop('Cover_Type',axis = 1,inplace = True)


# In[ ]:


test.head().T


# In[ ]:


w1 = test.Wilderness_Area1.value_counts().to_dict()
w1[1] = 'Wild_area1'
w1[0] = '0'
test.Wilderness_Area1 = test.Wilderness_Area1.map(w1)
test.Wilderness_Area1.value_counts()

w2 = test.Wilderness_Area2.value_counts().to_dict()
w2[1] = 'Wild_area2'
w2[0] = '0'
test.Wilderness_Area2 = test.Wilderness_Area2.map(w2)
test.Wilderness_Area2.value_counts()

w3 = test.Wilderness_Area3.value_counts().to_dict()
w3[1] = 'Wild_area3'
w3[0] = '0'
test.Wilderness_Area3 = test.Wilderness_Area3.map(w3)
test.Wilderness_Area3.value_counts()

w4 = test.Wilderness_Area4.value_counts().to_dict()
w4[1] = 'Wild_area4'
w4[0] = '0'
test.Wilderness_Area4 = test.Wilderness_Area4.map(w4)
test.Wilderness_Area4.value_counts()


# In[ ]:


test['Wild_area'] = test[['Wilderness_Area1', 'Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']].apply(lambda x: ''.join(x), axis = 1) 


# In[ ]:


test.drop(['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'],axis = 1,inplace = True)


# In[ ]:


for i in columns:
    test = decode(test,i)


# In[ ]:


test['soil_type'] = test[columns].apply(lambda x: ''.join(x), axis = 1) 


# In[ ]:


frequency_map = test.Wild_area.value_counts().to_dict()
test.Wild_area = test.Wild_area.map(frequency_map)


# In[ ]:


test.Wild_area = test.Wild_area/565892*100


# In[ ]:


test_frequency_map = test.soil_type.value_counts().to_dict()
test.soil_type = test.soil_type.map(test_frequency_map)
test.soil_type = test.soil_type/565892*100


# In[ ]:


test.drop(columns,axis = 1,inplace =True)


# In[ ]:


test.head()


# In[ ]:


#####################################Train Data ################################################
df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
df['HF2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
df['HR1'] = abs(df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
df['HR2'] = abs(df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
df['FR1'] = abs(df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
df['FR2'] = abs(df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
df['ele_vert'] = df.Elevation-df.Vertical_Distance_To_Hydrology

df['slope_hyd'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
df.slope_hyd=df.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
df['Mean_Fire_Hyd']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2 

####################### Test data #############################################
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=350)  
etc.fit(df,y)
sub = pd.DataFrame({"Id": sam_submit['Id'],"Cover_Type": etc.predict(test)})


# In[ ]:


from catboost import CatBoostClassifier
catclf = CatBoostClassifier(random_state=42)
catclf.fit(df,y)


# In[ ]:


train_predict = catclf.predict(df)
accuracy_score(y,train_predict)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y,train_predict)


# In[ ]:


test_pred = catclf.predict(test)


# In[ ]:


test_pred = test_pred.ravel()


# In[ ]:


test_pred.shape


# In[ ]:


sub = pd.DataFrame({"Id": sam_submit['Id'],"Cover_Type": test_pred})


# In[ ]:


sub.to_csv("submission.csv", index=False) 

