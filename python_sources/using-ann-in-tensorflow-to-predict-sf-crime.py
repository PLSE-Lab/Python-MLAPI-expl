#!/usr/bin/env python
# coding: utf-8

# <h1>Predicting Crime in San Francisco Using ANN and Keras</h1>
# =========
# <br>
# <h2> 1. Introduction</h2>
# ---------------------------------
# <p>In this project, I predicted the category of crimes that happened in San Fransisco based on different variables such as the latitude, longitude, date and time. This resulted in 2.51353 kaggle score (or top 33%).</p>
# 
# <p>Pandas is used for data manipulation. Numpy is the fundamental package for scientific computation in Python. 
# Neural Networks is the classification algorithm used to make the final predictions. Seaborn is a nice tool for data visualisation built on top of matplotlib. 
# The import code is as follows:</p>
# 
# 
# <h3>Libraries I used :</h3>
# <p>import pandas as pd<br>
# import numpy as np<br>
# import matplotlib.pyplot as plt<br>
# import tensorflow as tf<br>
# from tensorflow import keras<br>
# from datetime import datetime<br>
# from sklearn.model_selection import train_test_split<br>
# from sklearn.model_selection import GridSearchCV<br>
# from sklearn.tree import DecisionTreeClassifier<br>
# from sklearn import preprocessing<br>
# from sklearn.metrics import log_loss<br>
# from sklearn.metrics import make_scorer<br>
# from sklearn.model_selection import StratifiedShuffleSplit<br>
# from matplotlib.colors import LogNorm<br>
# from sklearn.decomposition import PCA<br>
# from keras.layers.advanced_activations import PReLU<br>
# from keras.layers.core import Dense, Dropout, Activation<br>
# from keras.layers.normalization import BatchNormalization<br>
# from keras.models import Sequential<br>
# from keras.utils import np_utils<br>
# from copy import deepcopy<br>
# %matplotlib inline<p>
# 
# 

# ## Step 1 : EDA
# During the EDA stage, I plotted the dataset to identify any trends/correlations and then identify outliers. 
# I used the learnings from EDA to decide how I will treat my independent variables, clean the data, and to build some feature engineering and put them in my model.<br>
# 
# Based on the EDA, below are the key takeaways: <br>
#     1. There was no ordinal trend between month and number of reported crime. 
#        Therefore, I will treat the month variable as a categorical variable and create dummy variable.
#     2. There was no ordinal trend between day of week and number of reported crime as well. Therefore, I will create create dummy variable.
#     3. There were outlier Longitude and I will be removing that from my train dataset.
#     4. There were a lot of crimes happening in the same address at the same district.

# ![image.png](attachment:image.png)

# ## Step 2 : Data Cleaning
# 
# I decided to combine both the train and test files together so that it is easier when I create the one-hot dummy variables. <br>
# I created a function to parse the datetime variables, create dummy variables for categorical data, and normalize all the continuous values. <br>
# Once all the data have been cleaned and organized in the same format, I splitted both the train and test data.

# ## Data Cleaning and Feature Engineering

# In[ ]:


##Categorizing the season column using Dummy Vars : the reason is because there is no Hierarchy..
#meaning that, "Fall IS NOT Higher or Better than Summer"

def data_prep(df_clean):
    
    def parse_time(x):
        DD=datetime.strptime(x,"%m/%d/%y %H:%M")
        time=DD.hour 
        day=DD.day
        month=DD.month
        year=DD.year
        mins=DD.minute
        return time,day,month,year,mins
    
    
    
    parsed = np.array([parse_time(x) for x in df_clean.Dates])
    
    df_clean['Dates'] = pd.to_datetime(df_clean['Dates'])
    df_clean['WeekOfYear'] = df_clean['Dates'].dt.weekofyear
    #df_clean['n_days'] = (df_clean['Dates'] - df_clean['Dates'].min()).apply(lambda x: x.days)
    df_clean['HOUR'] = parsed[:,[0]]
    df_clean['day'] = parsed[:,[1]]
    df_clean['month'] = parsed[:,[2]]
    df_clean['year'] = parsed[:,[3]]
    df_clean['mins'] = parsed[:,[4]]
    
    
    #adding season variable
    def get_season(x):
        if x in [5, 6, 7]:
            r = 'summer'
        elif x in [8, 9, 10]:
            r = 'fall'
        elif x in [11, 12, 1]:
            r = 'winter'
        elif x in [2, 3, 4]:
            r = 'spring'
        return r
    
    df_clean['season'] = [get_season(i) for i in df_clean.month] 
    
    
    df_clean['Block'] = df_clean['Address'].str.contains('block', case=False)
    df_clean['Block'] = df_clean['Block'].map(lambda x: 1 if  x == True else 0)
    
    #creating dummy variables
    df_clean_onehot = pd.get_dummies(df_clean, columns=['season'], prefix = [''])
    s = (len(list(df_clean_onehot.columns))-len(df_clean.season.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,s:]], axis=1)

    ##Categorizing the DayOFWeek column using Dummy Vars 
    df_clean_onehot = pd.get_dummies(df_clean, columns=['DayOfWeek'], prefix = [''])
    
    l = (len(list(df_clean_onehot.columns))-len(df_clean.DayOfWeek.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,l:]],axis=1)

    ##Categorizing the MONTH column using Dummy Vars : the reason is because there is no Hierarchy..
    #meaning that, "FEB IS NOT Higher or Better than JAN"
    #This insight was shown from the EDA result (forecasting data with trend might be a different case)

    df_clean_onehot = pd.get_dummies(df_clean, columns=['month'], prefix = ['month'])
    n = (len(list(df_clean_onehot.columns))-len(df_clean.month.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,n:]],axis=1)

    ##Categorizing the District column using Dummy Vars 
    df_clean_onehot = pd.get_dummies(df_clean, columns=['PdDistrict'], prefix = [''])
    o = (len(list(df_clean_onehot.columns))-len(df_clean.PdDistrict.value_counts()))
    df_clean = pd.concat([df_clean,df_clean_onehot.iloc[:,o:]],axis=1)
    
    df_clean['IsInterection']=df_clean['Address'].apply(lambda x: 1 if "/" in x else 0)
    df_clean['Awake']=df_clean['HOUR'].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
    
    ##changing the Output Variables to integer
    labels = df_clean['Category'].astype('category').cat.categories.tolist()
    replace_with_int = {'Category' : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}
    df_clean.replace(replace_with_int, inplace=True)
    
    #Normalizing the columns
    def norm_func(i):
        r = (i-min(i))/(max(i)-min(i))
        return(r)

    df_clean['normHOUR']=norm_func(df_clean.HOUR)
    df_clean['normmins']=norm_func(df_clean.mins)
    df_clean['normdate_day']=norm_func(df_clean.day)
    df_clean['normLat']=norm_func(df_clean.X)
    df_clean['normLong']=norm_func(df_clean.Y)
    df_clean['normmonth']=norm_func(df_clean.month)
    df_clean['normyear']=norm_func(df_clean.year)
    df_clean['normWeekOfYear']=norm_func(df_clean.WeekOfYear)
    #df_clean['normNDAYS']=norm_func(df_clean.n_days)
    


    ##removing the unused columns
    df_clean.drop(columns = ['Dates','season','HOUR','day','X','Y'
                             ,'DayOfWeek','Address','PdDistrict','mins','month','year','WeekOfYear','Resolution'], axis = 1,inplace=True)
                             #'Count_rec_x','Count_rec_y'], axis = 1,inplace=True)
    return(df_clean)


# ## Checking Train Data

# combined = data_prep(combined)
# 
# train_clean = combined[combined.Train == 1] <br>
# train_clean.drop(['Train','Test'], axis=1,inplace = True)

# train_clean.head()

# ![image.png](attachment:image.png)

# ## Training the neural network model

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)![](http://)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ## Checking Test Data

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# predictiondata.head()

# ![image.png](attachment:image.png)

# predictiondata.to_csv('.../SFprediction_dataSF.csv'
#                       ,encoding='utf-8', index=True)
