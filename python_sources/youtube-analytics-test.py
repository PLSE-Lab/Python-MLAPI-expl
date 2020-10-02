#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
dataset_CA=pd.read_csv("../input/youtube-new/CAvideos.csv")
dataset_DE=pd.read_csv("../input/youtube-new/DEvideos.csv")
dataset_FR=pd.read_csv("../input/youtube-new/FRvideos.csv")
dataset_GB=pd.read_csv("../input/youtube-new/GBvideos.csv")
dataset_IN=pd.read_csv("../input/youtube-new/INvideos.csv")
dataset_JP=pd.read_csv("../input/youtube-new/JPvideos.csv",encoding='latin1')
dataset_KR=pd.read_csv("../input/youtube-new/KRvideos.csv",encoding='latin1')
dataset_MX=pd.read_csv("../input/youtube-new/MXvideos.csv",encoding='latin1')
dataset_RU=pd.read_csv("../input/youtube-new/RUvideos.csv",encoding='latin1')
dataset_US=pd.read_csv("../input/youtube-new/USvideos.csv")
dataset_CA.isnull().sum()
dataset_DE.isnull().sum()
dataset_FR.isnull().sum()
dataset_GB.isnull().sum()
dataset_IN.isnull().sum()
dataset_JP.isnull().sum()
dataset_KR.isnull().sum()
dataset_MX.isnull().sum()
dataset_RU.isnull().sum()
dataset_US.isnull().sum()


# *Check for null values*
# 
# **We can clearly see that there are missing values from description of videos coming from different regions** 
# 
# **(Canada, Germany, France, Great Britan, India, Japan, South Korea, Mexico, Russia, United States)**

# In[ ]:


dataset_CA['region']='Canada'
dataset_DE['region']='Germany'
dataset_FR['region']='France'
dataset_GB['region']='Great Britan'
dataset_IN['region']='India'
dataset_JP['region']='Japan'
dataset_KR['region']='South Korea'
dataset_MX['region']='Mexico'
dataset_RU['region']='Russia'
dataset_US['region']='United States'
df=pd.DataFrame(pd.concat([dataset_CA,dataset_DE,dataset_FR,dataset_GB,dataset_IN,dataset_JP,dataset_KR,dataset_MX,dataset_RU,dataset_US], ignore_index=True))
df = df.sample(frac=1).reset_index(drop=True)
df


# *Cleaning up data*
# 
# **Here, i've created a column 'region' for each of the datasets to specify which region they come from. I then proceeded to combine all datasets into one dataframe.Furthermore, i shuffled DataFrame rows inorder to avoid any element of bias/patterns before training ML model. **

# In[ ]:


df.shape


# *Exploring the dataset*
# 
# **We have 375942 samples with 17 features.Approximately 5% of the entries have missing descriptions**

# In[ ]:


df['region'].value_counts(normalize=True) * 100


# *Distribution of observations per region*
# 

# In[ ]:


df=df.dropna()
df.shape


# *Handle Missing Values*
# 
# **Here I dropped all observations with null values since they make up 5% of the entire dataset -> insignificant **

# In[ ]:


print("Data types and their frequency\n{}".format(df.dtypes.value_counts()))


# *Investigate Categorical Columns*
# 
# **We have 9 object columns that have text**

# *Feature Selection*
# 
# **We select those features which contribute most to our prediction variable in which we are interested in.
# Using irrelevant features in our data can decrease the models prediction accuracy on unseen data, increase variance (model highly likely to overfit),and increase the algorithms complexity.**

# In[ ]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['region_cat'] = labelencoder.fit_transform(df['region'])
enc = OneHotEncoder(handle_unknown='ignore')
# passing region-cat column (label encoded values of region)

o=pd.DataFrame(enc.fit_transform(df[['region_cat']]).toarray())
#reducing the number of dummy values
df['category_id'].replace({18: 1, 30: 1,31: 1,32: 1,33: 1,34: 1,35: 1,36: 1,37: 1,38: 1,39: 1,40: 1,41: 1,42: 1,43: 1,44: 1,21: 22}, inplace=True)
#adding a new column that is title length
df['title_length'] = df['title'].str.len()
#here we are trying to delete the colummns(features) that we are not in need of in our model 
"""
del df['likes']
del df['dislikes']
del df['comment_count']
del df['ratings_disabled']
del df['video_error_or_removed']
del df['description']
del df['publish_time']
del df['trending_date']
"""

df
def name_col(row):
    if row['category_id'] == 2:
        val = "Autos & Vehicles"
    elif row['category_id'] == 1|row['category_id'] == 18|row['category_id'] == 30|row['category_id'] == 31|row['category_id'] == 32|row['category_id'] == 33|row['category_id'] == 34|row['category_id'] == 35|row['category_id'] == 36|row['category_id'] == 37|row['category_id'] == 38|row['category_id'] == 39|row['category_id'] == 40|row['category_id'] == 41|row['category_id'] == 42|row['category_id'] == 43|row['category_id'] == 44:
        val = "Film & Animation"
    elif row['category_id'] == 10:
        val="Music"
    elif row['category_id']==15:
        val="Pets & Animals"
    elif row['category_id']==17:
        val="Sports"
    elif row['category_id']==19:
        val="Travel & Events"
    elif row['category_id']==20:
        val="Gaming"
    elif row['category_id']==22|row['category_id']==21:
        val="People & Blogs"
    elif row['category_id']==23:
        val="Comedy"
    elif row['category_id']==24:
        val="Entertainment"
    elif row['category_id']==25:
        val="News & Politics"
    elif row['category_id']==26:
        val="Howto & Style"
    elif row['category_id']==27:
        val="Education"
    elif row['category_id']==28:
        val="Science & Technology"
    elif row['category_id']==29:
        val="Nonprofits & Activism"
    return val
o=df
df['category_names'] = df.apply(name_col, axis=1)
df



# 

# In[ ]:



#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
#splitting the dataset into source variables (independent variables) and the target variable (dependent variable)
X=df.loc[:, df.columns != 'views']
y=df['views']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# *Boxplots*

# In[ ]:


# import the required library  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
df.boxplot(by ='day', column =['total_bill'], grid = False)

