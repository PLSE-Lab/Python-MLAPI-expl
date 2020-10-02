#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew 
from scipy import stats
import datetime
import ast
import json
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#Standardization
from sklearn.preprocessing import StandardScaler


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("."))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
df_test = pd.read_csv("../input/tmdb-box-office-prediction/test.csv")

print(df_train.index)
print(df_test.index)


# In[ ]:


df_all = pd.concat([df_train, df_test], sort=False).reset_index()


print(df_all.isnull().sum())


# preparing for loading posters

# In[ ]:


import cv2

def loadPosterImages(df, _path_to_base):
    
    images = []
    
    for i in df.id:
        path_to_img = _path_to_base + str(i) + ".jpeg"
        if os.path.exists(path_to_img) == False:
            print(path_to_img + " is null")
            image = np.zeros([64,64,3],dtype=np.uint8)
        else:
            image = cv2.imread(path_to_img)
            image = cv2.resize(image, (64, 64))
        
        images.append(image)
    
    
    return np.array(images)


# In[ ]:


use_npy=0

if use_npy != 1:
    poster_train_img = loadPosterImages(df_train, "../input/tmdb-box-office-prediction-posters/tmdb_box_office_prediction_posters/tmdb_box_office_prediction_posters/train/")
    poster_test_img = loadPosterImages(df_test, "../input/tmdb-box-office-prediction-posters/tmdb_box_office_prediction_posters/tmdb_box_office_prediction_posters/test/")

    np.save('poster_train_img.npy', poster_train_img)
    np.save('poster_test_img.npy', poster_test_img)
else:
    #Loading images needs long time, so I save them as numpy binary and load from the npy files instead.
    poster_train_img = np.load('poster_train_img.npy')
    poster_test_img = np.load('poster_test_img.npy')


poster_train_img = poster_train_img / 255.0
poster_test_img = poster_test_img / 255.0


# There are missing posters in the dataset. <br>
# Train : 2303 <br>
# Test : 3829, 4925 <br>
# Their titles are...
# 

# In[ ]:


print(df_train.loc[df_train["id"] == 2303, "title"])
print(df_test.loc[df_test["id"] == 3829, "title"])
print(df_test.loc[df_test["id"] == 4925, "title"])

df_train.drop(df_train.loc[df_train["id"] == 2303].index, inplace=True)
print(df_train.loc[df_train["id"] == 2303, "title"])


# In[ ]:


target_col_name = "revenue"
sc = StandardScaler()


# **Preprocessing**

# In[ ]:


def visualize_distribution(y):
    
    plt.figure(figsize=(15, 8))
    sns.distplot(y,fit=norm)
    mu,sigma=norm.fit(y)
    plt.legend(["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})".format(mu,sigma)])
    plt.title("Distribution")
    plt.ylabel("Frequency")
    plt.show()
    
    
def visualize_probplot(y):
    plt.figure(figsize=(15, 8))
    stats.probplot(y,plot=plt)
    plt.show()


# In[ ]:


def showJointPlot(df, col_name):
    

    if not col_name in df.columns:
        print(col_name + " is not inside columns")
        return

    print("***[" + col_name + "]***")
    print("describe : ")
    print(df[col_name].describe())
    print("skew : ")
    print(df[col_name].skew())

    #correlation
    corrmat = df.corr()
    num_of_col = len(corrmat.columns)
    cols = corrmat.nlargest(num_of_col, col_name)[col_name]
    print("*****[ corr : " + col_name + " ]*****")
    print(cols)
    print("*****[" + col_name + "]*****")
    print("\n")


    visualize_distribution(df[col_name].dropna())
    visualize_probplot(df[col_name].dropna())


    if col_name != target_col_name and target_col_name in df.columns:
        plt.figure(figsize=(15, 8))
        sns.jointplot(col_name, target_col_name, df)

    print("******\n")
    return 


# In[ ]:


def showValueCount(df, col_name):

    if not col_name in df.columns:
        print(col_name, " is not inside columns")
        return

    print("***[" + col_name + "]***")
    print("describe :")
    print(df[col_name].describe())

    df_value = df[col_name].value_counts(dropna=False)
    print("value_counts :")
    print(df_value)

    plt.figure(figsize=(15,8))
    sns.barplot(df_value.index, df_value.values, alpha=0.8)
    plt.ylabel('Number of each element', fontsize=12)
    plt.xlabel(col_name, fontsize=12)
    plt.xticks(rotation=90, size='small')
    plt.show()


    if col_name != target_col_name and target_col_name in df.columns:
        plt.figure(figsize=(15, 8))
        plt.xticks(rotation=90, size='small')
        sns.boxplot(x=df[col_name], y =df[target_col_name])
        plt.show()

    print("******\n")
    return 


# In[ ]:


def countAndExpandFromDictToColumns(df, expanded_col_name, _val_name):
    
    df[expanded_col_name] = df[expanded_col_name].dropna().map(lambda x : ast.literal_eval(x))
    
    def connectToString(x, prefix, val_name):

        str_names = []
        for val in x:

            str_names.append(prefix + "_" + val[val_name])

        return ",".join(str_names)


    df[expanded_col_name] = df[expanded_col_name].dropna().map(lambda x : connectToString(x, prefix=expanded_col_name, val_name=_val_name))
    #print(df[expanded_col_name].head())
    df_tmp = df[expanded_col_name].dropna().str.get_dummies(sep=',')
    #print(df_tmp.info())
    df = pd.concat([df, df_tmp], axis=1, sort=False)

    df.drop(expanded_col_name, axis=1, inplace=True)


    return df


# In[ ]:


def countFromDict(df, count_col_name, _val_name=None):
    
    count_dic = {}

    def countFromDf(x):
        for val in ast.literal_eval(x):

            if _val_name != None:
                element_val = val[_val_name]
            else:
                element_val = json.dumps(val)
            if count_dic.get(element_val, 0) == 0:
                count_dic[element_val] = 1
            else:
                count_dic[element_val] = count_dic[element_val] + 1
        
        return x

    _ = df[count_col_name].dropna().map(lambda x : countFromDf(x))
    
    df_count = pd.DataFrame(list(count_dic.items()),columns=['key_name','num'])
    df_count.sort_values("num", ascending=False, inplace=True)

    return df_count


# **revenue**

# In[ ]:


col_name = "revenue"

showJointPlot(df_all, col_name)


# revenue's minimum is 1 dollar...?

# In[ ]:


df_million_index = df_all.loc[df_all[col_name] < 100].index
df_thousand_index = df_all.loc[df_all[col_name] < 1000].index

df_all.loc[df_million_index, col_name] = df_all.loc[df_million_index, col_name].map(lambda x: x * 1000000)
df_all.loc[df_thousand_index, col_name] = df_all.loc[df_thousand_index, col_name].map(lambda x: x * 1000)

df_all[col_name] = np.log(df_all[col_name])
showJointPlot(df_all, col_name)


# **Budget**

# In[ ]:


col_name = "budget"

showJointPlot(df_all, col_name)


# budget contains many 0...

# In[ ]:


df_million_budget_index = df_all.loc[df_all[col_name] < 100].index
df_thousand_budget_index = df_all.loc[df_all[col_name] < 1000].index

df_all.loc[df_million_budget_index, col_name] = df_all.loc[df_million_budget_index, col_name].map(lambda x: x * 1000000)
df_all.loc[df_thousand_budget_index, col_name] = df_all.loc[df_thousand_budget_index, col_name].map(lambda x: x * 1000)




#df_all[col_name] = pd.Series(sc.fit_transform(df_all[col_name].values.reshape(-1, 1)).flatten())
df_all[col_name] = np.log1p(df_all[col_name])
showJointPlot(df_all, col_name)


# **release_date**

# In[ ]:


col_name = "release_date"

            
print("\n**fillNanValues : " + col_name +  "***")
print(df_all.loc[df_all["release_date"].isnull() == True, ["title", "imdb_id", col_name]])

#fill from IMDB
df_all.loc[df_all["imdb_id"] == "tt0210130", col_name] = "3/20/01"


# In[ ]:


df_all[col_name + "_year"] = df_all[col_name].map(lambda x: int(x.split("/")[2]))
df_all[col_name + "_month"] = df_all[col_name].map(lambda x: int(x.split("/")[0]))
df_all[col_name + "_day"] = df_all[col_name].map(lambda x: int(x.split("/")[1]))


#we assume that release years are between 1900 and 2017
df_all.loc[(df_all[col_name + "_year"] < 18), col_name + "_year"] += 2000
df_all.loc[(df_all[col_name + "_year"] >= 18) & (df_all[col_name + "_year"] < 100), col_name + "_year"] += 1900

df_all[col_name] = df_all.apply(lambda x: datetime.datetime(x[col_name + "_year"], x[col_name + "_month"], x[col_name + "_day"]), axis=1)

df_all[col_name + "_month"] = df_all[col_name + "_month"].replace({1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"})

dow = ["Mon","Tue","Wed","Thr","Fri","Sat","Sun"]
df_all[col_name + "_dayofweek"] = df_all[col_name].map(lambda x: dow[x.weekday()])

df_all[col_name + "_week"]= pd.Series(len(df_all[col_name]), index=df_all.index)
df_all.loc[df_all[col_name + "_day"] <= 7, col_name + "_week"] = "w1"
df_all.loc[(df_all[col_name + "_day"] > 7) & (df_all[col_name + "_day"] <= 14), col_name + "_week"] = "w2"
df_all.loc[(df_all[col_name + "_day"] > 14) & (df_all[col_name + "_day"] <= 21), col_name + "_week"] = "w3"
df_all.loc[(df_all[col_name + "_day"] > 21) & (df_all[col_name + "_day"] <= 28), col_name + "_week"] = "w4"
df_all.loc[(df_all[col_name + "_day"] > 28), col_name + "_week"] = "w5"

df_all[col_name + "_MonthWeek"] = df_all[col_name + "_month"] + df_all[col_name + "_week"]


df_all.drop(col_name + "_month", inplace=True, axis=1)
df_all.drop(col_name + "_week", inplace=True, axis=1)
df_all.drop(col_name + "_day", inplace=True, axis=1)
df_all.drop(col_name, inplace=True, axis=1)


# In[ ]:


showValueCount(df_all, col_name + "_year")
showValueCount(df_all, col_name + "_dayofweek")
showValueCount(df_all, col_name + "_MonthWeek")


# **belongs_to_collection**

# In[ ]:


col_name = "belongs_to_collection"

df_all["inCollection"] = 0
df_all.loc[df_all[col_name].isnull() == False, "inCollection"] = 1

showValueCount(df_all, "inCollection")
df_all.drop(col_name, inplace=True, axis=1)


# I add the feature "inCollection" which means each movie belongs its collection or not.

# **genres**

# In[ ]:


col_name = "genres"

print(df_all.loc[df_all[col_name].isnull(), ["imdb_id", "title", col_name]])
print("\n**fillNanValues : " + col_name +  "***")

#fill from IMDB
df_all.loc[df_all["imdb_id"] == "tt0349159", col_name] = "[{'id': 12, 'name': 'Adventure'}]"
df_all.loc[df_all["imdb_id"] == "tt0261755", col_name] = "[{'id': 18, 'name': 'Drama'}, {'id': 35, 'name': 'Comedy'}]"
df_all.loc[df_all["imdb_id"] == "tt0110289", col_name] = "[{'id': 35, 'name': 'Comedy'}]"
df_all.loc[df_all["imdb_id"] == "tt0352622", col_name] = "[{'id': 10749, 'name': 'Romance'}, {'id': 18, 'name': 'Drama'}]"
df_all.loc[df_all["imdb_id"] == "tt0984177", col_name] = "[{'id': 28, 'name': 'Action'}, {'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]"
df_all.loc[df_all["imdb_id"] == "tt0833448", col_name] = "[{'id': 53, 'name': 'Thriller'}]"
df_all.loc[df_all["imdb_id"] == "tt1766044", col_name] = "[{'id': 18, 'name': 'Drama'}, {'id': 14, 'name': 'Fantasy'}, {'id': 9648, 'name': 'Mystery'}]"
df_all.loc[df_all["imdb_id"] == "tt0090904", col_name] = "[{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}, {'id': 53, 'name': 'Thriller'}]"
df_all.loc[df_all["imdb_id"] == "tt0086405", col_name] = "[{'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]"
df_all.loc[df_all["imdb_id"] == "tt0044177", col_name] = "[{'id': 12, 'name': 'Adventure'}, {'id': 99999999, 'name': 'Biography'}, {'id': 18, 'name': 'Drama'}, {'id': 10749, 'name': 'Romance'}]"
df_all.loc[df_all["imdb_id"] == "tt0108234", col_name] = "[{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}, {'id': 18, 'name': 'Drama'}, {'id': 53, 'name': 'Thriller'}]"
df_all.loc[df_all["imdb_id"] == "tt1572916", col_name] = "[{'id': 18, 'name': 'Drama'}]"
df_all.loc[df_all["imdb_id"] == "tt1569465", col_name] = "[{'id': 35, 'name': 'Comedy'}]"
df_all.loc[df_all["imdb_id"] == "tt0405699", col_name] = "[{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}]"
df_all.loc[df_all["imdb_id"] == "tt0461892", col_name] = "[{'id': 28, 'name': 'Action'}, {'id': 18, 'name': 'Drama'}]"
df_all.loc[df_all["imdb_id"] == "tt3121604", col_name] = "[{'id': 18, 'name': 'Drama'}]"
df_all.loc[df_all["imdb_id"] == "tt1164092", col_name] = "[{'id': 99, 'name': 'Documentary'}, {'id': 99999999, 'name': 'Biography'}, {'id': 10751, 'name': 'Family'}]"
df_all.loc[df_all["imdb_id"] == "tt0250282", col_name] = "[{'id': 35, 'name': 'Comedy'}]"
df_all.loc[df_all["imdb_id"] == "tt1620464", col_name] = "[{'id': 35, 'name': 'Comedy'}, {'id': 80, 'name': 'Crime'}, {'id': 9648, 'name': 'Mystery'}]"
df_all.loc[df_all["imdb_id"] == "tt0073317", col_name] = "[{'id': 35, 'name': 'Comedy'}, {'id': 80, 'name': 'Crime'}, {'id': 18, 'name': 'Drama'}]"
df_all.loc[df_all["imdb_id"] == "tt0361498", col_name] = "[{'id': 35, 'name': 'Comedy'}]"
df_all.loc[df_all["imdb_id"] == "tt0361596", col_name] = "[{'id': 99, 'name': 'Documentary'}, {'id': 18, 'name': 'Drama'}, {'id': 10752, 'name': 'War'}]"
df_all.loc[df_all["imdb_id"] == "tt2192844", col_name] = "[{'id': 18, 'name': 'Drama'}]"


# In[ ]:


df_all[col_name + "_len"] = df_all[col_name].map(lambda x: len(ast.literal_eval(x)) if pd.isnull(x) == False else 0)
df_all.loc[df_all[col_name + "_len"] > 5, col_name + "_len"] = 5
showValueCount(df_all, col_name + "_len")


# In[ ]:


df_all = countAndExpandFromDictToColumns(df_all, col_name, "name")
print(df_all.shape)


# **runtime**

# In[ ]:


col_name = "runtime"

print("\n**fillNanValues : " + col_name +  "***")
print(df_all.loc[df_all["runtime"].isnull() == True, ["title", "imdb_id", "release_date", col_name]])


showJointPlot(df_all, col_name)


# the longest and shortest movies are ...

# In[ ]:


df_all.sort_values(col_name, ascending=False)[["title", col_name]]


# In[ ]:


df_all.loc[df_all[col_name] == 0, col_name] = np.nan
df_all[col_name].fillna(df_all[col_name].median(), inplace=True)

#df_all[col_name] = pd.Series(sc.fit_transform(df_all[col_name].values.reshape(-1, 1)).flatten())
df_all[col_name] = np.log(df_all[col_name])
#df_all.drop(index=df_all.loc[df_all[col_name] < 2.4, ["title", col_name]].index, inplace=True)
#print(df_all.loc[df_all[col_name] < 2.4, ["title", col_name]])
showJointPlot(df_all, col_name)


# **spoken_languages**

# In[ ]:


col_name = "spoken_languages"

print("\n**fillNanValues : " + col_name +  "***")

df_count_sLangugaes = countFromDict(df_all, col_name)
df_count_sLangugaes["iso"] = df_count_sLangugaes["key_name"].map(lambda x: ast.literal_eval(x)["iso_639_1"])
#print(df_count_sLangugaes.head(len(df_count_sLangugaes)))


def returnStr(x):
    
    lis = df_count_sLangugaes.loc[df_count_sLangugaes["iso"] == x.original_language, "key_name"].values
    if len(lis) == 0:
        #print(x, lis)
        x.spoken_languages = np.nan
    else:
        str_dic = lis[0]
        x.spoken_languages =  str("[" + str_dic + "]")
    
    return x

df_all.loc[(df_all[col_name].isnull() == True), [col_name, "original_language"]] = df_all.loc[(df_all[col_name].isnull() == True), [col_name, "original_language"]].apply(lambda x: returnStr(x), axis=1)
df_all["lang_len"] = df_all[col_name].map(lambda x: len(ast.literal_eval(x)))

df_all = countAndExpandFromDictToColumns(df_all, col_name, "iso_639_1")
df_all.shape


# **original_language**

# In[ ]:


col_name = "original_language"

#print(df_all[col_name].isnull().sum())

#df_all["isEnglish"] = 0
#df_all.loc[(df_all[col_name] == "en") | (df_all["spoken_languages_en"] == 1), "isEnglish"] = 1

showValueCount(df_all, col_name)


# I add new feature "isEnglish"

# **overview**

# In[ ]:


col_name = "overview"
df_all[col_name].fillna(0, inplace=True)


# this feature is used in "Keywords"

# **popularity**

# In[ ]:


col_name = "popularity"

showJointPlot(df_all, col_name)
df_all[col_name].sort_values(ascending=False)


# In[ ]:


print(df_all.shape)
df_all[col_name + "_RANK"] = 2
df_all.loc[df_all[col_name] > 50, col_name + "_RANK"]= 3
df_all.loc[df_all[col_name] <= 3, col_name + "_RANK"] = 1

#df_all.drop(df_all.loc[(df_all["revenue"] < 10) & (df_all[col_name] > 25)].index, inplace=True)
df_all.loc[(df_all[col_name] < 0.0003), col_name] = 0.0003

#df_all[col_name] = pd.Series(sc.fit_transform(df_all[col_name].values.reshape(-1, 1)).flatten())
df_all[col_name] = np.log(df_all[col_name])

print(df_all.shape)
#showJointPlot(df_all, col_name)


# **crew**

# In[ ]:


col_name = "crew"

df_all[col_name + "_num"] = df_all[col_name].map(lambda x: len(ast.literal_eval(x)) if pd.isnull(x) == False else np.nan)


# I add "crew_num"

# In[ ]:


#fill na and 0 
df_all[col_name + "_num"].fillna(df_all[col_name + "_num"].median(), inplace=True)

#df_all[col_name + "_num"] = pd.Series(sc.fit_transform(df_all[col_name + "_num"].values.reshape(-1, 1)).flatten())
df_all[col_name + "_num"] = np.log(df_all[col_name + "_num"])

showJointPlot(df_all, col_name + "_num")


df_all.drop(col_name, axis=1, inplace=True)


# **cast**

# In[ ]:


col_name = "cast"
        
df_all[col_name + "_num"] = df_all[col_name].map(lambda x: len(ast.literal_eval(x)) if pd.isnull(x) == False else np.nan)
df_all.loc[df_all[col_name + "_num"] == 0, col_name + "_num"] = np.nan


# In[ ]:



cast_dict = {}

def countName(x):
    cast_list = ast.literal_eval(x.cast)
    
    for each_cast in cast_list:
        cast_name = each_cast["name"]
        
        
        if cast_name not in cast_dict.keys():
            cast_dict[cast_name] = {"train":0, "test":0}
        
        which_data = "train"
        if x.id > 3000:
            which_data = "test"
        cast_dict[cast_name][which_data] += 1
    
    return x

df_all = df_all.apply(lambda x: countName(x) if pd.isnull(x.cast) == False else x, axis=1)

df_cast_count = pd.DataFrame(cast_dict).T
df_cast_count.reset_index(inplace=True)
df_cast_count.rename(columns={'index': 'name'}, inplace=True)
df_cast_count["total"] = df_cast_count["train"] + df_cast_count["test"]
df_cast_count.sort_values("total", inplace=True)
print(df_cast_count.shape)
print(df_cast_count.columns)
print(df_cast_count.index)
df_cast_count
    


# In[ ]:


#fill na and 0 
df_all[col_name + "_num"].fillna( df_all[col_name + "_num"].median(), inplace=True)


#df_all[col_name + "_num"] = pd.Series(sc.fit_transform(df_all[col_name + "_num"].values.reshape(-1, 1)).flatten())
df_all[col_name + "_num"] = np.log(df_all[col_name + "_num"])

showJointPlot(df_all, col_name + "_num")

df_all.drop(col_name, axis=1, inplace=True)


# **Keywords**

# In[ ]:


col_name = "Keywords"

#df_all[col_name + "_len"] = 0
#df_count_keywords = countFromDict(df_all, col_name, "name")

            
def serachOverview(x):

    val = 0
    freq_keywords = []
    keywords_len = 0
    if pd.isna(x[col_name]) == False:
        overview_text = x["overview"]
        keyword_list = ast.literal_eval(x[col_name])
        keywords_len = len(keyword_list)


        for keyword in keyword_list:
            word = str(keyword["name"])
            #print(word)
            freq_num = df_count_keywords.loc[df_count_keywords["key_name"] == word, "num"].values[0]
            #print(freq_num)
            if freq_num > 100:
                freq_keywords.append(col_name + "_" + word)

            if overview_text != 0 & str(overview_text).find(word) != -1:
                val = val + 1

    x["overview"] = val
    x[col_name] = ",".join(freq_keywords)
    x[col_name + "_len"] = keywords_len
    return x



#df_all = df_all.apply(lambda x: serachOverview(x), axis=1)

#df_tmp = df_all[col_name].str.get_dummies(sep=',')
#print(df_tmp.info())
#df_all = pd.concat([df_all, df_tmp], axis=1, sort=False)
df_all.drop(["Keywords", "overview"], axis=1, inplace=True)


# **production_countries**

# In[ ]:


col_name = "production_countries"
        
print("\n**fillNanValues : " + col_name +  "***")

#df = fillNAN_productionC(df)
df_all["production_countries"].fillna("[{'iso_3166_1': 'US', 'name': 'United States of America'}]", inplace=True)
df_all = countAndExpandFromDictToColumns(df_all, col_name, "name")
df_all.shape


# **homepage**

# In[ ]:


col_name = "homepage"

df_all["hasHomepage"] = 1
df_all.loc[df_all[col_name].isnull() == True, "hasHomepage"] = 0

df_all.drop(col_name, axis=1, inplace=True)


# **others**

# In[ ]:


df_all["budget_year_ratio"] = df_all["budget"] / df_all["release_date_year"]

drop_cols = ["imdb_id", "poster_path", "original_title", "tagline", "title", "status", "production_companies"]
df_all.drop(drop_cols, axis=1, inplace=True)

df_all = pd.get_dummies(df_all, drop_first=True, columns=['release_date_dayofweek'])

df_all = pd.get_dummies(df_all)
df_all.shape


# **model**

# In[ ]:


from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score, make_scorer
import keras.backend as K
from keras.layers import Input, Dense, Activation, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.layers import concatenate
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping 
from keras.callbacks import LearningRateScheduler


# In[ ]:


train_idx = df_all[~df_all[target_col_name].isnull()].index.values
df_train = df_all.loc[train_idx].set_index('id')
df_train.drop(["index"], axis=1, inplace=True)

max_price = df_train["revenue"].max()
df_train["revenue"] = df_train["revenue"] / max_price
            
test_idx = df_all[df_all[target_col_name].isnull()].index.values
df_test = df_all.loc[test_idx].set_index('id')
df_test.drop([target_col_name, "index"], axis=1, inplace=True)

df_train_X = df_train.drop(target_col_name, axis=1)
df_train_y = df_train[target_col_name]
print(df_train.shape)
print(df_test.shape)
print(df_train.columns)
print(df_test.columns)


scorer = make_scorer(mean_squared_error, greater_is_better = False)


# In[ ]:



indices = np.array(range(df_train.shape[0]))
valid_train_X, valid_test_X, valid_train_y, valid_test_y, indices_valid_train, indices_valid_test  = train_test_split(df_train_X, df_train_y, indices, test_size=0.2, shuffle=True, random_state=64)

print("valid_train_X", valid_train_X.shape)
print("valid_test_X", valid_test_X.shape)
print("valid_train_y", valid_train_y.shape)
print("valid_test_y", valid_test_y.shape)

 
valid_train_img_X = poster_train_img[indices_valid_train]
valid_test_img_X = poster_train_img[indices_valid_test]
test_img_X = poster_test_img

print("valid_train_img_X", valid_train_img_X.shape)
print("valid_test_img_X", valid_test_img_X.shape)
print("test_img_X", test_img_X.shape)


# Neural Network model creation

# In[ ]:


def createNNmodel(_input_dim, regress=False):
    
    model = Sequential()
    model.add(Dense(5, input_dim=_input_dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(2, activation="relu"))
    
    if regress:
        model.add(Dense(1, activation="linear"))
    
    return model




# In[ ]:


def createCNNmodel(width, height, depth, filters, regress=False):

    inputShape = (height, width, depth)
    chanDim = -1

    inputs = Input(shape=inputShape)
    
    for (i, f) in enumerate(filters):

        if i == 0:
            x = inputs
 
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(data_format='channels_last', pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
    
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)
        

    model = Model(inputs, x)
    
    return model  


# In[ ]:


def createBoxOfficeNetModel(_input_dim_NN, width_img, height_img, depth_img, filters_CNN):
    
    model_NN = createNNmodel(_input_dim_NN)
    model_CNN = createCNNmodel(width_img, height_img, depth_img, filters=filters_CNN)

    combinedInput = concatenate([model_NN.output, model_CNN.output])

    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    
    
    model = Model(inputs=[model_NN.input, model_CNN.input], outputs=x)
    
    return model


# In[ ]:


# my loss function for RMSE
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


# In[ ]:


def drawResultCurves(_history):
    # Plot the loss and accuracy curves for training and validation 

    fig, ax = plt.subplots(2,1) 
    ax[0].plot(_history.history['loss'], color='b', label="Training loss")
    ax[0].plot(_history.history['val_loss'], color='r', label="Validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(_history.history['mean_squared_logarithmic_error'], color='b', label="Training accuracy")
    ax[1].plot(_history.history['val_mean_squared_logarithmic_error'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

    plt.show()


# In[ ]:


my_filters_CNN=[8]
model = createBoxOfficeNetModel(valid_train_X.shape[1], 64, 64, 3, filters_CNN=my_filters_CNN)


opt = Adam(lr=1e-5, decay=1e-4)
model.compile(loss=root_mean_squared_error, optimizer=opt, metrics=['msle'])

#model.summary()


# In[ ]:


_epochs=100

def step_decay(epoch):
    x = 1e-3
    if epoch >= 30: x = 1e-4
    return x
lr_decay = LearningRateScheduler(step_decay)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=0, mode='auto')
# train the model
history = model.fit([valid_train_X, valid_train_img_X], valid_train_y, 
                    validation_data=([valid_test_X, valid_test_img_X], valid_test_y),
                    #callbacks=[early_stopping],
                    callbacks=[early_stopping, lr_decay],
                    epochs=_epochs, batch_size=8, verbose = 2)

              
drawResultCurves(history)


# In[ ]:


# predict results with trained parameters
         
preds = model.predict([df_test.values, test_img_X])
#pred_test_all_y = np.exp(preds)
pred_test_all_y = np.exp(preds * max_price)

df_test.loc[:, "revenue"] = pred_test_all_y


# **Submission**

# In[ ]:


submission_Price = pd.DataFrame({
            "id": df_test.index,
            "revenue": np.nan
        }, index=df_test.index)
        
               
submission_Price.loc[df_test.index, "revenue"] = df_test["revenue"]

submission_Price.to_csv('submission.csv', index=False)


submission_Price
#for i in range(len(submission_Price)):
    #print(submission_Price.iloc[i])

