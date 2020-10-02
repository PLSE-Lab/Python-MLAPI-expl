#!/usr/bin/env python
# coding: utf-8

# **The purpose of this notebook is to create a DATASET that includes**
# 
# ** Characteristics of patients like - **
# 
# *age
# 
# *sex
# 
# *Country
# 
# and so
# 
# 
# 
# 
# **The condition of the patients and their characteristics - **
# 
# * Disease time (from diagnosis date)
# 
# * Have been cured
#  
# * Deaths
# 
# 
# The database is designed to allow easy exploration of the data
# 
# Anyone interested can use and donate

# In[ ]:


import nltk
import pandas as pd 
import  numpy as np
from collections import Counter
from ds_exam import *
from update_time import *
from bag_words import *


# In[ ]:


france = pd.read_csv("/kaggle/input/coronavirus-france-dataset/patient.csv")
tunisia = pd.read_csv("../input/coronavirus-tunisia/Coronavirus_Tunisia.csv")
japan = pd.read_csv("/kaggle/input/close-contact-status-of-corona-in-japan/COVID-19_Japan_Mar_07th_2020.csv")
indonesia = pd.read_csv("/kaggle/input/indonesia-coronavirus-cases/patient.csv")
korea = pd.read_csv("/kaggle/input/coronavirusdataset/PatientInfo.csv")
Hubei = pd.read_csv("/kaggle/input/covid19official/Hubei.csv")
outside_Hubei = pd.read_csv("/kaggle/input/covid19official/outside_Hubei.csv")


# In[ ]:


datasets = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
datasets_name = ["france", "tunisia", "japan", "indonesia", "korea", "Hubei", "outside_Hubei"]

garbge = [print("\n"+datasets_name[i], [i for i in datasets[i].columns]) for i in range(len(datasets_name))]


# # datasets.shape

# In[ ]:


o = []
for i in range(len(datasets)):
    print(datasets_name[i],datasets[i].shape)
    o.append(datasets[i].shape[0])
print("\nnum of i " + str(sum(o)))


# # change name of col

# In[ ]:


france.rename(columns={"health":"severity_illness","status":"treatment","infection_reason":"infection_place"}
              , inplace = True)

tunisia.rename(columns={"date":"confirmed_date", "gender":"sex", "situation":"severity_illness", 
                        "return_from":"infection_place", "health":"background_diseases", "hospital_name":"treatment"}, inplace = True)

japan.rename(columns={"No.":"id", "Fixed date":"confirmed_date","Age":"age", "residence":"region",
                      "Surrounding patients *":"infected_by"}, inplace = True)

indonesia.rename(columns={"patient_id":"id","gender": "sex", "province":"region", "hospital":"treatment",
                          "contacted_with":"infected_by", "current_state":"severity_illness"}, inplace = True)

korea.rename(columns={"patient_id":"id", "disease":"background_diseases_binary", "state":"severity_illness",
                      "province":"region", "infection_case" :"infection_place",
                      "symptom_onset_date":"date_onset_symptoms"}, inplace = True)

Hubei = Hubei.rename(columns={ "province":"region","date_confirmation": "confirmed_date",
                              "chronic_disease_binary":"background_diseases_binary", 
                              "chronic_disease":"background_diseases", "outcome":"severity_illness",
                               "travel_history_location":"infection_place"})

outside_Hubei = outside_Hubei.rename(columns={ "province":"region", "date_confirmation": "confirmed_date",
                              "chronic_disease_binary":"background_diseases_binary", 
                              "chronic_disease":"background_diseases", "outcome":"severity_illness", 
                              "travel_history_location":"infection_place",'travel_history_dates': "return_date", 
                              "travel_history_location":"infection_place" })


# # exam_df = columns vs dfs

# In[ ]:


datasets = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
columns_name = Exam.build_columns_name_ls(datasets)
exam_df = Exam.df_exam_columns_dfs(datasets,datasets_name,columns_name)


# In[ ]:


def full_common(exam_df):
    """
    Returns columns that all DATASETS have
    """
    full_common = []
    for j in exam_df.columns:
        boolyan = exam_df[j].all()
        if boolyan == True:
            full_common.append(j)
    return full_common


# In[ ]:


common = []
unique = []
blank = []
for i in exam_df.columns:
    if exam_df[i].value_counts()[True]>1:
        common.append(i)
    elif exam_df[i].value_counts()[True]==1:
        unique.append(i)
    else:
        blank.append(i)
        
        
print(common)
print(unique)
print(blank)   


# # drop feature

# Tests for columns' usefulness before drop

# country_new

# In[ ]:


for x in [outside_Hubei, Hubei]:
    l = x.index[x.country_new.notnull() == True]
    p = []
    for i in l:
        if x.country_new[i] == x.country[i]:
            p.append(i)

    print("country_new == country",len(p))
    print("country_new.notnull",len(l),"\n")


# The column has no new information to give

# ID

# In[ ]:


for x in [outside_Hubei, Hubei]:
    m =[]
    for i in range(len(x)):
        if x.ID[i] != str(i+1):
            m.append(i)
    print(m[0],x.ID[m[0]])


# As you can see the ID is not arranged in any numerical order.
# and because there is no column that needs another row identifier
# I drop ID

# # drop

# In[ ]:


france = france.drop(['id', "departement","source","comments","contact_number"],axis=1)


indonesia = indonesia.drop(["id", 'nationality'],axis=1)
japan = japan.drop(["id"],axis=1)

korea = korea.drop(["id", "age","contact_number"],axis=1)

Hubei = Hubei.drop(["ID",'location', 'admin3', 'admin2', "admin1" ,'latitude', 'longitude',
                    'geo_resolution','admin_id', "country_new","source","additional_information","geo_resolution"
                    ,"notes_for_discussion"],axis=1)

outside_Hubei = outside_Hubei.drop(["ID",'location', 'admin3', 'admin2', "admin1" ,'latitude', 'longitude',
                                    'geo_resolution', 'admin_id', "country_new", "data_moderator_initials",
                                    "source","additional_information","geo_resolution",
                                    "notes_for_discussion"],axis=1)


# # Examining values - v1 

# In[ ]:


def examining_values_by_col (datasets, datasets_name, col):
    """
    Prints values of each DF per column
    """
    counter = 0
    
    for i in datasets:
        if col in i.columns:
            print("\n" + datasets_name[counter])
            print(i[col].value_counts())
        counter =counter + 1


# In[ ]:


datasets = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
columns_name = Exam.build_columns_name_ls(datasets)
exam_df = Exam.df_exam_columns_dfs(datasets,datasets_name,columns_name)

for j in exam_df.columns[1:len(exam_df.columns)]:
    print(j)
    examining_values_by_col (datasets , datasets_name , j) 


# # format col

# >datetime

# In[ ]:


l1= tunisia.index[tunisia["return_date"] == "Local"]
l2 = tunisia.index[ tunisia["return_date"].notnull()]

index = l2.drop(l1)

for indx in index:
    tunisia.loc[indx,"return_date"] = pd.to_datetime(tunisia.loc[indx,"return_date"])


# In[ ]:


### CPU
cols = ["confirmed_date","released_date", "deceased_date"]

france[cols] = france[cols].apply(pd.to_datetime)
indonesia[cols] = france[cols].apply(pd.to_datetime)
japan[cols] = france[cols].apply(pd.to_datetime)
korea[cols] = korea[cols].apply(pd.to_datetime)

#### different#####

# korea
korea_col = ["date_onset_symptoms"]
korea[korea_col] = korea[korea_col].apply(pd.to_datetime)

#  tunisia
tunisia_col = ["confirmed_date"]
tunisia[tunisia_col] = tunisia[tunisia_col].apply(pd.to_datetime)

# Hubei
Hubei_col = ["confirmed_date", "date_death_or_discharge", "date_onset_symptoms"]
Hubei[Hubei_col] = Hubei[Hubei_col].apply(pd.to_datetime)

# outside_Hubei
outside_Hubei_col = ["date_death_or_discharge"]
outside_Hubei[outside_Hubei_col] = outside_Hubei[outside_Hubei_col].apply(pd.to_datetime)

# 'travel_history_dates'
for j in ["confirmed_date", "date_onset_symptoms"]:
    indexs = outside_Hubei.index[outside_Hubei[j].notnull()]
    indexs_ , error = UpdateTime.updte_time(outside_Hubei, j, j, indexs,".",[ "-", ','])
    print(j , error)


# In[ ]:


indexs =  outside_Hubei.index[outside_Hubei["return_date"].notnull()]

for indx in indexs:
    i = outside_Hubei.loc[indx, "return_date"]
    i = i.split("-")

    for x in range(len(i)):
        i[x] = pd.to_datetime(i[x], errors='ignore')


        if len(i) == 1:
            pass
            outside_Hubei.loc[indx , "return_date"] = i[0]

        elif len(i)>1 and type(i[0]) == type(i[1]):
            outside_Hubei.loc[indx , "return_date"] =  pd.DataFrame({"t":i}).max()[0]
        
        
outside_Hubei["return_date"].value_counts()


# In[ ]:


# error
outside_Hubei.index[outside_Hubei["date_onset_symptoms"] == "- 25.02.2020"]


# In[ ]:


examining_values_by_col (datasets , datasets_name , "date_onset_symptoms") 


# sex

# In[ ]:


tunisia_sex = {"F":"female", "M":"male",np.nan:np.nan}
tunisia.sex = [tunisia_sex[item] for item in  tunisia.sex] 

japan_sex = {"Woman":"female", "Man":"male",np.nan:np.nan, "Checking":np.nan, "investigating":np.nan}
japan.sex = [japan_sex[item] for item in  japan.sex] 

france_sex = {"female":"female", "male":"male","Female":"female", "Male":"male", "male\xa0?":"male", 
              np.nan:np.nan }
france.sex = [france_sex[item] for item in  france.sex] 


# In[ ]:


examining_values_by_col(datasets, datasets_name, "sex")


# In[ ]:


def update_index(dataset, col, indexs, data):
    """
    Value change according index
    
    dataset: df
    
    col : str
        name of col you want to change
        
    indexs: pd.index
    
    data: int/ str/ float
        data you want to into
    
    """
    for indx in indexs:
        dataset.loc[indx,col] = data


# background_diseases_binary

# In[ ]:


indexs = korea.index[korea.background_diseases_binary == True]
update_index(korea, "background_diseases_binary", indexs, 1.0)


# In[ ]:


tunisia["background_diseases_binary"] = np.nan

for dataset in [tunisia, Hubei, outside_Hubei]:
    indexs = dataset.index[dataset.background_diseases.notnull()]
    update_index(dataset,"background_diseases_binary",indexs,1.0) 


# In[ ]:


examining_values_by_col (datasets, datasets_name, "background_diseases_binary")


# background_diseases

# In[ ]:


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def remove(dict_a, keys_remove ):
    for key in keys_remove:
        if key in dict_a.keys():
            dict_a.pop(key)


# In[ ]:


ps = nltk.stem.SnowballStemmer('english')
r = []
o = []
for ind in outside_Hubei.index[outside_Hubei.background_diseases.notnull()]:
    i = outside_Hubei.loc[ind, "background_diseases"]

    i = BagWords.clean_str(i)
    l = [ps.stem(x) for x in i]

    for x in l:
        if x.isalpha():
            r.append(x)
            
keys_remove = ['to', 'a','like',  'no', 'and',  'yes', 'then','complaint',"great", "even", 
         "for", "the", "non",  'of' , "this",  'on' ,'with', "was", 'c',
         "cannot", "recommend", "as", "a", "i", "did", "not", "want", "to", "have", "to", "do", "this"]


            
test_dict = dict(Counter(r))
remove(test_dict, keys_remove )
print(test_dict)


# #                                                     Complete features

# severity_illness

# In[ ]:


bag_words= {"good":["good","stabl", "follow"],
            "critical":["critic", "intens", "sever"], 
            "deceased": ["death","dead", "die", "deceas" ],
            "cured":["discharg", "releas", "cure", "recov", 'health'],
            np.nan: ["isol"]}

sentences_bag = {"good":[['not', 'hospit'],['in', 'progress']],
                "critical":[], 
                "deceased": [ ],
                "cured":[]}



datasets2 = [france, tunisia, indonesia, korea, Hubei, outside_Hubei]
datasets_name1 = ["france", "tunisia", "indonesia", "korea", "Hubei", "outside_Hubei"]

for ind in range(len(datasets_name1)):
    dataset = datasets2[ind]
    indexs = dataset.index[dataset.severity_illness.notnull()]
    no_guess,multi_guess = BagWords.guess_category(dataset, "severity_illness", "severity_illness",indexs, ps, bag_words, sentences_bag)
    
    print(datasets_name1[ind])
    print(no_guess)
    print(multi_guess)

for x in [indonesia, france]:  
    indexs = x.index[x.deceased_date.notnull()]
    update_index(x,"severity_illness",indexs,"deceased") 

    indexs = x.index[x.released_date.notnull()]
    update_index(x,"severity_illness",indexs,"cured") 


# In[ ]:


datasets2 = [france, tunisia, indonesia, korea, Hubei, outside_Hubei]
datasets_name1 = ["france", "tunisia", "indonesia", "korea", "Hubei", "outside_Hubei"]
examining_values_by_col(datasets2, datasets_name1,  "severity_illness")


# released_date / deceased_date

# In[ ]:


categories = ["deceased", "cured"]
cols = ["deceased_date","released_date"]
for indx_j in range(len(cols)) :
    j  = cols[indx_j]
    category = categories[indx_j]
    
    for x in [outside_Hubei, Hubei]:
        x[j] = np.nan 
        indexs = x.index[x["severity_illness"] == category]

        for i in indexs:
            x.loc[i, j]= pd.to_datetime(x.loc[ i, "date_death_or_discharge"])


# released_date exam

# In[ ]:


datasets2 = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
examining_values_by_col(datasets2, datasets_name, "released_date")


# deceased_date exam

# In[ ]:


datasets2 = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
examining_values_by_col(datasets2, datasets_name, "deceased_date")


# test

# In[ ]:


for x in [outside_Hubei, Hubei]:
    l = x.date_death_or_discharge.notnull().sum()
    y = x.severity_illness.notnull().sum()
    p = x.released_date.notnull().sum() +x.deceased_date.notnull().sum()
    print(l,y,p)


# In[ ]:


for x in [outside_Hubei]:
    complete_features =list(x.index[x.released_date.notnull()]) + list(x.index[x.deceased_date.notnull()])
    date_death_or_discharge = list(x.index[x.date_death_or_discharge.notnull()])
    severity_illness = list(x.index[x.severity_illness.notnull()])

    if complete_features == date_death_or_discharge:
        print("==")
    else:
        print("not ==")
    
    for i in severity_illness:
        if i not in complete_features:
            print(i)


# age

# exam 

# In[ ]:


indexs =  outside_Hubei.index[outside_Hubei.age.notnull()]

def int_num(dataset, col, indexs):
    to_float = []

    for i in indexs:
        if dataset.loc[i, col].isdigit() == True:
            to_float.append(i)

    for indx in  to_float:
        dataset.loc[indx, col] = float(dataset.loc[indx, col])
    
    return to_float
        

to_float= int_num(outside_Hubei, "age",indexs)
indexs = indexs.drop(to_float)

print(indexs)


# In[ ]:


### NEED AGE CPU 

def birth_year_to_age(data):
    age_ls = []

    for i in range(len(data)):
        age_ls.append(data.confirmed_date[i].year - data.birth_year[i])
    return age_ls

korea["age"] = birth_year_to_age(korea)
france["age"] = birth_year_to_age(france)


# country

# In[ ]:


tunisia["country"] = ["tunisia" for i in range(len(tunisia))]
japan["country"] = ["japan" for i in range(len(japan))]
indonesia["country"] = ["indonesia" for i in range(len(indonesia))]


# In[ ]:


for dataset in [outside_Hubei,Hubei]:

    indexs = dataset.index[dataset["country"].notnull()]
    for indx in indexs:

        dataset.loc[indx, "country"] = dataset.loc[indx, "country"].lower()


# In[ ]:


outside_Hubei["country"].value_counts()


# outside_Hubei data VS country data

# In[ ]:


print(len(korea))
print(outside_Hubei.country.value_counts()["south korea"])
print()

print(len(france))
print(outside_Hubei.country.value_counts()["france"])
print()

print(len(Hubei))
print(outside_Hubei.country.value_counts()["china"])


# **infection_place

# In[ ]:


datasets2 = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
examining_values_by_col(datasets2, datasets_name, "infection_place")


# In[ ]:


print(Hubei["wuhan(0)_not_wuhan(1)"].value_counts())
print(outside_Hubei["infection_place"].value_counts())


# infection_case
# 
# = Community \abroad \ Nan

# duplicate

# In[ ]:


index_chack =  outside_Hubei.index[outside_Hubei["country"] == "France" ]


# In[ ]:


l = ['sex',  'city', 'confirmed_date',  'age']
df1 = france.loc[:, l]
 
df2 = outside_Hubei.loc[index_chack, l]

x = pd.concat([df2,df1])

print(x.shape)

df_diff = x.index[x.duplicated(keep="last") == True]

len(df_diff)


# In[ ]:


p = pd.DataFrame({"r": [0,9,0,88,7,6,0,0,0],
                 "o": [0,9,0,88,7,6,6,7,4]})

n = pd.DataFrame({"r": [0,9,4,886,77,6,607,0,0],
                 "o": [0,9,0,88,7,65,6,7,4]})

t = pd.concat([n,p])
k = t.index[t.duplicated(keep="last") == False]
k


# treatment

# In[ ]:


index = tunisia.index[tunisia["hospital_place"].notnull()]

y = tunisia.index[tunisia["treatment"] == "Self-insulation"]
index = index.drop(y)
print(index)

update_index(tunisia, "treatment", y,"home isolation")
update_index(tunisia, "treatment", index ,"hospital")


# In[ ]:


index = indonesia.index[indonesia["treatment"].notnull()]
update_index(tunisia, "treatment", index ,"hospital")

indonesia["treatment"].value_counts()


# In[ ]:


y = france.index[france["treatment"] == "deceased"] 
x = france.index[france["treatment"] == "released"]

update_index(france, "treatment", y ,np.nan)
update_index(france, "treatment", x ,np.nan)


# In[ ]:


outside_Hubei["treatment"] = np.nan
index = outside_Hubei.index[outside_Hubei['date_admission_hospital'].notnull()]
update_index(outside_Hubei, "treatment",  index,"hospital")

Hubei["treatment"] = np.nan
index = Hubei.index[Hubei['date_admission_hospital'].notnull()]
update_index(Hubei, "treatment", index ,"hospital")


# del country

# In[ ]:


for i in ['china',"japan","france","south korea"]:
    ind = outside_Hubei.index[outside_Hubei["country"] == i]
    outside_Hubei = outside_Hubei.drop(ind, axis=0)

outside_Hubei["country"].value_counts()


# #  Garbage drop 
# - Features that have only one dataset or  built with Engineered another feature with them

# In[ ]:


france = france.drop(["birth_year", "group"],axis=1)
tunisia = tunisia.drop(["hospital_place"],axis=1)
japan = japan.drop(["Close contact situation"],axis=1)
korea = korea.drop(["birth_year","global_num"],axis=1)
Hubei = Hubei.drop(["date_death_or_discharge"],axis=1)


# # drop Non-baked features

# In[ ]:


Hubei.columns


# In[ ]:


Hubei = Hubei.drop(['date_admission_hospital',"wuhan(0)_not_wuhan(1)",'travel_history_dates',
                    'lives_in_Wuhan', "reported_market_exposure","sequence_available"],axis=1)
outside_Hubei = outside_Hubei.drop( ['wuhan(0)_not_wuhan(1)', 'date_admission_hospital', "date_death_or_discharge", 'lives_in_Wuhan',
 'reported_market_exposure','sequence_available'],axis=1)


# **> Feature sum**

# In[ ]:


datasets2 = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
columns_name = Exam.build_columns_name_ls(datasets2)
exam_df = Exam.df_exam_columns_dfs(datasets2, datasets_name, columns_name)
print(columns_name)
exam_df.infection_place


# unipue

# common feature

# In[ ]:


datasets3 = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei]
columns_name = Exam.build_columns_name_ls(datasets3)
exam_df2 = Exam.df_exam_columns_dfs(datasets3,datasets_name,columns_name)


# In[ ]:


for i in exam_df2.columns:
    print("\n"+i)
    examining_values_by_col (datasets, datasets_name, i)


# # build final DS

# In[ ]:


exam_df.sex


# In[ ]:


datasets_final = [france, tunisia, japan, indonesia, korea,outside_Hubei, Hubei ]
final_DS = pd.concat(datasets_final, axis=0)


# # orgnaze DS

# index

# In[ ]:


final_DS.index = range(len(final_DS))


# In[ ]:


final_DS.to_csv(r'/kaggle/working/Characteristics_Corona_patients2.csv', index = False)


# In[ ]:


final_DS.to_csv()

