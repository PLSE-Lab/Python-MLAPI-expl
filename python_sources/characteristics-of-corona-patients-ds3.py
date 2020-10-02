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
world_a = pd.read_csv("/kaggle/input/covid19-outbreak-realtime-case-information/latestdata.csv")


# In[ ]:


datasets = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei, world]
datasets_name = ["france", "tunisia", "japan", "indonesia", "korea", "Hubei", "outside_Hubei", "world"]

garbge = [print("\n"+datasets_name[i], [i for i in datasets[i].columns]) for i in range(len(datasets_name))]


# In[ ]:


p = ['age', 'sex', 'city', 'province', 'country',"age", 'date_confirmation', 'date_onset_symptoms',
     'date_admission_hospital', 'symptoms', 'source']
l = ['ID', 'age', 'sex', 'city', 'province', 'country', 'wuhan(0)_not_wuhan(1)', 'geo_resolution',
     'date_onset_symptoms', 'date_admission_hospital', 'date_confirmation', 'symptoms',
     'lives_in_Wuhan', 'travel_history_dates', 
      'chronic_disease_binary']


df1 = Hubei.loc[:, l]

df2 =  outside_Hubei.loc[:, l]

df_diff = pd.concat([df2,df1]).drop_duplicates()

www =  df_diff


df2 =  world_a.loc[:, l]

df_diff = pd.concat([df2,www]).drop_duplicates()

mm = df_diff


# In[ ]:


e = []
for ind in range(len(mm['date_confirmation'])):
    if type(mm['date_confirmation'][ind]) == pd.core.series.Series:
        e.append(ind)


# In[ ]:


mm['date_confirmation'][(mm['date_confirmation'][21164].notnull()) == ]


# In[ ]:


# ls_i = [i  for i in mm.date_confirmation[mm['date_confirmation'].notnull()]]

# indexs = mm.index[mm['date_confirmation'].notnull()]
# dataset =mm
# input_col = 'date_confirmation'
# output_col ='date_confirmation'
# date_character= "."
# character_separator =  "-"

# d = []

# for i in ls_i:

#     p = i.split(character_separator)
#     p = UpdateTime.del_str_equal_x_from_ls(p, "")
#     boo = UpdateTime.redundant_numbers_date(p, date_character)
#     if len(p) == 1:
#             d.append(p)
            
# mm['date_confirmation']            
# # mm['date_confirmation'].max()


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
                        "return_from":"infection_place", "health":"background_diseases"}, inplace = True)

japan.rename(columns={"No.":"id", "Fixed date":"confirmed_date","Age":"age", "residence":"region",
                      "Surrounding patients *":"infected_by"}, inplace = True)

indonesia.rename(columns={"patient_id":"id","gender": "sex", "province":"region", "hospital":"hospital_name",
                          "contacted_with":"infected_by", "current_state":"severity_illness"}, inplace = True)

korea.rename(columns={"patient_id":"id", "disease":"background_diseases_binary", "state":"severity_illness",
                      "province":"region", "infection_case" :"infection_place",
                      "symptom_onset_date":"date_onset_symptoms"}, inplace = True)

Hubei = Hubei.rename(columns={ "province":"region","date_confirmation": "confirmed_date",
                              "chronic_disease_binary":"background_diseases_binary", 
                              "chronic_disease":"background_diseases", "outcome":"severity_illness"})

outside_Hubei = outside_Hubei.rename(columns={ "province":"region", "date_confirmation": "confirmed_date",
                              "chronic_disease_binary":"background_diseases_binary", 
                              "chronic_disease":"background_diseases", "outcome":"severity_illness" })


# In[ ]:


def format_datatime(dataset, input_col, output_col, indexs, date_character, character_separator, earliest=False):

    drop = []
    indexs_error = []
    for indx in indexs:
        i = dataset.loc[indx, input_col]
        print(i)
#         ls = i.split(character_separator)
#         ls = UpdateTime.del_str_equal_x_from_ls(ls, "")
#         boo = UpdateTime.redundant_numbers_date(ls, date_character)
#         if boo == False:
#                 indexs_error.append(indx)
#         else:
#             if len(ls) > 1:
#                 ls = UpdateTime.make_ls_of_str_datatime(ls)
#                 value = UpdateTime.time_range_extremity(ls, earliest)
#                 dataset.loc[indx, output_col] = value
#                 drop.append(indx)
#     indexs = indexs.drop(drop)
#      return indexs, indexs_error

format_datatime(mm,'date_confirmation','date_confirmation' ,indexs ,".",[ "-"])


# In[ ]:


# 'travel_history_dates'

# world['date_confirmation'] = world['date_confirmation'].apply(pd.to_datetime)
indexs = world.index[world['date_confirmation'].notnull()]
indexs_ , error = UpdateTime.updte_time(world,'date_confirmation','date_confirmation' ,[1],".",[ "-"])


# # exam_df = columns vs dfs

# In[ ]:


datasets = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei, world]
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


france = france.drop(["departement","source","comments","contact_number"],axis=1)


indonesia = indonesia.drop(['nationality'],axis=1)

korea = korea.drop(["age","contact_number"],axis=1)

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


datasets = [france, tunisia, japan, indonesia, korea, Hubei, outside_Hubei, world]
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


# cols = ["confirmed_date","released_date", "deceased_date"]

# france[cols] = france[cols].apply(pd.to_datetime)
# indonesia[cols] = france[cols].apply(pd.to_datetime)
# japan[cols] = france[cols].apply(pd.to_datetime)
# korea[cols] = korea[cols].apply(pd.to_datetime)

# #### different#####

# # korea
# korea_col = ["date_onset_symptoms"]
# korea[korea_col] = korea[korea_col].apply(pd.to_datetime)

# #  tunisia
# tunisia_col = ["confirmed_date"]
# tunisia[tunisia_col] = tunisia[tunisia_col].apply(pd.to_datetime)

# # Hubei
# Hubei_col = ["confirmed_date", "date_death_or_discharge", "date_onset_symptoms"]
# Hubei[Hubei_col] = Hubei[Hubei_col].apply(pd.to_datetime)

# # outside_Hubei
# outside_Hubei_col = ["date_death_or_discharge"]
# outside_Hubei[outside_Hubei_col] = outside_Hubei[outside_Hubei_col].apply(pd.to_datetime)

# # 'travel_history_dates'
# for j in ["confirmed_date", "date_onset_symptoms"]:
#     indexs = outside_Hubei.index[outside_Hubei[j].notnull()]
#     indexs_ , error = UpdateTime.updte_time(outside_Hubei, j, j, indexs,".",[ "-", ','])
#     print(j , error)


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


# city

# In[ ]:


# indexs = Hubei.index[Hubei.city.notnull()]
# for indx in indexs:
#     i = Hubei.loc[indx, "city"]
#     i = i.split(" ")
#     print(i)
#     if len(i)> 1:
#         ls_value = del_str_equal_x_from_ls(i, "City")
#         for i in ls_value:
#         Hubei.loc[indx, "city"] = value


# In[ ]:


Hubei.loc[5, "city"]


# In[ ]:


examining_values_by_col(datasets, datasets_name, "city")


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


examining_values_by_col (datasets, datasets_name, "background_diseases")


# In[ ]:


# ps = nltk.stem.SnowballStemmer('english')

# for i in outside_Hubei.background_diseases[outside_Hubei.background_diseases.notnull()]:
#     print("\n"+i)
#     i = BagWords.clean_str(i)
    
#     print([ps.stem(x) for x in i ])
    
# # chronic obstructive pulmonary disease  'obstruct', 'pulmonari', 'diseas'


# In[ ]:


# a_bag_words= {"hypertension":["hypertens",  ],
            
#             "coronary heart disease":["stenocardia" ],
            
#             "diabetes":['diabet', "mellitus"],
               
#             "tuberculosis": ["tuberculosi"],
            
#             "parkinson": ["parkinso", "madopar"],
           
#             "hypertriglyceridemia": ['hypertriglyceridemia',],
            
#             "obesity": ['obes',],
             
#             "Chronic obstructive pulmonary": ['copd',],
            
#             "hiv": ["hiv",],
#            "asthma": ["asthma",],}



# a_sentences_bag = {"hypertension":[['high', 'blood', 'pressur'],],

#                 "coronary heart disease": [['coronari', 'heart'], ['coronari', 'stent']],

#                 "chronic bronchitis":[['chronic', 'bronchiti'],],

#                 "chronic renal insufficiency":[[ 'chronic', 'renal', "insuffici"],],

#                 "hemorrhage of digestive tract":[['hemorrhag', 'of', 'digest', "tra"],],

#                 "colon cancer":[['colon', 'cancer'],],
                
#                  "lung cancer":[['lung', 'cancer'],],

#                 "prostate hypertrophy": [['prostat', 'hypertrophi'],],
                 
#                 "hip replacement": [['hip', "replac"],],

#                 "cerebral infarction":[['hypertens', 'cerebr', 'infarct'],],
                
#                  "hepatitis B": [["hepat", 'b'],]}
                                 
# # 'encephalomalacia'  encephalomalacia   coronary bypass                               
# # time = {['20', 'year'], ['four', 'year'], ['five', 'year']9 years}


# In[ ]:


# datasets2 = [tunisia, Hubei, outside_Hubei]
# datasets_name1 = [ "tunisia", "Hubei", "outside_Hubei"]

# for ind in range(len(datasets_name1)):
#     dataset = datasets2[ind]
#     print(ind)
#     dataset["guess"]= [np.nan for i in range(len(dataset.background_diseases)) ]
#     indexs = dataset.index[dataset.background_diseases.notnull()]
#     no_guess,multi_guess = BagWords.guess_category(dataset, "background_diseases", "guess",indexs, ps, a_bag_words, a_sentences_bag)
    
#     print(datasets_name1[ind])
#     print(no_guess)
#     print(multi_guess)


# In[ ]:


# indexs = Hubei.index[Hubei.background_diseases.notnull()]

# indexs


# symptoms

# In[ ]:


# bag_words= {"pneumonia":["pneumonia","pneumon"],
#             "fever":["fever"], 
#             "cough": ["cough" ],
#             "fatigue":["fatigu"],
#             "discomfort": ["discomfort"],
#            "weakness": ["weak", ['lack', 'of', 'energi]],
#            "dizziness": ["dizzi"],
#            "rhinorrhoea": ["rhinorrhoea",['runni', "nose"]],
#            "sneezing": ["sneez"],
#            "diarrhea": ["diarrhea"],
#             "expectoration": ["expector"],
#             "headache": ["headach"],
#             "diarrhea": ["diarrhea"],
#             "chills": ["chill"],
#             "dyspnea": ["dyspnea"],
#             "rigor": [" rigor"],
#                                  "pharyngalgia": ["pharyngalgia",],
#                                  "no_symptom": ["asymptomat"],
#                                  "nausea": ["nausea"],
                                 
            
#            }

# sentences_bag = {"nasal congestion":[['nasal', 'congest'],['in', 'progress']],
#                 "sore throat":[['sore', "throat"],], 
#                 "pleuritic chest pain": [ [ 'pleurit', 'chest', 'pain']],
#                 "muscular soreness":[['muscular',"sore"]],
#                  "chest distress":[[' chest', 'distress']],
#                  "muscular stiffness":[['muscular', 'stiff']],
#                  "muscular soreness":[['muscular',"sore"],[ "muscl", 'ach'],['muscl', "pain"], ["myalgia"]],
#                  "joint pain":[['muscular', 'stiff']],
#                  "sore limbs":[['muscular', 'stiff']],
                 
                 
#                }
                 
# # pleuritic chest pain , 


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
for ind in outside_Hubei.index[outside_Hubei.symptoms.notnull()]:
    i = outside_Hubei.loc[ind, "symptoms"]

    i = BagWords.clean_str(i)
    l = [ps.stem(x) for x in i]

    for x in l:
        if x.isalpha():
            r.append(x)
            
keys_remove = ['to', 'a','like',  'no', 'and',  'yes', 'then','complaint',"great", "even", 
         "for", "the", "non",  'of' , "this",  'on' ,'with', "was", 'c',
         "cannot", "recommend", "as", "a", "i", "did", "not", "want", "to", "have", "to", "do", "this"]


            
# r = columns_name = list(dict.fromkeys(o))
test_dict = dict(Counter(r))
remove(test_dict, keys_remove )
print(test_dict)



# In[ ]:


for key in test_dict.keys():
    
    o = test_dict[key]
    print(key, o)
    if o < 9:
        test_dict.pop(key)
        
print(test_dict)


# _____________________________________________________lab

# In[ ]:


# def remover(my_string =""):
#     values = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
#     for item in my_string:
#         if item not in values:
#             my_string = my_string.replace(item,"")
#     return my_string




# ps = nltk.stem.SnowballStemmer('english')

# for i in Hubei.severity_illness[Hubei.severity_illness.notnull()]:
#     i = remover(i)
#     i = i.split(" ")
#     print(i)
# #     print([ps.stem(x) for x in i ])
    


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

        
        if "." in i:
            to_float.append(i)

#     for indx in  to_float:
#         dataset.loc[indx, col] = int(dataset.loc[indx, col])
    
    return to_float
        

to_float= int_num(outside_Hubei, "age",indexs)
indexs = indexs.drop(to_float)

print(indexs)


# In[ ]:


float_indx = []

for indx in indexs:
    
   print(type(outside_Hubei.loc[indx, "age"]))


    
    


# In[ ]:


for i in indexs:
    ls = i.split("-")
    if len(ls)>1:
        y = ls[0]-ls[1]
        print(y)
    print(outside_Hubei.loc[i, "age"], i)


# In[ ]:



outside_Hubei.head(11714)


# In[ ]:



# outside_Hubei_age = {"investigating":np.nan, "Checking":np.nan, "Under 10":"0s", "Under teens":"0s","305":"30s",
#             "10s":"10s","20s":"20s", "30-39":"30s", "40-49":"40s", "50-59":"50s", "60-69":"60s", "70s":"70s" ,
#              "80s":"80s","90s":"90s" }
# outside_Hubei.age = [outside_Hubei_age[item] for item in outside_Hubei.age] 


# In[ ]:


# Hubei.age.value_counts()


# In[ ]:


# Hubei_age = {"15-88":np.nan, "25-89":np.nan, "21-39":np.nan, "40-49":"40s", "50-59":"50s", "60-69":"60s",
#              "70-82":"70s" }
# Hubei.age = [Hubei_age[item] for item in Hubei.age] 


# In[ ]:


# japan.age.value_counts()


# In[ ]:


# japan_age = {"investigating":np.nan, "Checking":np.nan, "Under 10":"0s", "Under teens":"0s","305":"30s",
#             "10s":"10s","20s":"20s", "30s":"30s", "40s":"40s", "50s":"50s", "60s":"60s", "70s":"70s" ,
#              "80s":"80s","90s":"90s" }
# japan.age = [japan_age[item] for item in japan.age] 


# In[ ]:


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


# outside_Hubei data VS country data

# In[ ]:


print(len(korea))
print(outside_Hubei.country.value_counts()["South Korea"])
print()

print(len(france))
print(outside_Hubei.country.value_counts()["France"])


# infection_place

# In[ ]:


print(Hubei["wuhan(0)_not_wuhan(1)"].value_counts())
print(outside_Hubei["travel_history_location"].value_counts())


# infection_case
# 
# = Community \abroad \ Nan

# In[ ]:


# TODO


# #  Garbage drop 
# - Features that have only one dataset or  built with Engineered another feature with them

# In[ ]:


france = france.drop(["birth_year", "treatment","group"],axis=1)
tunisia = tunisia.drop(["hospital_place"],axis=1)
japan = japan.drop(["Close contact situation"],axis=1)
korea = korea.drop(["birth_year","global_num"],axis=1)
Hubei = Hubei.drop(["date_death_or_discharge"],axis=1)


# # drop Non-baked features

# In[ ]:


france = france.drop(["infection_place", "infected_by","infection_order"],axis=1)
tunisia = tunisia.drop(["hospital_name"],axis=1)
indonesia = indonesia.drop(["infected_by","hospital_name"],axis=1)
korea = korea.drop(["infection_place", "infected_by","infection_order"],axis=1)


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


for i in dfs:
    print(i[col].isnull().sum())


# In[ ]:


for i in datasets:
    print(i.sex.isnull().sum())


# In[ ]:





# In[ ]:


datasets_final = [france, tunisia, japan, indonesia, korea]
final_DS = pd.concat(datasets_final, axis=0)


# In[ ]:


final_DS.status.value_counts()


# # orgnaze DS

# index

# In[ ]:


final_DS.index = range(len(final_DS))


# In[ ]:


final_DS.to_csv(r'/kaggle/working/Characteristics_Corona_patients1.csv', index = False)


# In[ ]:


final_DS.to_csv()


# infected by

# In[ ]:




