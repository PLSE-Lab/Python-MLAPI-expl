#!/usr/bin/env python
# coding: utf-8

# # DATASCIENCE4GOOD : City of LA

# ## Help the City of Los Angeles to structure and analyze its job descriptions

# ### **Please do not hesitate to comment, critic and vote so it will help to improve the approach by doing better or differently the things :)** <br>

# ## Table of contents
# 1. [Introduction](#intro)<br>
#     1.1. [Libraries](#lib)<br>
#     1.2. [Purpose & strategy](#purp)<br>
#     1.3. [Folder content](#Folcont)<br>
#     
# 2. [Structure of the bulletins](#struct)<br>
#     2.1. [Example](#expl)<br>
#     2.2. [Load the structure](#load)<br>
#     2.3. [Standardization of the structure](#stand)<br>
#     
# 3. [Extraction of data](#extract)<br>
#     3.1. [Administration data : <br>CLASS CODE, OPEN DATE, REVISED DATE, EXAM OPEN TO](#admin)<br>
#     3.2. [Salary data : <br> *QUANTITATIVE* : DEPARTEMENT (WATER POWER, HARBOR, AIRPORT) & GENERAL SALARY, RANGE & FLAT SALARY,<br> *QUALITATIVE* : LOWER & HIGHER PAY REASON](#sala)<br>
#     3.3. [Minimum requirement qualification : ](#minireq)<br>

# ## 1. Introduction
# 
# <a id="intro"></a>

# ### 1.1. Libraries
# 
# <a id="lib"></a>

# In[ ]:


#Librairies for data treatment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections as col
from collections import Counter

#Libraries for text treatment
import re
import string
import nltk
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from textblob import TextBlob
from difflib import SequenceMatcher

#Library for file loading
import os
import csv

#Library for ML
from sklearn import feature_extraction

pd.options.display.max_colwidth = 100


# ### 1.2. Purpose & strategy
# <a id="purp"></a>

# The puporse of this challenge is to struture, extract and analyse the data of the job descriptions contained in the bulletins.
# Our strategy to achieve thoses challenges both on the form and the content:
# * Standardize the structure of the bulletins - Form
# * Extract the data from the bulletins - Content
# * Analyse how the form and the content influence the applicants

# ### 1.3. Folder content
# <a id="Folcont"></a>

# The main folder CityofLA include two subfodlers:

# In[ ]:


for subfold in os.listdir("../input/cityofla/CityofLA/"):
    print(subfold)


# The subfolder Aditionnal data contain th following files or folders :

# In[ ]:


for subfold in os.listdir("../input/cityofla/CityofLA/Additional data"):
    print(subfold)


# The three csv files job_titles, kaggle_data_dictionnary and sample_job_class_export_template.
# Let's explore them to get information about the documents.

# In[ ]:


for subfold in os.listdir("../input/cityofla/CityofLA/Additional data") :
    if ".csv" in subfold :
        file_csv=pd.read_csv("../input/cityofla/CityofLA/Additional data/"+subfold)
        print(subfold + ', ' + str(file_csv.shape[0]) + ' rows, ' + str(file_csv.shape[1]) + ' columns')
        display(file_csv.head(5))


# > 
# We will focus on the 668 job positions that we should retrieve in the folder Job Bulletins

# In[ ]:


file_txt=os.listdir("../input/cityofla/CityofLA/Job Bulletins/")
print('Number of elements in the folder Job Bulletins : ' + str(len(file_txt)))
expl=5
print('Here the {0} first txt job bulletins : '.format(str(expl)))
for name_txt in file_txt[:expl]:
    print(name_txt)


# We notice here that there is a difference between the number of the job positions specified by the csv file job_titles and the number of job descriptions as txt file.
# Let'sb check the gap between the two list.

# We can expect that several job descriptions rely to a same job position.

# In[ ]:


frequencyBulletin=col.Counter([re.split("\s+[0-9]|\.+",x)[0] for x in sorted(file_txt)])
frequencyBulletin=pd.DataFrame.from_dict(frequencyBulletin,orient='index',columns=['Count_txt'])
frequencyBulletin.index.names = ['Name_txt']
frequencyBulletin.reset_index(inplace=True)
frequencyBulletin.loc[frequencyBulletin.Count_txt>1]


# We can expect that some job bulettins are not reprensented in the list of job title positions contained in the csv.

# In[ ]:


jobTitle=pd.read_csv("../input/cityofla/CityofLA/Additional data/job_titles.csv", names=['Name_csv'])
jobTitle.Name_csv=jobTitle.Name_csv.replace(to_replace="'|&", value='_', regex=True)
titleJobBulletin=frequencyBulletin.merge(jobTitle, how='outer', left_on='Name_txt', right_on='Name_csv')
titleJobBulletin.loc[(titleJobBulletin.Name_txt.isnull())|(titleJobBulletin.Name_csv.isnull())]


# In[ ]:


jobTitle=pd.read_csv("../input/cityofla/CityofLA/Additional data/job_titles.csv", names=['Name_csv'])
jobTitle.Name_csv=jobTitle.Name_csv.replace(to_replace="'|&", value='_', regex=True)
titleJobBulletin=frequencyBulletin.merge(jobTitle, how='outer', left_on='Name_txt', right_on='Name_csv')
print("""Thus we know that 
for {0} job titles we have a bulletin;
for {1} job bulletins we don't have any reference in the job title csv file.""".format(1+len(titleJobBulletin.loc[(~titleJobBulletin.Name_txt.isnull())&(~titleJobBulletin.Name_csv.isnull()),'Name_txt']),
                                                                                     len(titleJobBulletin.loc[(~titleJobBulletin.Name_txt.isnull())&(titleJobBulletin.Name_csv.isnull()),'Name_csv'])-1))


# ## 2. Structure of the bulletin
# 
# <a id="struct"></a>

# ### 2.1. Example
# 
# <a id="expl"></a>

# We take as example a bulletin to have a first understanding of the structure.

# In[ ]:


job="SENIOR REAL ESTATE OFFICER 1961 0413018 (2).txt"
print(job)
file=open(r"../input/cityofla/CityofLA/Job Bulletins/"+job,"r")
txt_file=file.read()
txt_file[:500]


# ### 2.2. Load the structure
# 
# <a id="load"></a>

# We analyse the bulletin txt file and different elements structure the document :
# * Elements in CAPITAL letters
# * \n for new line symbol
# * \t for tab symbol
# * Succesions of dots
# * List of points 1., 2., 3.,... ect

# In[ ]:


def fromTxt2Dataframe (string,job):
    """
    Input (string) : content of the bulletin txt file
    Output (dataframe) : 
                        header the title part (capital letter) ; 
                        row the  content between two title parts (list with each item is a line)
    This function transform the txt file into a semi structure dataframe as first step    
    """

    # we seperate each line

    superList=[]
    cellList=[]
    key=''
    note_flag=0
    
    #string=re.sub(re.split("\s+[0-9]|\.+",job)[0].replace(' ','[\n\s]*'), "", string)
    
    string=re.sub("\t", " ", string)
    string=re.sub(" +", " ", string)
    string=re.sub("\s\.+", "", string)
    string=re.sub("[\s ]*\n[\s ]*", "\n", string)
    string=re.sub("^\s+", "", string)
    string=[x for x in re.split("[\n\t]",string) if not (x==''or x==' ')]
    
    stringLen=len(string)-1
    
    superList.append(('JOB_NAME_TXT',re.sub(" +", " ",re.split("\s+[0-9]|\.+",job)[0]).upper()))  
    
    if string[0]=='CAMPUS INTERVIEWS ONLY':
        start=2
        superList.append((string[0],string[0]))
        title=string[1]
        item=string[start]
        while item.isupper() and re.search("[0-9]", item)==None:
            title=title + " " + item
            start += 1
            item=string[start]
        superList.append(('JOB_TITLE_TXT',title.upper()))

    else :
        start=1
        title=string[0]
        item=string[start]
        while item.isupper() and re.search("[0-9]", item)==None:
            title=title + " " + item
            start += 1
            item=string[start]
        superList.append(('JOB_TITLE_TXT',title.upper()))
    
    key='ADMINISTRATION'
    
    for index, item in enumerate(string[start:]):
        # we keep capital letters as title part
        if item.isupper() and re.search("[0-9]", item)==None:
            if cellList:
                if re.search(r"^NOTES*",item):
                    if note_flag==0:
                        superList.append((key,cellList))
                        cellList=[]
                        key=key + " " + item
                        note_flag=1
                else :
                    superList.append((key,cellList))
                    cellList=[]
                    key=item
                    note_flag=0
            else:
                key=key+' '+item
            
            #print(note_flag, key)
        # otherwise it's the content of the part
        else:
            cellList.append(item)
            if index==stringLen:
                superList.append((key,cellList))
    
    df_position=pd.DataFrame([dict(superList)],columns=dict(superList).keys())
    return df_position


# In[ ]:


fromTxt2Dataframe(txt_file,job)


# ### 2.3. Standardization of the structure
# 
# <a id="stand"></a>

# We need to standardize the headers of the dataframe we built :
# * Remove stop words and ponctuation if any
# * Correct wording mistake
# * Singularize words
# * Combine NOTE(S) headers 

# In[ ]:


def standardize2Aggregate(listOfFile,dictOfStandard=dict(), step='standardize'):
    
    list_metaData=[]
    df1_metaData=pd.DataFrame()
    
    for job in listOfFile:
        file=open(r"../input/cityofla/CityofLA/Job Bulletins/"+job,"r")
        try:
            txt_file=file.read()
            df=fromTxt2Dataframe(txt_file,job)

            # we map the headers of the document with the standardize header before to aggregate
            if step=='aggregate':
                #print(df.columns)
                df.columns=[ x if ((x=="JOB_NAME_TXT") | (x=="JOB_TITLE_TXT")) else dictOfStandard[x]
                        #" ".join(x.translate(str.maketrans(string.punctuation,' '*len(string.punctuation))).split())
                        for x in list(df.columns)]
                df1_metaData = df1_metaData.append(df, sort=False)

            list_metaData.append([x  for x in list(df.columns) if((x!="JOB_NAME_TXT") & (x!="JOB_TITLE_TXT"))])

        except:print(job)

        file.close()
        
    if step=='aggregate':
        return list_metaData,df1_metaData
    else:
        return list_metaData


# In[ ]:


listOfHeaders=standardize2Aggregate(file_txt)


# We will ignore the bulletin related to the Police Job Commander position that failed to be read
# 
# We collected the headers of the bulletins that are going to be standardized by lemmatization
# 

# In[ ]:


def lemmatizeSentence(sentence):
    sent = sentence
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return lemmatized_list

def reduceSentence(sentence):
    sentence=TextBlob(sentence.translate(str.maketrans(string.punctuation,' '*len(string.punctuation))).lower())
    sentence=sentence.correct()
    sentence=lemmatizeSentence(sentence)
    sentenceback=[re.sub("MORTIFICATION","CERTIFICATION",x.upper()) for x in list(sentence) if x not in stopWords]
    return sentenceback

def add_or_append(dictionary, key, value):
    dictionary[key]=value


# In[ ]:


mappingHeaders=dict()
for value in set(x for l in listOfHeaders for x in l):
    add_or_append(mappingHeaders,value, "_".join(reduceSentence(value)))


# In[ ]:


listOfHeaders, dfStandarized =standardize2Aggregate(file_txt,mappingHeaders,'aggregate')


# We would like to check the matching between the name of the file and the title position they content.
# So we use the similarity string between JOB_TITLE_TXT and JOB_NALE_TXT

# In[ ]:


def ratio_sim(row) :
    return SequenceMatcher(None,row['JOB_NAME_TXT'],row['JOB_TITLE_TXT']).ratio()
df=dfStandarized[['JOB_NAME_TXT','JOB_TITLE_TXT']]
df['similarity']=df.apply(ratio_sim,axis=1)
df.sort_values('similarity').head(5)


# We note that 3 job bulletins title does not match the file name :
# * ANIMAL CARE TECHNICIAN SUPERVISOR 4313 122118.txt
# * SENIOR EXAMINER OF QUESTIONED DOCUMENTS 3231 072216 REVISED 072716.txt
# * WASTEWATER COLLECTION SUPERVISOR 4113 121616.txt
# 
# The bulletin VOCATIONAL WORKER DEP OF PUBLIC WORKS does not have a common framework shared with the other files. We remove it from our analysis.

# In[ ]:


display(dfStandarized.loc[dfStandarized.JOB_NAME_TXT!='VOCATIONAL WORKER DEPARTMENT OF PUBLIC WORKS'].head(5))
dfStandarized=dfStandarized.loc[dfStandarized.JOB_NAME_TXT!='VOCATIONAL WORKER DEPARTMENT OF PUBLIC WORKS']


# The last action to perform is to make "hand" correction by analysing the similarity between some headers to merge columns to enhance the completion of information.

# In[ ]:


a=dfStandarized.count().sort_values(ascending=False).reset_index()
a.columns=['HEADERS','Count']
display(a.head(10))
list_ratio=[]
a=a.sort_values('Count',ascending=False)
list_name=a['HEADERS']
list_valeur=a['Count']
for x in range(len(list_name)):
    for y in range(x,len(list_name)):
        if not (bool(re.search("_NOTE$",list_name[x])) ^ bool(re.search("_NOTE$",list_name[y]))):
            correl=dfStandarized.loc[(~dfStandarized[list_name[x]].isnull()) & (~dfStandarized[list_name[y]].isnull()),[list_name[x],list_name[y]]].shape[0]
            list_ratio.append([list_name[x],list_name[y],round(SequenceMatcher(None,list_name[x], list_name[y]).ratio(),4)*100,
                                  list_valeur[x],list_valeur[y],correl])
df_ratio=pd.DataFrame(list_ratio, columns=['Col1','Col2','Similarity','Num_Col1','Num_Col2','Overlap'])
df_ratio.loc[(df_ratio['Col1']!=df_ratio['Col2'])
             &(df_ratio['Similarity']>0)
             &(df_ratio['Overlap']==0)
            ].sort_values(['Similarity'], ascending=False).head(5)


# In[ ]:


listToCorrect=["ANNUALSALARY_NOTE","ANNUALSALARY",
               "EXAMINATION_GIVE_INTERDEPARTMENTAL_PROMOTION_BASIS_NAVY",
               "PASS_SCORE_QUALIFY_TEST_NOTE","REQUIREMENT_MIMINUMUM_QUALIFICATION",
               "PASS_SCORE_QUALIFY_TEST","EXAM_GIVE_INTERDEPARTMENTAL_PROMOTION_BASIS",
               "EXAMINATION_GIVE_OPEN_COMPETITIVE_INTERDEPARTMENTAL_PROMOTION_BASIS","SELECTION_PROCEDURE",
               "EQUAL_OPPORTUNITY_EMPLOYER","EQUAL_EMPLOYMENT_OPPORTUNITY_EMPLOYER_EQUAL_EMPLOYMENT_OPPORTUNITY_EMPLOYER",
               "REQUIREMENT","REQUIREMENT_MINIMUM_REQUIREMENT",
               "MINIMUM_REQUIREMENT",
               "REQUIREMENT_NOTE","APPLICATION_DEADLINE_NOTE_EXPERT_REVIEW_COMMITTEE"]
listByCorrect=["ANNUAL_SALARY_NOTE","ANNUAL_SALARY",
               "EXAMINATION_GIVE_INTERDEPARTMENTAL_PROMOTION_BASIS",
               "PASS_SCORE_QUALIFYING_TEST_NOTE","REQUIREMENT_MINIMUM_QUALIFICATION",
               "PASS_SCORE_QUALIFYING_TEST","EXAMINATION_GIVE_INTERDEPARTMENTAL_PROMOTION_BASIS",
               "EXAMINATION_GIVE_INTERDEPARTMENTAL_PROMOTION_OPEN_COMPETITIVE_BASIS","SELECTION_PROCESS",
               "EQUAL_EMPLOYMENT_OPPORTUNITY_EMPLOYER","EQUAL_EMPLOYMENT_OPPORTUNITY_EMPLOYER",
               "REQUIREMENT_MINIMUM_QUALIFICATION","REQUIREMENT_MINIMUM_QUALIFICATION",
               "REQUIREMENT_MINIMUM_QUALIFICATION",
               "REQUIREMENT_MINIMUM_QUALIFICATION_NOTE","EXPERT_REVIEW_COMMITTEE"]
for index, item in enumerate(listToCorrect):
    dfStandarized.loc[~dfStandarized[listToCorrect[index]].isnull(),listByCorrect[index]]=dfStandarized.loc[~dfStandarized[listToCorrect[index]].isnull(),listToCorrect[index]]
    dfStandarized.drop([listToCorrect[index]], axis=1,inplace=True)
dfStandarized.reset_index(drop=True,inplace=True)


# In[ ]:


a=dfStandarized.count().sort_values(ascending=False)
a.columns=['HEADERS','Count']
a.head(30)


# ## 3. Extraction and analyse of data
# 
# <a id="extract"></a>

# ### 3.1. Administration data
# <a id="admin"></a>
# * CLASS CODE
# * OPEN DATE
# * REVISED DATE
# * EXAM OPEN TO

# In[ ]:


dfStandarized['CLASS_CODE']=dfStandarized['ADMINISTRATION'].apply(lambda x: ' '.join(x)).str.lower().apply(lambda x : re.search("class code: (\S+)*",x).group(1) if re.search("class code: (\S+)*",x) else np.NaN )

dfStandarized.loc[dfStandarized.CLASS_CODE.isnull(),'CLASS_CODE']=dfStandarized.loc[dfStandarized.CLASS_CODE.isnull(),'JOB_TITLE_TXT'].str.lower().apply(lambda x : re.search("class code: (\S+)*",x).group(1) if re.search("class code: (\S+)*",x) else np.NaN )
dfStandarized['JOB_TITLE_TXT']=dfStandarized['JOB_TITLE_TXT'].apply(lambda x : re.sub(" CLASS CODE: (\S+)*","",x))

dfStandarized['OPEN_DATE']=dfStandarized['ADMINISTRATION'].apply(lambda x: ' '.join(x)).str.lower().apply(lambda x : re.search("open date: (\S+)*",x).group(1) if re.search("open date: (\S+)*",x) else None )

dfStandarized['REVISED_DATE']=dfStandarized['ADMINISTRATION'].apply(lambda x: ' '.join(x)).str.lower().apply(lambda x : re.search("revised: (\S+)*",x).group(1) if re.search("revised: (\S+)*",x) else None )

dfStandarized['EXAM_OPEN_TO']=dfStandarized['ADMINISTRATION'].apply(lambda x: ' '.join(x)).str.lower().apply(lambda x : re.search("\(exam (.+)*\)",x).group(1) if re.search("\(exam ([\w\s]+)*",x) else None )


# We can have a look now to the administration data structured in our dataframe

# In[ ]:


dfStandarized[['JOB_TITLE_TXT','ADMINISTRATION','CLASS_CODE','OPEN_DATE','REVISED_DATE','EXAM_OPEN_TO']].head(5)


# We are going to have to standardized the verbatim related to the EXAM_OPEN_TO in order to get only the two related designation :
# * *open to all, including current city employees* (as well as *open to all, including city employees* and *open to all including current city employees*
# * *open to current city employees* (as similar with *open to all current city employees*)

# In[ ]:


dfStandarized.EXAM_OPEN_TO.value_counts()


# ### 3.2. Salary data
# <a id="sala"></a>
# * ANNUAL SALARY
# * NOTE

# #### Quantitative data

# Let's have a quick look to how the ANNUAL_SALARY data is presented :

# In[ ]:


dfStandarized.head(10)['ANNUAL_SALARY'].values


# So we can notice than some salary are expressed in term on range **salary1 to salary2** and other in term of flat rate **salary (flat-rated)**<br>
# Thus we will focus first to retreive those two kind of information.

# In[ ]:


regex_range_salary="\$*\s*(\d+\,*\s*\d+)+\** to \$*\s*(\d+\,*\s*\d+)+"
regex_flat="(flat\srate)+"
regex_flat_salary="\$*\s*(\d+\,*\d+)+"


# In[ ]:


dfStandarized['ANNUAL_SALARY']=dfStandarized['ANNUAL_SALARY'].apply(lambda x: list(filter(lambda x: x!='',[' '.join(x[i:i+2]) if bool(re.search(" (to)$",x[i])) 
                                              else x[i]  if (i==0)|
                                              ((bool(re.search(" (to)$",x[i-1]))==False) & (i>0)) else '' for i in range(len(x)) ])))

dfStandarized['ANNUAL_SALARY_REDUCED_BIS']=dfStandarized['ANNUAL_SALARY'].apply(lambda x: [re.sub(regex_range_salary,"xx",y.lower()) if re.search(regex_range_salary,y.lower()) else y.lower() for y in x]).apply(lambda x: [re.sub(regex_flat_salary,"ff",y) if re.search(regex_flat,' '.join(reduceSentence(y)).lower()) else y for y in x]).apply(lambda x :[re.sub('(LOS ANGELES)|(SALARY)|(RANGE)|(DEPARTMENT)|(WORLD)|(POSITION)|\d', r'',' '.join(reduceSentence(y))) for y in x] ).apply(lambda x: [re.sub(r'\b(\w+)( \1\b)+', r'\1',' '.join(reduceSentence(y))) for y in x]).apply(lambda x: [re.sub(r'\b([\w\s]+)( \1\b)+', r'\1',' '.join(reduceSentence(y))) for y in x]).apply(lambda x: [re.sub(r'\b([\w\s]+)( \1\b)+', r'\1',' '.join(reduceSentence(y))) for y in x]).apply(lambda x :[re.sub('\s+', r' ',y) for y in x] ).apply(lambda x :[re.sub('^\s+|\s+$', r'',y) for y in x])


# Thus we get a list of the most common typology of salary as in the bulletin:
# * Some just contain salaries as range (XX), other as flat rated (FF FLAT RATE)
# * Other are related to specific Departements : Water and Power, Airport, Harbor

# In[ ]:


a=dfStandarized['ANNUAL_SALARY_REDUCED_BIS']
x=Counter([y for x in a for y in x])
sorted(x.items(), key=lambda pair: pair[1], reverse=True)[:10]


# Here below some example of verbatim reduction from ANNUAL_SALARY as in the bulletin to ANNUAL_SALARY_REDUCED_BIS which get the pattern of the salary

# In[ ]:


dfStandarized.head(10)[['ANNUAL_SALARY','ANNUAL_SALARY_REDUCED_BIS']].values


# In[ ]:


dict_salary=dict()
dict_salary['AIRPORT FLAT RATE FF']='AIRPORT FF FLAT RATE'
dict_salary['WATER POWER FLAT RATE FF']='WATER POWER FF FLAT RATE'
dict_salary['XX WATER POWER XX']='WATER POWER XX'
dict_salary['XX FLAT RATE']='FF FLAT RATE'
dfStandarized['ANNUAL_SALARY_REDUCED_BIS']=dfStandarized['ANNUAL_SALARY_REDUCED_BIS'].apply(lambda x: [dict_salary[y] if y in dict_salary.keys() else y for y in x]).apply(lambda x :[re.sub('(WATER POWER FF FLAT RATE XX)|(WATER POWER XX FF FLAT RATE)', r'WATER POWER XX WATER POWER FF FLAT RATE',y) for y in x] )


# In[ ]:


dict_get_salary={
    
    "WATER_POWER_SALARY_RANGE" : ["(WATER POWER XX)" , regex_range_salary],
    "WATER_POWER_SALARY_FLAT" : ["(WATER POWER FF FLAT RATE)", regex_flat_salary],
    
    "AIRPORT_SALARY_RANGE" : ["(AIRPORT XX)" , regex_range_salary],
    "AIRPORT_SALARY_FLAT" : ["(AIRPORT FF FLAT RATE)" , regex_flat_salary],
    
    "HARBOR_SALARY_RANGE" : ["(HARBOR XX)" , regex_range_salary],
    "HARBOR_SALARY_FLAT" : ["(HARBOR FF FLAT RATE)", regex_flat_salary],
    
    "GENERAL_SALARY_RANGE" : ["(XX)",regex_range_salary],
    "GENERAL_SALARY_FLAT" : ["(FF FLAT RATE)", regex_flat_salary]
}


# In[ ]:


def f(x,y):
    a=[re.findall(y[1],x['ANNUAL_SALARY_MODIFIED'][i].lower())
       for i,z in enumerate(x['ANNUAL_SALARY_REDUCED_BIS'])
       if (re.search(y[0],z)) and (re.search(y[1],x['ANNUAL_SALARY_MODIFIED'][i].lower()))]
    return [item for sublist in a for item in sublist] if a!=[] else None

def g(x,y):
    b=[re.sub(y[1],'Substitute',x['ANNUAL_SALARY_MODIFIED'][i].lower()) 
       if re.search(y[0],z) else x['ANNUAL_SALARY_MODIFIED'][i].lower()
       for i,z in enumerate(x['ANNUAL_SALARY_REDUCED_BIS'])]
    return b


# In[ ]:


dfStandarized['ANNUAL_SALARY_MODIFIED']=dfStandarized['ANNUAL_SALARY']

for new_column in dict_get_salary.keys():
    print(new_column)
    dfStandarized[new_column]=dfStandarized[['ANNUAL_SALARY_MODIFIED','ANNUAL_SALARY_REDUCED_BIS']]    .apply(lambda x : f(x,dict_get_salary[new_column]), axis=1)
    dfStandarized['ANNUAL_SALARY_MODIFIED']=dfStandarized[['ANNUAL_SALARY_MODIFIED','ANNUAL_SALARY_REDUCED_BIS']]    .apply(lambda x : g(x,dict_get_salary[new_column]), axis=1)


# We detect that for one offer, none salary is mentionned : 

# In[ ]:


dfStandarized[dfStandarized[['WATER_POWER_SALARY_RANGE','WATER_POWER_SALARY_FLAT',
                            'AIRPORT_SALARY_RANGE','AIRPORT_SALARY_FLAT',
                            'HARBOR_SALARY_RANGE','HARBOR_SALARY_FLAT',
                            'GENERAL_SALARY_RANGE','GENERAL_SALARY_FLAT']].isna().all(1)][['JOB_NAME_TXT','ANNUAL_SALARY_MODIFIED']]


# We can have a look now to the salary data structured in our dataframe

# In[ ]:


dfStandarized.head(5)[['ANNUAL_SALARY',
                       'GENERAL_SALARY_RANGE','WATER_POWER_SALARY_RANGE', 'AIRPORT_SALARY_RANGE', 'HARBOR_SALARY_RANGE',
                       'GENERAL_SALARY_FLAT','WATER_POWER_SALARY_FLAT', 'AIRPORT_SALARY_FLAT', 'HARBOR_SALARY_FLAT']]


# How about to start a small visualization that those data ? <br>
# We will try to first if we can make any conclusion about how are the range salary for specific departments compare with salary without specification fo departement

# In[ ]:


dfSalaryComparaison=dfStandarized[dfStandarized[['WATER_POWER_SALARY_RANGE',
                                                                     'AIRPORT_SALARY_RANGE',
                                                                     'HARBOR_SALARY_RANGE',
                                                                     'GENERAL_SALARY_RANGE']].isnull().sum(axis=1)<3]
dfSalary=pd.DataFrame()
for x in ['WATER_POWER_SALARY_RANGE', 'AIRPORT_SALARY_RANGE', 'HARBOR_SALARY_RANGE','GENERAL_SALARY_RANGE']:
    a=pd.DataFrame(dfSalaryComparaison.loc[~dfSalaryComparaison[x].isnull(),x].tolist())
    a['JOB_NAME']=dfSalaryComparaison.loc[~dfSalaryComparaison[x].isnull(),'JOB_NAME_TXT'].values
    a=a.set_index('JOB_NAME').stack().reset_index(name='new')[['JOB_NAME','new']]
    b=pd.DataFrame(a['new'].values.tolist(),columns=['LOWER_RANGE','UPPER_RANGE']).applymap(lambda x : re.sub('[^0-9]','',x))
    b['JOB_NAME']=  a['JOB_NAME']
    b['DEPARTEMENT_POSITION']=x
    dfSalary=dfSalary.append(b, ignore_index=True, sort=True)

dfSalary[['LOWER_RANGE','UPPER_RANGE']]=dfSalary[['LOWER_RANGE','UPPER_RANGE']].astype(int)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x="LOWER_RANGE", y="UPPER_RANGE",
                hue="DEPARTEMENT_POSITION",style="DEPARTEMENT_POSITION",
                data=dfSalary)


# Maybe some initial conclusions :
# * as general statement, the lowest salary for water and power departement are higher than the lowest salary for unspecified department
# * netherless, the range ratio is higher for unspecified departement than for specific departement

# Let's compare now the ranged salaries with the flated ones

# In[ ]:


dfSalaryComparaison=dfStandarized

dfSalary=pd.DataFrame()
for x in ['WATER_POWER_SALARY_RANGE', 'AIRPORT_SALARY_RANGE', 'HARBOR_SALARY_RANGE','GENERAL_SALARY_RANGE']:
    a=pd.DataFrame(dfSalaryComparaison.loc[~dfSalaryComparaison[x].isnull(),x].tolist())
    a['JOB_NAME']=dfSalaryComparaison.loc[~dfSalaryComparaison[x].isnull(),'JOB_NAME_TXT'].values
    a=a.set_index('JOB_NAME').stack().reset_index(name='new')[['JOB_NAME','new']]
    b=pd.DataFrame(a['new'].values.tolist(),columns=['LOWER_RANGE','UPPER_RANGE']).applymap(lambda x : re.sub('[^0-9]','',x))
    b['JOB_NAME']=  a['JOB_NAME']
    b['DEPARTEMENT_POSITION']='SALARY_RANGE'#x
    dfSalary=dfSalary.append(b, ignore_index=True, sort=True)

for x in ['WATER_POWER_SALARY_FLAT', 'AIRPORT_SALARY_FLAT', 'HARBOR_SALARY_FLAT','GENERAL_SALARY_FLAT']:
    a=pd.DataFrame(dfSalaryComparaison.loc[~dfSalaryComparaison[x].isnull(),x].tolist())
    a['JOB_NAME']=dfSalaryComparaison.loc[~dfSalaryComparaison[x].isnull(),'JOB_NAME_TXT'].values
    a=a.set_index('JOB_NAME').stack().reset_index(name='LOWER_RANGE')[['JOB_NAME','LOWER_RANGE']]
    b=pd.DataFrame(a['LOWER_RANGE']).applymap(lambda x : re.sub('[^0-9]','',x))
    b['JOB_NAME']=  a['JOB_NAME']
    b['DEPARTEMENT_POSITION']='SALARY_FLAT'#x
    b['UPPER_RANGE']=b['LOWER_RANGE']
    dfSalary=dfSalary.append(b, ignore_index=True, sort=True)

dfSalary[['LOWER_RANGE','UPPER_RANGE']]=dfSalary[['LOWER_RANGE','UPPER_RANGE']].astype(int)


# In[ ]:


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
sns.scatterplot(x="LOWER_RANGE", y="UPPER_RANGE",
                hue="DEPARTEMENT_POSITION",style="DEPARTEMENT_POSITION",
                data=dfSalary)


# Heree also maybe some initial conclusions :
# * as general statement, the flated salaries are lower than the ranged salaries

# #### Qualitative

# In addition to quantitative data related to salary, there are qualitative data contained for the mostly contained in the NOTES part attached to the ANNUAL SALARY part.

# Let's have a quick look to how the ANNUAL_SALARY_NOTE data is presented :

# In[ ]:


dfStandarized.head(5)[['JOB_NAME_TXT','ANNUAL_SALARY_NOTE']].values


# As previously done with the qualitative data, we will reduce the dimentionnality of the sentence in order to regularize the text and find the most common elements

# In[ ]:


regex_url='https?:\/\/.*[\r\n]*'

dfStandarized['ANNUAL_SALARY_NOTE_REDUCED_BIS']=dfStandarized['ANNUAL_SALARY_NOTE'].apply(lambda x : x if type(x) == list else []).apply(lambda x: [re.sub(regex_url,"URLPDF",y.lower()) if re.search(regex_url,y.lower()) else y.lower() for y in x]).apply(lambda x: [re.sub('^\d. ',"",y.lower()) if re.search('^\d. ',y.lower()) else y.lower() for y in x]).apply(lambda x :[re.sub('(LOS ANGELES)|(SALARY)|(RANGE)|(DEPARTMENT)|(WORLD)|(POSITION)|\d', r'',' '.join(reduceSentence(y))) for y in x] ).apply(lambda x: [re.sub(r'\b(\w+)( \1\b)+', r'\1',' '.join(reduceSentence(y))) for y in x]).apply(lambda x: [re.sub(r'\b([\w\s]+)( \1\b)+', r'\1',' '.join(reduceSentence(y))) for y in x]).apply(lambda x: [re.sub(r'\b([\w\s]+)( \1\b)+', r'\1',' '.join(reduceSentence(y))) for y in x]).apply(lambda x :[re.sub('\s+', r' ',y) for y in x] ).apply(lambda x :[re.sub('^\s+|\s+$', r'',y) for y in x])


# Thus we get a list of the most common typology of informations related to the note salary as in the bulletin:
# * Grade position,
# * Night and shift work bonus,
# * Part time position,
# * Start level salary,
# * Multiple range grade,
# * Salary to confirm,
# * Possible salary change,
# * Confirm hiring salary

# In[ ]:


a=dfStandarized['ANNUAL_SALARY_NOTE_REDUCED_BIS']
x=Counter([y for x in a for y in x])
sorted(x.items(), key=lambda pair: pair[1], reverse=True)[:10]


# In[ ]:


columns_reduced=['ANNUAL_SALARY_REDUCED_BIS','ANNUAL_SALARY_NOTE_REDUCED_BIS']

dfStandarized['LOWER_PAY_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'LOWER PAY GRADE POSITION' if sum([bool(re.search("LOW",y))for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0 
                                                                                else None, axis=1)
dfStandarized['HIGHER_PAY_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'NIGHT WORK' if sum([bool(re.search("NIGHT",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                 else('SHIFT WORK'if sum([bool(re.search("ASSIGN",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                      else None), axis=1)
dfStandarized['PART_TIME_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'PART TIME' if sum([bool(re.search("PART TIME",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                else None, axis=1)
dfStandarized['BEGIN_RANGE_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'BEGIN SALARY RANGE' if sum([bool(re.search("(START|BEGIN) PAY",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                  else None, axis=1)
dfStandarized['MULTIPLE_PAY_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'COVER MULTIPLE PAY GRADE' if sum([bool(re.search("MULTIPLE",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                   else None, axis=1)
dfStandarized['CONFIRM_PAY_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'CONFIRM SALARY BEFORE' if sum([bool(re.search("ACCEPT",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                  else None, axis=1)
dfStandarized['CHANGE_PAY_SALARY_CRITERIA']=dfStandarized[columns_reduced].apply(lambda x: 'CURRENT SALARY SUBJECT TO CHANGE' if sum([bool(re.search("CHANGE",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                 else None, axis=1)
dfStandarized['RECIPROCITY_SALARY_INFORMATION']=dfStandarized[columns_reduced].apply(lambda x: 'CITY LA AND LADWP' if sum([bool(re.search("URLPDF",y)) for y in x[columns_reduced[1]]+x[columns_reduced[0]]])>0
                                                                                 else None, axis=1)


# In[ ]:


dfStandarized.head(5)[['JOB_NAME_TXT',
                       'LOWER_PAY_SALARY_CRITERIA','HIGHER_PAY_SALARY_CRITERIA', 'PART_TIME_SALARY_CRITERIA', 'BEGIN_RANGE_SALARY_CRITERIA',
                       'MULTIPLE_PAY_SALARY_CRITERIA','CONFIRM_PAY_SALARY_CRITERIA', 'CHANGE_PAY_SALARY_CRITERIA', 'RECIPROCITY_SALARY_INFORMATION']]


# For some positions no additional information is provided about the salay :

# In[ ]:


dfStandarized[dfStandarized[['LOWER_PAY_SALARY_CRITERIA','HIGHER_PAY_SALARY_CRITERIA', 'PART_TIME_SALARY_CRITERIA', 'BEGIN_RANGE_SALARY_CRITERIA',
                       'MULTIPLE_PAY_SALARY_CRITERIA','CONFIRM_PAY_SALARY_CRITERIA', 'CHANGE_PAY_SALARY_CRITERIA', 'RECIPROCITY_SALARY_INFORMATION']].isna().all(1)][['JOB_NAME_TXT','ANNUAL_SALARY','ANNUAL_SALARY_NOTE']]


# ### 3.3. Requirement minimum qualification data
# <a id="minireq"></a>
# * REQUIREMENT MINIMUM QUALIFICATION
# * NOTE

# We can notice here that the text of this part is much more unstructured and diversed that the annual salsry data:

# In[ ]:


dfStandarized.head(5)[['JOB_NAME_TXT','REQUIREMENT_MINIMUM_QUALIFICATION']].values


# So our strategy here wil be a little bit more advanced through the following steps:
# 1. Apply first a stemmatization and correction of the texts
# 2. Tokenize then the texts
# 3. Vectorize the corpus using TFIDF method
# 4. Clusterize using hierarchical aggregation algorithm
# 5. Visualize the main words behind those clusters

# **Stemmatization and correction**

# In[ ]:


dfStandarized['REQUIREMENT_MINIMUM_QUALIFICATION_REDUCED_BIS']=dfStandarized['REQUIREMENT_MINIMUM_QUALIFICATION'].apply(lambda x: [re.sub('^\d\. ',"",y) if re.search('^\d\. ',y) else y for y in x]).apply(lambda x: [re.sub('^[a-z]{1}\. ',"",y) if re.search('^[a-z]{1}\. ',y) else y for y in x]).apply(lambda x :' '.join(x)).apply(lambda x :' '.join(reduceSentence(x)).lower())


# In[ ]:


dfStandarized.loc[np.random.randint(1, 682, size=(1, 3))[0],
                  ['REQUIREMENT_MINIMUM_QUALIFICATION_REDUCED_BIS']].values


# **Tokenization**

# In[ ]:


def tokenize_only(text):
    tokens=[word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens=[]
    
    for token in tokens:
        if re.search('[a-zA-Z]',token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[ ]:


totalvocab_tokenized=[]

for i in dfStandarized['REQUIREMENT_MINIMUM_QUALIFICATION_REDUCED_BIS'].values:
    allwords_tokenized=tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)    
    


# In[ ]:


vocab_frame=pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_tokenized)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')


# **Vectorization TFIDF**

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=0.2,max_df=0.8,
                                   use_idf=True, tokenizer= tokenize_only, ngram_range=(1,1))

tfidf_matrix = tfidf_vectorizer.fit_transform(dfStandarized['REQUIREMENT_MINIMUM_QUALIFICATION_REDUCED_BIS'].values)

terms=tfidf_vectorizer.get_feature_names()

print(tfidf_matrix.shape)


# **Clustering hierarchical**

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
dist = 1 - cosine_similarity(tfidf_matrix)
linkage_matrix = ward(dist)


# In[ ]:


fig,ax = plt.subplots(figsize=(15, 20))
ax= dendrogram(linkage_matrix, orientation="right",labels=dfStandarized['JOB_NAME_TXT'].values )

plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.tight_layout()


# By a visual analysis we can quickly decide to cluster the corpus in 18 groups (this is a first approach)

# In[ ]:


k=18


# In[ ]:


fig,ax = plt.subplots(figsize=(15, 20))
ax= dendrogram(
    linkage_matrix,orientation="right",
    truncate_mode='lastp',  # show only the last p merged clusters
    p=k,  # show only the last p merged clusters
    show_contracted=True  # to get a distribution impression in truncated branches
)
plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

plt.tight_layout()


# In[ ]:


from scipy.cluster.hierarchy import fcluster
dfStandarized['REQUIREMENT_MINIMUM_QUALIFICATION_CLUSTER_HIER']=fcluster(linkage_matrix, k, criterion='maxclust')


# **Visualization**

# In[ ]:


from sklearn.cluster import KMeans
from wordcloud import WordCloud

tfidf_vectorizer_bis = TfidfVectorizer(min_df=0.2,max_df=0.8,
                                       use_idf=True, ngram_range=(1,1))

funct = lambda x: ' '.join([vocab_frame.loc[y].values.tolist()[0][0] for y in x.split(' ')])

dictResultCluster=dict()

for i in range(k):
    df=dfStandarized.loc[dfStandarized.REQUIREMENT_MINIMUM_QUALIFICATION_CLUSTER_HIER==i+1,
                          'REQUIREMENT_MINIMUM_QUALIFICATION_REDUCED_BIS']
    tfidf_matrix_bis = tfidf_vectorizer_bis.fit_transform(df.values)
    terms_bis=tfidf_vectorizer_bis.get_feature_names()

    km=KMeans(n_clusters=1)
    km.fit(tfidf_matrix_bis)
    order_centroids = km.cluster_centers_.argsort()[:,::-1]
    
    print('Cluster '+str(i+1) + ' : size '+ str(df.shape[0]))
    list_cluster=[]
    for ind in order_centroids[0,:]:
        try : 
            list_cluster.append(funct(terms_bis[ind])) 
        except : 
            pass
    
    comment_words=' '.join([ x 
                            for y in df
                            for x in tokenize_only(y) if x in list_cluster
                            ])
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',
                min_font_size = 10).generate(comment_words) 
    
    plt.figure(figsize = (4, 4), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    plt.show()

    dictResultCluster[i+1]   =[df.shape[0],' ,'.join(list_cluster)] 

    #df=dfStandarized.loc[dfStandarized.REQUIREMENT_MINIMUM_QUALIFICATION_CLUSTER_HIER==i+1,
    #                        ['REQUIREMENT_MINIMUM_QUALIFICATION']]
    #print(df.values[np.random.randint(1, df.shape[0], size=(1, 3))[0]])


# In[ ]:


pd.DataFrame(dictResultCluster, index=['Cluster size','Words embeded'])


# # Coming Soon --  Under construction
# ![](https://scholarblogs.emory.edu/ranews/files/2017/02/under-construction-image-850x478.jpeg)

# In[ ]:




