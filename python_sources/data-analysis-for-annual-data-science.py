#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
dt= pd.read_csv('../input/freeFormResponses.csv')[0:10].fillna(0)
display(dt)


# In[ ]:





# # Analysis 1:
# #This  first analysis is to know how which tools are common and frequently used by professionals and students for data analysis

# In[ ]:


import nltk
import itertools as it
from operator import itemgetter
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words= (stopwords.words('english'))
l=['','-','None','I','Use','No','Dont','"Dont','"I','Software','Tools','Nothing','+','Student','Custom','Not']
stop_words.extend(l)
#print(stop_words)
#print(len(stop_words))
dt= pd.read_csv('../input/freeFormResponses.csv')[1:].fillna(0)
X= dt.iloc[:,1:2].values
ListData=[]
for each in X:
   # print(each)
    if str(each).find(',') >0 :
        repStr=str(each).replace('[','').replace(']','').replace("'",'').split(",")
    elif str(each).find('/') > 0 :
        repStr=str(each).replace('[','').replace(']','').replace("'",'').split("/")
        
    else:
        repStr= str(each).replace('[','').replace(']','').replace("'",'').split(' ')
    #print(type(str(each)),repStr[0])
    for each in repStr:
        #print(each)
        if each != '0':
            #print(each)
            if each not in stop_words: 
                ListData.append(each.lower().title().strip())
#print(ListData)
DataStorage=dict()
DataList=[]
for each in ListData:
    if each not in DataStorage.keys():
        DataStorage[each]=(ListData.count(each))
DataList = sorted(DataStorage.items(),key=itemgetter(1),reverse=True)
#print(DataList)
cleanedDataList=[]
indexList=[]
for each in DataList:
    if each[0] not in stop_words:
        cleanedDataList.append(each[1])
        indexList.append(each[0])
#print(cleanedDataList)  

cleanedDataFrame=pd.DataFrame(cleanedDataList[0:10],index= indexList[0:10],columns=['Data_Analysis_tool_used'])
#print(indexList)
display(cleanedDataFrame)
plt.figure(figsize=(16,8))
sns.barplot(x=indexList[0:10],y=cleanedDataList[0:10],palette='rocket')
plt.ylabel('Total_no_Students_used')
plt.title('Top 10 Students usage of Analytics Tool Usage ')
plt.show()
plt.figure(figsize=(16,8))
plt.pie(cleanedDataList[0:10],labels=indexList[0:10],explode=(0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05),shadow=True,autopct='%.2f')
plt.show()


# #This proves that students favour Python as the most valuable and wide-ranged data analysis toll to work upon.
# Over the years , Python has been the crux platform for solving various real-life problems involving a wide range of datas. 
# Henceforth, usage of Python should be incorporated in daily life. Although Matlab is though second on the list, so is SQL and R.
# These technologies are emerging and may soon provide alternate routes of dealing with categorical and numerical variables

# # Analysis 2:
# This is to find out what age group finds which programming language most suitable and worked on it

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
dt= pd.read_csv('../input/multipleChoiceResponses.csv').fillna(0)
#display(dt)
#print(dt.columns[3])
finalList1=[]
finalList2=[]
Age=pd.Series(dt.iloc[1:,3].values)
#Age=dt['What is your age (# years)?']
ProgramTaken = pd.Series(dt.iloc[1:,84].values)
#ProgramTaken = dt['What specific programming language do you use most often? - Selected Choice']
AgeList=[]
ProgramList=[]
for each in Age:
    AgeList.append(each)
SetAgeList= set(AgeList)
print(SetAgeList)
for each in ProgramTaken:
    ProgramList.append(each)
setProgramList = set(ProgramList)
print(set(ProgramList))
#print(ProgramList,AgeList)
AgeWithProgramZipped = list(zip(AgeList,ProgramList))
#print(AgeWithProgramZipped)
for each in SetAgeList:
    strNew=''
    for elem in AgeWithProgramZipped:
        if each == elem[0]:
            #print(elem[0],elem[1])
            if strNew =='':
                strNew= str(elem)
            else:
                strNew = strNew + '::' + str(elem)
    finalList1.append(strNew)
#print(finalList1) 
dictProgram=dict()
indexFinal=[]
k=0
from operator import itemgetter
paletteColors= ['Accent','Blues','BuGn', 'Dark2', 'gist_heat','inferno','mako','nipy_spectral','ocean','pink','vlag', 'winter']
matColors=['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','aliceblue','chartreuse','chocolate','darkgrey','violet','darkorange','crimson','darkgoldenrod']
for each in finalList1:
    listk=[]
    #print(each)
    listTemp= each.split('::')
    for each in listTemp:
        listk.append(each.split(",")[1].replace("'",'').replace(')','').replace("'",'').strip())
    #print("listk:",listk) 
    temp=0
    for each in setProgramList:
        dictProgram[each]=listk.count(each)
    #print(sorted(dictProgram.items(),key=itemgetter(1),reverse=True)[0:12])
    kList = sorted(dictProgram.items(),key=itemgetter(1),reverse=True)[0:12]
    index=[]
    nameOfLang=[]
    for each in kList:
        #print(each)
        nameOfLang.append(each[0])
        index.append(each[1])
    plt.figure(figsize=(18,6))
    sns.barplot(x=nameOfLang,y= index,palette=paletteColors[k])
    plt.xlabel('For ' + list(SetAgeList)[k] +'years')
    plt.ylabel('Frequency of Language Used')
    plt.title('Language Frequency in respective ages')
    k+=1
    plt.show()
    plt.figure(figsize=(18,6))
    plt.pie(index,labels=nameOfLang,explode=(0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05),shadow=True,autopct='%.2f')
    plt.show()
    indexFinal.append(index)
dataFrameNet = pd.DataFrame(indexFinal,columns = nameOfLang,index=list(SetAgeList))         
display(dataFrameNet)     


# # This Analysis basically gathers its data from two Columns:Age and 'What specific programming language do you use most often? - Selected Choice'
# What i have done over here is analysed which age group is inclined to which programming language and which programing language is gaining popularity in that age bar .
# This will eventually help us gather information regarding the popularity of certain language and lacking ability of the other language.

# # Analysis 3:
# Now i am going to analyse the freuency and usage of programming language among different professions.
# It will be interesting as it will unfold what occupation majorly deals with what kind of language. Thus giving information of companies best practices

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
dt= pd.read_csv('../input/multipleChoiceResponses.csv').fillna(0)
#display(dt)
#print(dt.columns[3])
finalList1=[]
finalList2=[]
Profession = pd.Series(dt.iloc[1:,7].values)
#Profession=dt['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']
ProgramTaken =pd.Series(dt.iloc[1:,84].values)
#ProgramTaken = dt['What specific programming language do you use most often? - Selected Choice']
ProfessionList=[]
ProgramList=[]
for each in Profession:
    ProfessionList.append(str(each))
SetProfessionList= set(ProfessionList)
print(SetProfessionList)
for each in ProgramTaken:
    ProgramList.append(str(each))
setProgramList = set(ProgramList)
print(set(ProgramList))
#print(ProgramList,AgeList)
ProfessionWithProgramZipped = list(zip(ProfessionList,ProgramList))
#print(AgeWithProgramZipped)
for each in SetProfessionList:
    strNew=''
    for elem in ProfessionWithProgramZipped:
        if each == elem[0]:
            #print(elem[0],elem[1])
            if strNew =='':
                strNew= str(elem)
            else:
                strNew = strNew + '::' + str(elem)
    finalList1.append(strNew)
#print(finalList1) 
dictProgram=dict()
indexFinal=[]
k=0
from operator import itemgetter
paletteColors= ['Accent','Blues','BuGn', 'Dark2', 'gist_heat','inferno','mako','nipy_spectral','ocean','pink','vlag', 'winter']
matColors=['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','aliceblue','chartreuse','chocolate','darkgrey','violet','darkorange','crimson','darkgoldenrod']
for each in finalList1:
    listk=[]
    #print(each)
    listTemp= each.split('::')
    for each in listTemp:
        listk.append(each.split(",")[1].replace("'",'').replace(')','').replace("'",'').strip())
    #print("listk:",listk) 
    temp=0
    for each in setProgramList:
        dictProgram[each]=listk.count(each)
    #print(sorted(dictProgram.items(),key=itemgetter(1),reverse=True)[0:12])
    kList = sorted(dictProgram.items(),key=itemgetter(1),reverse=True)[0:12]
    index=[]
    nameOfLang=[]
    for each in kList:
        #print(each)
        nameOfLang.append(each[0])
        index.append(each[1])
    plt.figure(figsize=(18,6))
    sns.barplot(x=nameOfLang,y= index,palette='spring')
    plt.xlabel('For ' + list(SetProfessionList)[k])
    plt.ylabel('Frequency of Language Used')
    plt.title('Language Frequency in respective professions')
    k+=1
    plt.show()
    plt.figure(figsize=(18,6))
    plt.pie(index,labels=nameOfLang,explode=(0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05),shadow=True,autopct='%.2f')
    plt.show()
    indexFinal.append(index)
dataFrameNet = pd.DataFrame(indexFinal,columns = nameOfLang,index=list(SetProfessionList))         
display(dataFrameNet)     


# #Here 0 means those people who have denied to write their professions.
# This analysis is to give an idea which profession uses which language the most and which language is in the uprising.
# Clearly Python is the standout performer and i think for data analysis, it is moving in a rapid pace

# In[ ]:




