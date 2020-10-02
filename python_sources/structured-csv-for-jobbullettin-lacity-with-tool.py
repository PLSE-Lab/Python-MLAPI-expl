#!/usr/bin/env python
# coding: utf-8

# # Data Science for Good: City of Los Angeles

# In[ ]:


get_ipython().system('pip install word2number')


# In[ ]:


import pandas as pd 
import numpy as np
import re
import os
from word2number import w2n


# This function remove all duplicates from list.

# In[ ]:


def my_function(x):
    return (list(dict.fromkeys(x)))


# Function usefull to find word between two other words

# In[ ]:


def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


# I'll set the directory

# In[ ]:


path ="../Job Bulletins/"
cwd= os.chdir(path)


# Set the output dataframe

# In[ ]:


structured_csv= pd.DataFrame(index=np.arange(len([name for name in os.listdir('.') if os.path.isfile(name)])),
                             columns=['FILE_NAME','JOB_CLASS_TITLE','JOB_CLASS_NO','REQUIREMENT_SET_ID','REQUIREMENT_SUBSET_ID',
                                     'JOB_DUTIES','EDUCATION_YEARS','SCHOOL_TYPE','EDUCATION_MAJOR','EXPERIENCE_LENGHT',
                                     'FULL_TIME_PART_TIME','EXP_JOB_CLASS_TITLE','EXP_JOB_CLASS_FUNCTION','COURSE_COUNT',
                                     'COURSE_LENGHT','COURSE_SUBJECT','DRIVERS_LICENSE_REQ','DRIV_LIC_TYPE','EXAM_TYPE',
                                     'ENTRY_SALARY_GEN','ENTRY_SALARY_DWP','OPEN_DATE'])


# In[ ]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# The code below do all the Job of this request. For each file txt content in the directory Job Bullettins, this alghorithm will extract the following information:
# * 'FILE_NAME'
# * 'JOB_CLASS_TITLE'
# * 'JOB_CLASS_NO'
# * 'REQUIREMENT_SET_ID'
# * 'REQUIREMENT_SUBSET_ID'
# * 'JOB_DUTIES'
# * 'EDUCATION_YEARS' 
# * 'SCHOOL_TYPE
# * 'EDUCATION_MAJOR'
# * 'EXPERIENCE_LENGHT'
# * 'FULL_TIME_PART_TIME'
# * 'EXP_JOB_CLASS_TITLE'
# * 'EXP_JOB_CLASS_FUNCTION' 
# * 'COURSE_COUNT',
# * 'COURSE_LENGHT'
# * 'COURSE_SUBJECT'
# * 'DRIVERS_LICENSE_REQ'
# * 'DRIV_LIC_TYPE' 
# * 'EXAM_TYPE'
# * 'ENTRY_SALARY_GEN'
# * 'ENTRY_SALARY_DWP' 
# * 'OPEN_DATE'
# 
# Informations are extracted using text mining, that is very usefull for this task. One of the best library used for text mining is nltk, in the code you will find an explenation of what it do, to make it more understandable.

# In[ ]:



z= 0
for filename in os.listdir(cwd):    

    with open(filename, 'r') as f:
        try:
            lower=f.readlines()
        except UnicodeDecodeError:
            pass
    with open(filename, 'r') as f:   
        try:
            upper= [word for word in f.readlines() if word.isupper()]
        except UnicodeDecodeError:
            pass
        lower= [word.replace("\n","") for word in lower]
        lower= [x for x in lower if x]

        upper= [word.replace("\n","") for word in upper]
        upper= [word.replace("\t","") for word in upper]
        diz_job= dict((el,"") for el in upper)

        # FOR EACH FILE THE PART OF CODE ABOVE, READS LINE BY LINE THE DOCUMENT AND IT PUTS THE LOWER PART OF THE FILE IN A LIST AND THE UPPER WORDS 

        # ANOTHER ONE. THIS IS USEFULL TO CREATE A DICTIONARY TO ACCESS AT THE INFORMATION. AS WE CAN SE IN THE CODE ABOVE.
        
        for i in range (0,len(lower)):
            if lower[i] in diz_job.keys():
                name = lower[i]
                diz_job[lower[i]]=" "
            else:
                try:
                    diz_job[name]= diz_job[name] + " " + lower[i]
                except:
                    pass
        #ON THIS PART I PUT THE KEY "SALARY,REQUIREMENT,DUTIE" ON A VARIABLE. THIS IS USEFULL TO AUTOMATE EVERYTHING.
        fi_sal = str([x for x in upper if 'SALARY' in x])
        salary_k= fi_sal.replace("['","").replace("']","")

        re_sal=str([x for x in upper if 'REQUIREMENT' in x])
        req_k= re_sal.replace("['","").replace("']","")

        dut_sal=str([x for x in upper if 'DUTIE' in x])
        dut_k= dut_sal.replace("['","").replace("']","")
        
        # HERE I FIND THE NAME OF THE JOB, FIRST THING IN UPPER FOR EACH DOCUMENT
        JOB_CLAS_TITLE=upper[0]
        
        #TO FIND THE CLASS CODE, I FIND WHERE THE STRING 'Class Code' IS IN THE LOWER LIST AND I TAKE THE NUMERIC WORD.
        for i in range (0,4):
            if "Class Code" in str(lower[i]):
                JOB_CLASS_NO= "".join([word for word in lower[i] if word.isnumeric()])
        
        school=[]
        school="university","high school","college","apprenticeship"
        
        # AFTER DECLARED A LIST WITH ALL POSSIBLE SCHOOL TYPE, I'LL FIND IT IN THE DOCUMENT
        SCHOOL_TYPE=[]
        for word in school:
            if word in str(lower):
                SCHOOL_TYPE.append(word)
        #AS DONE FOR CLASSE CODE, OPEN DATE IS EXTRACTED IN THE SAME WAY AND USING THE PD.TO_DATETIME TO TRANSFORM IT IN A DATE 
        for i in range (0,4):
            if "Open Date" in str(lower[i]):
                data= "".join([word for word in lower[i] if word.isnumeric()])
        try:
            data=pd.to_datetime(data,format="%m%d%y")
            OPEN_DATE=str(data.month) + "-" + str(data.day) + "-" + str(data.year)
        except:
            OPEN_DATE=data
        
        #NOW THE DICTIONARY CREATED BEFORE IS USEFULL TO RETRIEVE THE (FIRST) SALARY, FOR EACH JOB
        salary=[]
        try:
            if "and" in str(diz_job[salary_k]):
                salary= diz_job[salary_k].split(" and")
                ENTRY_SALARY_GEN=salary[0]
            else:
                ENTRY_SALARY_GEN = diz_job[salary_k]
        except:
            ENTRY_SALARY_GEN=""
        try:
            JOB_DUTIES=diz_job[dut_k]
        except:
            JOB_DUTIES=""
        
        # FIND IF LICENSE IS REQUIRED, MAY REQUIRE OR NO REQUIRED 
        if "driver's license is required" in str(lower):
                DRIVERS_LICENSE_REQ="R"
        elif "may require a valid California driver's license" in str(lower):
            DRIVERS_LICENSE_REQ="P"
        else:
            DRIVERS_LICENSE_REQ="NoR"
        
        # IF EXIST EXTRACT THE TYPE OF DRIVER'S LICENSE NEEDED 
        DRIV_LIC_TYPE=" "
        if "Class C driver's" in str(lower):
            DRIV_LIC_TYPE= "C"
            DRIVERS_LICENSE_REQ="R"
        elif "Class B" in str(lower):
            DRIV_LIC_TYPE="B"
            DRIVERS_LICENSE_REQ="R"
        elif "Class A" in str(lower):
            DRIV_LIC_TYPE="A"
            DRIVERS_LICENSE_REQ="R"
        elif "Class M1" in str(lower):
            DRIV_LIC_TYPE="M1"
            DRIVERS_LICENSE_REQ="R"
        elif "Class M2" in str(lower):
            DRIV_LIC_TYPE="M2"
            DRIVERS_LICENSE_REQ="R"
        else:
            DRIV_LIC_TYPE=" "
        
        #TO HAVE NE NUMBER OF REQUESTS AND SUB_REQUEST, I WILL FIND IT IN LOWER AND IF LOWER[i] IS LIKE THE DICTIONARY.KEY OF REQUIREMENT
        # I WILL TAKE EVERITHING BEFORE AND SPLIT IT TO SEE IF IT IS A NUMBER(REQUIREMENT) ORA A LETTER (SUBREQUIREMENT)
        list2=[]
        list3=[]
        for i in range(0,len(lower)):
            if lower[i]== str(req_k):
                for j in range(1,(len(lower)-i)):
                    if lower[i+j].isupper():
                        break
                    else:
                        list2.append([word for word in lower[i+j][0:2] if word.isnumeric()])
                        list3.append([word for word in lower[i+j][0:1] if word.isalpha() & word.islower()])

        D1= len([x for x in list2 if x])

        str1 = ''.join(str(e) for e in list3 if e)
        D2= str1.replace("['","").replace("']","").upper()
        
        #IN THE CODE BELOW I WILL SAVE THE REQUIREMENT TEXT IN A STRING, IF A NEED SOMETHING ELSE 
        lista_req=[]
        k=0

        for i in range(0,len(lower)): 
            if lower[i]==str(req_k):
                for j in range(1,(len(lower)-i)):
                    if lower[i+j].isupper():
                        break
                    else:
                        lista_req.append(lower[i+j])
                        k=k+1
        #TO HAVE THE EDUCATION YEAR, I WILL USE THE FOLLOWING LIST OF NUMBER, AND USING THE ATTRIVUTE CREATED BEFORE IF A JOB HAVE A SCHOOL_TYPE
        #I WILL FIND IT IN THE LIST OF REQUIREMENTS AND I WILL TAKE THE NUMBER, THEN I TRANSFORM IT IN A NUMBER
        nums_0_19 = ['One','Two','Three','Four','Five','Six','Seven','Eight',"Nine", 'Ten',
                    'one','two','three','four','five','six','seven','eight',"nine", 'ten']
        take_num=[]
        for el in SCHOOL_TYPE:
            for i in range(0,len(lista_req)):
                if el in str(lista_req[i]):
                    take_num= lista_req[i].split(el,1)

        year_ed=[]

        if len(take_num) > 0:
            year_ed.append([w2n.word_to_num(word) for word in nums_0_19 if word in take_num[0]])
        else:
            year_ed=[]
        # FOR COUNT OF SEMESTER AND QUARTER, WORD TOKENIZE IS USED, SO AFTER SPLITTED THE STRING I TAKE THE FIRST PART OF THE SPLIT
        # AND IF IS NUMERIC I WILL TAKE IT OTHERWISE I TRANSFROM IT USING THE FUNCTION w2n.word_to_num(). SAME FOR QUARTER
        for i in range(0,len(lista_req)):
            if 'semester' in lista_req[i]:
                n_sem= lista_req[i].split('semester',1)
                num_s=[word.lower() for word in word_tokenize(n_sem[0],'english',False)][-1]
                if num_s.isnumeric():
                    sem= "#S"+" " + num_s
                else:
                    try:
                        sem= "#S"+" " + str(w2n.word_to_num(num_s))
                    except:
                        sem= ""
            else:
                n_sem=" "
                sem= " "
            if 'quarter' in lista_req[i]:
                n_quar= lista_req[i].split('quarter',1)
                num_q= [word.lower() for word in word_tokenize(n_quar[0],'english',False)][-1]
                if num_q.isnumeric():
                    quar= "#Q" + " " + num_q
                else:
                    try:
                        quar= "#Q" + " " + str(w2n.word_to_num(num_q))
                    except:
                        quar=""
            else:
                n_quar=" "
                quar=" "

        word_exp=[]
        year_exp=[]
        ftpt=[]
        job_exp=[]
        job_fun1=[]
        period=['full-time','part-time',"apprenticeship"]
        # FOR EXP_JOB_TITLE, FULL_TIME_PART_TIME AND EXPERIENCE LENGHT. THE SAME LOGIC IS USED TO EXTRACT THE INFORMATION 
        for i in range(0,len(lista_req)):
            exp=lista_req[i].split('experience',1)
            if 'experience' in lista_req[i]:

                if "City of Los Angeles" in exp[1]:
                    year_exp.append([word for word in nums_0_19 if word in exp[0]])
                    ftpt.append([word for word in period if word in exp[0]])        
                    try:
                        bck= exp[1].split('as',1)[1]
                    except:
                    #bck= exp[1].split('in',1)[1]
                        pass
                    job_exp.append(bck.split('with',1)[0])
                else:
                    try:
                        if " as " in exp[1]:
                            year_exp.append([word for word in nums_0_19 if word in exp[0]])
                            ftpt.append([word for word in period if word in exp[0]])
                            job_exp.append(" ")
                            job_fun= exp[1].split('as a',1)[1]
                            fun_j= job_fun.split(',')
                            for j in range(0,len(fun_j)):
                                job_fun1.append(fun_j[j])
                        else:
                            year_exp.append([word for word in nums_0_19 if word in exp[0]])
                            ftpt.append([word for word in period if word in exp[0]])
                            job_exp.append(" ")
                            job_fun= exp[1].split(' in ',1)[1]
                            fun_j= job_fun.split(',')

                            for j in range(0,len(fun_j)):
                                job_fun1.append(fun_j[j])
                    except:
                        pass
        try:
            EXPERIENCE_LENGTH=year_exp[0]
            FULL_TIME_PART_TIME=ftpt[0]
            EXP_JOB_CLASS_TITLE=job_exp[0]
        except:
            EXPERIENCE_LENGTH=""
            FULL_TIME_PART_TIME=""
            EXP_JOB_CLASS_TITLE=""
        #FOR JOB CLASS FUNCTION THE ONLY WAY TO EXTRACT THE INFORMATION IS THE FOLLOWING ONE, I'LL TAKE IT IF WE DON'T HAVE THE 
        #"City of Los Angeles" FOR THE WORK AND I'LL SPLIT IT AND I TAKE EACH ITEMS IN JOB_FUN1.
        
        if "job_fun1" in locals():
            job_fun1 = my_function(job_fun1)

            EXP_JOB_CLASS_FUNCTION=""

            for item in job_fun1:
                EXP_JOB_CLASS_FUNCTION=EXP_JOB_CLASS_FUNCTION +"|"+ item
        else:
            job_fun1=""
        
        #TO UNDERSTAND THE EDUCATION MAJOR I WILL TAKE EVERYTHING AFTER THE STRING "major in","college in","university in" AND AFTER I WILL
        #CLEAN IT IN THE DATA CLEANING CHUNKS
        EDUCATION_MAJOR=" "
        for i in range(0,len(lista_req)):
            if "apprenticeship" in SCHOOL_TYPE:
                EDUCATION_MAJOR= EXP_JOB_CLASS_FUNCTION
            elif "major in" in lista_req[i]:
                #result= re.search('major in(.*);', lista_req[i]
                EDUCATION_MAJOR = find_between(lista_req[i],"major in",";")
            elif "college in" in lista_req[i]:
                #result= re.search('college in(.*);', lista_req[i]
                EDUCATION_MAJOR = find_between(lista_req[i],"college in",";")
            elif "university in" in lista_req[i]:
                #result= re.search('university in(.*);', lista_req[i]
                EDUCATION_MAJOR = find_between(lista_req[i],"university in",";")
        
        #EXAM TYPE IS OFTEN IN THE END OF THE FILE AND IS ALWAYS UPPER, I TAKE THE LAST 5 STRING IN UPPER AND I COMPARE IT WITH SOME STRING
        #TO FIND THE FINAL EXAM TYPE
        get_exam=[]
        get_exam.append([word for word in word_tokenize(str(upper[-5:]),'english',False)])
        
        for i in range(0,len(get_exam)):
            if "OPEN" and ("INTERDEPARTMENTAL" in get_exam[i] or "'INTERDEPARTMENTAL" in get_exam[i]):
                EXAM_TYPE = "OPEN_INT_PROM"
            elif "OPEN" in get_exam[i] and ("INTERDEPARTMENTAL" not in get_exam[i] or "'INTERDEPARTMENTAL" not in get_exam[i]) :
                EXAM_TYPE = "OPEN"
            elif ("INTERDEPARTMENTAL" in get_exam[i] or "'INTERDEPARTMENTAL" in get_exam[i]) and "OPEN" not in get_exam[i]:
                EXAM_TYPE = "INT_DEPT_PROM"
            else:
                EXAM_TYPE = "DEPT_PROM"

        #FOR SALARY_DWP THE SAME LOGIC OF SALARY IS USED IF "Department of Water and Power" IS IN THE ELEMENT "ANNUAL SALARY" OF THE DICTIONARY
        try:    
            if "Department of Water and Power" in diz_job[salary_k]:
                try:
                    if len(diz_job[salary_k].split("is",1)[1]) > 21:
                        try:
                            ENTRY_SALARY_DWP=diz_job[salary_k].split("is",1)[1][1:20]
                        except:
                            ENTRY_SALARY_DWP=diz_job[salary_k].split("is",1)[1][1:19]
                    else:
                        ENTRY_SALARY_DWP=diz_job[salary_k].split("is",1)[1]
                except:
                    ENTRY_SALARY_DWP = diz_job[salary_k]
            else:
                ENTRY_SALARY_DWP=" "
        except:
            ENTRY_SALARY_DWP=" "
        
       #COURSE SUBJECT IS EXTRACTED SPLITTING THE STRING WHERE IS "semester" or "quarter", SPLITTED IN "units" AND TAKE THE WORDS AFTER.
        for i in range(0,len(lista_req)):
            if "semester" in lista_req[i]:
                semester=True
            else:
                semester=False
            if "quarter" in lista_req[i]:
                quarter=True
            else:
                quarter=False

            try:
                if quarter==True | semester==True:
                    COURSE_SUBJECT= lista_req[i].split("units",1)[1].split(",",1)[0]
                else:
                    COURSE_SUBJECT = " "
            except:
                COURSE_SUBJECT= " "
         
        #TO FIND UNITS OF COURSE NEEDED IS USED THE CODE BELOW, SAME LOGICO FO YEAR EDUCATION OR EXPERIENCE, IF THERE IS "courses" TAKE THE NUMBER
         #BEFORE OTHERWISE IF THERE IS "semester" or "quarter" PUT 1
            if quarter==True | semester==True:
                if 'courses' in lista_req[i]:
                    n_courses= lista_req[i].split('courses',1)
                    COURSE_COUNT = [word.lower() for word in word_tokenize(n_courses[0],'english',False)][-1]
                else:
                    COURSE_COUNT= "1"
            else:
                COURSE_COUNT=" "
        
        #CREATE THE DATAFRAME
        structured_csv['FILE_NAME'][z]= filename
        structured_csv['JOB_CLASS_TITLE'][z]=JOB_CLAS_TITLE
        structured_csv['JOB_CLASS_NO'][z]=JOB_CLASS_NO
        structured_csv['REQUIREMENT_SET_ID'][z]=D1
        structured_csv['REQUIREMENT_SUBSET_ID'][z]= str(D2)
        structured_csv['JOB_DUTIES'][z]= JOB_DUTIES
        structured_csv['EDUCATION_YEARS'][z]=str(year_ed)
        structured_csv['SCHOOL_TYPE'][z]=str(SCHOOL_TYPE)
        structured_csv['EDUCATION_MAJOR'][z]=EDUCATION_MAJOR
        structured_csv['EXPERIENCE_LENGHT'][z]=str(EXPERIENCE_LENGTH)
        structured_csv['FULL_TIME_PART_TIME'][z]=str(FULL_TIME_PART_TIME)
        structured_csv['EXP_JOB_CLASS_TITLE'][z] = str(EXP_JOB_CLASS_TITLE)
        structured_csv['EXP_JOB_CLASS_FUNCTION'][z] = str(EXP_JOB_CLASS_FUNCTION)
        structured_csv['COURSE_COUNT'][z]=COURSE_COUNT
        structured_csv['COURSE_LENGHT'][z] = str(quar) + " " + str(sem)
        structured_csv['COURSE_SUBJECT'][z]=str(COURSE_SUBJECT)
        structured_csv['DRIVERS_LICENSE_REQ'][z]=DRIVERS_LICENSE_REQ
        structured_csv['DRIV_LIC_TYPE'][z]=DRIV_LIC_TYPE
        structured_csv['EXAM_TYPE'][z]=EXAM_TYPE
        structured_csv['ENTRY_SALARY_GEN'][z]=ENTRY_SALARY_GEN
        structured_csv['ENTRY_SALARY_DWP'][z]=ENTRY_SALARY_DWP
        structured_csv['OPEN_DATE'][z]=OPEN_DATE

        z=z+1
        

        
        
     
    


# # DATA CLEANING
# 
# In this section, after that we have our structured dataframe we need to clean it, to make it better and more understandable. So stopwords in the EXP_JOB_TITLE,FUNCTION,SCHOOL_TYPE are deleted and Salary is made like this 123,456-145,678. 

# In[ ]:


for i in range(0,len(structured_csv)):
    
    if len(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_GEN'][i].replace(",","."))) > 1:
        st= str(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_GEN'][i].replace(",","."))[0])
        en= str(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_GEN'][i].replace(",","."))[1])
        structured_csv['ENTRY_SALARY_GEN'][i]=st.replace(".",",") + " - " + en.replace(".",",")
    elif len(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_GEN'][i].replace(",","."))) ==1:
        st= re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_GEN'][i].replace(",","."))[0]
        en= "(flat-rated)"
        structured_csv['ENTRY_SALARY_GEN'][i]=st.replace(".",",") + " " + en.replace(".",",")
    else:
        st= ""
        en= ""
        structured_csv['ENTRY_SALARY_GEN'][i]=st.replace(".",",") + " " + en.replace(".",",")
        
    if len(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_DWP'][i].replace(",","."))) > 1:
        st= str(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_DWP'][i].replace(",","."))[0])
        en= str(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_DWP'][i].replace(",","."))[1])
        structured_csv['ENTRY_SALARY_DWP'][i]=st.replace(".",",") + " - "  + en.replace(".",",")
    elif len(re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_DWP'][i].replace(",","."))) ==1:
        st= re.findall('\d*\.?\d+',structured_csv['ENTRY_SALARY_DWP'][i].replace(",","."))[0]
        en= "(flat-rated)"
        structured_csv['ENTRY_SALARY_DWP'][i]=st.replace(".",",") + " " + en.replace(".",",")
    else:
        st= ""
        en= ""
        structured_csv['ENTRY_SALARY_DWP'][i]=st.replace(".",",") + " " + en.replace(".",",")
    
    


# In[ ]:


for j in range(0,len(structured_csv)):
    try:
        structured_csv['EXPERIENCE_LENGHT'][j]=w2n.word_to_num(structured_csv['EXPERIENCE_LENGHT'][j].replace("[","").replace("'","").replace("]",""))
    except:
        structured_csv['EXPERIENCE_LENGHT'][j]= " "


# In[ ]:


for j in range(0,len(structured_csv)):
    structured_csv['EDUCATION_YEARS'][j]=structured_csv['EDUCATION_YEARS'][j].replace("[","").replace("]","")
    structured_csv['SCHOOL_TYPE'][j]=structured_csv['SCHOOL_TYPE'][j].replace("[","").replace("]","").replace("'","")
    structured_csv['FULL_TIME_PART_TIME'][j]=structured_csv['FULL_TIME_PART_TIME'][j].replace("[","").replace("]","").replace("'","")


# In[ ]:


for i in range(0,len(structured_csv)):
    structured_csv['EXP_JOB_CLASS_TITLE'][i]=' '.join(re.findall(r'\b[A-Z][a-z]+|\b[A-Z]\b', structured_csv['EXP_JOB_CLASS_TITLE'][i]))


# In[ ]:


l_prova=[]
for i in range(0,len(structured_csv)):
    try:
        text_after= structured_csv['COURSE_SUBJECT'][i].split(' in ',1)[1]
        l_prova.append([word for word in word_tokenize(text_after,'English',False) if word not in stopwords.words('English')])    
    
    except:
        l_prova.append("")               


# In[ ]:


for i in range(0,len(l_prova)):
    if len(l_prova[i])> 0:
        if l_prova[i][0:2] not in stopwords.words('English'):
            if l_prova[i][0]=="following":
                l_prova[i]=l_prova[i][-2:]
            elif l_prova[i][0]=="accredited":
                l_prova[i]=l_prova[i][-2:]
            else:
                if len(l_prova[i])> 1:
                    if re.findall(r'\b[A-Z][a-z]+|\b[A-Z]\b',l_prova[i][1]):
                        l_prova[i]=l_prova[i][0] + " " + l_prova[i][1]
                    else:
                        l_prova[i]=l_prova[i][0]
                else:
                    l_prova[i]=l_prova[i][0]
        else:
            l_prova[i]=" "

structured_csv['COURSE_SUBJECT']=l_prova


# In[ ]:


structured_csv.head(8)


# # Information for visualization
# To make some insight, a new df is created with the attributes, in my opinion,that has a better comparison. In the end a tool is generated and it is easy to use.

# In[ ]:


#def extract_mean:
salary_list4plot=[]
for i in range(0,len(structured_csv)):
    salary_list4plot.append([word for word in word_tokenize(structured_csv['ENTRY_SALARY_GEN'][i],'english',False)])


# In[ ]:


media=[]
for i in range(0,len(salary_list4plot)):    
    if len(salary_list4plot[i]) ==3:
        salary_list4plot[i][0]=salary_list4plot[i][0].replace(",",".")
        salary_list4plot[i][2]=salary_list4plot[i][2].replace(",",".")
        media.append(round((float(salary_list4plot[i][0]) + float(salary_list4plot[i][2])/2),3))
    elif len(salary_list4plot[i]) == 4:
        salary_list4plot[i][0]=salary_list4plot[i][0].replace(",",".")
        media.append(float(salary_list4plot[i][0]))
    elif len(salary_list4plot[i])== 0:
        media.append(0)


# In[ ]:


df_4viz=pd.DataFrame()


# In[ ]:


df_4viz['JOB_TITLE'] = structured_csv['JOB_CLASS_TITLE']
df_4viz['DATE']= structured_csv['OPEN_DATE']
df_4viz['AVG_SALARY']= media
df_4viz['EXAM_TYPE']= structured_csv['EXAM_TYPE']
df_4viz['REQUIREMENTS']= structured_csv['REQUIREMENT_SET_ID']
df_4viz['SCHOOL']= structured_csv['SCHOOL_TYPE']
df_4viz['EDUC_YEAR']=structured_csv['EDUCATION_YEARS']


# In[ ]:


df_4viz['DATE']=pd.to_datetime(df_4viz['DATE'],format='%m-%d-%Y',errors="coerce")


# In[ ]:


df_4viz['AVG_SALARY']=pd.to_numeric(df_4viz['AVG_SALARY'])


# In[ ]:


df_4viz.head()


# # A little tool with Tableau

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1561137697641' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;JO&#47;JOB_LACITY&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='JOB_LACITY&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;JO&#47;JOB_LACITY&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1561137697641');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


structured_csv.to_csv("../../../../working/structured_csv.csv",sep=";")

