#!/usr/bin/env python
# coding: utf-8

# **Quick note about aiofiles**
# 
# Ordinary local file IO is blocking, and cannot easily and portably made asynchronous. This means doing file IO may interfere with asyncio applications, which shouldn't block the executing thread. aiofiles helps with this by introducing asynchronous versions of files that support delegating operations to a separate thread pool.

# In[ ]:


#Make sure to install aiofiles
get_ipython().system('pip install aiofiles')


# In[ ]:


get_ipython().system('pip install word2number')


# In[ ]:


#Dependencies 
import os
import pandas as pd,numpy as np
import matplotlib.pyplot as plt
import re
import asyncio
import time
import aiofiles
from tqdm import tqdm
from word2number import w2n


# **Extracting content of each key element of text files:**
# 
#     FILE_NAME
#     JOB_CLASS_TITLE
#     JOB_CLASS_NO
#     REQUIREMENT_SET_ID
#     REQUIREMENT_SUBSET_ID
#     JOB_DUTIES
#     EDUCATION_YEARS
#     SCHOOL_TYPE
#     EDUCATION_MAJOR
#     EXPERIENCE_LENGTH
#     FULL_TIME_PART_TIME
#     EXP_JOB_CLASS_TITLE
#     EXP_JOB_CLASS_ALT_RESP
#     EXP_JOB_CLASS_FUNCTION
#     COURSE_COUNT
#     COURSE_LENGTH
#     COURSE_SUBJECT
#     MISC_COURSE_DETAILS
#     DRIVERS_LICENSE_REQ
#     DRIV_LIC_TYPE
#     ADDTL_LIC
#     EXAM_TYPE
#     ENTRY_SALARY_GEN
#     ENTRY_SALARY_DWP
#     OPEN_DATE

# ![image.png](https://arlennav.github.io/Asyncio/job.PNG)

# In[ ]:


columns=['FILE_NAME','JOB_CLASS_TITLE','JOB_CLASS_NO','EXPERIENCE_LENGTH','FULL_TIME_PART_TIME',
        'JOB_DUTIES','EDUCATION_YEARS','SCHOOL_TYPE','REQUIREMENT','WHERE TO APPLY','DRIVERS_LICENSE_REQ',
         'DRIV_LIC_TYPE','EXAM_TYPE','APPLICATION DEADLINE','SELECTION PROCESS','ENTRY_SALARY_GEN','ENTRY_SALARY_DWP','OPEN_DATE','REVISED_DATE']

bulletin_dir = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'


# **''''''''''''''''''''''''''''''
# EXAM_TYPE
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#EXAM_TYPE
#''''''''''''''''''''''''''''''
def examType(content):
    '''Code explanation:
    OPEN: Exam open to anyone (pending other requirements)
    INT_DEPT_PROM: Interdepartmental Promotional
    DEPT_PROM: Departmental Promotional
    OPEN_INT_PROM: Open or Competitive Interdepartmental Promotional
    '''
    exam_type=''
    if 'INTERDEPARTMENTAL PROMOTIONAL AND AN OPEN COMPETITIVE BASIS' in content:
        exam_type='OPEN_INT_PROM' 
    elif 'OPEN COMPETITIVE BASIS' in content:
         exam_type='OPEN'
    elif 'INTERDEPARTMENTAL PROMOTIONAL' or 'INTERDEPARMENTAL PROMOTIONAL' in content:
        exam_type='INT_DEPT_PROM'
    elif 'DEPARTMENTAL PROMOTIONAL' in content:
        exam_type='DEPT_PROM' 
    return exam_type


# **''''''''''''''''''''''''''''''
# DRIVERS_LICENSE_REQ
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#DRIVERS_LICENSE_REQ
#Whether a driver's license is required, 
#possibly required, or not required (note: the job class will most likely not explicitly say if a license is not required)
#P,R
#''''''''''''''''''''''''''''''
def drivingLicenseReq(content):
    try:
        result= re.search("(.*?)(California driver\'s license|driver\'s license)", content)
        if result:
            exp=result.group(1).strip()
            exp=' '.join(exp.split()[-10:]).lower()
            if 'may require' in exp:
                return 'P'
            else:
                return 'R'
        else:
            return ''
    except Exception as e:
        return '' 


# **''''''''''''''''''''''''''''''
# DRIV_LIC_TYPE
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#DRIV_LIC_TYPE
#''''''''''''''''''''''''''''''
def drivingLicense(content):
    driving_License=[]
    result= re.search("(valid California Class|valid Class|valid California Commercial Class)(.*?)(California driver\'s license|driver\'s license)", content)
    if result:
        dl=result.group(2).strip()
        dl=dl.replace("Class","").replace("commercial","").replace("or","").replace("and","")
        if 'A' in dl:
            driving_License.append('A')
        if 'B' in dl:
            driving_License.append('B') 
        if 'C' in dl:
            driving_License.append('C')  
        if 'I' in dl:
            driving_License.append('I')   
        return ','.join(driving_License)
    else:
        return ''


# **''''''''''''''''''''''''''''''
# FULL_TIME_PART_TIME
# ''''''''''''''''''''''''''''''**

# In[ ]:



#''''''''''''''''''''''''''''''
#FULL_TIME_PART_TIME
#''''''''''''''''''''''''''''''
def fulltime_parttime(content):
    worktype=''
    content=content.lower()
    if 'full-time' in content:
        worktype='FULL-TIME' 
    elif 'part-time' in content:
        worktype='PART-TIME'
    return worktype


# **''''''''''''''''''''''''''''''
# EXPERIENCE_LENGTH
# ''''''''''''''''''''''''''''''**

# In[ ]:


def experienceLength(content):
    try:
        result= re.search("(.*?)(of full-time)", content)
        if result:
            exp=result.group(1).strip()
            exp= exp.replace('.',' ')
            exp=' '.join(exp.split()[-2:])
            if 'month' in exp:
                result=re.findall(r'\d+', exp)
                if result:
                    year=float(result[0])/12
                else:
                    year=float(w2n.word_to_num(exp))/12
            else:
                result=re.findall(r'\d+', exp)
                if result:
                    year=float(result[0])
                else:
                    year=float(w2n.word_to_num(exp))
            year=round(year,1)
            return year
    except Exception as e:
        return np.nan    


# **''''''''''''''''''''''''''''''
# ENTRY_SALARY_GEN
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#ENTRY_SALARY_GEN
#''''''''''''''''''''''''''''''
def salary(content):
    try:
        salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?') #match salary
        sal=re.search(salary,content)
        if sal:
            range1=sal.group(1)
            if range1 and '$' not in range1:
                range1='$'+range1
            range2=sal.group(2)
            if range2:
                range2=sal.group(2).replace('to','')
                range2=range2.replace('and','')
            if range1 and range2:
                return f"{range1}-{range2.strip()}"
            elif range1:
                return f"{range1} (flat-rated)"
        else:
            return ''
    except Exception as e:
        return ''  


# **''''''''''''''''''''''''''''''
# ENTRY_SALARY_DWP
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#ENTRY_SALARY_DWP
#''''''''''''''''''''''''''''''
def salaryDWP(content,filename):
    try:
        result= re.search("(Department of Water and Power is)(.*)", content)
        if result:
            salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?') #match salary
            sal=re.search(salary,result.group(2))
            if sal:
                range1=sal.group(1)
                if range1 and '$' not in range1:
                    range1='$'+range1
                range2=sal.group(2)
                if range2:
                    range2=sal.group(2).replace('to','')
                    range2=range2.replace('and','')
                if range1 and range2:
                    return f"{range1}-{range2.strip()}"
                elif range1:
                    return f"{range1} (flat-rated)"
            else:
                return ''
    except Exception as e:
        return ''  


# **''''''''''''''''''''''''''''''
# EDUCATION_YEARS
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#EDUCATION_YEARS
#''''''''''''''''''''''''''''''
def educationYears(content):
    try:
        result= re.search("(.*?)(college)", content)
        if result:
            exp=result.group(1).strip()
            exp= exp.replace('.',' ')
            exp=' '.join(exp.split()[-2:])
            if 'month' in exp:
                result=re.findall(r'\d+', exp)
                if result:
                    year=float(result[0])/12
                else:
                    year=float(w2n.word_to_num(exp))/12
            else:
                result=re.findall(r'\d+', exp)
                if result:
                    year=float(result[0])
                else:
                    year=float(w2n.word_to_num(exp))
            year=round(year,1)
            return year
    except Exception as e:
        return np.nan


# **''''''''''''''''''''''''''''''
# SCHOOL_TYPE
# ''''''''''''''''''''''''''''''**

# In[ ]:


#''''''''''''''''''''''''''''''
#SCHOOL_TYPE
#''''''''''''''''''''''''''''''
def schoolType(content):
    school_Type=''
    content=content.lower()
    #COLLEGE OR UNIVERSITY, HIGH SCHOOL, APPRENTICESHIP
    if 'college' in content or 'university' in content:
        school_Type='COLLEGE OR UNIVERSITY' 
    elif 'high school' in content:
        school_Type='HIGH SCHOOL'
    elif 'apprenticeship' in content:
        school_Type='APPRENTICESHIP'  
    elif 'certificate' in content:
        school_Type='CERTIFICATE'
    return school_Type


# More Fields
# ******
# *  FILE_NAME
# 
# * JOB_CLASS_TITLE
# 
# * JOB_CLASS_NO
# 
# * OPEN_DATE
# 
# * REVISED
# 
# * JOB_DUTIES
# 
# * REQUIREMENT
#  
# * WHERE TO APPLY
#  
# * APPLICATION DEADLINE
# 
# * SELECTION PROCESS or Examination Weight
# ****
# 
# Basically processContent function generates a list of dictionary of all the required features.
# 

# In[ ]:


listdic=[]
def processContent(filename,content):
    dicJobs={}
    content=content.replace("\n","").replace("\t","").strip()
    #print(filename)
    #FILE_NAME           
    dicJobs['FILE_NAME']=filename.split("/")[-1]

    #JOB_CLASS_TITLE
    dicJobs['JOB_CLASS_TITLE']=''
    result= re.search("(.*?)(Class Code:|Class  Code:)", content,flags=re.IGNORECASE)
    if result:
        dicJobs['JOB_CLASS_TITLE']=result.group(1).strip()
    
    #JOB_CLASS_NO
    dicJobs['JOB_CLASS_NO']=''
    result=re.search("(Class Code:|Class  Code:)(.*)Open Date:", content,flags=re.IGNORECASE)
    if result:
        dicJobs['JOB_CLASS_NO']=result.group(2).strip()

    #DRIV_LIC_TYPE
    dicJobs['DRIV_LIC_TYPE']=drivingLicense(content) 
    
    
    #FULL_TIME_PART_TIME
    dicJobs['FULL_TIME_PART_TIME']=fulltime_parttime(content)
    
    #EXAM_TYPE
    dicJobs['EXAM_TYPE']=examType(content)
    
    #OPEN_DATE
    dicJobs['OPEN_DATE']=''
    result= re.search("(Class Code:|Class  Code:)(.*)(ANNUAL SALARY|ANNUALSALARY)", content)
    shortContent=''
    if result:
        shortContent=result.group(2).strip()
        result= re.search("Open Date:(.*)REVISED", shortContent,flags=re.IGNORECASE)
        if result:
            dicJobs['OPEN_DATE']=result.group(1).strip()
        if dicJobs['OPEN_DATE']=='':
            result= re.search("Open Date:(.*)\(Exam", shortContent,flags=re.IGNORECASE)
            if result:
                dicJobs['OPEN_DATE']=result.group(1).strip()
        if dicJobs['OPEN_DATE']=='':
            result= re.search("Open Date:(.*)", shortContent,flags=re.IGNORECASE)
            if result:
                dicJobs['OPEN_DATE']=result.group(1).strip()
                
    #REVISED:
    dicJobs['REVISED_DATE']=''
    result= re.search("(REVISED:|revised)(.*?)\(Exam", shortContent,flags=re.IGNORECASE)
    if result:
        dicJobs['REVISED_DATE']=result.group(2).strip()
    else:
        result= re.search("(REVISED:|revised)(.*)", shortContent,
                          flags=re.IGNORECASE)
        if result:
            dicJobs['REVISED_DATE']=result.group(2).strip() 
    
    #ENTRY_SALARY_GEN
    #ENTRY_SALARY_DWP
    dicJobs['ENTRY_SALARY_GEN']=''
    result=re.search("(ANNUAL SALARY|ANNUALSALARY)(.*?)DUTIES", content)
    if result:
        salContent= result.group(2).strip()
        dicJobs['ENTRY_SALARY_GEN']=salary(salContent)   
        dicJobs['ENTRY_SALARY_DWP']=salaryDWP(salContent,filename) 
    else:
        result=re.search("(ANNUAL SALARY|ANNUALSALARY)(.*?)REQUIREMENT", content,flags=re.IGNORECASE)
        if result:
            salContent= result.group(2).strip()
            dicJobs['ENTRY_SALARY_GEN']=salary(salContent)
            dicJobs['ENTRY_SALARY_DWP']=salaryDWP(salContent,filename) 
            

    #JOB_DUTIES
    dicJobs['JOB_DUTIES']=''
    result=dicJobs['JOB_DUTIES']= re.search("DUTIES(.*?)REQUIREMENT", content)
    if result:
        dicJobs['JOB_DUTIES']= result.group(1).strip()
  
    #REQUIREMENT
    req='|'.join(["REQUIREMENT/MIMINUMUM QUALIFICATION",
                  "REQUIREMENT/MINUMUM QUALIFICATION",
                  "REQUIREMENT/MINIMUM QUALIFICATION",
                  "REQUIREMENT/MINIMUM QUALIFICATIONS",
                  "REQUIREMENT/ MINIMUM QUALIFICATION",
                  "REQUIREMENTS/MINUMUM QUALIFICATIONS",
                  "REQUIREMENTS/ MINIMUM QUALIFICATIONS",
                  "REQUIREMENTS/MINIMUM QUALIFICATIONS",
                  "REQUIREMENTS/MINIMUM REQUIREMENTS",
                  "REQUIREMENTS/MINIMUM QUALIFCATIONS",
                  "REQUIREMENT/MINIMUM QUALIFICAITON",
                  "MINIMUM REQUIREMENTS:",
                  "REQUIREMENTS",
                  "REQUIREMENT"])
    
    result= re.search(f"({req})(.*)(WHERE TO APPLY|HOW TO APPLY)", content)
    dicJobs['REQUIREMENT']=''
    if result:
        dicJobs['REQUIREMENT']= result.group(2).strip()
        
    
    #EXPERIENCE_LENGTH
    dicJobs['EXPERIENCE_LENGTH']=experienceLength(dicJobs['REQUIREMENT'])
    
    #EDUCATION_YEARS
    dicJobs['EDUCATION_YEARS']=educationYears(dicJobs['REQUIREMENT'])
    
    #SCHOOL_TYPE
    dicJobs['SCHOOL_TYPE']=schoolType(dicJobs['REQUIREMENT'])
    
    #DRIVERS_LICENSE_REQ
    dicJobs['DRIVERS_LICENSE_REQ']=drivingLicenseReq(dicJobs['REQUIREMENT']) 
    
    #WHERE TO APPLY
    dicJobs['WHERE TO APPLY']= ''
    result= re.search("(HOW TO APPLY|WHERE TO APPLY)(.*)(APPLICATION DEADLINE|APPLICATION PROCESS)", content)
    if result:
        dicJobs['WHERE TO APPLY']= result.group(2).strip()
    else:
        result= re.search("(HOW TO APPLY|WHERE TO APPLY)(.*)(SELECTION PROCESS|SELELCTION PROCESS)", content)
        if result:
            dicJobs['WHERE TO APPLY']= result.group(2).strip()
        

    #APPLICATION DEADLINE
    dicJobs['APPLICATION DEADLINE']=''
    result= re.search("(APPLICATION DEADLINE|APPLICATION PROCESS)(.*?)(SELECTION PROCESS|SELELCTION PROCESS)", content)
    if result:
        dicJobs['APPLICATION DEADLINE']= result.group(2).strip()
    else:
        result= re.search("(APPLICATION DEADLINE|APPLICATION PROCESS)(.*?)(Examination Weight:)", content)
        if result:
            dicJobs['APPLICATION DEADLINE']= result.group(2).strip() 
    
    
    #SELECTION PROCESS or Examination Weight: 
    dicJobs['SELECTION PROCESS']=''
    result=dicJobs['SELECTION PROCESS']= re.search("(SELECTION PROCESS|Examination Weight:)(.*)(APPOINTMENT|APPOINTMENT IS SUBJECT TO:)", content)
    if result:
        dicJobs['SELECTION PROCESS']= result.group(2).strip()
    else:
        result=dicJobs['SELECTION PROCESS']= re.search("(SELECTION PROCESS|Examination Weight:)(.*)", content)
        if result:
            dicJobs['SELECTION PROCESS']= result.group(2).strip()
    
    listdic.append(dicJobs)


# **The Asynchronous I/O feature enhances performance by allowing applications to overlap processing with I/O operations.**

# ![image.png](https://arlennav.github.io/Asyncio/Asyncio.PNG)

# In[ ]:


async def processFile(filename):
    async with aiofiles.open(filename, mode='r',encoding="ISO-8859-1") as f:
        try:            
            content = await f.read()
            processContent(filename,content)
        except Exception as exp:
            print(f"Error file {filename} {exp}")
            pass
async def process_all():
    for filename in tqdm(os.listdir(bulletin_dir)):
        await processFile(f"{bulletin_dir}{filename}")
    df=pd.DataFrame(listdic,columns=columns)
    df['JOB_CLASS_NO'] = df['JOB_CLASS_NO'].apply('="{}"'.format)
    df['OPEN_DATE'] = pd.to_datetime(df['OPEN_DATE'])
    df['REVISED_DATE'] = pd.to_datetime(df['REVISED_DATE'])
    df.to_csv("JobBulletins.csv",index=False)   
    print('Process complete.')

    
if __name__ == "__main__":
    asyncio.ensure_future(process_all())


# In[ ]:


df_JobBulletins= pd.read_csv('../input/jobbulletinsla/JobBulletins.csv')
df_JobBulletins.head()


# **Lets do some visualization**

# In[ ]:


# Number of missing values in each column of training data
missing_val_count_by_column = (df_JobBulletins.isnull().sum())
missing_val_count_by_column


# In[ ]:


#FULL_TIME vs PART_TIME
df_JobBulletins['FULL_TIME_PART_TIME'].value_counts().plot(kind="pie", y='FULL_TIME_PART_TIME',autopct='%1.1f%%',legend=False)
plt.axis("equal")
plt.tight_layout()


# In[ ]:


#SCHOOL_TYPE
df_JobBulletins['SCHOOL_TYPE'].value_counts().plot(kind="pie", y='SCHOOL_TYPE',autopct='%1.1f%%',legend=False)
plt.axis("equal")
plt.tight_layout()


# In[ ]:


#DRIVERS_LICENSE_REQ
df_JobBulletins['DRIVERS_LICENSE_REQ'].value_counts().plot(kind="pie", y='DRIVERS_LICENSE_REQ',autopct='%1.1f%%',legend=False)
plt.axis("equal")
plt.tight_layout()


# In[ ]:


#DRIV_LIC_TYPE
df_JobBulletins['DRIV_LIC_TYPE'].value_counts().plot(kind="pie", y='DRIV_LIC_TYPE',autopct='%1.1f%%',legend=False)
plt.axis("equal")
plt.tight_layout()


# In[ ]:


#EXAM_TYPE
df_JobBulletins['EXAM_TYPE'].value_counts().plot(kind="pie", y='EXAM_TYPE',autopct='%1.1f%%',legend=False)
plt.axis("equal")
plt.tight_layout()


# **Experiense (year) vs Number of Jobs**

# In[ ]:


#EXPERIENCE_LENGTH

df=pd.DataFrame(df_JobBulletins['EXPERIENCE_LENGTH'].value_counts()).reset_index(). rename(columns={'index': 'experiense (year)', 'EXPERIENCE_LENGTH': 'Number of Jobs'}).sort_values('experiense (year)').reset_index(drop=True)

ax = df.plot(kind='bar',x='experiense (year)',y='Number of Jobs', title ='Experiense (year) vs Number of Jobs', figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("experiense (year)", fontsize=12)
ax.set_ylabel("Number of Jobs", fontsize=12)
plt.show()

df


# Separate **ENTRY_SALARY_GEN** column into **Starting Salary** and **Ending Salary** columns**

# In[ ]:


# new data frame with split value columns 
df_JobBulletins["ENTRY_SALARY_GEN_NEW_FORMAT"] = df_JobBulletins["ENTRY_SALARY_GEN"].str.replace('$','').str.replace('\(flat-rated\)','').str.replace(',','')

new_salary=df_JobBulletins["ENTRY_SALARY_GEN_NEW_FORMAT"].str.split("-", n = 1, expand = True) 

# making separate Starting Salary column from new data frame 
df_JobBulletins["Starting Salary"]= new_salary[0]
# making separate Ending Salary column from new data frame 
df_JobBulletins["Ending Salary"]= new_salary[1]

df_JobBulletins["Starting Salary"] = pd.to_numeric(df_JobBulletins["Starting Salary"], errors='coerce')
df_JobBulletins["Ending Salary"] = pd.to_numeric(df_JobBulletins["Ending Salary"], errors='coerce')


# **Top 10 Jobs With Highest Salaries**

# In[ ]:


df_JobBulletins_sorted=df_JobBulletins.sort_values('Starting Salary',ascending=False).head(10)

ax=df_JobBulletins_sorted[['JOB_CLASS_TITLE','Starting Salary','Ending Salary']].plot(kind='bar',
                                                            title ='', label='JOB_CLASS_TITLE',                                                             
                                                            figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Title", fontsize=12)
ax.set_ylabel("Salary", fontsize=12)
ax.set_xticklabels(df_JobBulletins_sorted['JOB_CLASS_TITLE'])
plt.show()

