#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install word2number vaderSentiment')


# In[ ]:


import re
import os
# import tika
import string
import shutil
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from word2number import w2n
from datetime import datetime
from os import walk
# from PIL import Image
# from tika import parser
from collections import Counter
from nltk.corpus import stopwords
# from wand.image import Image as Img
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from shutil import copytree, ignore_patterns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
get_ipython().run_line_magic('matplotlib', 'inline')

inputFolder = '../input/cityofla/CityofLA/Additional data/PDFs'
outputFolder = '/kaggle/working/pdfs/'


# In[ ]:


bulletins_df = pd.DataFrame(columns = ['FILE_NAME','JOB_CLASS_TITLE','JOB_CLASS_NO','REQUIREMENT_SET_ID',
                                       'REQUIREMENT_SUBSET_ID','JOB_DUTIES','EDUCATION_YEARS','SCHOOL_TYPE',
                                       'EDUCATION_MAJOR','EXPERIENCE_LENGTH','FULL_TIME_PART_TIME',
                                       'EXP_JOB_CLASS_TITLE','EXP_JOB_CLASS_ALT_RESP','EXP_JOB_CLASS_FUNCTION',
                                       'COURSE_COUNT','COURSE_LENGTH','COURSE_SUBJECT','MISC_COURSE_DETAILS',
                                       'DRIVERS_LICENSE_REQ','DRIV_LIC_TYPE','ADDTL_LIC','EXAM_TYPE',
                                       'ENTRY_SALARY_GEN','ENTRY_SALARY_DWP','OPEN_DATE'])

######################## Patterns to match #######################
bulletinspath = '../input/cityofla/CityofLA/Job Bulletins/'
classcodepattern = re.compile( r'[C,c]lass [C,c]ode:(\s+)([0-9a-zA-Z]+)' )
driversLicencePattern = re.compile( r'(.*[D,d]river\'s(\s*)[L,l]icense)' )
datepattern = re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')       #match open date
salarypattern = re.compile(r'(\$\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')       #match salary
salaryDwpPattern = re.compile(r' (Water and Power is) (\$\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')
requirementspattern = re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')      #match requirements
yearsexperiencepattern = re.compile( r'(.*)[Y,y]ears(.*)full(.*)experience(.*)' )
educationexppatern = re.compile( r'(.*)[A,a]ccredited(.*)[C,c]ollege(.*)[U,u]niversity(.*)' )
schooltypepattern = re.compile( r'(.*)[Q,q]ualifying(.*)[E,e]ducation(.*)from(.*)accredited(.*)' )
educationmajorpattern = re.compile( r'(major|degree) in(.*)' )
fulltime_parttime = re.compile( r'(.*)(full|part)(.*)experience(.*)' )
exptitlepattern = re.compile( r'(.*)experience as(.*)' )
coursecountpattern = re.compile( r'(.*)completion of(.*)(course?)(.*)' )
semestercountpattern = re.compile( r'(.*)completion of(.*)(semester)(.*)' )
quartercountpattern = re.compile( r'(.*)completion of(.*)(quarter)(.*)' )
miscLicencePattern = re.compile( r'as a ([L,l]icensed)(.+[;,.])' )
examPattern = re.compile( r'THIS EXAM( |INATION )IS TO BE GIVEN(.*)BASIS' )

i = 0
for file in os.listdir( bulletinspath ):
    
    i = i + 1
    with open ( bulletinspath + file, 'r', encoding = "ISO-8859-1" ) as f :
        try :
            
            filedata = f.read().replace('\t','')

            data = filedata.replace( '\n', '' )
            
            position = filedata.replace('CAMPUS INTERVIEWS ONLY','').split('\n')[0]
            
            try :
                ClassCode = re.search(classcodepattern, filedata).group(2)
            except :
                ClassCode = '-'

            try :
                YearsExp = re.search(yearsexperiencepattern, filedata).group(1)         
                YearsExp = YearsExp.split()
                if( 0 < len(YearsExp)) :
                    YearsExp = YearsExp[-1]
                    if '.' in YearsExp :
                        YearsExp = YearsExp.split('.')[-1]
                    YearsExp = w2n.word_to_num(re.sub( r'[^a-zA-Z]+','',YearsExp ) )
                else:
                    YearsExp = ''
            except :
                YearsExp = ''
            
            try :
                EducationExp = re.search(educationexppatern, filedata).group(2)
                EducationExp = EducationExp.strip()
                EducationExp = EducationExp.replace( ' ', '-' )
                EducationExp = EducationExp.split('-')
                EducationExp = w2n.word_to_num(EducationExp[0])
            except :
                EducationExp = ''

            try :
                SchoolType = re.search(schooltypepattern, filedata).group(4)
                SchoolType = SchoolType.split('accredited')
                SchoolType = SchoolType[0]
            except :
                SchoolType = ''

            DriverLicenseTypes = []
            try :
                DriversLicense = re.search(driversLicencePattern, filedata).group()

                LicenseTypes = re.findall( r'([C,c]lass ([a-zA-Z]))', DriversLicense )
                if( 0 != len(LicenseTypes) ) :
                    for LicenseType in LicenseTypes :
                        DriverLicenseTypes.append(LicenseType[1])

                if( 'may' not in DriversLicense and ( 'valid' in DriversLicense or 'require' in DriversLicense ) ) :
                    DriversLicense = 'R'
                elif( 'may' in DriversLicense and ( 'valid' in DriversLicense or 'require' in DriversLicense ) ) :
                    DriversLicense = 'P'
            except :
                DriversLicense = ''
 
            Salary = 0
            SalaryDwp = 0
            try :
                if ( None != re.search( salarypattern, data ).group() ) :
                    Salary = []
                    if None != re.search( salarypattern, data ).group(1) :
                        Salary.append( (re.search( salarypattern, data ).group(1)).strip() )
                    if None != re.search( salarypattern, data ).group(5) :
                        Salary.append( re.search( salarypattern, data ).group(5).strip() )
                    Salary = '-'.join( Salary )
            except:
                Salary = 0

            try :
                if ( None != re.search( salaryDwpPattern, data ).group() ) :
                    SalaryDwp = []
                    if None != re.search( salaryDwpPattern, data ).group(2) :
                        SalaryDwp.append( (re.search( salaryDwpPattern, data ).group(2)).strip() )

                    if None != re.search( salaryDwpPattern, data ).group(6) :
                        SalaryDwp.append( (re.search( salaryDwpPattern, data ).group(6)).strip() )
                    SalaryDwp = '-'.join( SalaryDwp )
            except:
                SalaryDwp= 0

            try :
                opendate = re.search(datepattern, data).group(3)
            except :
                opendate = '-'
            
            try :
                duties = re.search(r'(DUTIES)(.*)(REQ[A-Z])',data).group(2)
            except:
                duties = "-"
            
            ExamPattern = '-'
            ExamPatterns = []
            try :
                ExamPattern = (re.search( examPattern, data ).group(2)).upper()
                if( 'OPEN COMPETITIVE' in ExamPattern and 'INTERDEPARTMENTAL PROMOTION' in ExamPattern ) :
                    ExamPatterns.append( 'OPEN_INT_PROM' )
                else :    
                    if( 'OPEN COMPETITIVE' in ExamPattern ) :
                        ExamPatterns.append( 'OPEN' )
                    if( 'INTERDEPARTMENTAL PROMOTION' in ExamPattern ) :
                        ExamPatterns.append( 'INT_DEPT_PROM' )

                if( ' DEPARTMENTAL PROMOTION' in ExamPattern ) :
                    ExamPatterns.append( 'DEPT_PROM' )
            except Exception as e :
                ExamPatterns.append('-')
                
            selection = [z[0] for z in re.findall('([A-Z][a-z]+)((\s\.\s)+)',data)]     ##match selection criteria
            
            requirements = '-'
            EduMajor = '-'
            CourseSubjects = []
            miscCourseDetails = ''
            expJobAltResp = ''

            try:
                requirements = re.search(requirementspattern,data).group(2)
            except Exception as e:
                requirements = re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',data)[0][1][:1200]).group(1)
            
            subrequirements = ''
            totalRequirements = 1
            expJobClassFunc = ''
            try :
                lstMatches = re.findall( r'([0-9])(\.[^\S\n\t])', requirements )
                superset = []
                for match in lstMatches :
                    superset.append(int(match[0]))
                superset = list(set(superset))
                superset.sort()
                intCount = 1
                if 1 < len(superset) :
                        
                    while intCount < len(superset) :
                        if 1 != superset[intCount] - superset[intCount-1] :
                            break
                        end = requirements.index(str(superset[intCount])+'.')
                        start = requirements.index(str(superset[intCount-1])+'.')
                        lstSubMatches = re.findall( r'([a-zA-Z])(\.[^\S\n\t])', requirements[start:end].replace('U.S.','').strip())
                        altReqs = re.findall( r'(.*)with the City of Los Angeles(.*)or in(.*)(;|.)', requirements[start:end])
                        if 0 < len(altReqs) :
                            for altReq in altReqs :
                                if 0 < len(altReq) :
                                    expJobAltResp = (altReq[2].strip())
                        if 0 < len(lstSubMatches) :
                            matched = list(lstSubMatch[0] for lstSubMatch in lstSubMatches)
                            if 'a' == matched[0].lower() :
                                subrequirements = matched[-1].upper()
                                jobClassStart = requirements.index(str(matched[0])+'.')
                                expJobClassFunc = (requirements[jobClassStart:end].replace(str(superset[intCount-1])+'.', '')).strip()
                                AltExpStart = requirements.index(str(matched[0])+'.')
                                expJobAltResp = (requirements[AltExpStart:end].replace(str(superset[intCount])+'.', '')).strip()
                                miscCourses = re.findall( r'(certified|course)(.*)', expJobAltResp )
                                miscCourseDetails = ' '.join(list( ' '.join(matchMiscCourse) for matchMiscCourse in miscCourses ))

                        intCount = intCount + 1
                    totalRequirements = intCount
                start = requirements.index(str(superset[intCount-1])+'.')
                end = len(requirements)
                lstSubMatches = re.findall( r'([a-zA-Z])(\.[^\S\n\t])', requirements[start:end].replace('U.S.','').strip())
                altReqs = re.findall( r'(.*)with the City of Los Angeles(.*)or in(.*)(;|.)', requirements[start:end])
                if 0 < len(altReqs) :
                    for altReq in altReqs :
                        if 0 < len(altReq) :
                            expJobAltResp = (altReq[2].strip())
                if 0 < len(lstSubMatches) :
                    expJobClassFunc = (requirements[start:end].replace(str(superset[intCount-1])+'.', '')).strip()
                    matched = list(lstSubMatch[0] for lstSubMatch in lstSubMatches)
                    if 'a' == matched[0] :
                        jobClassStart = requirements.index(str(matched[0])+'.')
                        expJobClassFunc = (requirements[jobClassStart:end].replace(str(superset[intCount-1])+'.', '')).strip()
                        subrequirements = matched[-1].upper()
                        AltExpStart = requirements.index(str(matched[0])+'.')
                        expJobAltResp = (requirements[AltExpStart:end].replace(str(superset[intCount])+'.', '')).strip()
                        miscCourses = re.findall( r'(certified|course)(.*)', expJobAltResp )
                        miscCourseDetails = ' '.join(list( ' '.join(matchMiscCourse) for matchMiscCourse in miscCourses ))
            except :
                subrequirements = ''
                totalRequirements = 1
                    
            try :
                if None != re.search(educationmajorpattern,requirements) :
                    EduMajor = re.search(educationmajorpattern,requirements).group(2)
                    EduMajor = EduMajor.strip().strip('.')
                    EduMajor = EduMajor.split(';')
                    EduMajor = EduMajor[0]
                    EduMajor = EduMajor.replace('and/or', 'or')
                    EduMajor = EduMajor.replace('landscape', 'lnadscape')
                    EduMajor = EduMajor.split('and')
                    EduMajor = EduMajor[0]
                    EduMajor = EduMajor.replace('lnadscape', 'landscape')
                    EduMajor = EduMajor.split( 'or in' )
                    EduMajor = EduMajor[0]
                    EduMajor = EduMajor.split( 'or upon' )
                    EduMajor = EduMajor[0]
                    EduMajor = EduMajor.split( '.' )
                    EduMajor = string.capwords( EduMajor[0] )

                    if 0 < len(EduMajor.replace(' or ', ',').replace(' Or ', ',').replace('A Related Field','').strip(',').split(',')) :
                        for major in EduMajor.replace(' or ', ',').replace(' Or ', ',').replace('A Related Field','').strip(',').split(',') :
                            if 0 < len(major.split(' or')) :
                                for submajor in major.split(' or') :
                                    CourseSubjects.append( submajor.strip() )
                            else :
                                CourseSubjects.append( major.strip() )                   
            except :
                EduMajor = '-'
            
            TimeExp = '-'

            try :
                TimeExp = re.search(fulltime_parttime, requirements).group(2)         
                TimeExp = (TimeExp + '_Time').upper()
            except :
                TimeExp = ''
            
            ExpTitle = ''
            SemCount = 0
            QuarterCount = 0
            CourseCount = 0
            if None != re.search(exptitlepattern, requirements) :
                try :
                    ExpTitle = re.search(exptitlepattern, requirements).group(2)         
                    ExpTitle = ExpTitle.strip().strip('.').lstrip('a ').lstrip('an ')
                    broken = 0 
                    if 0 < len(ExpTitle.split(';')) :
                        for strLine in ExpTitle.split(';') :
                            if 0 < len(strLine.split('with')):
                                for strSubLine in strLine.split('with') :
                                    if 'Los Angeles' not in strSubLine :
                                        strSubLine = strSubLine.strip().strip('.').lstrip('a ').lstrip('an ')
                                        strSubLine = strSubLine.split(' or in')
                                        strSubLine = strSubLine[0]
                                        if '.' not in strSubLine :
                                            ExpTitle = string.capwords(strSubLine.strip().strip('.').strip(',') )
                                            ExpTitle = ExpTitle.split('.')
                                            ExpTitle = ExpTitle[0]
                                            broken = 1 
                                            break
                            if broken == 1 :
                                break
                    else : 
                        ExpTitle = '-'
                except :
                    ExpTitle = ''
                    
                CourseCount = 0
                if None != re.search( coursecountpattern, requirements ) :
                    try :
                        CourseCount = re.search( coursecountpattern, requirements ).group( 2 )        
                        CourseCount = CourseCount.split( 'course' )
                        CourseCount = CourseCount[0]

                        if 0 < len(CourseCount.strip().split( ' ' )) :
                            for strLine in CourseCount.split( ' ' ) :
                                try :
                                    CourseCount = w2n.word_to_num(strLine)
                                    break
                                except:
                                    continue
                            if str(CourseCount).isdigit() != True :
                                CourseCount = 0
                        else :
                            CourseCount = 0

                    except Exception as e :
                        CourseCount = 0

            try :
                SemCount = re.search( semestercountpattern, requirements ).group(2)
                SemCount = SemCount.strip().split(' ')
                SemCount = SemCount[-1]
                if True != str(SemCount).isdigit() :
                    SemCount = w2n.word_to_num(str(SemCount))

                QuarterCount = re.search( quartercountpattern, requirements ).group(2)
                QuarterCount = QuarterCount.strip().split(' ') 
                QuarterCount = QuarterCount[-1].strip()
                if True != str(QuarterCount).isdigit() :
                    QuarterCount = w2n.word_to_num(str(QuarterCount))

            except Exception as e :
                SemCount = 0
                QuarterCount = 0
  
            try:
                enddate = re.search(r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})',data).group()
            except Exception as e:
                enddate = np.nan

            MiscLicense = '-'
            try :
                MiscLicense = re.search( miscLicencePattern, data ).group(2)
                MiscLicense = MiscLicense.strip().split('.')
                MiscLicense = MiscLicense[0].strip().split('issued')
                MiscLicense = MiscLicense[0].strip().split(';')
                MiscLicense = MiscLicense[0]
            except Exception as e :
                MiscLicense = '-' 
      
            bulletins_df = bulletins_df.append( {
                                'FILE_NAME'       : file,
                                'JOB_CLASS_TITLE' : position.strip(), 
                                'JOB_CLASS_NO' : ClassCode.strip(),
                                'REQUIREMENT_SET_ID' : totalRequirements,
                                'REQUIREMENT_SUBSET_ID' : subrequirements,
                                'JOB_DUTIES' : duties.strip(),
                                'EDUCATION_YEARS' : EducationExp,
                                'SCHOOL_TYPE' : SchoolType,
                                'EDUCATION_MAJOR' : EduMajor,
                                'EXPERIENCE_LENGTH' : YearsExp,
                                'FULL_TIME_PART_TIME' : TimeExp,
                                'EXP_JOB_CLASS_TITLE' : ExpTitle,
                                'EXP_JOB_CLASS_ALT_RESP' : expJobAltResp,
                                'EXP_JOB_CLASS_FUNCTION' : expJobClassFunc,
                                'COURSE_COUNT' : CourseCount,
                                'COURSE_LENGTH' : str(SemCount) + '|' + str(QuarterCount),
                                'COURSE_SUBJECT': '|'.join( CourseSubjects ),
                                'MISC_COURSE_DETAILS' : miscCourseDetails,
                                'DRIVERS_LICENSE_REQ' : DriversLicense,
                                'DRIV_LIC_TYPE': ','.join( set(DriverLicenseTypes)),
                                'ADDTL_LIC' : string.capwords(MiscLicense),
                                'EXAM_TYPE' : ','.join(ExamPatterns),
                                'ENTRY_SALARY_GEN' : Salary,
                                'ENTRY_SALARY_DWP' : SalaryDwp,
                                'OPEN_DATE' : opendate,
                                'Criteria' : ','.join(selection)
                            },ignore_index=True )
                
        except Exception as e:
            print(file)
            print(e)

bulletins_df.to_csv('bulletins.csv')


# In[ ]:


jobs_df = pd.read_csv('bulletins.csv')
jobs_count = jobs_df.groupby('JOB_CLASS_TITLE').JOB_CLASS_TITLE.count().sort_values(ascending=False)

salaries = jobs_df['ENTRY_SALARY_GEN'].str.replace(',','').str.replace('$','').str.split('-',n=1,expand=True)
jobs_df['min_salary'], jobs_df['max_salary'] = salaries[0], salaries[1]
jobs_df['max_salary'].fillna(jobs_df['min_salary'], inplace=True)
jobs_df['max_salary'] = jobs_df['max_salary'].astype('int64')
salries_by_job_title = jobs_df.groupby('JOB_CLASS_TITLE').max_salary.agg(['max']).sort_values(by=['max'],ascending=False)


# **Top paid profiles**

# In[ ]:


salries_by_job_title.head(10)


# **Max Salary Range**

# In[ ]:


salries_by_job_title.plot.hist()


# Top

# In[ ]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(20,25),dpi=80, facecolor='w', edgecolor='k')

stopwords = set(STOPWORDS)
stopwords.update(['work', 'may', 'city', 'including', 'assigns', 'los', 'angeles'])

JobDuties = ' '.join([descr for descr in jobs_df.JOB_DUTIES])

wc = WordCloud(stopwords= stopwords, background_color = 'white',max_words=100,width=800, height=400).generate(JobDuties)

plt.imshow(wc)
plt.axis("off")
plt.show()

figure(num=None, figsize=(20,25),dpi=80, facecolor='w', edgecolor='k')
wordcount = collections.defaultdict(int)
wordpattern= r"\W"
for word in JobDuties.lower().split() :
    word = re.sub(wordpattern,'',word)
    if word not in stopwords :
        wordcount[word] = wordcount[word] + 1
mostCommonWords = sorted(wordcount.items(), key=lambda k_v:k_v[1], reverse=True)[:50]
mostCommonWords = dict(mostCommonWords)

wordsForCount = list(mostCommonWords.keys())
wordsCount = list(mostCommonWords.values())
wordsForCount.reverse()
wordsCount.reverse()

plt.barh(wordsForCount, wordsCount)
plt.show()


# In[ ]:


analyser = SentimentIntensityAnalyzer()

analyser.polarity_scores(jobs_df.iloc[10]['JOB_DUTIES'])

sentiments = pd.DataFrame([analyser.polarity_scores(duties) for duties in jobs_df['JOB_DUTIES']])

sentiments.plot.hist()


# Looks like the job duties are neutral, if it can be more on positive side would be better,
