#!/usr/bin/env python
# coding: utf-8

# In our dataset we have differnet type of data. There are:
# 1. Job Bulletins in plain text stored in .txt format(in folder ./cityofLA/Job Bulletins).
# 2. Checking requirements for csv and processing csv.
# 2. PDFs with Job Bulletins for last years stored in PDF(in folder ./cityofLA/Additional Data/PDFs).
# 3. And City Job Paths (as i understood it's data which we can structured in one tree as cool example career opportunities)
# 4. Other data for help us with structuring our dataset into single csv file(job_title.csv, kaggle_data_dictionary.csv, sample job class export template.csv, Description of promotions in job bulletins.docx in ./cityofLA/Additional Data and all files in ./cityofLA/Additional Data/job bulletins with annotations).

# 1. [](http://)As i understand the most important of this challenge is creating fully structured CSV file. I attempted and nearly finished. I saw this challenge so later and i am afraid that I did not make it in time(and can make fully converted CSV with a bit more time) Check my notebook. This notebook was written with GOOD emotions.
# P.S I deleted all print elements for good, readable, code

# In[ ]:


import os
import re
import numpy as np
from datetime import datetime
import pandas as pd


# In[ ]:


#We need to create a specific start for the future table.
headerslist =[]
bulletin_dir = '../input/cityofla/CityofLA/Job Bulletins/'
bulletinlist = []
bulletinsnamelist = []
for filename in os.listdir(bulletin_dir):
    bulletinsnamelist.append(filename)
    with open(bulletin_dir + "/" + filename, 'r', errors = 'ignore') as file:
        file = file.read()
        headersdata = file.replace('\t','').split('\n')
        bulletindata = file.replace('\t','').replace('\n','')
        headersdata = [head for head in headersdata if head.isupper()]
        headerslist.append(headersdata)
        bulletinlist.append(bulletindata)


# In[ ]:


#compile regular expressions and creating table 
date = re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')
plaintextDF=pd.DataFrame(columns=['File Name','Position','salary_start','salary_end','opendate','requirements','duties','deadline'])


# In[ ]:


sal = re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')
req=re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')
opendate = re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')
badbulletins = []
#creating first dateframe with regular expressions (this datframe can be create with a lot of ways and this way is not better:)
for bulletin in bulletinlist:
    try:
        iterator = bulletinlist.index(bulletin)
        salary = re.search(sal,bulletin)
        date=datetime.strptime(re.search(opendate,bulletin).group(3),'%m-%d-%y')
        try:
            requirement=re.search(req,bulletin).group(2)
        except Exception as e:
            requirement=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',bulletin)[0][1][:1200]).group(1)
        duties=re.search(r'(DUTIES)(.*)(REQ[A-Z])',bulletin).group(2)
        
        try:
            enddate=re.search(r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})',bulletin).group()
        except Exception as e:
            enddate=np.nan
        
        selection= [z[0] for z in re.findall(r'([A-Z][a-z]+)((\s\.\s)+)',bulletin)]
        plaintextDF=plaintextDF.append({'File Name':bulletinsnamelist[iterator],'Position':headerslist[iterator][0].lower(),'salary_start':salary.group(1),'salary_end':salary.group(5),"opendate":date,"requirements":requirement,'duties':duties,'deadline':enddate,'selection':selection},ignore_index=True)
        reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\s(years?)\s(of\sfull(-|\s)time)')
        
        plaintextDF['EXPERIENCE_LENGTH']=plaintextDF['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None else np.nan)
        plaintextDF['FULL_TIME_PART_TIME']=plaintextDF['EXPERIENCE_LENGTH'].apply(lambda x:  'FULL_TIME' if x is not np.nan else np.nan )
        reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\s|-)(years?)\s(college)')
        plaintextDF['EDUCATION_YEARS']=plaintextDF['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)
        plaintextDF['SCHOOL_TYPE']=plaintextDF['EDUCATION_YEARS'].apply(lambda x : 'College or University' if x is not np.nan else np.nan)
    except Exception as e:
        badbulletins.append(bulletinsnamelist[bulletinlist.index(bulletin)])
print(badbulletins)#we get list of bad bulletins filename and can create own parser for bad bulletins or insert info about bad bulletins with hands


# In[ ]:


plaintextDF.head() #see result


# In[ ]:


#see basic criterias to csv and need to rename
plaintextDF.rename(columns={"File Name":"FILE_NAME","Position":"JOB_CLASS_TITLE"}, inplace =True)
#create class code column
plaintextDF.insert(2,"JOB_CLASS_NO","")


# In[ ]:


#it is important because in future we will use class code for join rows
#getting class code
text = plaintextDF["FILE_NAME"]
iterator = 0
for i in text:
    textlist = text[iterator].split(' ')
    for element in textlist:
        if element.isdigit():
            if not ".txt" in element:
                if len(element) == 4:
                    plaintextDF.loc[iterator,"JOB_CLASS_NO"] = element
                    break
    iterator+=1


# In[ ]:


#we parse bad class code
from nltk.tokenize import word_tokenize
rightclasscode = []
filenames = []
for i in plaintextDF["FILE_NAME"][plaintextDF["JOB_CLASS_NO"] == ""].head(660):
    filenames.append(i)
    with open(bulletin_dir+i, 'r') as f:
        f = f.readlines()
        for l in f:
            l = word_tokenize(l)
            eliter = 0
            for el in l:
                if len(el) == 4 and el.isdigit() and l[eliter-1] == ":":
                    rightclasscode.append(el)
                eliter+=1
print(rightclasscode)
print(filenames)
for i in filenames:
    plaintextDF.loc[plaintextDF["FILE_NAME"] == i,"JOB_CLASS_NO"] = rightclasscode[filenames.index(i)]


# In[ ]:


plaintextDF#see our created class code


# In[ ]:


plaintextDF.insert(14,"OPEN_DATE","")


# In[ ]:


plaintextDF['OPEN_DATE'] = plaintextDF['opendate'].dt.strftime('%m/%d/%Y')#parse date,meeting the criterias


# In[ ]:


plaintextDF.head()#check our date column


# In[ ]:


plaintextDF.insert(12,"ENTRY_SALARY_GEN","")
#creating column for salary (main salary)


# In[ ]:


#parse our salary 
text = plaintextDF["salary_start"]
iterator = 0
for i in text:
    if plaintextDF.loc[iterator,"salary_end"] not in ["",None]:
        element = i.replace(",","") + "-" + plaintextDF.loc[iterator,"salary_end"].replace("$","").replace(",","")
    plaintextDF.loc[iterator,"ENTRY_SALARY_GEN"] = element
    iterator+=1


# In[ ]:


plaintextDF.head()


# In[ ]:


plaintextDF = plaintextDF.drop(columns = ["salary_start","opendate","salary_end"],)#columns which we must delete for meeting the criterias


# In[ ]:


plaintextDF = plaintextDF.drop(columns = ["deadline"])


# In[ ]:


plaintextDF.shape #see number of columns


# In[ ]:


plaintextDF


# In[ ]:


plaintextDF.rename(columns={"duties":"JOB_DUTIES"}, inplace =True)# name of columns is not matching with criterias and we need rename it
plaintextDF


# In[ ]:


plaintextDF.shape


# In[ ]:


data = plaintextDF["EXPERIENCE_LENGTH"].head(660)#pars experience length from plain text to numbers
iterator = 0
convertdict ={
    "ONE":1,"TWO":2,"THREE":3,"FOUR":4,"FIVE":5,"SIX":6,"SEVEN":7,"EIGHT":8,"NINE":9,"TEN":10
}
for i in data:
    if str(i).upper() in convertdict.keys():
        rightval = convertdict[i.upper()]
        plaintextDF.loc[iterator,"EXPERIENCE_LENGTH"] = rightval
    iterator+=1


# In[ ]:


plaintextDF


# In[ ]:


data = plaintextDF["EDUCATION_YEARS"].head(660)#create parser for education years finding
iterator = 0
convertdict ={
    "ONE":1,"TWO":2,"THREE":3,"FOUR":4,"FIVE":5,"SIX":6,"SEVEN":7,"EIGHT":8,"NINE":9,"TEN":10
}
for i in data:
    if str(i).upper() in convertdict.keys():
        rightval = convertdict[i.upper()]
        plaintextDF.loc[iterator,"EDUCATION_YEARS"] = rightval
    iterator+=1


# In[ ]:


plaintextDF


# In[ ]:


plaintextDF.insert(12,"ENTRY_SALARY_DWP","")# insert column with additional salary


# In[ ]:



from nltk.tokenize import word_tokenize
checkiter = 0
bulletinlist = []
for bulletinname in plaintextDF["FILE_NAME"]:
    bulletinsnamelist.append(bulletinname)
    print(bulletinname)
    try:
        with open(bulletin_dir +bulletinname, 'r', errors = 'ignore')as file:
            file = file.readlines()
            bulletinlist.append(file)
    except TypeError:
        pass
def parsefromline(line):
    tokenlist = word_tokenize(line)
    for element in tokenlist:
        if element == "$":
            first  = tokenlist[tokenlist.index(element)+1]
            second = ""
            try:
                if tokenlist[tokenlist.index(element) + 2] == 'TO':
                    second = tokenlist[tokenlist.index(element) + 4]
                    return (first.replace(",","") + "-" + second.replace(",",""))
            except IndexError:
                return first.replace(",","")
            return first.replace(",","")
iterator = 0
for i in bulletinlist:
    for line in i:
        line = line.upper()
        if "SALARY" in line:
            linetokens = word_tokenize(line)
            for token in linetokens:
                if token == '$':
                    #print(line)
                    coolline = parsefromline(line)
                    #print(coolline)
                    if coolline == '70908-88092':
                        print("match: ",iterator)
                    plaintextDF.loc[iterator,"ENTRY_SALARY_DWP"] = coolline
                    break
        else:continue
    iterator = iterator + 1


# In[ ]:


plaintextDF.head()


# In[ ]:


plaintextDF.insert(9,"EXAM_TYPE","")


# In[ ]:


#parsing exam type
bulletiniterator = 0 
for bulletin in bulletinlist:
    lineiterator = 0
    examination = ""
    for line in bulletin:
        line = line.upper()
        if ("EXAM OPEN TO ALL" in line) and examination =="":
            examination = "OPEN"
        if ("EXAMINATION IS" and "GIVEN") in line:
            if "INTERDEPARTMENTAL PROMOTIONAL" in bulletin[lineiterator+1].upper():
                if (examination =="OPEN") or ("OPEN" in bulletin[lineiterator+1].upper()):
                    examination = "OPEN_INT_PROM"
                else:
                    examination = "INT_DEPT_PROM"
            elif "DEPARTMENTAL PROMOTIONAL" in bulletin[lineiterator+1].upper():
                examination = "DEPT_PROM"
            elif "OPEN" in bulletin[lineiterator+1].upper():
                examination = "OPEN"
        lineiterator +=1
    if examination == "":
        print("ALARM")#i found the bug in the job bulletin and change "INTERDEPARMENTAL" to "INTERDEPARTMENTAL" with my hands:)
        break
    plaintextDF.loc[bulletiniterator,"EXAM_TYPE"] = examination
    bulletiniterator +=1


# In[ ]:


plaintextDF.head(660)


# as i saw where i analyzed systems analyst file and template export i saw that most of oher information base on requirements.(And saome columns which we can analyze later)

# In[ ]:


plaintextDF = plaintextDF.drop(columns = ["selection"])


# In[ ]:


plaintextDF


# In[ ]:


plaintextDF.insert(4,"REQUIREMENT_SET_ID","")
plaintextDF.insert(4,"REQUIREMENT_SUBSET_ID","")


# we need to parse requirement set and subset and copy requirements then parse requirement

# In[ ]:


#for easy understanding code i created function that add subset to fullsubset without repeating
def addfullsubset(fullsubset, subset):
    subsetiter = 1
    for l in string.ascii_uppercase:
        if subsetiter == subset:
            if fullsubset[-1:] == "A" and l == "A":
                return fullsubset
            fullsubset+= l 
            return fullsubset
        subsetiter+=1

import string
DF=pd.DataFrame(columns=["REQUIREMENT_SET_ID","REQUIREMENT_SUBSET_ID"])
#we will created fullsubset like 1AB2A3AB then will get from it subset and set requirement
#generator main requirement
requirementiterator = 0
for requirementtext in plaintextDF["requirements"]:
    tokenizedrequirement = word_tokenize(requirementtext)
    #finding 1. 2. and other
    tokenelementiterator = 0
    mainrequirement = 1
    newdfiterator = 0
    newdfiterator+=1
    subset = 0
    setlist = []
    checkstatus = False
    fullsubset = ""
    for element in tokenizedrequirement:
        #check requirement have any main requirement
        if element.isdigit()  and (tokenizedrequirement[tokenelementiterator+1] == "." or "." in element) and (element == 1 or element == "1" or "1." in element) and tokenelementiterator < 3:
            checkstatus = True
            fullsubset = "1A"#default subset
        if checkstatus == True:
            try:
                #parser subset requiremnet
                if (((len(element.replace("or","")) == 1) and ((element.replace("or","") in string.ascii_lowercase) or (element.replace("or","") in string.ascii_uppercase)) ) or ((len(element.replace("and","")) in [1,2]) and (element.replace("and","") in string.ascii_lowercase or element.replace("and","") in string.ascii_uppercase))) and (tokenizedrequirement[tokenelementiterator+1] in [".",")"] or element in [".",")"]):
                    subset+=1
                    fullsubset = addfullsubset(fullsubset,subset)
                elif len(element.replace(".","")) == 1 and ((element.replace(".","") in string.ascii_lowercase) or (element.replace(".","") in string.ascii_uppercase)) and "." in element:
                    subset+=1
                    fullsubset = addfullsubset(fullsubset,subset)
                elif "." in element:
                    for w in element.split("."):
                        if len(w) == 1 and w in "abcd":
                            subset+=1
                            fullsubset = addfullsubset(fullsubset,subset)
                #we can append elif if something was missed
            except IndexError as E:#just in case
                print(element)
        if element.isdigit() and ((tokenizedrequirement[tokenelementiterator+1] == ".") or "." in element) and (element != 1) and (element != "1") and (int(element)>mainrequirement):          
            mainrequirement+=1
            fullsubset += str(mainrequirement) + "A"
            subset = 0
            setlist.append(tokenizedrequirement.index(element))
        if any(word in element for word in ["or","and","OR","AND"]):
            for i in element:
                if i.isdigit() and ((tokenizedrequirement[tokenelementiterator+1] == ".") or "." in element) and (i != 1) and (i != "1"):
                    setlist.append(tokenizedrequirement.index(element))
                    mainrequirement+=1
                    fullsubset += str(mainrequirement) + "A"
                    subset = 0
        tokenelementiterator+=1
    if checkstatus == False:
        fullsubset = "1A"
    DF = DF.append({"REQUIREMENT_SET_ID":mainrequirement,"REQUIREMENT_SUBSET_ID":fullsubset},ignore_index = True)
    requirementiterator+=1
plaintextDF.update(DF)


# In[ ]:


plaintextDF


# In[ ]:


#forms for appending
#just copy rows for subset requirement
#we read fullsubset and append
newDF = pd.DataFrame(columns = plaintextDF.columns)
subset = ""
for requirementiterator in  range(plaintextDF.shape[0]):
    if plaintextDF["REQUIREMENT_SUBSET_ID"][requirementiterator] == "1A":
        #change our data for further append right data
        plaintextDF.loc[requirementiterator,"REQUIREMENT_SUBSET_ID"]  = "A"
        plaintextDF.loc[requirementiterator,"REQUIREMENT_SET_ID"] = 1
        newDF = newDF.append(plaintextDF.iloc[requirementiterator],ignore_index = True)
        continue
    else:
        subset = plaintextDF["REQUIREMENT_SUBSET_ID"][requirementiterator]
        mainiter = 1
        for i in subset:
            if i.isdigit() and i != "1":
                mainiter +=1
            elif i.isdigit() and i == "1":
                continue
            elif i != "A" and not i.isdigit():
                #change our data for further append right data
                plaintextDF.loc[requirementiterator,"REQUIREMENT_SUBSET_ID"]  = i
                plaintextDF.loc[requirementiterator,"REQUIREMENT_SET_ID"] = mainiter
                #append subset and set requirement to new dataframe
                newDF = newDF.append(plaintextDF.iloc[requirementiterator],ignore_index = True)
            elif i == "A" and not i.isdigit():
                #change our data for further append right data
                plaintextDF.loc[requirementiterator,"REQUIREMENT_SUBSET_ID"]  = "A"
                plaintextDF.loc[requirementiterator,"REQUIREMENT_SET_ID"] = mainiter
                #append subset and set requirement to new dataframe
                newDF = newDF.append(plaintextDF.iloc[requirementiterator],ignore_index = True)
newDF.shape


# In[ ]:


newDF


# we created subset requirement and can get from "requirements" other columns which needeed

# In[ ]:


newDF.insert(8,"EDUCATION_MAJOR","")


# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords # for cleaning
from nltk import pos_tag # for underatnding type of words 
requirementiter = 0
badrequirement = []
for i in newDF["requirements"]:
    if str(newDF["REQUIREMENT_SET_ID"][requirementiter]) == "1":
        try:
            endindex = i.index((str(int(newDF["REQUIREMENT_SET_ID"][requirementiter]) + 1) + "."))
            startindex = 1
        except:
            startindex = 0
            endindex = -1
    else:
        try:
            startindex = i.index((str(newDF["REQUIREMENT_SET_ID"][requirementiter])+"."))
        except:
            badrequirement.append(requirementiter)
        try:
            endindex = i.index((str(int((newDF["REQUIREMENT_SET_ID"][requirementiter])) + 1) + "."))
        except Exception as e:
            endindex = -1
    substring = i[startindex:endindex]
    if endindex == 0 and startindex == 0:
        substring = i
    elif endindex == 0:
        substring = i[startindex:]
    majoredu = ""
    newstring =  ""
    if "major" in substring:
        try:
            newstring = substring[(substring.index("in")+2):]
        except:
            pass
    #newDF.loc[requirementiter,"EDUCATION_MAJOR"]
    if newstring == "":
        requirementiter+=1
        continue
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(newstring)
    wordsFiltered = []
    for w in words:
        if w not in stopWords and (len(w) > 1 or w in ",/\orand"):
            wordsFiltered.append(w)
    wordsfiltered = []# do not confuse with wordsFiltered
    for w in wordsFiltered:
        if w == ";":
            break
        wordsfiltered.append(w)
    for w in wordsfiltered:
        if w == ",":
            majoredu += " |"
            continue
        majoredu  = majoredu + " " + w.upper()
    rightedu = ""
    for el in majoredu.split("|"):
        goodedu = True
        for i in pos_tag(word_tokenize(el.lower())):
            if i[1] in ["RB","VBD","MD","VBZ","RP","JJS","IN","POS","VBP","CD"]:#type of words that will be missing
                goodedu = False
                break
            else:
                goodedu = True
                continue
        if goodedu:
            print(el)
            rightedu+="|"+el
    rightedu = rightedu[1:]
    newDF.loc[requirementiter,"EDUCATION_MAJOR"] = rightedu
    requirementiter+=1


# In[ ]:


newDF.insert(9,"EXP_JOB_CLASS_FUNCTION","")


# In[ ]:


#we use try except but we could use if, else with check like if (str(newDF["REQUIREMENT_SET_ID"][requirementiter])+".") in i 
requirementiter = 0
for i in newDF["requirements"]:
    if str(newDF["REQUIREMENT_SET_ID"][requirementiter]) == "1":
        try:
            endindex = i.index((str(int(newDF["REQUIREMENT_SET_ID"][requirementiter]) + 1) + "."))
            startindex = 1
        except:
            startindex = 0
            endindex = -1#end line in array iteration
    else:
        try:
            startindex = i.index((str(newDF["REQUIREMENT_SET_ID"][requirementiter])+"."))
        except:
            badrequirement.append(requirementiter)
        try:
            endindex = i.index((str(int((newDF["REQUIREMENT_SET_ID"][requirementiter])) + 1) + "."))
        except Exception as e:
            endindex = -1 #end line in array iteration
    substring = i[startindex:endindex]
    newstring= substring
    startindex = 0
    endindex = -1
    try:
        startindex = newstring.index("experience in") + 13
    except:
        requirementiter+=1
        continue
    substring = newstring[startindex:endindex]
    data = word_tokenize(substring)
    #print(data)
    stopwords = [(newDF["REQUIREMENT_SUBSET_ID"][requirementiter].lower() + "."),(newDF["REQUIREMENT_SUBSET_ID"][requirementiter].lower() + ")")]#we create delimiter from which we will get substring
    try:
        nextstopwords = [(newDF["REQUIREMENT_SUBSET_ID"][requirementiter+1].lower() + "."),(newDF["REQUIREMENT_SUBSET_ID"][requirementiter+1].lower() + ")")]#we create next delimiter from which we will get substring
    except:
        pass
    if any(word in substring.lower() for word in stopwords):
        for word in stopwords:
            if word in substring:
                substring = substring[substring.index(word)+2:]
    if any(word in substring.lower() for word in nextstopwords):
        for word in nextstopwords:
            if word in substring:
                substring = substring[:substring.index(word)]
    if ";" in substring:
        substring = substring[:substring.index(";")]
    newDF.loc[requirementiter,"EXP_JOB_CLASS_FUNCTION"] = substring
    requirementiter+=1


# In[ ]:


newDF[newDF["JOB_CLASS_NO"] == "1596"]#check our export template bulletin


# In[ ]:



newDF.insert(10,"COURSE_COUNT","")


# In[ ]:


requirementiter = 0
for i in newDF["requirements"]:
    if str(newDF["REQUIREMENT_SET_ID"][requirementiter]) == "1":
        try:
            endindex = i.index((str(int(newDF["REQUIREMENT_SET_ID"][requirementiter]) + 1) + "."))
            startindex = 1
        except:
            startindex = 0
            endindex = -1
    else:
        try:
            startindex = i.index((str(newDF["REQUIREMENT_SET_ID"][requirementiter])+"."))
        except:
            badrequirement.append(requirementiter)
        try:
            endindex = i.index((str(int((newDF["REQUIREMENT_SET_ID"][requirementiter])) + 1) + "."))
        except Exception as e:
            endindex = -1
    substring = i[startindex:endindex]
    coursecount = ""
    if "courses" in word_tokenize(substring):
        indexer = word_tokenize(substring).index("courses")
        for i in range(1,5):
            if pos_tag(word_tokenize(substring))[indexer-i][1] == "CD":
                coursecount = pos_tag(word_tokenize(substring))[indexer-i][0]
    elif "course" in word_tokenize(substring):
        indexer = word_tokenize(substring).index("course")
        for i in range(1,3):
            if pos_tag(word_tokenize(substring))[indexer-i][1] == "CD":
                coursecount = pos_tag(word_tokenize(substring))[indexer-i][0]
    numberlist = ["one","two","three","four"]
    if coursecount in numberlist:
        coursecount = numberlist.index(coursecount)+1
    elif coursecount.isdigit() and int(coursecount)<8:
        coursecount = coursecount
    else:
        coursecount = ""
    newDF.loc[requirementiter,"COURSE_COUNT"] = coursecount
    requirementiter +=1


# In[ ]:


newDF[newDF["JOB_CLASS_NO"] == "1596"]#see our result


# In[ ]:


#and get our CSV
newDF.to_csv()


# Thank you for watching through to the end. Hope my notebook brought help to the city of Los Angeles. And if it's not difficult, you can help me decide whether to finish the solution for all 26 columns, because the deadline has already passed. All questions: sachakas03@gmail.com
