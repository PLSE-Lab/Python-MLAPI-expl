#!/usr/bin/env python
# coding: utf-8

# ## The goal of this project was to first extract all the "real" words from all the job decriptions in the Kaggle Challenge (link below) and then to look up the level of difficulty of the words to determine if language could be a barrier to diversity in hiring.
# ### https://www.kaggle.com/c/data-science-for-good-city-of-los-angeles/overview?utm_medium=email&utm_source=intercom&utm_campaign=data-science-for-good-2019.  

# Note:  While this code worked successfully in my environment it required updates to run in Kaggle from not using the Kaggle API to replacing Unirest with requests (which would not install) to finding a way to not leverage Textract (which also would not install).

# ## Part 1: Get the data from Kaggle and tokenize

# Set up libaries

# In[ ]:


# Standard imports and imports for creating a .csv
import numpy as np
import os
import csv
import urllib
import json 
import zipfile
import pandas as pd

# Import libraries for parsing
import PyPDF2 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import fnmatch


# Read files and write to .csv and extract word parts

# ### Read the files and extract all the words 
# ##### Reference article: https://medium.com/@rqaiserr/how-to-convert-pdfs-into-searchable-key-words-with-python-85aab86c544f

# Removed function to see if that would help with output issues

# In[ ]:


path = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/PDFs'
filesread = 0


files = [os.path.join(dirpath, filename)
for dirpath, dirnames, files in os.walk(path)
for filename in files if filename.endswith('.pdf')]

filecount=len(files)
joblist = []

while filesread <  filecount:
    for file in files:
            fileerror = 0
            filegood = 0
            namesonly = file.split("/")
            arraylen=len(namesonly)
            try:
                fileyear = namesonly[8]
                fileonly = namesonly[arraylen-1]
                jobdata = (fileonly, fileyear)
                joblist.append(jobdata)
                filegood +=1
            except:
                fileerror +=1

    filesread +=1

joblist_df = pd.DataFrame(joblist, columns =['Job Description', 'Year']) 
joblist_df.to_csv('job_descriptions.csv' , index=False)


# In[ ]:


print(os.listdir("."))


# In[ ]:


pdfFileObj = open(file,'rb')               ## open the file
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)   ## read the data in the file
num_pages = pdfReader.numPages                 ## figure out the number of pages
count = 0
text = ""

while count < num_pages:                       ## get a page
    pageObj = pdfReader.getPage(count)          
    count +=1
    text += pageObj.extractText()              ## extract the data from the page
    if text != "":
            text = text
                                    
    else:
            text = textract.process(fileurl, method='tesseract', language='eng')
                            
    global tokens
    tokens = word_tokenize(text)
    textlen = len(text) 
                    
print (filesread, "files read")
print ("words in file: ",textlen)


# Specify words to exclude from level of difficulty tagging

# In[ ]:


punctuations = ['(',')',';',':','[',']',',', "$", "%", "/"]
web_characters =[ "http//", "http", ".", ['.'], "-", "www", "//" "https"]
stop_words = stopwords.words('english')


# Create a process to extract keywords by exclusion and ensure the words are unique

# In[ ]:


def get_key_words():  
    
    keywords = [word for word in tokens if not word in stop_words                 and not word in punctuations and not word in web_characters]

    keywordct = len(keywords)
    print("keyword count", keywordct)


    unique_keywords = set(keywords) 
    uniquect = len(unique_keywords)
    print("unique count", uniquect)


    wordread = 0
    global goodwords
    goodwords=[]

    while wordread < keywordct:
            wordread +=1
            
    goodwords = [word.lower() for word in unique_keywords if word.isalpha()]

    global goodwordct
    goodwordct = len(goodwords)
    print("good words", goodwordct) 


# Run the keyword extraction process

# In[ ]:


get_key_words()


# ## Part 2: Detemine level of difficulty 

# Look up each word for level of difficulty (Twinword API) and write words and difficulty levels to a file

# Iterate through words looking up the level of difficulty and write to list.   Code replaced as Unirest could not be installed n Kaggle.

# In[ ]:


import requests

index = 0
wordcode = 0 
# goodwordct=
dictlist = []
dropword = 0
twinword_key = "null"

while wordcode < goodwordct:
    myword = goodwords[index]
    wordcode +=1
    index +=1    

    response = requests.get("https://twinword-language-scoring.p.rapidapi.com/word/",
            headers={
                "X-RapidAPI-Host": "twinword-language-scoring.p.rapidapi.com",
                "X-RapidAPI-Key": twinword_key,
                "Content-Type": "application/x-www-form-urlencoded"
                       },
                params={
                  "entry": myword
                }
        )
    
    if response.status_code == 200:
        try:
            data =response.json()
            worddiff = data.get("ten_degree")
            if type(worddiff) == int:
                listdata = (worddiff, myword)
                dictlist.append(listdata)
            else:
                print ("Word Difficulty not found for: ", myword)
        except:
            dropword +=1
        
print ("words dropped", dropword)


# Create a process to sort and output the data to a .csv file with the most difficult words at the top.

# In[ ]:


def write_to_csv():
    dict_list_sorted = sorted(dictlist, key=None, reverse=True)
    output_dataframe = pd.DataFrame(dict_list_sorted,columns=["Word Difficulty",'Word']) 
    csv_outfile_name = 'word_difficulty.csv'
    output_dataframe.to_csv(csv_outfile_name , index=False)
    


# Run the process to create the .csv

# In[ ]:


write_to_csv()


# ## Part 3: Gender Bias
# Referencing this research article and acknowledging this tool: http://gender-decoder.katmatfield.com/about#masculine, 
# look up words for contains male or femal bias words in the job description.

# Import gender bias words from .csv

# In[ ]:


Male_Bias_csv = "../input/bais-word-lists/Female-Bias-wordparts.csv"
with open(Male_Bias_csv, 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    male_bias_words = list(reader)
Female_Bias_csv = "../input/bais-word-lists/Female-Bias-wordparts.csv"
with open(Female_Bias_csv, 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    female_bias_words = list(reader)


# #### Compare words to list of known gender bias words and write output to files. 

# Transform to lists for matching

# In[ ]:


dict_list_sorted = sorted(dictlist, key=None, reverse=True)
gender_clean_dataframe = pd.DataFrame(dict_list_sorted)
gender_clean_dataframe.columns = ["Difficulty","Word"]
gender_df =  gender_clean_dataframe["Word"]
gender_list = gender_df.values.tolist()


# Process for Male bias words and write matches to file

# In[ ]:


male_matchbias = []
try:
    male_matchbias = [s for s in gender_list if any(xs in s for xs in male_bias_words)]
except:
    pass

if (len(male_matchbias)) == 0:
    print ("No male bias words were found")
else:
    print("Male bias words were found and exported")
    male_bias_out= pd.DataFrame(male_matchbias)
    male_bias_out.columns = ["Male Bias Words"]
    csv_outfile_name = 'Male Bias Words.csv'
    male_bias_out.to_csv(csv_outfile_name , index=False, header =False)


# Process for Female bias words and write matches to file

# In[ ]:


female_matchbias = []
try:
    female_matchbias = [s for s in gender_list if any(xs in s for xs in female_bias_words)]
except:
    pass


if (len(female_matchbias)) == 0:
    print ("No female bias words were found")
else:
    print("female bias words were found and exported")
    female_bias_out= pd.DataFrame(female_matchbias)
    female_bias_out.columns = ["female Bias Words"]
    csv_outfile_name = 'female Bias Words.csv'
    female_bias_out.to_csv(csv_outfile_name , index=False, header =False)


# ## Part 4: Age Bias

# Finally as this article points out you need to look for age bias.  
# https://www.linkedin.com/in/alison-doucette-5b40374/

# Read list of age bias words from .csv (list compiles from articles)

# In[ ]:


Age_Bias_csv = "../input/bais-word-lists/age-bias-wordparts.csv"
with open(Age_Bias_csv, 'r') as my_file:
    reader = csv.reader(my_file, delimiter='\t')
    age_bias_words = list(reader)


# Compare words for age bias and write age bias words detected to .csv

# In[ ]:


age_matchbias = []
try:
    age_matchbias = [s for s in gender_list if any(xs in s for xs in age_bias_words)]
except:
    pass


if (len(age_matchbias)) == 0:
    print ("No age bias words were found")
else:
    print("Age bias words were found and exported")
    age_bias_out= pd.DataFrame(age_matchbias)
    age_bias_out.columns = ["Age Bias Words"]
    csv_outfile_name = 'Age Bias Words.csv'
    female_bias_out.to_csv(csv_outfile_name , index=False, header =False)


# Check output.  Note:  no bias output .csv files produced if no bias is found.

# In[ ]:


print(os.listdir("."))


# In conclusion, the City of Los Angeles Job descriptions showed no gender or age bias.
# However some words which may not be essential could limit accessibklity for those for whom English is not a first language.
