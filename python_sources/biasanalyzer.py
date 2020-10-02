#!/usr/bin/env python
# coding: utf-8

# # Finding Bias in Job Bulletins
# 
# 
# <b>98% of the words used in the City of Los Angeles Job Bulletins are FREE from gender-bias and non-native speakers bias.</b>  
# 
# Job Bulletins are consistent and easy to read.  Only 12% percent of the time it appears that biased words are used.  The most common biased word was "competitive" which occurs 1293 times.  
# 
# ### Gender bias research
# 
# Gender-bias means using words to target male or female job applicants.  Gaucher, Friesen, and Kay researched gender bias in job advertisements and made a list of gender-coded words.  They published this list in "Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality" in the Journal of Personality and Social Psychology, July 2011, Vol 101(1), p109-28.  The gender-bias app finds these gender-coded words.  I customized this app to read the Los Angeles Job Bulletins and check for gender-bias.  gender-bias has three bias detectors.  I added two more.  
# 
# ### Easier to read 
# 
# "Non-native speakers" bias detector check for complex words that might discourage people who speak English as a second language.  I based this bias detector on my experience staffing project teams in 27 different countries where I was not a native speaker.  To find complex words, I modified a text analyzer from Allen Downey's book, Think Python, 2nd Edition. It read the Job Bulletins and reported a list of the most commonly used words.  It also evaluated the 50 most complex words over 10 characters long.  This script is explained in the [WordCounts](https://www.kaggle.com/nelsondata/wordcounts/edit/run/15428692) kernel.  WordCounts is a plug-in modules.  
# 
# ### Plugging-in other tools and techniques
# 
# Knowing the Kaggle community, I realize the City of Los Angeles will receive many valuable submissions.  My prototype is modular so you can plug-in these tools or mix and match to meet your needs.
# 
# ### Technology
# 
# NOTICE:  Please do not confuse any files in this kernel with the Job Bulletin CSV file requested for this competition.  The Job Bulletin CSV file is in the [Text2CSV](https://www.kaggle.com/nelsondata/text2csv) kernel.
# 
# ```
# Ubuntu 18.04 LTS
# Python
# Pandas
# Matplotlib
# GraphViz
# BigHugeThesaurus API
# WordCloud
# regex
# NLTK
# Gender-bias app
# ```

# ### Good results 
# 
# The Job Bulletins contain 1,060,919 words. However, only 8,200 different words are used. This indicates that these Job Bulletins consistently use a select group of words. The most used words are grammar words, such as "the" (used 52,261 times). Grammar words are also called "stopwords".  Looking past stopwords, the Job Bulletins focus on candidates because "candidates" was used 6,647  times and "applicants" was used 5,150 times. Many jobs require an "examination" (used 5,691 times) and experience (used 4,115 times) plus qualifications (used 3,810  times).  
# 
# Of the original 186 possible biased words, 108 words were verified as likely to be biased.  Biased words were verified by looking up the phrase where the word was used.  If the word was not found, then a false positive was reported.  Initially it appeared that biased words were used 65,134 times.  After verifying, it is more likely that biased words are used 8,001 times.  Zero false positives were found.
# 
# Only 2% of the words appear biased meaning 98% are bias free.  12% percent of the time it appears that biased words are used.
# 
# The most common biased word was "competitive" which occurs 1293 times.  Words such as "competitive" or "compete" attract male applicants more frequently. 
# 
# 1434 complex words were found meaning these words had more than 10 characters.  Using shorter words can make Job Bulletins easier to read.  For example, "examination" was used 5,691 times.  Consider using "exam" or "test".
# 
# ### Pointing out where Job Bulletins are missing data
# 
# One Job Bulletin has embedded characters that prevent it from opening properly:  POLICE COMMANDER 2251 092917.txt
# 
# These Job Bulletins do not have job class numbers in the file name:
# ```
# ELECTRIC SERVICE REPRESENTATIVE
# REFUSE COLLECTION TRUCK OPERATOR
# ELEVATOR REPAIR SUPERVISOR
# REHABILITATION CONSTRUCTION SPECIALIST
# ELECTRICAL SERVICES MANAGER
# ```
# This Job Bulletin was removed because it has no Job Class in the Job Bulletin or the file name:
# ```Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt```
# 
# ### Recommending improvements for Human Resources
# 
# Consider replacing the word 'competitive'.  For example, "open competitive basis", might be called "new-hire basis" or "open hiring basis".  Or the City of Los Angeles can make a decision to add this word to the non-bias list.

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/results.png")


# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/nonnative-2.png")


# ### Use shorter words when possible to make it easier to read Job Bulletins.

# Here are specific examples from app:  
# 
# ```
# Consider replacing 'abilities' with 'skills' or 'talents'
# Consider replacing 'accommodation' with 'aid' or 'help'
# Consider replacing 'accordance' with 'following' or 'like'
# Consider replacing 'activities' with 'tasks' or 'work'
# Consider replacing 'additional' with 'also' or 'plus'
# Consider replacing 'anticipated' with 'expected' or 'future'
# Consider replacing 'application' with 'use' or 'cure'
# Consider replacing 'appointment' with 'job' or 'date'
# Consider replacing 'available' with 'ready' or 'here'
# Consider replacing 'compensation' with 'pay' or 'salary'
# Consider replacing 'competencies' with 'skills' or 'expertise'
# Consider replacing 'competitive' with 'fair' or 'matched'
# Consider replacing 'determine' with 'choose' or 'select'
# Consider replacing 'disqualified' with 'not qualified'
# Consider replacing 'evaluation' with 'review' or 'exam'
# Consider replacing 'examination' with 'exam' or 'test'
# Consider replacing 'following' with 'after' or 'next'
# Consider replacing 'information' with 'data' or 'info'
# Consider replacing 'necessary' with 'needed' 
# Consider replacing 'opportunity' with 'chance 
# Consider replacing 'personnel' with 'staff' or 'team'
# Consider replacing 'positions' with 'jobs'
# Consider replacing 'principles' with 'rules' or 'guidelines'
# Consider replacing 'professional' with 'expert 
# Consider replacing 'promotional' with 'advanced 
# Consider replacing 'qualification' with 'making' or 'fitness'
# Consider replacing 'qualifying' with 'matching' or 'meeting'
# Consider replacing 'questionnaire' with 'form 
# Consider replacing 'requirement' with 'duty' or 'thing'
# Consider replacing 'submitted' with 'turned in' or 'given'
# Consider replacing 'sufficient' with 'enough' 
# ```

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/nonnative.png")


# Here's a list of the 20 most used complex words:
# 
# ```
# Word       Number of times used
# 
# examination       5682
# qualifications    3804
# application       3387
# applications      3025
# promotional       2765
# positions         2284
# appointment       2237
# information       2043
# accommodation     2030
# qualifying        1947
# personnel         1725
# opportunity       1472
# requirements      1368
# competitive       1298
# accordance        1282
# available         1243
# principles        1229
# following         1213
# activities        1168
# disqualified      1166
# ```

# ### Use gender-neutral terms

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/male.png")


# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/male2.png")


# ```
# These words tend to be biased towards men:
# 
# Word       Number of times used
# competitive        1298
# determined          822
# competencies        479
# analysis            256
# analyze             180
# analyst             178
# analytical          100
# leadership           88
# analyzing            55
# analyses             49
# analyzes             35
# independently        35
# determining          32
# logical              23
# confidentiality      22
# logically            19
# competency           13
# determines           12
# competent            10
# fighting              8
# ```

# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/female.png")


# In[ ]:


from IPython.display import Image
Image("../input/jobbulletindata/female2.png")


# ```
# These words tend to be biased towards women:
# 
# Word       Number of times used
# 
# responsibilities     641
# responses            178
# understanding        173
# committee            146
# responsible           97
# connection            85
# response              80
# responsibility        47
# connections           23
# agreements            23
# responsiveness        16
# committees            15
# conscientiousness     14
# family                13
# shares                12
# supporting            10
# responding             8
# children               8
# agreement              7
# ```
# 

# ## Technical details
# 
# Gender-bias is an app that uses the Natural Language Toolkit (NLTK) to find gender-biased words.  I customized it to read and evaluate the City of Los Angeles Job Bulletins.  Words were checked against a not-biased list.  This list was based on my estimates.  It can and should be customized by Human Resources.  Six hiring bias detectors were used but more can be added:
# 
# - native_speakers - detects words that could be simpler, for example, "exam" rather than "examination"
# 
# - gender_terms - detects unnecessary use of gendered pronouns such as "she" or "he"
# 	
# - effort_accomplishment - detects subtle focus towards men or women.  Studies show that women are more associated with "effort" and men with "accomplishment" 
# 	
# - personal_life - detects subtle bias towards women because women are more likely to speak about personal life
# 	
# - male_terms - detects words that attract men, such as, "competitive"
# 	
# - female_terms - detects words that attract women, such as, "interpersonal"
# 
# ### gender-bias app finds the location of biased words
# 
# Gender-bias outputs text files.  Take a look at a snippet from ```9206.txt```, based on the review of ```311 DIRECTOR  9206 041814.txt```.  The numbers in brackets show the location in the file where the word was found.
# 
# ```
# Consider replacing ' information ' with ' data '  or 'info'
# Consider replacing ' professional ' with ' expert ' 
# Consider replacing ' principles ' with ' rules '  or 'guidelines'
# Consider replacing ' opportunity ' with ' chance ' 
# Consider replacing ' requirement ' with ' duty '  or 'thing'
# 
# Unnecessary use of gender terms
#  SUMMARY: None
# 
# Terms biased towards native speakers:
#  [216-227]: information
#  [497-509]: professional
#  [662-672]: principles
#  [757-768]: opportunity
#  [813-825]: requirements
#  SUMMARY: To encourage non-native speakers, use short words and simple sentences
# 
# Terms focusing on effort vs accomplishment
#  [367-376]: efficient
#  SUMMARY: None
# 
# Terms about personal life
#  SUMMARY: None
# 
# Terms biased towards men:
#  [893-900]: analyst
#  [2876-2887]: competitive
#  SUMMARY: Depending on context, these words may be biased towards recruiting men
# 
# Terms biased towards women
#  [131-142]: responsible
#  SUMMARY: Depending on context, these words may be biased towards recruiting women
# 
# ```

# ### Pre-processed files
# 
# To save time, I pre-processed one gender-bias text file for each Job Bulletin.  I stored them in the ```/JBR_BiasWords``` folder.  If you want to pre-process files, follow this manual installation process.  First, install gender-bias by following these instructions: 
# 
# ```
# git clone https://github.com/gender-bias/gender-bias
# cd gender-bias
# pip3 install -e .
# ```
# 
# Then, go to https://github.com/NelsonPython/TextAnalyzer and copy these folders into the gender-bias folder.  You will replace the original bias detectors because I fixed a bug.  And you will add the new bias detectors.
# 
# ```
# /effort  (replace)
# /genderedwords (replace)
# /personal_life (replace)
# /malewords    (add)
# /femalewords    (add)
# /nonnativewords    (add)
# ```
# 
# Create a folder called JBR_BiasText in your home folder.  Then, run gender-bais using this command.
# 
# ```
# cat /home/YOUR_HOME_FOLDER/gender-bias/YOUR_JOB_BULLETINS_FOLDER/'311 DIRECTOR  9206 041814.txt' | genderbias > JBR_BiasText/3119.txt
# ```

# ### Finding biased words and getting synonyms from BigHugeThesaurus
# 
# First, ```biasFinder.py``` creates a Pandas DataFrame containing data from the gender-bias raw text files.  Next, the ```biasAnalyzer.py``` evaluates the data and reports statistics.  It uses the [BigHugeThesaurus](https://www.kaggle.com/nelsondata/bighugethesaurusapi-jobads) plug-in to find synonyms for complex words.  It recommends the shortest two synonyms.
# 

# ### Verifying biased words
# 
# ```biasVerifier.py``` verifies each word.  It reports the sentence where each word is found in a log and a CSV file.
# 
# BiasWords.csv has these fields:
# ```
# FILE_NAME
# JOB_CLASS
# WORD_LOCATION
# WORD
# BIAS
# SENTENCE
# ```
# 
# BiasStudyResults.txt uses this format:
# ```
# PHRASE FROM JOB BULLETIN MAY HAVE MULTIPLE LINES:
#   >>> competitive <<<  list.  However, if open competitive candidates receive a higher score, without military  ...
# VERIFIED that 'competitive' can be found at offset: 9254 for Job Class: 3734 in Job Bulletin: JobBulletins/'EQUIPMENT SPECIALIST 3734 111717 revised 11.21.txt'
# ```
# 
# #### REMEMBER:  JobBulletin.csv required for the competition is explained in [Text2CSV](https://www.kaggle.com/nelsondata/text2csv)
# 
# 

# In[ ]:


'''
Script:     biasFinder.py
Project:    City of Los Angeles Job Bulletin analysis
Purpose:    Extract data from bias terms analysis into a csv file:

CSV database row types:

JobClass   TermLocation   Term            Bias
1117       None           None            gender_terms       #example when no biased terms are found by the detector
1117       [1554-1565]    examination     native_speakers    #when possibly biased terms are found, record term location offsets, term, and detector
'''

import os
import pandas as pd
import re
import string

def process_file(jobClass, file, b, skip_language_recommendations):
    '''
    Parse bias detector files generated by gender-bias toolkit

    jobClass - 4 digit number
    file - file generated by gender-bias toolkit
    b - list for storing results
    skip_language_recommendations - gender-bias toolkit stores recommendations for synonyms in these files but this information is not used here so skip it
    
    '''
    r = []  # list for each row of terms
    bias = ''
    strippables = "!#$%&'()*+,-./;<=>?@\^_`{|}~" + string.whitespace

    f = open(file)
    for line in f:
        line = line.strip(strippables)
        if skip_language_recommendations:
            if line.startswith('Consider replacing'):
                continue
        if line.startswith('Unnecessary use of gender terms'):
                bias = 'gender_terms'
        elif line.startswith('Terms biased towards native speakers'):
                bias = 'native_speakers'
        elif line.startswith('Terms focusing on effort vs accomplishment'):
                bias = 'effort_accomplishment'            
        elif line.startswith('Terms about personal life'):
                bias = 'personal_life'
        elif line.startswith('Terms biased towards men'):
                bias = 'male_terms'
        elif line.startswith('Terms biased towards women'):
                bias = 'female_terms'

        elif 'SUMMARY: Too few words about concrete accomplishments' in line:
            r.append(jobClass)
            r.append('0-0')
            r.append('none')
            r.append(bias)
            b.append(r)
            r = []

        elif 'SUMMARY: None' in line:
            r.append(jobClass)
            r.append('0-0')
            r.append('none')
            r.append(bias)
            b.append(r)
            r = []

        elif line.startswith('['):
            terms = line.split(':')
            termLocation = terms[0].strip('[]')
            term = terms[1].strip(' ')
            if "=" in term:
                continue
            elif "/" in term:
                continue
            else:
                r.append(jobClass)
                r.append(termLocation)
                r.append(term)
                r.append(bias)
                b.append(r)
            r = []
    return b

def main():
    LAJobFiles = [os.path.join(root, file) for root, folder, LAJobFile in os.walk('../input/jobbulletindata/JBR_BiasText/') for file in LAJobFile]
    print("\n",len(LAJobFiles), "LA Job Files\n")

    b = []    
    biasAnalysis = [process_file(LAJobFile[:4], LAJobFile, b, skip_language_recommendations=True) for LAJobFile in LAJobFiles]
    ba = pd.DataFrame(biasAnalysis[0], columns=('JobClass','TermLocation','Term','Bias'))

    #print("\nSample rows")
    #print(ba.head(5))
    # TODO - consider replacing job classes with ###. with 0###
    ba.to_csv('biasTerms.csv', index=False)

    baTerms = ba['Term'].unique()
    print("\n")
    print((len(baTerms)-1), " possible biased words\n")
    #print(baTerms[1:])

if __name__ == '__main__':
    main()


# In[ ]:


'''
Script:     biasTermsAnalyzer.py
Purpose:    Evaluated possible bias terms found in City of Los Angeles Job Bulletins
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

pd.options.display.max_columns=5

def main():
    '''
    input files:

        ba             bias analysis DataFrame input from: /JBR_Output/biasTerms.csv
        simpleWords    list of English synonyms for complex terms input from /genderbias/nativespeakers/Englishwords.wordlist
        nonBiasTerms   terms specific to City of Los Angeles input from /JBR_Resources/nonBiasTerms.csv

    output files:

        fo             output file with recommendations        JBR_Output/BiasStudyRecommendations.txt       
    '''
    
    fo = open("BiasStudyRecommendations.txt", "w+")
    ba = pd.read_csv('../input/jobbulletindata/JBR_Output/biasTerms.csv', dtype='str')

    print("STATISTICS:\n", file=fo)
    print("STATISTICS:\n")

    # Number of job classes processed
    jobClasses = ba['JobClass'].unique()
    possiblyUsed = len(ba)
    a = len(jobClasses)
    print(a, " job classes evaluated\n", file=fo)
    print("Biased terms may have been used ", possiblyUsed," times\n", file=fo)
    print(a, " job classes evaluated\n")
    print("Biased terms may have been used ",possiblyUsed," times\n")

    # Bias categories (also called bias detectors)
    biasCats = ba['Bias'].unique().tolist()
    print(len(biasCats), ' bias detectors used:', file=fo)
    print(len(biasCats), ' bias detectors used:')
    print(biasCats, "\n", file=fo)
    print(biasCats, "\n")
 
    # Possible biased words found
    biasTerms = ba['Term'].unique().tolist()
    biasTerms.sort()
    possiblyBiased = len(biasTerms) - 1
    print(possiblyBiased, " unique words were found :\n ", biasTerms[1:], file=fo)
    print(possiblyBiased, " unique words were found:\n ", biasTerms[1:])


    # NATIVE LANGUAGE BIASES
    complexTermsCnt = pd.value_counts(ba['Term'].loc[ba['Bias']=='native_speakers'], ascending=True)

    # WordCloud
    wordcloud = WordCloud(max_font_size=30, max_words=20,background_color="white").generate(complexTermsCnt.sort_values(ascending=False).head(20).to_string())
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Words that can be simplifed for non-native speakers')
    plt.show()

    # Horizontal bar graph
    complexTermsCnt = complexTermsCnt.tail(20)
    complexTermsCnt.plot(figsize=(10,5), x='Count', y='Complex Term', kind='barh', color='black', title='Words that can be simplified for non-native speakers')
    plt.show()

    print("\nNATIVE LANGUAGE BIASES", file=fo)

    # Words that can be simplified for non-native speakers
    print("\nWords that can be simplifed for non-native speakers:", file=fo)
    print("\nWord       Number of times used", file=fo)
    print("\nWords that can be simplifed for non-native speakers:")
    print("\nWord       Number of times used")
    print("\n", complexTermsCnt.sort_values(ascending=False), file=fo)
    print("\n", complexTermsCnt.sort_values(ascending=False))

    # Recommend shorter synonyms
    print("\nRECOMMENDATIONS: use simple language:\n", file=fo)
    print("\nRECOMMENDATIONS: use simple language:\n")
    # lookup synonyms
    simpleWords = pd.read_csv('../input/jobbulletindata/Englishwords.wordlist')
    c = ba['Term'].loc[ba['Bias']=='native_speakers'].unique()
    complexWords = pd.DataFrame(data=c.flatten(), columns = ['Used'])
    # find the shortest two synonyms for complex words
    synonyms = pd.merge(complexWords, simpleWords, how='left')
    synonyms.dropna(subset=['Recommend1'], inplace=True)
    print(synonyms, file=fo)
    print(synonyms)
    print("\nBiasStudyResults.txt contains instructions for finding sentences where biased words are used.")

    # GENDER-BIASED TERMS
    maleTerms = pd.value_counts(ba['Term'].loc[ba['Bias']=='male_terms'], ascending=True)
    maleTerms = maleTerms.tail(20)

    # WordCloud
    wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white", colormap="Blues").generate(maleTerms.to_string())
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Top 20 Possible Male-biased Terms")
    plt.show()

    # Horizontal bar graph
    maleTerms.plot(figsize=(10,6), x='Count', y='Male Biased', kind='barh', title='Top 20 possible male biased terms')
    plt.show()
    print("\nGENDER BIASES", file=fo)
    print('\nPossible male-biased terms:', file=fo)
    print("\nWord       Number of times used", file=fo)

    print("\nGENDER BIASES")
    print('\nPossible male-biased terms:')
    print("\nWord       Number of times used")
    print(maleTerms.sort_values(ascending=False), file=fo)
    print(maleTerms.sort_values(ascending=False))
        
    femaleTerms = pd.value_counts(ba['Term'].loc[ba['Bias'].isin(['female_terms','personal_life','gender_terms','effort_accommplishment'])], ascending=True)
    femaleTerms = femaleTerms.tail(20)

    # WordCloud
    wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white", colormap="PuRd").generate(femaleTerms.to_string())
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Top 20 Possible Female-biased Terms")
    plt.axis('off')
    plt.show()

    # Horizontal bar graph
    femaleTerms[:-1].plot(figsize=(10,6), x='Count', y='Female Biased', kind='barh', color='red', title='Top 20 possible female biased terms')
    plt.show()
    print('\nPossible female-biased terms:', file=fo)
    print("\nWord       Number of times used", file=fo)
    print('\nPossible female-biased terms:')
    print("\nWord       Number of times used")
    print(femaleTerms[:-1].sort_values(ascending=False), file=fo)
    print(femaleTerms[:-1].sort_values(ascending=False))

    fo.close()
    
if __name__ == '__main__':
    main()


# In[ ]:


'''
Script:  biasVerifier
Purpose: create log and csv file with biased words

Files:

BiasStudyResults.txt
biasTerms.csv
'''
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def getJobClass(JobBulletins):
    '''
        get the Job Class number from the filename where filename is JobBulletin and Job Class is always the first 4 digits after the name and before the version and date
    '''
    jbDict = {}
    for JobBulletin in JobBulletins:
        #JobBulletin = re.sub("[/]","/'",JobBulletin)   # original folder structure.  use JobClass[1:5]
        JobBulletin = re.sub("../input/jobbulletindata/JBR_Output/", "", JobBulletin)  # kaggle folder structure us JobClass[2:6]
        JobBulletin = re.sub("txt", "txt'",JobBulletin)
        JobClass = re.sub('[a-zA-Z /]','',JobBulletin)
        jbDict[JobClass[2:6]] = JobBulletin
    return jbDict

def getWordLocation(loc):
        ''' get the beginning and ending offsets'''
        return loc.split('-')

def getNLPstats():
    # get the stats from JBR_NLP.py
    with open ("../input/jobbulletindata/JBR_Output/totWords.txt") as fs:
        return [line.strip("\n") for line in fs]

def printLog(fo,msg, fileOnly=False):
    if fileOnly:
        print(msg,file=fo)
    else:
        print(msg, file=fo)
        print(msg)

def main():

    fo = open("BiasStudyResults.txt", "w+")
    ba = pd.read_csv('../input/jobbulletindata/JBR_Output/biasTerms.csv', dtype='str')  

    # find the number of job classes processed
    jobClasses = ba['JobClass'].unique()
    possiblyUsed = len(ba)

    # get the Job Bulletins from the JobBulletin folder
    JobBulletins = [os.path.join(root, file) for root, folder, JobBulletin in os.walk('../input/jobbulletindata/JobBulletins') for file in JobBulletin]

    # get jobclass from filename
    jbDict = getJobClass(JobBulletins)
    #jb = pd.DataFrame.from_dict(jbDict.items())   # code works in Ubuntu
    #jb.columns=('JobClass', 'jbFilename')         # code works in Ubuntu
    jb = pd.DataFrame(list(jbDict.items()),columns=['JobClass','jbFilename'])    # code works on Kaggle

    printLog(fo,"VERIFYING WHETHER WORDS ARE BIASED: ")
    printLog(fo,"\nFirst, I read " + str(len(JobBulletins)) + " Job Bulletins and found words that may be biased") 

    # get a list of words that we know are not biased
    notBias = pd.read_csv('../input/jobbulletindata/JBR_Resources/nonBiasTerms.csv')
    printLog(fo,"Then, I checked these words against the not-biased list which contains " + str(len(notBias))+ " words")
    
    # get the list of possibly biased terms from gender-bias output
    b = ba['Term'].loc[ba['Bias'].isin(['female_terms','personal_life','gender_terms','effort_accommplishment', 'male_terms'])].unique()
    b = b[1:]     # remove 'none'
    possibleBias = pd.DataFrame(data=b.flatten(), columns=['Term'])

    # "subtract" the not-biased words so we have a list of words to be verified
    bias = pd.merge(possibleBias,notBias,how='left')
    printLog(fo,"After checking the not-biased list, there are " + str(len(bias)) + " words that need to be verified.  ")

    # find the location of each biased word in each Job Bulletin
    verifyWords = pd.merge(bias,ba,how='inner')
    printLog(fo,"Biased words may have been used " + str(len(verifyWords)) + " times, but more analysis is needed.")

    # find the Job Bulletins where biased words are located
    vWordsFilename = pd.merge(verifyWords, jb, how='inner')
    printLog(fo,"\nHere's a partial list:")
    pd.options.display.max_columns=5
    printLog(fo,vWordsFilename.head(10))
    
    biasedTerms = []                # for reporting a summary of biased terms
    biasedRow = []                  # for recording each row of biased terms
    kF = 0                          # count the number of files opened
    kFP = 0                         # count the number of false positives

    # look up each biased word and print out the phrase where it is used
    # open each Job Bulletin file only once and process all the biased words in that file then close it
    priorFilename = ''
    for i, row in vWordsFilename.iterrows():
        filename = re.sub('[\']','',row["jbFilename"])  # strip the single quote so the file will open
        biasedRow.append(filename)  # store the filename WITHOUT quote so it matches the filename from JBR_NLP.py.  Thus the final statistics will be computed properly
        
        # open the Job Bulletin once but verify many words
        if priorFilename != filename:
            try:
                f.close()
                priorFilename = filename
            except:
                if len(priorFilename) > 0:
                    print(priorFilename, " STILL OPEN")
            try:
                f = open(filename)
                kF +=1
            except:
                print("DOUBLE CHECK THIS FILE: ", filename)
                continue

        # get the phrase where the biased word is located
        loc = getWordLocation(row['TermLocation'])
        msg = "VERIFIED that '" + row['Term'] + "' can be found at offset: " + str(int(loc[0])) + " for Job Class: " + row['JobClass'] + " in Job Bulletin: " + row['jbFilename']

        z = 100             # read z characters after the biased word
        sentence = ''       # capture the phrase with biased word
        offset = 0          # beginning offset

        _text = f.read()
        offset = _text.find(row['Term'], offset)
        f.seek(int(loc[0]))
        for k in range(offset, offset+z):
            ch = f.read(1)
            if k == offset:
                sentence += " >>> "
            if k == offset + len(row['Term']):
                sentence += " <<< "
            sentence += ch
        row["SENTENCE"] = sentence

        # store the results
        biasedRow.append(row["JobClass"])
        biasedRow.append(row["TermLocation"])
        biasedRow.append(row["Term"])
        biasedRow.append(row["Bias"])
        biasedRow.append(row["SENTENCE"])
        biasedTerms.append(biasedRow)
        biasedRow = []
        
        printLog(fo,"\n\nPHRASE FROM JOB BULLETIN MAY HAVE MULTIPLE LINES:\n" + sentence + "...", fileOnly=True)
        printLog(fo,msg, fileOnly=True)

        # report false positives
        searchTerm = row['Term'].lower()
        falsePositive = re.search(searchTerm, sentence.lower())
        if not falsePositive:
            printLog(fo,"FALSE REPORT WARNING:  the term found doesn't match the search term", fileOnly=True)
            kFP += 1

    # REPORT TERMS LIKELY TO BE BIASED
    likelyBiased = len(biasedTerms)
    biasedDF = pd.DataFrame(biasedTerms, columns=('FILE_NAME', 'JOB_CLASS', 'WORD_LOCATION','WORD','BIAS','SENTENCE'))
    biasedDF.index.name = 'IDX'
    results = biasedDF['WORD'].unique()
    
    score = ((likelyBiased - kFP) / possiblyUsed * 100)
    totals = getNLPstats()
    percentBiased = (len(results) / int(totals[1])) * 100

    # print results
    printLog(fo,"\nWhen I began, I thought that biased words were used  " + str(possiblyUsed) + "  times.  ")
    printLog(fo,"After verifying, I believe that biased words are used  " + str(likelyBiased - kFP) + "  times")
    printLog(fo,"I verified each word by looking up the phrase where the word was used.  If the word was not found, then I reported a false positive")
    printLog(fo, str(kFP) + "  false positives were found")
    printLog(fo,"\n" + str(np.round(score,0)) + "% percent of the time biased words are used in Job Bulletins (A lower score is better.  0% is a perfect score)\n")
    printLog(fo, str(np.round(percentBiased,0)) + "% of the words are biased")
    printLog(fo,"\n" + str(len(results)) + " words that may be biased:\n" + str(results))

    # WordCloud
    wordcloud = WordCloud(max_font_size=50, max_words=100,background_color="white").generate(biasedDF['WORD'].to_string())
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Words used in Job Bulletins that discourage applicants')
    plt.show()

    # save results in a csv file
    biasedDF.to_csv("BiasedWords.csv")

if __name__ == '__main__':
    main()


# ### 108 Words that may be biased
# 
#  ['determined' 'responsibilities' 'understanding' 'determines' 'analyst'
#  'analysis' 'competitive' 'competencies' 'analytical' 'connection'
#  'shares' 'leading' 'responsible' 'responses' 'analyzing' 'guy'
#  'understandably' 'leadership' 'responsiveness' 'connecting' 'supporting'
#  'cooperation' 'independently' 'analyze' 'understandable'
#  'interdependence' 'competency' 'response' 'determinations' 'connections'
#  'her' 'logical' 'decisiveness' 'thoroughness' 'analytic'
#  'conscientiousness' 'confidentiality' 'confidential' 'analyses'
#  'analyzes' 'supportive' 'responsibility' 'connected' 'determining'
#  'sharpening' 'competent' 'committee' 'women' 'family' 'carefully'
#  'connectors' 'determination' 'shared' 'cooperative' 'fighting'
#  'responding' 'logically' 'committees' 'responsibilites' 'assertive'
#  'assertiveness' 'Family' 'decision-making' 'athletic' 'children'
#  'agreements' 'sensitivity' 'competing' 'analytics' 'decisively'
#  'sharepoint' 'leader' 'analyzed' 'sharing' 'child' 'analyzers'
#  'agreement' 'competitve' 'cooperatively' 'respondents' 'commitment'
#  'responds' 'analyzer' 'challenging' 'collaborate' 'depending' 'warming'
#  'individualized' 'leaders' 'analysts' 'sensitively' 'determinate'
#  'childcare' 'childhood' 'challenges' 'yielding' 'thoroughly' 'forced-air'
#  'competitors' 'hierarchies' 'Child' 'diligently' 'hierarchical'
#  'competence' 'cooperates' 'competition' 'independence' 'commitments']
