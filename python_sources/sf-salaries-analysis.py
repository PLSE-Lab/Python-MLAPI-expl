#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # A rapid analysis of salaries in San Fransisco 
# #### This data contains the names, job title, and compensation for San Francisco city employees on an annual basis from 2011 to 2014.
# 
# ## Questions to be answered are:
# ###1. In what professions are the internal pay gaps the largest and how did it change over time?
# ###2. 
#SF salaries analysis

#Jonatan H. Bergqvist

import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import matplotlib.pyplot as plt
import nltk as nl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# ### Let's explore the features!

salaries = pd.read_csv('../input/Salaries.csv')
salaries.info()


# ###Now let's do some basic pre-processing

# In[ ]:


salaries = salaries.drop(['Notes', 'Agency', 'Status', 'Id'], axis=1);
#Remove unimportant columns (empty or all rows containing the same information or uninteresting)
salaries = salaries[salaries['JobTitle']!='Not provided']
salaries.JobTitle = salaries.JobTitle.str.lower()
salaries = salaries[salaries['TotalPay']>0.00]
#Clean up JobTitles and TotalPay columns


# In[ ]:


# Let's look at the different professions and try to classify them into departments
#Split into different years not to get duplicate persons
sal2011 = salaries[salaries['Year']==2011]
sal2012 = salaries[salaries['Year']==2012]
sal2013 = salaries[salaries['Year']==2013]
sal2014 = salaries[salaries['Year']==2014]

jobTitlesByYear = pd.DataFrame({'2011': sal2011.JobTitle.value_counts(),
                            '2012': sal2012.JobTitle.value_counts(),
                            '2013': sal2013.JobTitle.value_counts(),
                            '2014': sal2014.JobTitle.value_counts(),})
jobTitlesByYear.describe()


# In[ ]:


jobTitlesByYear.sort_values("2011",axis=0, ascending=False)[:30].plot(kind='bar')


# In[ ]:



jobnames=salaries['JobTitle']
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
wordsinjobs=""
for word in jobnames:
    wordsinjobs=wordsinjobs+word
tokens=tokenizer.tokenize(wordsinjobs)
vectorizer=CountVectorizer(tokens)
dtm=vectorizer.fit_transform(salaries['JobTitle'])

jobwords = salaries.JobTitle.str.split(r'[ -]',expand=True)
#jobwords


# In[ ]:


jobnamesByYear=pd.DataFrame([sal2011['JobTitle'], sal2012['JobTitle'], sal2013['JobTitle'], sal2014['JobTitle']], index=[2011, 2012,2013,2014])


# In[ ]:


jobnamesByYear = jobnamesByYear.T


# In[ ]:


jobnamesByYear


# ###It looks like there are many titles that are common, "transit operator" being the most common one. 
# 
# ### Now we want to classify each job title according into a department, but without having to do it manually. 
# ### Let's use some unsupervised classification!

# In[ ]:


#Feature extraction
Cvect=CountVectorizer(max_df=0.95, min_df=2, max_features=30,
                                stop_words='english');
X = Cvect.fit_transform(salaries.JobTitle)


# In[ ]:


def professional_features(listOfWords):
    if ('department' in listOfWords):
        return listOfWords[listOfWords.index('department')-1]
    elif 
        
    


# In[ ]:



jobwords 


# In[ ]:



jobwords


# In[ ]:


'iii' in jobwords


# In[ ]:




