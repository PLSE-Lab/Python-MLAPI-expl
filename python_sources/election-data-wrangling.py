#!/usr/bin/env python
# coding: utf-8

# <h1> Data Wrangling of Electoral Data</h1>
# <h2>Get our environment set up</h2>
# The first thing we'll need to do is load in the libraries we'll be using.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import fuzzywuzzy
from fuzzywuzzy import process
import chardet

from subprocess import check_output


# <h2>Load Data</h2>
# 

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

NA2 = pd.read_csv("../input/National Assembly 2002 - Updated.csv", encoding = "ISO-8859-1")
NA8 = pd.read_csv("../input/National Assembly 2008.csv", encoding = "ISO-8859-1")
NA13 = pd.read_csv("../input/National Assembly 2013.csv", encoding = "ISO-8859-1")
print("Data Dimensions are: ", NA2.shape)
print("Data Dimensions are: ", NA8.shape)
print("Data Dimensions are: ", NA13.shape)


# <h2>Data Info</h2>
# 
# Let's look into the info of provided dataset.

# In[ ]:


print("NA 2002.csv")
NA2.info()
print("\nNA 2008.csv")
NA8.info()
print("\nNA 2013.csv")
NA13.info()


# All three files have 11 columns. Row details are as follow:
# 
# * NA 2002: 1,792
# * NA 2008: 2,315
# * NA 2013: 4,541
# 
# NA 2002.cv file has fine data types. 
# 
# In NA 2008 & NA 2013 files:
# * 1st column name is missing and showing that it have int64 data type. 
# * Turnout column has read as object.
# 
# <h2>Data Wrangling </h2>
# Lets observe the 1st file in order to fix next two and merge them all into one dataframe.

# In[ ]:


print(NA2.head())
print(NA8.head())
print(NA13.head())
print(NA8.columns, "\n>>\n", NA13.columns)


# So the first column should be District. We will extract district names from Seat column.
# We will drop last column from NA13 because it contain no value..
# 
# <b>Rename Column and Replace Values </b>

# In[ ]:



NA8.rename(columns={'Unnamed: 0':'District'}, inplace=True)
NA13.rename(columns={'Unnamed: 0':'District'}, inplace=True)
print("NA 8: ", NA8.columns, "\nNA 13: ", NA13.columns)
#NA13 = NA13.drop('Unnamed: 11', axis=1)


# In[ ]:


NA8.District = NA8.Seat#.str.split("-", expand=True)[0]
#Add District column
#NA8['District'] = NA8['Seat']
NA8['District'] = NA8['District'].str.replace("."," ") # to deal with D.I. Khan
# remove all those substring with () 
NA8['District'] = NA8['District'].str.replace(r"\(.*\)","")
# remove numeric
NA8['District']  = NA8['District'] .str.replace('[^a-zA-Z -]', '')
#NA8['District'] = NA8['District'].str.replace(r"Cum.*","")
#NA8['District'] = NA8['District'].str.replace(r"cum.*","")
#na18['District'] = na18['District'].str.replace(r"KUM.*","")
# to convert Tribal Area III - Mohman into Tribal Area III
NA8['District'] = NA8['District'].str.replace(r"-.*","")
NA8['District']  = NA8['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA8['District']  = NA8['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA8['District'].unique()


# In[ ]:


NA13.District = NA13.Seat #.str.split("-", expand=True)[0]
#Add District column
#NA13['District'] = NA13['Seat']
NA13['District'] = NA8['District'].str.replace("."," ") # to deal with D.I. Khan
# remove all those substring with () 
NA13['District'] = NA13['District'].str.replace(r"\(.*\)","")
# remove numeric
NA13['District']  = NA13['District'] .str.replace('[^a-zA-Z -]', '')
NA13['District'] = NA13['District'].str.replace(r"Cum.*","")
#na18['District'] = na18['Distirct'].str.replace(r"KUM.*","")
# to convert Tribal Area III - Mohman into Tribal Area III
NA13['District'] = NA13['District'].str.replace(r"-.*","")
NA13['District']  = NA13['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA13['District']  = NA13['District'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
NA13['District'].unique()


# In[ ]:


NA13.head()


# We are all set with first issue. Turnout column has % symbol in it which makes it object datatype. We will remove non-numeric characters and change datatype.
# 
# <b>Change the datatype of Turnout column </b>

# In[ ]:


NA8['Turnout'] = NA8['Turnout'].str.rstrip('%').str.rstrip(' ')
NA13['Turnout'] = NA13['Turnout'].str.rstrip('%').str.rstrip(' ')
NA8['Turnout'] = pd.to_numeric(NA8['Turnout'], errors='coerce')
NA13['Turnout'] = pd.to_numeric(NA13['Turnout'], errors='coerce')


# Now our dataset is aligned and ready to merge. But before merging, lets add another column  'Year'. 

# In[ ]:


NA2['Year'] = "2002"
NA8['Year'] = "2008"
NA13['Year'] = "2013"


# In[ ]:


print(NA2.head(), "\n", NA8.head(), "\n", NA13.head())


# <h2>NAN Values</h2>
# Lets check the status of NA values

# In[ ]:


print("NA2", NA2.isnull().any(), "\nNA8: ", NA8.isnull().any(), "\nNA13:", NA13.isnull().any())


# There is no null record.
# 
# Final step before merging:
# 
# Just to confirm that column names are similar in all files.

# In[ ]:


print("\n NA2", NA2.columns, "\n NA8", NA8.columns, "\n NA13", NA13.columns)


# In[ ]:


NA2.rename(columns={'Constituency_title':'ConstituencyTitle', 'Candidate_Name':'CandidateName', 'Total_Valid_Votes':'TotalValidVotes', 'Total_Rejected_Votes':'TotalRejectedVotes', 'Total_Votes':'TotalVotes', 'Total_Registered_Voters':'TotalRegisteredVoters', }, inplace=True)
NA2.columns


# <h2>Concatenate All 3 Datasets </h3>

# In[ ]:


df = pd.concat([NA2, NA8, NA13])
df.shape
df.head()


# In[ ]:


df.isnull().any()


# <h2>Some Preliminary Text Pre-processing</h2>
# Here, I'm interested in cleaning up all Text columns to make sure there's no data entry inconsistencies in it. We could go through and check each row by hand, of course, and hand-correct inconsistencies when we find them. There's a more efficient way to do this though!

# In[ ]:


# get all the unique values in the 'District' column
#df['District'] = df['District'].astype(str)
dist = df['District'].unique()
#dist.sort()
dist


# Just looking at this, We can see some problems due to inconsistent data entry: 'PESHAWAR' and Peshawar ', for example, or 'Charsadda' and 'Charsdda'.
# 
# The first thing we are going to do is make everything lower case (we can change it back at the end if need) and remove any white spaces at the beginning and end of cells. Inconsistencies in capitalizations and trailing white spaces are very common in text data and you can fix a good 80% of your text data entry inconsistencies by doing this.

# In[ ]:


# convert to lower case
df['District'] = df['District'].str.lower()
# remove trailing white spaces
df['District'] = df['District'].str.strip()


# <h2>Use fuzzy matching to correct inconsistent data entry</h2>
# Alright, let's take another look at the district column and see if there's any more data cleaning we need to do.

# In[ ]:


dist = df['District'].unique()
#dist.sort()
dist


# It does look like there are some remaining inconsistencies: 'charsadda' and 'charsdda' should probably be the same. 
# 
# We are going to use the fuzzywuzzy package to help identify which string are closest to each other. 
# > <b>Fuzzy matching:</b> The process of automatically finding text strings that are very similar to the target string. In general, a string is considered "closer" to another one the fewer characters you'd need to change if you were transforming one string into another. So "apple" and "snapple" are two changes away from each other (add "s" and "n") while "in" and "on" and one change away (rplace "i" with "o"). You won't always be able to rely on fuzzy matching 100%, but it will usually end up saving you at least a little time.
# 
# Fuzzywuzzy returns a ratio given two strings. The closer the ratio is to 100, the smaller the edit distance between the two strings. Here, we're going to get the ten strings from our list of districts that have the closest distance to "charsadda".

# In[ ]:


# get the top 10 closest matches to "charsadda"
matches = fuzzywuzzy.process.extract("charsadda", dist, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches


# We can see that two of the items in the districts are very close to "charsadda": 'charsadda; and 'charsdda'.
# 
# Let's replace all rows in our District column that have a ratio of > 90 with "charsadda".
# 
# For the reusability,  I'm going to write a function to fix all these challenges ASAP.

# In[ ]:


# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    


# To test the funtion

# In[ ]:


# use the function we just wrote to replace close matches to "charsadda" 
replace_matches_in_column(df=df, column='District', string_to_match="charsadda")


# In[ ]:


dist = df['District'].unique()
#dist.sort()
dist


# Lets fix few more

# In[ ]:


replace_matches_in_column(df=df, column='District', string_to_match="nowshera")
replace_matches_in_column(df=df, column='District', string_to_match="rawalpindi")
replace_matches_in_column(df=df, column='District', string_to_match="sheikhupura")
replace_matches_in_column(df=df, column='District', string_to_match="shikarpur")
replace_matches_in_column(df=df, column='District', string_to_match="nankana sahib")


# <h3>Lets Clean data around Party & Candidates Name </h3>

# In[ ]:


del dist

pty = df['Party'].unique()
pty.sort()
pty


# In[ ]:


df['Party'] = df['Party'].replace(['MUTTHIDA\xa0MAJLIS-E-AMAL\xa0PAKISTAN'], 'Muttahidda Majlis-e-Amal Pakistan')
df['Party'] = df['Party'].replace(['Pakistan Muslim League'], 'Pakistan Muslim League (QA)')
#converting text to lower case & removing white spaces
df['Party'] = df['Party'].str.lower()
df['Party'] = df['Party'].str.strip()


# In[ ]:


# As I coded this earlier, I wouldn't change it due to lower case letters. 
replace_matches_in_column(df=df, column='Party', string_to_match="Balochistan National Movement")
replace_matches_in_column(df=df, column='Party', string_to_match="Independent")
replace_matches_in_column(df=df, column='Party', string_to_match="Istiqlal Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Jamote Qaumi Movement")
replace_matches_in_column(df=df, column='Party', string_to_match="Labour Party Pakistan")
replace_matches_in_column(df=df, column='Party', string_to_match="Mohib-e-Wattan Nowjawan Inqilabion Ki Anjuman (MNAKA)")
replace_matches_in_column(df=df, column='Party', string_to_match="Muttahida Qaumi Movement") # Muttahida Qaumi Movement Pakistan
replace_matches_in_column(df=df, column='Party', string_to_match="Muttahidda Majlis-e-Amal") # Muttahidda Majlis-e-Amal Pakistan
replace_matches_in_column(df=df, column='Party', string_to_match="National Peoples Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Nizam-e-Mustafa Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Pak Muslim Alliance")
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Awami Party")
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Democratic Party")
# After analyzing each of the below strings.
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (QA)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (N)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (J)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Muslim League (F)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Peoples Party Parliamentarians", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Peoples Party(Shaheed Bhutto)", min_ratio =95)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Peoples Party(Sherpao)", min_ratio =97)
replace_matches_in_column(df=df, column='Party', string_to_match="Pakistan Tehreek-e-Insaf", min_ratio =95)
replace_matches_in_column(df=df, column='Party', string_to_match="Saraiki Sooba Movement Pakistan", min_ratio =95)


# In[ ]:


#fuzzywuzzy.process.extract("Pakistan Muslim League (QA)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Muslim League (N)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Muslim League (J)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Muslim League (F)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Peoples Party Parliamentarians", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Peoples Party(Shaheed Bhutto)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >95
#fuzzywuzzy.process.extract("Pakistan Peoples Party(Sherpao)", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >97
#fuzzywuzzy.process.extract("Pakistan Tehreek-e-Insaf", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >95
#fuzzywuzzy.process.extract("Saraiki Sooba Movement Pakistan", pty, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >95


# In[ ]:


df['Party'] = df['Party'].str.lower()
# few fixes taken from https://www.kaggle.com/usman786/exploratory-data-analysis-for-interesting-insights/notebook
df['Party'].replace(['muttahida qaumi movement pakistan'], 'muttahida qaumi movement', inplace = True)
df['Party'].replace(['indeindependentdente','independent (retired)','indepndent'], 'independent',inplace = True)
df['Party'].replace(['indeindependentdente','independent (retired)','indepndent'], 'independent',inplace = True)
df['Party'].replace(['muttahidda majlis-e-amal pakistan','mutthida\xa0majlis-e-amal\xa0pakistan'
                     ,'mutthida�majlis-e-amal�pakistan'] 
                     ,'muttahidda majlis-e-amal' ,inplace = True)
df['Party'].replace(['nazim-e-mistafa'], 'nizam-e-mustafa party' ,inplace = True)

pty = df['Party'].unique()
pty.sort()
pty


# In[ ]:


#del pty
#convert textual content to lower case & remove trailing white spaces
df['CandidateName'] = df['CandidateName'].str.lower()
df['CandidateName'] = df['CandidateName'].str.strip()
df['CandidateName'].head(10)


# We will remove Mr Initial from the begining of names, But we will keep Dr Initial because it is worth gaining title. 

# In[ ]:


# remove mr at the beginning of names.
df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="mr ", value="")
df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="mrs ", value="")
df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="miss ", value="")
#df['CandidateName'] = df.loc[:, 'CandidateName'].replace(regex=True, to_replace="mis ", value="")
df['CandidateName'].head(10)


# In[ ]:





# In[ ]:


cn = df['CandidateName'].unique()
cn.sort()
print("cn size: ", cn.shape, "\nValues: ", cn) 


# In[ ]:


df['CandidateName']


# Lets observe few to set the threshold for fuzzywuzzy

# In[ ]:


fuzzywuzzy.process.extract("zumurad khan", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
fuzzywuzzy.process.extract("zobaida jalal", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >79
#fuzzywuzzy.process.extract("barkat ali", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
#fuzzywuzzy.process.extract("sher muhammad baloch", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
#fuzzywuzzy.process.extract("gulab baloch", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90
#fuzzywuzzy.process.extract("babu gulab", cn, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio) # acceptance value >90


# In[ ]:


replace_matches_in_column(df=df, column='CandidateName', string_to_match="zumurad khan", min_ratio=92)
replace_matches_in_column(df=df, column='CandidateName', string_to_match="zobaida jalal", min_ratio=80)
replace_matches_in_column(df=df, column='CandidateName', string_to_match="barkat ali", min_ratio=90)
replace_matches_in_column(df=df, column='CandidateName', string_to_match="muhammad yasin baloch", min_ratio=90)

for candi in df['CandidateName'].unique(): # 7000
    replace_matches_in_column(df=df, column='CandidateName', string_to_match=candi, min_ratio=90)

# let us know the loop is completed
print("All done!")


# In[ ]:


#del NA2, NA8, NA13
df.to_csv('NA2002-18.csv', index=None) 


# <h2>Candidate List & Parties of 2018 Election</h2>
# Lets se if these 2 files need some cleaning as well. We will merge both files. 

# In[ ]:


cc = pd.read_csv("../input/National Assembly Candidates List - 2018 Updated.csv", encoding = "ISO-8859-1")
na18 = pd.read_csv("../input/2013-2018 Seat Changes in NA.csv", encoding = "ISO-8859-1") 
pp = pd.read_csv("../input/Political Parties in 2018 Elections - Updated.csv", encoding = "ISO-8859-1")
print(cc.shape, na18.shape, pp.shape)


# In[ ]:


print(cc.columns, na18.columns)


# Adding "NA-" string in NA# column to merge it with na18 dataset.

# In[ ]:


cc['NA#'] = 'NA-' + cc['NA#'].astype(str)


# In[ ]:


print(cc['NA#'].unique().shape) # 272
print(na18['2018 Seat Number'].unique().shape) # 273
na18.rename(columns={'2018 Seat Number':'NA#'}, inplace=True)
na18.rename(columns={'Seat Name':'Seat'}, inplace=True)
na18[na18['NA#'] == "Old Constituency Changed Considerably"]


# In[ ]:


na18 = na18[na18['NA#'] != "Old Constituency Changed Considerably"]
na18['NA#'] = na18.loc[:, 'NA#'].replace(regex=True, to_replace="NA-", value="")
na18['NA#'] = pd.to_numeric(na18['NA#'])
na18['NA#'] = na18['NA#'].astype(np.int64)
na18['NA#'] = 'NA-' + na18['NA#'].astype(str)
#na18['NA#'] = na18.loc[:, 'NA#'].replace(regex=True, to_replace=".0", value="")
na18['NA#'].head()


# Lets add District column and do its cleaning

# In[ ]:


#Add District column & its cleani
na18['Distirct'] = na18['Seat']
# remove all those substring with () 
na18['Distirct'] = na18['Distirct'].str.replace(r"\(.*\)","")
# remove numeric
na18['Distirct']  = na18['Distirct'] .str.replace('[^a-zA-Z -]', '')
na18['Distirct'] = na18['Distirct'].str.replace(r"Cum.*","")
#na18['Distirct'] = na18['Distirct'].str.replace(r"KUM.*","")
# to convert Tribal Area III - Mohman into Tribal Area III
na18['Distirct'] = na18['Distirct'].str.replace(r"-.*","")
na18['Distirct']  = na18['Distirct'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
na18['Distirct']  = na18['Distirct'] .str.replace(r" (XX|IX|X?I{0,3})(IX|IV|V?I{0,3})$", '')
na18['Distirct'].unique()


# In[ ]:


cc = cc.join(na18.set_index('NA#'), on='NA#')
cc.info()


# In[ ]:


print(pp.shape)
pp['Name of Political Party'].unique()


# In[ ]:


pp.rename(columns={'Acronym':'PartyAcro'}, inplace=True)
cc.rename(columns={'Party':'PartyAcro'}, inplace=True)


# In[ ]:


cc[cc['PartyAcro']=='PTI'].head()
#pp[pp['PartyAcro']=='PTEI']


# In[ ]:


#del cnd
cnd = cc.join(pp.set_index('PartyAcro'), on='PartyAcro')


# In[ ]:


cnd[cnd['PartyAcro']=="PTI"].head()
#remove non-aplhabetic characters from Name
cnd['Name'] = cnd['Name'].str.replace('[^a-zA-Z ]', '')
cnd['Name'] = cnd['Name'].str.lower()
cnd['Name'] = cnd['Name'].str.strip()

cnd['Name of Political Party'] = cnd['Name of Political Party'].str.lower()
cnd['Name of Political Party'] = cnd['Name of Political Party'].str.strip()

cnd[cnd['PartyAcro']=="PTI"].head()


# Merging .. 

# In[ ]:


print(df.columns, cnd.columns)
df.info()
cnd.info()


# In[ ]:


cnd.rename(columns={'NA#':'ConstituencyTitle'}, inplace=True)
cnd.rename(columns={'Name of Political Party':'Party'}, inplace=True)
cnd.rename(columns={'Name':'CandidateName'}, inplace=True)


# In[ ]:


cnd.to_csv('Canditates2018.csv', index=None) 


# <h3><u>Note: Both files can not is mergered easily as NA mapping is changed for current year.  </u></h3>
# I will use both files in EDA and Feature Engineering.

# That's all from me. I tried to clean maximum of the data inconsistency issues. So, I am saving this file for the audience for the seek a reusability. You can fork kernel and continue from here.
# 
# We are all set to move towards <b>Exploratory Data Analysis </b>. 
# 
# Do share your comments and if you find it helpful, <b>please upvote! </b>
# <h2> Happy Exploratory Analsysis :-) </h2>
