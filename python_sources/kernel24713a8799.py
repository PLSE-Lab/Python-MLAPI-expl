# %% [markdown]
# We have Achieved to successfully classify the articles provided in Json to different tasks assigned. 

# %% [code]
#from IPython.display import Image
#Image("../input/related-papers-vs-tasks/related_papers.png")

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from datetime import date
import itertools
import re
import string
import pymongo 
import nltk
import os
import json

count=0
df=pd.DataFrame()


dir="/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/"
for file in os.listdir(dir):
    if file.endswith(".json"):
        file= os.path.join(dir,file)
        data = json.loads(open(file).read())
        df.loc[count,'subset_type']="common"
        try:
             df.loc[count,'paper_id']=data['paper_id']
        except:
             df.loc[count,'paper_id']=None
        try:
             df.loc[count,'title']=data['metadata']['title']
        except:
             df.loc[count,'title']=None
        try:
             df.loc[count,'abstract']=data['abstract'][0]['text']
        except:
             df.loc[count,'abstract']=None
        try:
             df.loc[count,'text']=data['body_text'][0]['text']
        except:
             df.loc[count,'text']=None
    count=count+1
    
dir="/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/"
for file in os.listdir(dir):
    if file.endswith(".json"):
        file= os.path.join(dir,file)
        data = json.loads(open(file).read())
        df.loc[count,'subset_type']="non-common"
        try:
             df.loc[count,'paper_id']=data['paper_id']
        except:
             df.loc[count,'paper_id']=None
        try:
             df.loc[count,'title']=data['metadata']['title']
        except:
             df.loc[count,'title']=None
        try:
             df.loc[count,'abstract']=data['abstract'][0]['text']
        except:
             df.loc[count,'abstract']=None
        try:
             df.loc[count,'text']=data['body_text'][0]['text']
        except:
             df.loc[count,'text']=None
    count=count+1


# Input data files 
#df=pd.read_csv('../input/input-covid-filecsv/input_covid_file.csv', encoding='iso-8859-1' )

# Any results you write to the current directory are saved as output.

# %% [markdown]
# The table contained the following columns:
# 1.	Abstract
# 2.	Paper_id
# 3.	Subset_type
# 4.	Text
# 5.	Title
# The number of rows were 10973. Each row was a paper from the COVID data provided in the kaggle datatset. Stored the data table in a dataframe called ‘df’.
# 

# %% [markdown]
#  ## Analysis 1 : Extracting Keywords from Text

# %% [markdown]
# Check for the parts of speech in the text and apply lemmatization accordingly to obtain the root words. Store the list of lemmatized words in a new column called ‘lemma’.

# %% [code]
##Lemmatize words based on parts of speech
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = nltk.stem.WordNetLemmatizer()
df['lemma'] = df['text'].apply(lambda x: [lemmatizer.lemmatize(y,get_wordnet_pos(y)) for y in x.split()])


# %% [markdown]
# Clean the text removing special characters and convert it to a single string.

# %% [code]
##Convert vector to string
df['lemma'] = df['lemma'].apply(lambda x:' '.join(map(str, x)))
df['lemma'].replace(regex=True, inplace=True, to_replace=r'[^\sA-Za-z0-9.%-]', value=r'')


# %% [markdown]
# Apply the tfidf vectorizer on the ‘lemma’ column for each individual row, to identify the key words from the text and filter out the top 25 important keywords. While applying the vectorizer we ensure to remove the stopwords as well as obtain meaningful results.

# %% [code]
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(analyzer='word', stop_words = 'english')
for i in range(0,len(df)):
    value = [df['lemma'][i]]
    try:
        x = v.fit_transform(value)#.values.astype('str'))
    except ValueError:
        continue
    #words = v.get_feature_names()
    feature_array = np.array(v.get_feature_names())
    tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:25]
    words=",".join(top_n)
    df.loc[i,'keywords']=words



# %% [code]
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

# %% [code]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white',  
                min_font_size = 10).generate(str(df['keywords']))

# %% [code]
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

# %% [markdown]
# ## Analysis 2 - Task Data Analysis to opbtain Top Related Papers

# %% [markdown]
# Prepared a dataset containing the tasks and the task details from kaggle. Stored this dataset into a dataframe called ‘data’. Cleaned the ‘task details’ column of any special characters using regular expressions. 

# %% [code]
#Import csv with task and task details
data = pd.read_csv("../input/task-data/task_data.csv",encoding='cp1252')
data['Task Details'].replace(regex=True, inplace=True, to_replace=r'[^\sA-Za-z0-9.-]', value=r'')

# %% [markdown]
# Separated parts of speech of each word of each task detail per task. Applied lemmatization on each of the procured word to obtain the root word.  Created a list of these words for each task and stored it in a column called ‘lemma’ in ‘data’.

# %% [code]
##Lemmatize words based on parts of speech
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = nltk.stem.WordNetLemmatizer()
data['lemma'] = data['Task Details'].apply(lambda x: [lemmatizer.lemmatize(y,get_wordnet_pos(y)) for y in x.split()])


# %% [markdown]
# Converted the list into a whitespace separated string. Created a temporary dataframe called ‘temp’ containing a single column called ‘lemma’. This column was the column ‘lemma’ from ‘data’ with index numbers starting from 12013 to 12024. This helped us join ‘temp[’lemma’]’ with ‘df[‘abstract’]’ containing rows from 0 to 12013. Temp now had rows  from 0 to  12024. Temp was cleaned for any special characters within the text.

# %% [code]
##Convert vector to string
data['lemma'] = data['lemma'].apply(lambda x: ' '.join(map(str, x)))
temp=pd.DataFrame(data['lemma'],columns=['lemma'])
temp=temp.rename(index={0:12014,1:12015,2:12016,3:12017,4:12018,5:12019,6:12020,7:12,8:12021,9:12022,10:12023,11:12024})
df_consolidated=pd.DataFrame(data=df['abstract'],columns=['abstract'])
for i in range(0,len(df)):
    if df_consolidated.loc[i,'abstract']==None:
        df_consolidated.loc[i,'abstract']=df.loc[i,'text']
df_consolidated=df_consolidated.rename(columns={'abstract':'lemma'})
df_consolidated['lemma'].replace(regex=True, inplace=True, to_replace=r'[^\sA-Za-z0-9.-]', value=r'')
df_consolidated=pd.concat([df_consolidated,temp])


# %% [markdown]
# Applying tfidf Vectorization on this column to obtain the important words from the article abstracts and the task details together. Stop words were removed from this data to help obtain better results. These keywords were then stored in the form of a dataframe containing the tfidf score of each article for that keyword. 

# %% [code]
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer(analyzer='word', min_df = 0.04, stop_words = 'english')
x = v.fit_transform(df_consolidated.lemma.values.astype('str'))
df2 = pd.DataFrame(x.toarray(), columns=v.get_feature_names())


# %% [markdown]
# The dataframe was then put into the cosine similarity function to obtain the relation of the tasks and the abstract of each article. The similarity score 2D list was then truncated to focus only on the task and the relationship of each task with the articles. 

# %% [code]
from sklearn.metrics.pairwise import cosine_similarity
new_similarity=cosine_similarity(df2)

# %% [markdown]
# Applying filtering process to identify the top 30 associated articles to the tasks, we store the results of the obtained indices in a list. The list for each task is prepared in a similar manner and stored as a column in ‘data’ called ‘top related papers’.

# %% [code]
new_similarity=new_similarity[12014:,:12014]

##Loop through each row to find the top 30 similar papers
final_new=[]
for  i in range(0,len(new_similarity)):
    b=[]
    a=new_similarity[i]
    if len(a)==0 or np.count_nonzero(a)==0:
            continue
    for j in range(0,31):
        maxi=max(a)
        x=[]
        s=[]
        for l,k in enumerate(a):
            if k==maxi:
                x.append(l)
                s.append(l+j)
        a=np.delete(a,x)
        b.append(s)
    if i in b[0]:
        b[0].remove(i)
    final_new.append(b)

##Eliminate anything  excess of 30
add=[]
for i in final_new:
    x=[]
    for j in i:
        for k in j:
            if len(x)>=30:
                del x[30:]
                break
            x.append(k)
    add.append(x)
top = pd.DataFrame(add)

top['Top Related Papers']='['+top[0].astype(str)+','+top[1].astype(str)+','+top[2].astype(str)+','+top[3].astype(str)+','+top[4].astype(str)+','+top[5].astype(str)+','+top[6].astype(str)+','+top[7].astype(str)+','+top[8].astype(str)+','+top[9].astype(str)+','+top[10].astype(str)+','+top[11].astype(str)+','+top[12].astype(str)+','+top[13].astype(str)+','+top[14].astype(str)+','+top[15].astype(str)+','+top[16].astype(str)+','+top[17].astype(str)+','+top[18].astype(str)+','+top[19].astype(str)+','+top[20].astype(str)+','+top[21].astype(str)+','+top[22].astype(str)+','+top[23].astype(str)+','+top[24].astype(str)+','+top[25].astype(str)+','+top[26].astype(str)+','+top[27].astype(str)+','+top[28].astype(str)+','+top[29].astype(str)+']'

top=top.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],axis=1)

data_final=pd.concat([data,top],axis=1)



# %% [markdown]
# ## Analysis 3. Task Analysis to obtain keywords from the related articles for each task

# %% [markdown]
# A.	Individual Article keywords

# %% [markdown]
# Once the index for each associated article is found, we iterate through the article with that index number. We target the keywords column from ‘df’ to identify the keywords already procured for these articles. We create a list of these keyword lists and add it as a column in ‘data’ called ‘keywords’.

# %% [code]
key=[]
for i in add:
    k=[]
    for  j in i:
        k.append([df.loc[j,'keywords']])
    #print(k)
    key.append(str(k))
#print(key)
data_final=pd.concat([data_final,pd.DataFrame(key,columns=['Keywords'])],axis=1)


# %% [markdown]
# B.	Overall keywords per task

# %% [markdown]
# Combining the ‘text’ column for the related articles, we create a temp column. The temp column is put through the tfidf vectorizer to obtain overall 25 keywords from the combined text of all these articles. These keywords are stored in a list inturn stored ina column called ‘Overall Keywords’

# %% [code]
from sklearn.feature_extraction.text import TfidfVectorizer
c=0
for i in add:
    text=""
    for j in i:
        text=text+" "+df.loc[j,'text']
    inp=[text]
    v = TfidfVectorizer(analyzer='word', stop_words = 'english')
    x = v.fit_transform(inp)#.values.astype('str'))
    #p=v.get_feature_names()
    feature_array = np.array(v.get_feature_names())
    tfidf_sorting = np.argsort(x.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:25]
    #print(p)
    data_final.loc[c,'Overall Keywords']=str(top_n)
    #print(top_n)
    c=c+1




# %% [code]
# importing all necessery modules 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

# %% [code]
#What do we know about COVID-19 risk factors?
indiv=data_final.loc[1,'Keywords']
stopwords=['human','virus','disease','different','health','study','infect','infection','cause','case']
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords=stopwords,
                min_font_size = 10).generate(str(indiv))

# %% [code]
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

# %% [code]
#What do we know about virus genetics, origin, and evolution?
indiv=data_final.loc[2,'Keywords']
stopwords=['human','virus','disease','different','health','study','infect','infection','cause','important','case']
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords=stopwords,
                min_font_size = 10).generate(str(indiv))

# %% [code]
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

# %% [code]
#What do we know about diagnostics and surveillance?
indiv=data_final.loc[8,'Keywords']
stopwords=['human','virus','disease','different','health','study','infect','infection','cause','important']
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords=stopwords,
                min_font_size = 10).generate(str(indiv))

# %% [code]
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

# %% [code]
#What is known about transmission, incubation, and environmental stability?

indiv=data_final.loc[0,'Keywords']
stopwords=['human','virus','disease','different','health','study','infect','infection','cause','important']
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords=stopwords,
                min_font_size = 10).generate(str(indiv))

# %% [code]
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

# %% [code]
##Loop through each row to find the similar papers
final_new_limitless=[]
for  i in range(0,len(new_similarity)):
    a=new_similarity[i]
    b=np.argsort(a)
    print(b)
    for i in range(0,len(b)):
        if a[i]>=0.5:
            print(a[i])
            print(i)           
            break
    b=b[i:]
    final_new_limitless.append(b)
    
    
print(final_new_limitless[2])
    
    
    
   # for j in range(0,len(a)):
   #     maxi=max(a)
   #     if maxi<=0.05:
   #         break
   #     x=[]
   #     s=[]
   #     for l,k in enumerate(a):
   #         if k==maxi:
   #             x.append(l)
   #             s.append(l+j)
   #     a=np.delete(a,x)
   #     b.append(s)
   #     print(b)
   # if i in b[0]:
   #     b[0].remove(i)
   # final_new_limitless.append(b)

# %% [code]
count=0
df['Related_to_Task_No']=''
for i in final_new_limitless:
    for j in i:
        df.loc[j,'Related_to_Task_No']= str(df.loc[j,'Related_to_Task_No']) + "," + str(count)  
    count=count+1

print(df)
# %% [code]
import matplotlib.pyplot as plt
labels=[]
values = []
for i in range(0,len(final_new_limitless)):
    labels.append(data_final.loc[i,'Tasks'])
    values.append(len(final_new_limitless[i]))
indexes = np.arange(len(labels))
width = 1    
    
for i in range(0,len(labels)):
    count=0
    length=0
    for j in labels[i].split():
        length=length+len(j)+1
        if count==5:
            res = list(labels[i])
            res.insert(length, '\n')
            res= ''.join(res) 
            break
        count=count+1
    labels[i]=res
        
    
my_color=['red','magenta','green','blue','pink','black','orange','violet','yellow','grey','purple','cyan']
ax= plt.bar(indexes, values, color=my_color)
#plt.color('rgb')
plt.xlabel('Tasks', fontsize=8)
plt.ylabel('No of Related Papers', fontsize=8)
plt.xticks(indexes, labels, rotation='vertical',fontsize=8)
plt.title('No of Related Papers per Task')
rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_x()
    y_value = rect.get_y() + rect.get_height() 

    # Number of points between bar and label. Change to your liking.
    space = 1

    # Use X value as label and format number with one decimal place
    label = "{:.1f}".format(x_value)

    # Create annotation
    plt.annotate(
        rect.get_height(),          
        (x_value, y_value),         
        xytext=(0, 5),              
        textcoords="offset points", 
        va='center')                      
                                    

# %% [code]

data_final.to_csv('Task_output_submission.csv')
df.to_csv('Article_output_submission.csv')
