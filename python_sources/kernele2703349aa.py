# %% [markdown]
# Accesing the board game review dataset

# %% [code]
import pandas as pd
df_13m = pd.read_csv(r"C:\Users\suchi\Downloads\boardgamegeek-reviews\bgg-13m-reviews.csv")
df_13m.head()

# %% [code]
import pandas as pd
df1 = pd.read_csv(r"C:\Users\suchi\Downloads\boardgamegeek-reviews\2019-05-02.csv")
df1.head()

# %% [code]
import pandas as pd
df_details = pd.read_csv(r"C:\Users\suchi\Downloads\boardgamegeek-reviews\games_detailed_info.csv",index_col=0)
df_details.head()


# ## Looking at the content and data type

# ### d.types gives the data types for each column stored in corresponding table

df_details.dtypes

df_13m.dtypes


df1.dtypes

# ### describe() funtion does all the below calculations over data


df1.describe()

df_13m.describe()

df_details.describe()


# ### isna().sum() are for calculating the NaN values


df1.isna().sum() #counting num of NAN values

df_details.isna().sum()


df_13m.isna().sum()


# ### a)  Plotting Histograms for better visualization


import matplotlib.pyplot as plt
df1['Year'].plot(kind='hist')
plt.show()

import matplotlib.pyplot as plt
df1['Average'].plot(kind='hist')
plt.show()

import matplotlib.pyplot as plt
df1['Bayes average'].plot(kind='hist')
plt.show()

import matplotlib.pyplot as plt
df1['Users rated'].plot(kind='hist')
plt.show()

df_details['Board Game Rank'].value_counts().head()


# ### plotting the reviews of the BGG 


# ### The plot explains that there a maximum reviews of approximatly between 6.5 to 8 


import matplotlib.pyplot as plt
df_13m['rating'].plot(kind='hist')
plt.show()


# ###  Describing the category features of the games



df_details.iloc[:, :16].describe()


# ### Repeated Columns Analysis
# IDs are present for every table (join condition)
# Ranking and game detail tables have published year, rank, average, bayes average, users rated and thumbnail columns
# Looking at Kaggle, the dataset was last updated on 2nd June, and looking at the number of users rated, the game detailed info table looks more updated
# Should drop ranking data's repeated columns and join the rest (BGG URL and Name) with game detail dataset

sub_df1 = df1[['ID', 'Year', 'Rank', 'Average', 'Bayes average', 'Users rated', 'Thumbnail']]
sub_df1.head()


sub_df_details = df_details[['id', 'yearpublished', 'Board Game Rank', 'average', 'bayesaverage', 'usersrated', 'thumbnail']]
sub_df_details.head()


merged_df = sub_df1.merge(sub_df_details, left_on='ID', right_on='id', how='left')
merged_df.head(20)

# ### Dropping all reviews that are missed

df_details =df_details.drop(axis=1, index=None, columns=['Abstract Game Rank','Accessory Rank', 'Amiga Rank','Arcade Rank', 'Atari ST Rank',                   
"Children's Game Rank", 'Commodore 64 Rank','Customizable Rank', 'Family Game Rank','Party Game Rank',                  
'RPG Item Rank', 'Strategy Game Rank', 'Thematic Rank', 'Video Game Rank',                  
'War Game Rank', 'alternate', 'boardgameartist', 'boardgamecategory', 'boardgamecompilation',             
'boardgamedesigner', 'boardgameexpansion',  'boardgamefamily',  'boardgameimplementation',          
'boardgameintegration', 'boardgamemechanic','boardgamepublisher', 'description','image',                               
'suggested_language_dependence', 'suggested_playerage', 'thumbnail','type' ])

df1 = df1.drop(axis=1, index=None, columns=['URL','Thumbnail'])

df_13m = df_13m.drop(axis=1 ,columns=['Unnamed: 0']) 

df_13m = df_13m.dropna(axis=0 ,subset=['comment'])    
# Drop all missing reviews 

rk_dl_df = pd.merge(left=df1, right=df_details, left_on='ID', right_on='id') 

merge_df = pd.merge(left=df_13m, right=rk_dl_df, how='left', left_on='ID', right_on='ID') # left merge
merge_df.shape

merge_df['playes_num'] = merge_df['maxplayers'] - merge_df['minplayers']
merge_df['play_time'] = merge_df['maxplaytime'] - merge_df['minplaytime']
merge_df['Time'] = merge_df[['playingtime','play_time']].max(axis=1)

#joint_df['Board Game Rank'] = joint_df['Board Game Rank'].apply(pd.to_numeric, errors='coerce')  # convert from object to numeric


merge_rk_dl = merge_df[['Name','Year','rating','numweights','usersrated',
                                           'numcomments','user','comment', 'wanting', 'wishing', 
                                           'trading', 'owned','Time','playes_num','minage']]
merge_rk_dl.shape

#for last time lets drop all the missing and dupliactes rows
merge_rk_dl = merge_rk_dl.dropna(axis=0)
merge_rk_dl = merge_rk_dl.drop_duplicates(subset=None, keep="first", inplace=False)
merge_rk_dl.shape


import seaborn as sns
corr = df_details.iloc[:, 16:].corr()
corr = corr.dropna(how='all', axis=1).dropna(how='all', axis=0).round(2)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(200, 10, as_cmap=True)

plt.subplots(figsize=(25,25))
sns.heatmap(corr, cmap='GnBu', annot=True, linewidths=.5)
plt.title('Plot without genre variables under correlation')
plt.show()


# ### Descriptive analysis of the board game


# ### The plot of maximum number of published board games were during 2010 to 2018

df_details['yearpublished'].loc[(df_details['yearpublished'] >= 2010) & (df_details['yearpublished'] < 2018)].value_counts().sort_index().plot()
plt.xlabel('Year')
plt.ylabel('Board Games Published')
plt.title('Total Board Games Published over Time (2010 to 2018)')
plt.show()



# ### Correlation 

# ths corolation of numeric values to the the target 'Rating' the main ranking
correlations = merge_df.corr()
correlations['rating'] 


# ### b) Seaborn Plots on data


# plot # 'rating', 'Average', 'Rank'
plt.figure(figsize=(12, 6))
sns.set(color_codes="True")
sns.distplot(merge_df["rating"])


plt.figure(figsize=(10, 8))
sns.set(color_codes="True")
sns.distplot(merge_df["Average"])


plt.figure(figsize=(8, 6))
sns.set(color_codes="True")
sns.distplot(merge_df["Rank"])

plt.figure(figsize=(12, 6))
sns.set(color_codes="True")
sns.distplot(merge_rk_dl["rating"])


# ## Data analysis 
# 


# ### Describing the rankings data


merge_rk_dl.rating.describe()


rate_grp = merge_rk_dl.groupby(['rating'])
name_grp = merge_rk_dl.groupby(['Name'])
year_grp = merge_rk_dl.groupby(['Year'])
user_grp = merge_rk_dl.groupby(['user'])


# ## Top 25 reviews 



plt.figure(figsize=(12, 6))
merge_rk_dl.loc[merge_rk_dl['comment'].notna()]['Name'].value_counts()[:25].plot(kind='bar')
plt.xlabel('Board Game Name')
plt.ylabel('Review Count')
plt.title('Top 25 Most Reviewed Board Games')
plt.show()



#  ### Text PreProcessing and Wordcloud Generation

import sys, sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import cross_val_score , train_test_split
import nltk
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize





# ### Defining funtions for ploting bag of words and removing less frequent words


def tokenize_del_punctuation(X):               
    wordfreq = {}
    for text in X:
        text = ' '.join(word for word in text.split() if word not in stop_words 
                                         if not word.isdigit() if len(word)>1 )   #if not word.isalpha())
        tokens = nltk.RegexpTokenizer(r"\w+").tokenize(text.lower())                                                 # len(word)>2   
        for token in tokens:                                    
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
                
    return wordfreq 



# Sort words in reverse order
def sort_allwords_freq(wordfreq):                        
    wordfreq_sorted = dict(sorted(wordfreq.items(), key=lambda x: x[1], reverse=True))
    return wordfreq_sorted
 
    
 # deleting keys with less occurence ie. value less than 5   
def remove_threshold(wordfreq_sorted):              
    delete = []                                      
    for key, val in wordfreq_sorted.items(): 
        if val < 5 : 
            delete.append(key) 

    for i in delete: 
        del wordfreq_sorted[i] 
    return   wordfreq_sorted
    
    
#  Bag of words   
def vector_matrix_review(X):
    sentence_vectors = []
    for txt,i  in zip(X,y):   #txt,i in train_set.iterrows():
        sentence_tokens = nltk.RegexpTokenizer(r"\w+").tokenize(txt)
        sent_vec = []
        for token in wordfreq_sorted_X:   
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append([sent_vec,i])
    return sentence_vectors    


x = tokenize_del_punctuation(merge_rk_dl['comment'])
x = sort_allwords_freq(x)
x = remove_threshold(x)


len(x)


# ## Top 50 words in the dictionary


#top 50 words of dictionary
top50 = {k: x[k] for k in list(x)[:50]}
print(top50)


from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

text = " ".join(review for review in merge_rk_dl['comment'].sample(250))


# Generate a word cloud
wordcloud = WordCloud(stopwords=stop_words, background_color="black").generate(text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# ## Splitting data into Train and Test


# split dataset to train, test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 35)


# ## References
# ### https://www.kaggle.com/jvanelteren/exploring-the-13m-reviews-bgg-dataset
# ### https://www.kaggle.com/ngrq94/boardgamegeek-reviews-eda-final
# ### https://www.kaggle.com/ngrq94/boardgamegeek-reviews-data-preparation


