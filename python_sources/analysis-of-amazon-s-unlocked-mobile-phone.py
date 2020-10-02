#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
from matplotlib import pyplot
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


Data=pd.read_csv("../input/Amazon_Unlocked_Mobile.csv")
Data=Data.dropna(axis=0)


# <h1>Code </h1>

# In[ ]:


def ReviewLength(Dat):
    
    Dat.dropna(axis=0)
    Price=np.asarray(Dat['Price'])
    Review_Length=np.asarray(Dat['Reviews'].str.len())
    return(Review_Length,Price)


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def Phone_Stat(Dataframe):

    Phones={}

    for i,j in Dataframe.iterrows():
    
          if(j['Product Name'] in Phones):
        
                  Phones[j['Product Name']].append(j['Rating'])
        
          else:
        
                  Phones[j['Product Name']]=[j['Rating']]
        
 
    Mean=[]
    Product=[]
    SD=[]
    N=[]

    for i in Phones:

        Mean.append(np.asarray(Phones[i]).mean())
        Product.append(i)
        SD.append(np.asarray(Phones[i]).std())
        N.append(len(Phones[i]))
        
    Phone_Stat={'Product':Product,
                 'Mean':Mean,
                 'SD':SD,
                 'Number':N}

    Phone_Stat=pd.DataFrame(Phone_Stat)
    Phone_Stat.sort_values(['Mean','Number'],ascending=False)
    
    return Phone_Stat


# In[ ]:


def Word_Freq(Data):
    
     Words={}
     for i in Data['Reviews']:
            Word_List=word_tokenize(i)
            for j in Word_List:
                    if(j in Words):
                             Words[j]+=1
                    else:
            
                             Words[j]=1
            
     Keys=[] 
     Values=[]
     
     Custom=[]
     stop_words=set(stopwords.words("english"))
     Custom.append("phone")
     Custom.append("The")

     for i in Words:
    
                if(i not in stop_words and i.isalpha() and i not in Custom):
        
                        Keys.append(i)
                        Values.append(Words[i])

     Word_Stat={'Word':Keys,'Count':Values}
     Word_Stat=pd.DataFrame(Word_Stat) 
     Word_Stat=Word_Stat.sort_values(['Count'],ascending=False)
        
     return(Word_Stat[1:30])
 


# In[ ]:


def Tokenize_n_Filter(Data,Word):
    
    New_Data=[]
    for i in Data['Reviews']:
        tokens=nltk.word_tokenize(i)
        for j in range(0,len(tokens)):
            
            if(tokens[j]==Word):
                
                New_Data.append(i)
                j=len(tokens)
    Data={'Reviews':New_Data}
    New_Data=pd.DataFrame(Data)
    return(New_Data)
            


# In[ ]:


def ReviewString(Data):
    
    Corpus=""
    
    for i in Data:
        
        Corpus+=Data
        
    return(Corpus)


# <h1>Overall Analysis</h1>

# <h2>Dataset</h2>

# In[ ]:


Data.sample(frac=0.1).head(n=7)


# **Product Name**: Name of the mobile phone  
# 
# **Brand Name**: Brand of the mobile phone 
# 
# **Price**: Price of the mobile phone 
# 
# **Rating**: Ratings given to the mobile phone by users who bought the phone on a scale(1-5)
# 
# **Reviews**: Review by the user who rated the given phone
# 
# **Review Rating**: Ratings given to the review indicating how useful the rating was
# 
# 
# 
# 

# In[ ]:


Data.describe()


# In[ ]:


fig, ax =plt.subplots(1,2)
sns.kdeplot(Data['Rating'],shade=True,color="yellow",ax=ax[0])
sns.kdeplot(Data['Price'],shade=True,color="blue",ax=ax[1])
fig.show()


# <h3>Unlocked Mobiles with highest Reviews</h3>

# In[ ]:


Data['Product Name'].value_counts().head(n=10)


# <h3>Reviews Distribution by Price Category </h3>

# Lets divide the data set into two classes: phones above $250  and below $250.

# In[ ]:


Expensive=Data[Data['Price']>250]
N_Expensive=Data[Data['Price']<250]


# In[ ]:


len(Expensive)


# In[ ]:


len(N_Expensive)


# In[ ]:


(len(Expensive)/float(len(Expensive)+len(N_Expensive)))*100


# ~25.4%  phones are above 250$

# In[ ]:


sns.kdeplot(Expensive['Rating'],shade=True,color="red")


# In[ ]:


sns.kdeplot(N_Expensive['Rating'],shade=True,color="green")


# **  Overall , It could be infered that pricing has some  effect on the ratings of phones.Phones with higher price have had higher ratings.**

# <h2>Review Votes Distribution</h2>

# In[ ]:


sns.kdeplot(Data['Review Votes'],shade=True,color="pink")


# <h2>Reviews Analysis</h2>

# <h3>Length of Reviews</h3>

# In[ ]:


sns.kdeplot(Data['Reviews'].str.len(),shade=True,color="pink")


# <h3>Relationship between Price and Ratings</h3>

# In[ ]:


sns.regplot(x='Price',y='Rating',data=Data)


# The trend noticed that the rating of phone is higher with  the price. 

# <h3>Relationship between Price of Phone and Reviews Rating</h3>

# In[ ]:


sns.regplot(x='Price',y='Review Votes',data=Data)


# <h3> Relationship between Price of Phone and Review Lengths</h3>

# In[ ]:


Review_Length,Price=ReviewLength(Data)
sns.regplot(x=Price,y=Review_Length)


# In[ ]:


print(Review_Length.mean())


# Average length of ~400,000 reviews is 218.076

# **It could be infered that review length , pricing and ratings have no relation.**

# <h3>Review Votes Distribution</h3>

# In[ ]:


sns.kdeplot(Data['Review Votes'],shade=True,color="pink")


# From the above plots we could infer that Ratings are less dependent on Pricing overall. It might be a differ from brand to brand though.

# <h1>Analysis By Brand</h1>

# While the above section was based on overall analysis of data. Lets further investigate phones by brands.For a reasonable amount of data per brand , we could infer various results. In the below section lets investigate various trends in few selected brands in the top 10 list.

# In[ ]:


Top_B=Data['Brand Name'].value_counts().head(n=5).index.tolist()
print(Data['Brand Name'].value_counts().head(n=10))


# In[ ]:


Length=(Data['Brand Name'].value_counts().sum())
print((Data['Brand Name'].value_counts().head(n=10).sum())/(Length)*100)


# The top 10 brands make up for 83.1788% of the phones in the dataset. 

# In[ ]:


new_Df=pd.DataFrame()

Phones_B=[]

for i in Data['Brand Name'].value_counts().head(n=10).index.tolist():
    
    Phones_B.append(i)
    
for j in Phones_B:
  
    new_Df=new_Df.append(Data[Data['Brand Name']==j])


# In[ ]:


new_Df.head(n=5)


# <h3>Top 10 Brands Ratings</h3>

# In[ ]:


fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(x="Brand Name",y="Rating",data=new_Df,ax=ax)


# <h3>Top 10 Brands Pricing</h3>

# In[ ]:


fig,ax=plt.subplots(figsize=(15,10))
sns.boxplot(x="Brand Name",y="Price",data=new_Df,ax=ax)


# <h3>Top 10 Brands Review Lengths</h3>

# In[ ]:


Data_RL,Data_P=ReviewLength(new_Df)
sns.boxplot(x="Brand Name",y=Data_RL,data=new_Df)


# <h2>Samsung</h2>

# In[ ]:


Samsung=Data[Data['Brand Name']=='Samsung']


# <h3>Rating Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='Samsung']['Rating'],shade=True,color="orange")


# <h3>Pricing Distributon</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='Samsung']['Price'],shade=True,color="blue")


# <h3>Top 10 Samsung Phones</h3>

# In[ ]:


print(Samsung['Product Name'].value_counts().head(n=10))


# In[ ]:


print(((Samsung['Product Name'].value_counts().head(n=10).sum())/len(Samsung))*100)


# The top 10 phones account for just 14.69% of Samsung phones in the dataset.

# <h2>Highest Rated Phones</h2>

# In[ ]:


Samsung_Phones=Samsung['Product Name']


# In[ ]:


S_Phone_Stat=Phone_Stat(Samsung)
four=S_Phone_Stat[S_Phone_Stat['Number']>800]
  
plt.figure(figsize=(12,10))    
for i in four.iterrows():

    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=10)
    print(i[1]['Product'])
     
plt.scatter('Number','Mean',data=four)
plt.show()


# From the above it could be noted that the most popular phones have been:
# 
#  1. Samsung Galaxy 5 Duos II S7582 DUAL SIM FACTORY 
#       Unlocked International Version-Black                
#  2. Samsung Galaxy S7 Edge G9350 32GB HK DUAL SIM Factory 
#      Unlocked GSM International Version no warranty  (BLUE CORAL)                                              
#  3. Samsung Galaxy S5 Mini G800H Unlocked Cellphone
#      International Version 16GB White

# <h2>Reviews Analysis</h2>

# <h3>Overall Word Frequencies</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(Samsung),color="b")


# <h3>Negative Word Frequencies</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(Samsung[Samsung['Rating']<3]),color="r")


# <h3>Positive Word Frequencies</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(Samsung[Samsung['Rating']>3]),color="g")


# </h3>Relation between Price and Length of Reviews</h3>

# In[ ]:


Samsung_RL,Samsung_P=ReviewLength(Samsung)
sns.regplot(x=Samsung_P,y=Samsung_RL)


# In[ ]:


from gensim.models import Word2Vec
import gensim.models.lsimodel 
from gensim.corpora import Dictionary
import gensim
from gensim import corpora


# In[ ]:


Samsung_Reviews=Samsung[Samsung['Rating']>3]['Reviews'].values.tolist()
Samsung_Tok=[nltk.word_tokenize(sent) for sent in Samsung_Reviews]


# In[ ]:


from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in ReviewString(Samsung['Reviews'])]


# In[ ]:


# Importing Gensim
import gensim
from gensim import corpora

# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# In[ ]:


# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
Lda = gensim.models.ldamodel.LdaModel
# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)


# <h2>Apple</h2>

# In[ ]:


Apple=Data[Data['Brand Name']=='Apple']


# <h3>Top 10 Apple Phones</h3>

# In[ ]:


print(Apple['Product Name'].value_counts().head(n=10))


# In[ ]:


print(((Apple['Product Name'].value_counts().head(n=10).sum())/len(Apple))*100)


# The top 10 phones account for just 16.076% of Apple phones in the dataset.

# <h3>Rating Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='Apple']['Rating'],shade=True,color="orange")


# <h3>Pricing Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='Apple']['Price'],shade=True,color="blue")


# <h3>Highest Rated Apple Phones</h3>

# In[ ]:


A_Phone_Stat=Phone_Stat(Apple)
four_A=A_Phone_Stat[A_Phone_Stat['Number']>600]
  
plt.figure(figsize=(12,10))    
for i in four_A.iterrows():

    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=10)
     
plt.scatter('Number','Mean',data=four_A)
plt.show()


# <h2>Reviews Analysis</h2>

# <h3>Overall Word Frequencies</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(Apple),color="b")


# <h3>Negative Word Frequencies</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(Apple[Apple['Rating']<3]),color="r")


# <h3>Positive Word Frequencies</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(Apple[Apple['Rating']>3]),color="g")


# </h3>Relation between Price and Length of Reviews</h3>

# In[ ]:


Apple_RL,Apple_P=ReviewLength(Apple)
sns.regplot(x=Apple_P,y=Apple_RL)


# <h2>HTC</h2>

# In[ ]:


HTC=Data[Data['Brand Name']=='HTC']


# <h3>Ratings Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='HTC']['Rating'],shade=True,color="orange")


# <h3>Pricing Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='HTC']['Price'],shade=True,color="blue")


# <h3>Top 10 HTC Phones</h3>

# In[ ]:


print(HTC['Product Name'].value_counts().head(n=10))


# <h3>Highest Rated HTC Phones</h3>

# In[ ]:


H_Phone_Stat=Phone_Stat(HTC)
four_H=H_Phone_Stat[H_Phone_Stat['Number']>400]
  
plt.figure(figsize=(12,10))    
for i in four_H.iterrows():

    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=11)
     
plt.scatter('Number','Mean',data=four_H)
plt.show()


# In[ ]:


print(((HTC['Product Name'].value_counts().head(n=10).sum())/len(HTC))*100)


# Top 10 HTC phones make up 39.916% of HTC phones in Amazon Unlocked sales.

# <h3>Reviews Analysis</h3>

# <h3>Overall Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(HTC),color="b")


# <h3>Negative Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(HTC[HTC['Rating']<3]),color="r")


# <h3>Positive Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(HTC[HTC['Rating']>3]),color="g")


# <h3>Relation between Price and Review Length</h3>

# In[ ]:


HTC_RL,HTC_P=ReviewLength(HTC)
sns.regplot(x=HTC_P,y=HTC_RL)


# <h2>CNPGD </h2>

# In[ ]:


CNPGD=Data[Data['Brand Name']=='CNPGD']


# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='CNPGD']['Rating'],shade=True,color="orange")


# <h3>Pricing Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='CNPGD']['Price'],shade=True,color="blue")


# <h3>Top 10 CNPGD Phones</h3>

# In[ ]:


print(CNPGD['Product Name'].value_counts().head(n=10))


# In[ ]:


print(((CNPGD['Product Name'].value_counts().head(n=10).sum())/len(CNPGD))*100)


# The top 10 CNPGD phones make up for 67.30% of all CNPGD phones in Amazon Unlocked sale.

# <h2>Reviews Analysis</h2>

# <h3>Overall Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(CNPGD)[1:20],color="b")


# <h3>Positive Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(CNPGD[CNPGD['Rating']>3])[1:20],color="g")


# <h2>Negative Frequency</h2>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(CNPGD[CNPGD['Rating']<3])[1:20],color="r")


# <h3>Relation between Price and Review Length</h3>

# In[ ]:


CNPGD_RL,CNPGD_P=ReviewLength(HTC)
sns.regplot(x=CNPGD_P,y=CNPGD_RL)


# <h3>Highest Rated CNPGD Phones</h3>

# In[ ]:


CNP_Phone_Stat=Phone_Stat(CNPGD)
four_CNP=CNP_Phone_Stat[CNP_Phone_Stat['Number']>400]
  
plt.figure(figsize=(12,10))    
for i in four_CNP.iterrows():

    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=11)
     
plt.scatter('Number','Mean',data=four_CNP)
plt.show()


# <h2>OtterBox</h2>

# In[ ]:


OtterBox=Data[Data['Brand Name']=='OtterBox']


# <h3>Ratings Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='OtterBox']['Rating'],shade=True,color="orange")


# <h3>Pricing Distribution</h3>

# In[ ]:


sns.kdeplot(Data[Data['Brand Name']=='OtterBox']['Price'],shade=True,color="blue")


# <h3> Top 10 OtterBox Phones</h3>

# In[ ]:


print(OtterBox['Product Name'].value_counts().head(n=10))


# In[ ]:


print(((OtterBox['Product Name'].value_counts().head(n=10).sum())/len(OtterBox))*100)


# Top 10 OtterBox phones make up for 68.65% of all OtterBox phones.

# <h2>Reviews Analysis</h2>

# <h3>Overall Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(OtterBox),color="b")


# <h3>Positive Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(OtterBox[OtterBox['Rating']>3]),color="g")


# <h3>Negative Frequency</h3>

# In[ ]:


sns.barplot(x="Count",y="Word",data=Word_Freq(OtterBox[OtterBox['Rating']<3]),color="r")


# <h3>Relation between Price and Length Review</h3>

# In[ ]:


HTC_RL,HTC_P=ReviewLength(HTC)
sns.regplot(x=HTC_P,y=HTC_RL)


# <h3>Top Rated Phones </h3>

# In[ ]:


OT_Phone_Stat=Phone_Stat(OtterBox)
four_OT=OT_Phone_Stat[OT_Phone_Stat['Number']>400]
  
plt.figure(figsize=(12,10))    
for i in four_OT.iterrows():

    plt.text(i[1]['Number'],i[1]['Mean'],i[1]['Product'],fontsize=11)
     
plt.scatter('Number','Mean',data=four_OT)
plt.show()


# In[ ]:




