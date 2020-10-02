import pandas as pd 
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os, sys, email, re, string
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from collections import defaultdict
from nltk.corpus import wordnet as wn
import csv


lemmatiser = WordNetLemmatizer()
nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}

def clean(text):
    stop = set(STOPWORDS)
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()) and len(i)>2)])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split() if word in nouns)
    
    uniqueWords = []
    
    words = normalized.split()

    for word in words:
        # Loop that define only unique terms
        if word not in uniqueWords:
            uniqueWords.append(word)
    
    return uniqueWords

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        elif(len(word)>2):
            counts[word] = 1

    return counts
    
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)
    
stopwords = set(STOPWORDS)

inputrows = 510000

emails_df = pd.read_csv("../input/emails.csv" ,nrows=inputrows) # Only n' top rows
messages = list(map(email.message_from_string, emails_df['message']))
keys = messages[0].keys()

for key in keys: emails_df[key] = [doc[key] for doc in messages]

currentname=emails_df['X-Origin'][1].lower()
odd=[]
even=[]
rowsPerUser=0
users=0
allOdd=[]
allEven=[]
Totalmails=0

with open('Terms.csv', 'w') as csvfile:
            fieldnames = ['Term', 'User', 'Count', 'Source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

emails_df['content'] = list(map(get_text_from_email, messages)) # Refer to content

users = []
users = ('mann-k', 'dasovich-j', 'kaminski-v', 'beck-s', 'shackleton-s', 'scott-s', 'germany-c', 'symes-k', 'jones-t', 'taylor-m', 'fossum-d', 'mcconnell-m', 'bass-e', 'arnold-j', 'perlingiere-d', 'nemec-g', 'kean-s', 'lay-k', 'sanders-r', 'farmer-d', 'stclair-c', 'skilling-j', 'love-p', 'delainey-d', 'rodrique-r', 'rogers-b', 'shankman-j', 'haedicke-m', 'cash-m', 'allen-p', 'lenhart-m', 'horton-s', 'hyvl-d', 'campbell-l', 'giron-d', 'smith-m', 'sager-e', 'ward-k', 'neal-s', 'steffes-j', 'blair-l', 'grigsby-m', 'dorland-c', 'lavorato-j', 'kitchen-l', 'mims-thurston-p', 'presto-k', 'corman-s', 'heard-m', 'watson-k', 'ruscitti-k', 'pereira-s', 'gay-r', 'whalley-l', 'forney-j', 'buy-r', 'hain-m', 'mclaughlin-e', 'kuykendall-t', 'tycholiz-b', 'tholt-j', 'hayslett-r', 'hyatt-k', 'causholli-m', 'lokay-m', 'shively-h', 'stokley-c', 'guzman-m', 'sturm-f', 'davis-d', 'derrick-j', 'richey-c', 'geaccone-t', 'quigley-d', 'white-s', 'lavorado-j', 'schoolcraft-d', 'keiser-k', 'martin-t', 'cuilla-m', 'mckay-j', 'wheldon-c', 'semperger-c', 'donoho-l', 'brawner-s', 'parks-j', 'carson-m', 'hernandez-j', 'weldon-v', 'ring-a', 'crandell-s', 'dickson-s', 'hodge-j', 'thomas-p', 'hendrickson-s', 'sanchez-m', 'schwieger-j', 'whitt-m', 'scholtes-d', 'storey-g', 'mccarty-d', 'lucci-p', 'mims-p', 'fischer-m', 'baughman-d', 'rapp-b', 'lewis-a', 'whalley-g')

for i in range(1,inputrows-1):
    
    if(emails_df['X-Origin'][i] and currentname==emails_df['X-Origin'][i].lower() and emails_df['X-Origin'][i].lower() in users):
        if(emails_df['X-Folder'][i] and "sent" in emails_df['X-Folder'][i].lower()):
            if(i%10==1):
                odd.append(clean(emails_df['content'][i].split("Forwarded")[0].split("From")[0].split("Original Message")[0]))
            else:
                even.append(clean(emails_df['content'][i].split("Forwarded")[0].split("From")[0].split("Original Message")[0])) 
    else:

        odd = re.sub(r'[^a-zA-Z]', ' ', str(odd))
        even = re.sub(r'[^a-zA-Z]', ' ', str(even))
        
        allOdd.append((Counter(word_count(str(odd))).most_common(10000)))
        allEven.append((Counter(word_count(str(even))).most_common(10000)))
        
        FileName = currentname + '.csv'
        
        with open('Terms.csv', 'a') as csvfile:
            fieldnames = ['Term', 'User', 'Count', 'Source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
           
            for item in allOdd[0]:
                writer.writerow({'Term': item[0], 'User': currentname, 'Count': item[1], 'Source': 'Odd'})   
        
            for item in allEven[0]:
                writer.writerow({'Term': item[0], 'User': currentname, 'Count': item[1], 'Source': 'Even'})    


        if(emails_df['X-Origin'][i]): currentname=emails_df['X-Origin'][i].lower()
        
        odd=[]
        even=[]
        allOdd=[]
        allEven=[]
        rowsPerUser=0
        
        
    
    
    
    
'''    
    if(emails_df['X-Folder'][i] and "sent" in emails_df['X-Folder'][i].lower() and len(clean(emails_df['content'][i]))>2):
        Totalmails += 1
        if(i%5!=1):
            odd.append(clean(emails_df['content'][i].split("to: ")[0]))
        else:
            even.append(clean(emails_df['content'][i].split("to: ")[0])) 
            
odd = re.sub(r'[^a-zA-Z]', ' ', str(odd))
allOdd.append((Counter(word_count(str(odd))).most_common(100)))
even = re.sub(r'[^a-zA-Z]', ' ', str(even))
allEven.append((Counter(word_count(str(even))).most_common(100)))

print(allOdd)
print(allEven)
print(Totalmails)


'''    



'''
# Fine

import pandas as pd 
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os, sys, email, re, string
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from collections import defaultdict

lemmatiser = WordNetLemmatizer()

def clean(text):
    stop = set(STOPWORDS)
    stop.update(('fwd'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()) and len(i)>2)])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    tokens = nltk.word_tokenize(normalized)
    tags = nltk.pos_tag(tokens)
    nouns =" ".join(word for word,pos in tags if (pos == 'NN'))
    return nouns

def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        elif(len(word)>2):
            counts[word] = 1

    return counts
    
def get_text_from_email(msg):
    #To get the content from email objects
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)
    
stopwords = set(STOPWORDS)

stopwords.update(('RE','FW','Hello','Meeting','Ga','Access','positions','list','forward','floor','collar','fixed'))

inputrows = 10000

emails_df = pd.read_csv("../input/emails.csv" ,nrows=inputrows) # Only n' top rows
messages = list(map(email.message_from_string, emails_df['message']))
keys = messages[0].keys()

for key in keys: emails_df[key] = [doc[key] for doc in messages]

currentname=emails_df['X-Origin'][1].lower()
odd=[]
even=[]
rowsPerUser=0
users=0
allOdd=[]
allEven=[]

emails_df['content'] = list(map(get_text_from_email, messages)) # Refer to content

for i in range(1,inputrows-1):
    if(emails_df['X-Origin'][i] and currentname==emails_df['X-Origin'][i].lower()):
        if(emails_df['X-Folder'][i] and "sent" in emails_df['X-Folder'][i]):
            if(i%2==1):
                odd.append(clean(emails_df['content'][i]))
            else:
                even.append(clean(emails_df['content'][i])) 
            rowsPerUser += 1
    else:

        users += 1
        
        odd = re.sub(r'[^a-zA-Z]', ' ', str(odd))
        even = re.sub(r'[^a-zA-Z]', ' ', str(even))
        
        print("#",users, " - ",rowsPerUser," Rows - ",currentname)
        
        allOdd.append((Counter(word_count(str(odd))).most_common(10000)))
        allEven.append((Counter(word_count(str(even))).most_common(10000)))
            
        if(emails_df['X-Origin'][i]): currentname=emails_df['X-Origin'][i].lower()
        
        odd=[]
        even=[]
        rowsPerUser=0

print(allOdd)


'''

'''
# Works Great !!!

for i in range(1,inputrows):
    if(emails_df['X-Origin'][i] and currentname==emails_df['X-Origin'][i].lower()):
        if(emails_df['X-Folder'][i] and "sent" in emails_df['X-Folder'][i]): 
            text_clean.append(clean(emails_df['content'][i]))
            subject_Clean.append(clean(emails_df['Subject'][i])) 
            rowsPerUser += 1
    else:
        if(len(text_clean)>100):
            
            users += 1
            
            text_clean = re.sub(r'[^a-zA-Z]', ' ', str(text_clean))
            subject_Clean = re.sub(r'[^a-zA-Z]', ' ', str(subject_Clean))
            
            print("#",users, " - ",rowsPerUser," Rows - ",currentname)
            
            print(Counter(word_count(str(text_clean))).most_common(50))
            print("In Subjects")
            print("- - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print(Counter(word_count(str(subject_Clean))).most_common(50))
            
            wordcloud = WordCloud(
                max_font_size=50,
                background_color='white',
                max_words=50
                ).generate(str(text_clean))
            fig = plt.figure(1)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.show()
            fig.savefig(currentname+" - Body")
            
            wordcloud = WordCloud(
                max_font_size=50,
                max_words=50
                ).generate(str(subject_Clean))
            fig = plt.figure(1)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.show()
            fig.savefig(currentname+"- Subjects")
        
        if(emails_df['X-Origin'][i]): currentname=emails_df['X-Origin'][i].lower()
        
        subject_Clean=[]
        text_clean=[]
        rowsPerUser=0
        





'''

        
'''

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD

n_words_to_select = 100

for i in range(1,inputrows):
    text_clean.append(emails_df['Subject'][i])

# calculate TF-IDF scores
tfidf_vect = TfidfVectorizer()
# sum over all documents to get importance score for each word
tfidf = tfidf_vect.fit_transform(text_clean).toarray().sum(axis=0)
tfidf_terms = tfidf_vect.get_feature_names()

# count terms in each document
count_vect = CountVectorizer()
count = count_vect.fit_transform(text_clean).toarray()
# sum them up across all documents
total_counts = count.sum(axis=0)
count_terms = count_vect.get_feature_names()

# perform LSA:
lsa = TruncatedSVD(n_words_to_select)
lsa.fit_transform(count)

# get largest component from each LSA vector
lsa_vectors = lsa.components_
top_components = []

for vector in lsa_vectors:
    term_importance_map = pd.Series(vector.flatten(), index=count_terms).abs().sort_values(ascending=False)
    i = 0
    component = term_importance_map.index[i]
    while component in top_components:
        i += 1
        component = term_importance_map.index[i]
    top_components.append(component)

# put terms, counts and TF-IDF scores into a dataframe to make writing out the results easier
output = pd.DataFrame({"Term": count_terms, "Occurence": total_counts})
output.index = output["Term"]
output["TF-IDF"] = tfidf

# pick top scoring TF-IDF terms, write to file (CSV and XML)
output.nlargest(n_words_to_select, "TF-IDF")[["Occurence"]].to_csv("TF.csv", sep=';', encoding='utf-8')

# select top LDA components, write to file (CSV and XML)
output.loc[top_components, ["Occurence"]].to_csv("LSA.csv", sep=';', encoding='utf-8')
'''

