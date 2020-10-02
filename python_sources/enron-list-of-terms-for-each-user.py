import pandas as pd 
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os, sys, email, re, string
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
import csv

lemmatiser = WordNetLemmatizer()

print("Starting...")

def clean(text):
    # Function that get string and return it after cleaning as pure terms
    stop = set(STOPWORDS)
    # Excludes irealevant words 
    stop.update(('fwd','RE','FW','Hello','Meeting','Ga','Access','positions','list','forward','floor','collar',
    'fixed','enron','hou','ect','corp','please','vince','time','mail','john','kay','day','message','week',
    'kaminski','year','meeting','enronxgate','question','group','work','call','scott','change','company','let',
    'mann','date','number','mark','today','david','mike','issue','houston','chris','subject','way','bass','jeff',
    'edu','office','doc','don','month','copy','name','comment','email','need','phone','point','thing','request',
    'look','ben','michael','list','help','delainey','fax','morning','use','tomorrow','thank','phillip','hotmail',
    'guy','robert','night','lon','part','talk','kate','home','mailto','person','address','form','jeffrey',
    'something','end','line','hour','place','march','love','anything','paul','giron','smith','hope','darron',
    'jim','kevin','weekend','george','north','someone','section','richard','discus','bob','jacoby','ena','room',
    'see','demand','desk','area','everyone','greg','detail','jason','afternoon','discussion','tom','kslaw',
    'check','basis','visit','mcconnell','miller','entity','location','peter','monday','response','show','page',
    'jennifer','lot','meet','respond','yesterday','pdx','house','june','larry','jan','dan','city','july','judy',
    'friday','julie','shirley','meter','level','fyi','addition','martin','anyone','generation','department',
    'type','rick','friend','period','word','lisa','think','class','johnson','org','robin','thompson',
    'columbiagas','didn','april','william','lee','thomas','hey','adam','stephen','man','sender','tim','taylor',
    'organization','center','everything','ferc','notice','start','davis','york','sorry','cell','return','street',
    'hernandez','thursday','campbell','care','content','curve','minute','floor','stinson','janet','head','move',
    'kind','kent','tuesday','sheila','send','suzanne','brenda','kim','matter','fgt','carolyn','cindy',
    'ccampbell','tell','fwd','crenshaw','baumbach','linda','side','clark','mind','hain','wharton','future',
    'errol','carlos','hand','matt','bruce','gossett','brian','try','wednesday','calendar','laura','nothing',
    'doug','llc','rebecca','rob','stephanie','austin','victor','join','joseph','couple', 'kean', 'arnold', 
    'var', 'keith', 'lucy', 'grigsby', 'allen'))
    # Punctuation (formerly sometimes called pointing) is the use of spacing, conventional signs, 
    # and certain typographical devices as aids to the understanding and the correct reading, 
    # both silently and aloud, of handwritten and printed texts. 
    exclude = set(string.punctuation) 
    # Lemmatize the terms
    lemma = WordNetLemmatizer()
    # The method rstrip() returns a copy of the string in 
    # which all chars have been stripped from the end of the string (default whitespace characters).
    text=text.rstrip()
    # Remaind only with letters without anything else
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Removing stopwords, digits and word lenth less than 3
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()) and len(i)>2)])
    # Exclude punctuation
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # Adds the terms after running Lemmatizer
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    # Building tokens from the words to remain only the common nouns as terms
    tokens = nltk.word_tokenize(normalized)
    tags = nltk.pos_tag(tokens)
    nouns =" ".join(word for word,pos in tags if (pos == 'NN' and word not in stop and len(word)>2))
    return nouns

def unique_words_count(str):
    # Function that get string and counts each word in the string 
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        elif(len(word)>2):
            counts[word] = 1

    return len(counts)
    
def get_text_from_email(msg):
    # Function that gets email and return the content as list of strings
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload().lower() )
    return ''.join(parts)

def add_terms(str):
    # Function that get string and insert and count each word to termsCountPerUser 
    uniqueWords = []
    words = str.split()

    for word in words:
        # Loop that define only unique terms
        if word not in uniqueWords:
            uniqueWords.append(word)
            
    for word in uniqueWords:
        # Searching the term and the user in the DB
        Found=0
        for t in range(0,len(tCPU)):
            if (word == tCPU[t][0] and currentname == tCPU[t][1]):
                Found=1
                tCPU[t][2] += 1
                break
                
        if (Found == 0):
            # Define the new term and user because it was not found in the data base
            tCPU.append([])
            tCPU[t+1].append(word)
            tCPU[t+1].append(currentname)
            tCPU[t+1].append(1)

CountTermsGlobaly=0
inputrows = 500000
user = 0
maxword = 50

# Load the CSV 
emails_df = pd.read_csv("../input/emails.csv" ,nrows=inputrows) 
# Transform the CSV to list of strings
messages = list(map(email.message_from_string, emails_df['message']))

keys = messages[0].keys()

print("Starting making the keys...")

# Define Keys as headers
for key in keys: emails_df[key] = [doc[key] for doc in messages]

# Load the emails content to list of strings
emails_df['content'] = list(map(get_text_from_email, messages)) 

currentname=emails_df['X-Origin'][0].lower()
usernum=0

if (CountTermsGlobaly==1):
    text_clean=[[]]
    text_clean[0].append(currentname)
    text_clean[0].append(0)

tCPU=[[]]
tCPU[0].append('forecast')
tCPU[0].append(currentname)
tCPU[0].append(0)

Users=[[]]
Users[0].append(currentname)
Users[0].append(0)

print("Starting building the Data-Base...")
emails_df['content'] = list(map(get_text_from_email, messages)) 
print("#",usernum,"User Name = ",currentname," | i = ",1)

for i in range(450000,inputrows):
    if(emails_df['X-Origin'][i] and emails_df['X-Origin'][i].lower() == currentname):
        if(emails_df['X-Folder'][i] and "sent" in emails_df['X-Folder'][i].lower()):
            if (CountTermsGlobaly==1):
                text_clean[usernum][1] += 1
                text_clean[usernum].append(clean(emails_df['content'][i].split("to: ")[0]))
            Users[usernum][1] += 1
            add_terms(clean(emails_df['content'][i].split("to: ")[0]))
    else:
        # The current user name had been changed
        if(emails_df['X-Origin'][i]): 
            
            currentname=emails_df['X-Origin'][i].lower()
            # Check if the current user name already exist in the list
            found=0
            for u in range(0,usernum+1):
                if(Users[u][0]==currentname):
                    Users[u][1] += 1
                    add_terms(clean(emails_df['content'][i].split("to: ")[0]))
                    found=1
                    break
                
                if (CountTermsGlobaly==1):
                    if(text_clean[u][0]==currentname):
                        text_clean[u][1] += 1
                        text_clean[u].append(clean(emails_df['content'][i].split("to: ")[0]))
                        found=1
                        break
                
            # The current user name is not exist in the list
            if(found==0):
                usernum+=1
                Users.append([])
                Users[usernum].append(currentname)
                Users[usernum].append(1)
                add_terms(clean(emails_df['content'][i].split("to: ")[0]))
                print("#",usernum,"User Name = ",Users[usernum][0]," | i = ",i)
                
                if (CountTermsGlobaly==1):
                    text_clean.append([])
                    text_clean[usernum].append(currentname)
                    text_clean[usernum].append(1)
                    text_clean[usernum].append(clean(emails_df['content'][i].split("to: ")[0]))
                    print("#",usernum,"User Name = ",text_clean[usernum][0]," | i = ",i)
                

usersnames = []
emailsnumber = []

#print(tCPU)

with open('TermsPerUser.csv', 'w') as csvfile:
    fieldnames = ['Term', 'User', 'Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    for r in range(0,len(tCPU)):
        writer.writerow({'Term': tCPU[r][0], 'User': tCPU[r][1], 'Count': tCPU[r][2]})

    if (CountTermsGlobaly==1):
        for u in range(0,usernum+1):
            usersnames.append(text_clean[u][0])
            emailsnumber.append(text_clean[u][1])
            text_clean[u]= re.sub(r'[^a-zA-Z]', ' ', str(text_clean[u]))
            print("#",u,"-",usersnames[u],"Sent",emailsnumber[u],"emails, and uses",
            unique_words_count(str(text_clean[u])),"unique words, and totaly uses",len(str(text_clean[u])),"words")

'''   
wordcloud = WordCloud(
    max_font_size=50,
    background_color='white',
    max_words=maxword
    ).generate(str(text_clean[user]))
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig(usersnames[user])
'''