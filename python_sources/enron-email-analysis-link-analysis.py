#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Let's read in the dataset and see how it looks like! However, with such a large number of data, it is very tough to do a proper network visualisation later on. How about, we just take 500 of the data first..

# In[ ]:


#=== initial set up
import pandas as pd
import numpy as np
import os

pd.set_option("display.max_column", None)
pd.set_option("display.max_row", 50)

#=== reading in of data
emails = pd.read_csv('/kaggle/input/enron-email-dataset/emails.csv')

#=== make a copy of the dataframe
emails_df = emails.copy()
print(emails_df.head())

#=== take a small portion of the data for better visualisation results
emails_df = emails_df.sample(n = 200, random_state = 0)


# We create functions to split the text and to extract the meaningful texts from the email body. Also, we have other functions created to clean up the data fields and to extract email addresses, subjects and the name of the entity (sender/recipient)

# In[ ]:


#=== create a function to split text
import re
def split_text(text, match):	
    text = re.sub(r"\n\t", "", text)
    return re.split(match, text)

#=== create a function to extract proper text from the email body
def extract_body(text, substr):	
    result = re.split(substr, text)[-1]
    result = re.sub(r"([\n-])", "", result)
    return result

#=== clean up the data fields
#- function to extract email addresses
def extract_emails(text, substr):
    result = re.findall("[^\s]+@[\w]+.[\w]+", str(text))
    if substr not in text:
        result = ""
    return result

#- function to extract subject
def extract_subject(text):

    list_of_words = re.split("\s", text)
    words_to_drop = ["Subject:","re:","Re:","RE:","fw:","Fw:", "FW:"]

    desired_words = []
    for word in list_of_words:
        if word not in words_to_drop:
            desired_words.append(word)

    r = re.compile("[\w]{3,}")
    final_list = list(filter(r.match, desired_words))

    return final_list 

#- function to extract the name of entity
def extract_entity(text):	
    string = ""
    for i in text:
        string = string + " " + i

    list_of_emails = list(re.findall(r"@[\w]+", string))	
    result = []
    for item in list_of_emails:		
        result.append(item[1:])

    return set(result)


# We applied the "split_text" function and stored the results into a new column called "message_tidy". Let us take a look at our output for now..

# In[ ]:


#=== store output in new column
emails_df["message_tidy"] = emails_df.message.apply(lambda x : split_text(x, "\n"))

#=== take a look at the output
print(emails_df.head())
print(emails_df.message.head(1))
print(emails_df.message_tidy.head(1))


# We do some "feature engineering", by extracting useful information that can be used for the analysis such as "sender email", "recipient email" (because we really want to know who sends/receives email from who; critical to establish relationships).
# 
# We also extracted the useful information from the body of the emails, leaving out all the "noises".

# In[ ]:


#=== pull out useful data and post them into columns
emails_df["date"] = emails_df.message_tidy.apply(lambda x : x[1])
emails_df["sender_email"] = emails_df.message_tidy.apply(lambda x : x[2])
emails_df["recipient_email"] = emails_df.message_tidy.apply(lambda x : x[3])
emails_df["subject"] = emails_df.message_tidy.apply(lambda x : x[4])
emails_df["cc"] = emails_df.message_tidy.apply(lambda x : x[5])
emails_df["bcc"] = emails_df.message_tidy.apply(lambda x : x[9])
emails_df["body"] = emails_df.message.apply(lambda x : extract_body(x, r"X-FileName: [\w]*[\s]*[(Non\-Privileged).pst]*[\w-]*[.nsf]*").strip())


# We extracted the date information and email addresses, as well as to count the number of recipients. Quite a lot of effort for the data cleaning parts

# In[ ]:


#- extract date info
emails_df["day_of_week"] = emails_df.loc[:,"date"].apply(lambda x : x[5:9])
emails_df.loc[:,"date"] = emails_df.loc[:,"date"].apply(lambda x : x[10:22])

#- extract sender and recipient email
emails_df.loc[:,"sender_email"] = emails_df.loc[:,"sender_email"].apply(lambda x : extract_emails(x, "From: "))
emails_df.loc[:,"recipient_email"] = emails_df.loc[:,"recipient_email"].apply(lambda x : extract_emails(x, "To: "))
emails_df.loc[:,"cc"] = emails_df.loc[:,"cc"].apply(lambda x : extract_emails(x, "Cc: "))
emails_df.loc[:,"bcc"] = emails_df.loc[:,"bcc"].apply(lambda x : extract_emails(x, "Bcc: "))
emails_df["all_recipient_emails"] = emails_df.apply(lambda x : list(x["recipient_email"]) + list(x["cc"]) + list(x["bcc"]), axis = 1)
emails_df["num_recipient"] = emails_df.recipient_email.apply(lambda x : len(x)) + emails_df.cc.apply(lambda x : len(x)) +                                 emails_df.bcc.apply(lambda x : len(x))
    
#- extract sender and recipient entity info
emails_df["sender_entity"]    = emails_df.loc[:,"sender_email"].apply(lambda x : extract_entity(x))
emails_df["recipient_entity_to"] = emails_df.loc[:,"recipient_email"].apply(lambda x : extract_entity(x))
emails_df["recipient_entity_cc"] = emails_df.loc[:,"cc" ].apply(lambda x : extract_entity(x))
emails_df["recipient_entity_bcc"] = emails_df.loc[:,"bcc"].apply(lambda x : extract_entity(x))
emails_df["all_recipient_entities"] = emails_df.apply(lambda x :                                                  x["recipient_entity_to" ] |                                                  x["recipient_entity_cc" ] |                                                  x["recipient_entity_bcc"], axis = 1)

emails_df["sender_entity"] = emails_df.sender_entity.apply(lambda x : list(x))
emails_df["all_recipient_entities"] = emails_df.all_recipient_entities.apply(lambda x : list(x))

#- extract subject
emails_df.loc[:,"subject"] = emails_df.loc[:,"subject"].apply(lambda x : extract_subject(x))

#=== select and reorder the colums
df = emails_df.loc[:,["date","day_of_week","subject","body","sender_email","all_recipient_emails",
                                 "sender_entity","all_recipient_entities","num_recipient"]]  

print(df.head())


# We notice that the "all_recipients_emails" column contains multiple emails addresses and entities in one chunk. It is ideal, from network analysis standpoint, to split them up into multiple rows. For that purpose, we create the "investigate" function to do just that.

# In[ ]:


#=== function to expand list into multiple rows
#- we examine just the "sender_email" and "all_recipient_emails"
#- we try to break up the list (in "all_recipient_emails") into multiple rows
from itertools import chain
def investigate(df_col1, df_col2):
    result_df = pd.DataFrame({ "send" : np.repeat(df_col1.values, df_col2.str.len()),"receive": list(chain.from_iterable(df_col2))})
    result_df.send = result_df.send.apply(lambda x : x[0])
    return result_df


# Let's store the emails and the names of the entities

# In[ ]:


df2a = investigate(df.sender_email, df.all_recipient_emails)
df2b = investigate(df.sender_entity, df.all_recipient_entities)


# We are finally at the network analysis step; this is the easy part of the whole notebook. We do two different network diagrams, one for emails and one for entities

# In[ ]:


#=== network analysis
import networkx as nx
import matplotlib.pyplot as plt

#=== for the sender and recipient emails
#- define the graph
G1 = nx.from_pandas_edgelist(df2a.sample(round(0.1*len(df2a)), random_state = 0), "send", "receive")

#- we define the closeness measure
closeness_G1 = nx.closeness_centrality(G1)
closeness_G1 = list(closeness_G1.values())

#- plot the network
plt.figure(figsize = (20,20))
pos1 = nx.spring_layout(G1, k=.1)
nx.draw(G1, pos1, node_size = 20, node_color = closeness_G1, with_labels = True)
plt.show()


# We see that some of the nodes are unlinked to the rest of the clusters. Be mindful that we are taking a random sample and it is not definite that these "standalone" nodes are indeed far from the clusters. To do a more meaningful visual analysis, you will need to do an extraction of the data containing particular entity names or email addresses. What we presented here is just a means to show what Python is capable of showing.

# In[ ]:


#=== for the sender and recipient entities
#- define the graph
G2 = nx.from_pandas_edgelist(df2b, "send", "receive")

#- we define the closeness measure
closeness_G2 = nx.closeness_centrality(G2)
closeness_G2 = list(closeness_G2.values())

#- plot the network
plt.figure(figsize = (20,20))
pos2 = nx.spring_layout(G2, k=.1)
nx.draw(G2, pos2, node_size = 300, node_color = closeness_G2,with_labels = True)
plt.show()

