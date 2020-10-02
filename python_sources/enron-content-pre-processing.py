#!/usr/bin/env python
# coding: utf-8

# # 1. Loading data in splits:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regex python
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

from nltk.tokenize import word_tokenize

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


filename = '/kaggle/input/enron-email-dataset/emails.csv'
n = 50  # every 100th line = read 1% of the emails (total emails = 517400)
df = pd.read_csv(filename, header=0, skiprows=lambda i: i % n != 0)

# df = pd.read_csv(filename)
print("shape of the dataset:",df.shape)
df.head()


# Let's see what individual messages looks like. They all have a set of pre-info containing various features and a body separated by two newline characters.

# In[ ]:


# for i in range(20):
#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")
#     print(df.message[i])


# # 2. Split the content and pre-info in message:

# In[ ]:


# Dropping the file column:
df = df.drop(['file'], axis=1)


# In[ ]:


# Splitting:
df['pre_info']= df.message.map(lambda r: r.split('\n\n', 1)[0])
df['content']= df.message.map(lambda r: r.split('\n\n', 1)[1])
df = df.drop(['message'], axis=1)
df.head()


# In[ ]:


# Check the pre-info part:
print(df.pre_info[0])
# Keep the message id for indexing later on:
# df['message_id'] = df.pre_info.map(lambda r: r.split('\n')[0].split('Message-ID: ')[1])
# df = df.drop(['pre_info'], axis=1)


# # 3. Dealing with *Forwarded by*:

# In[ ]:


# #Investigating the content first:
# for i in range(25):
#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")
#     print(df.content[i])


# From the above samples it is clear that there are a whole lot of messages that have `---------------------- Forwarded by ...` in them.
# Exactly how many? Let's see:

# In[ ]:


df.content.str.contains('[- ]*Forwarded by').value_counts()


# Let's deal with them first. How? Looking at the forwarded texts, they are accompanied by a `To, From and Subject` texts. We are interested in the body of the email that comes after the `Subject`.

# The regex below simply says: 
# * `[- ]*` means zero or more number of `-` and `:space:` characters, because of wildcard `*`
# * `\s` means all whitespace characters including newline and tabs
# * `\S` means complimentary set of all the whitespace characters, basically all the alphanumeric and special characters
# * Together `[\S\s]*` means zero or more characters (everything) 
# * `[\S\t ]*` means zero or more characters except for a newline

# In[ ]:


# Test the deal with one sample email:
email = df.content[df.content.str.contains('[- ]*Forwarded by')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = '[- ]*Forwarded by[\S\s]*Subject:[\S\t ]*'
print(re.sub(condition, '', email).strip())

# Do it for all the others:
def deal_forwarded(row):
    condition = '[- ]*Forwarded by[\S\s]*Subject:[\S\t ]*'
    return re.sub(condition, '', row).strip()
df['content1'] = df.content.map(deal_forwarded)


# Has it worked successfully?

# In[ ]:


print(df.content1.str.contains('[- ]*Forwarded by').value_counts())


# Almost. So let's look at the pattern for these emails:

# In[ ]:


# for email in df.content1[df.content1.str.contains('[- ]*Forwarded by')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# Looking at above emails, there's no clear pattern that can detected to remove from these emails. But we can just remove the `----- Forwarded by .. -----` alone.

# In[ ]:


# Test for one email:
email = df.content1[df.content1.str.contains('[- ]*Forwarded by')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = '[- ]*Forwarded by[\S\s]*---[-]+' 
print(re.sub(condition, '', email).strip())

# DO it for all the others:
def deal_forwarded_patternless(row):
    condition = '[- ]*Forwarded by[\S\s]*[-]+'
    return re.sub(condition, '', row).strip()
df['content2'] = df.content1.map(deal_forwarded_patternless)


# In[ ]:


print(df.content2.str.contains('[- ]Forwarded by').value_counts())


# Sweet, we've eliminated all the Forwards! 

# # 4. Dealing with *Original Message*:

# Investigating Content again:

# In[ ]:


# for i in range(10,50):
#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")
#     print(df.content2[i])


# Looks like emails contain `-----Original Message-----` in them a lot. Exactly how many?

# In[ ]:


df.content2.str.contains('[- ]Original Message').value_counts()


# Alright then the same thing applies here as well. Looking at the forwarded texts, they are accompanied by a `To, From and Subject` texts. 
# We are interested in the body of the email that comes after the `Subject`.

# In[ ]:


# Test the deal with one sample email:
email = df.content2[df.content2.str.contains('[- ]Original Message')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = '[- ]*Original Message[\S\s]*Subject:[\S\t ]*'
print(re.sub(condition, '', email).strip())

# Do it for all emails:
def deal_originals(row):
    condition = '[- ]*Original Message[\S\s]*Subject:[\S\t ]*'
    return re.sub(condition, '', str(row)).strip()
df['content3'] = df.content2.map(deal_originals)


# Has it worked successfully?

# In[ ]:


df.content3.str.contains('[- ]*Original Message').value_counts()


# Almost. So let's look at the failures:

# In[ ]:


# for email in df.content3[df.content3.str.contains('[- ]*Original Message')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# Looks like they contain patterns like this that end with `Sent:` or `Date:` instead of `Subject:`
# ```
# ----- Original Message -----
# From: <Eric.Bass@enron.com>
# To: <daphneco64@bigplanet.com>
# Sent: Monday, December 18, 2000 1:41 PM
# ```

# In[ ]:


# Test the deal with one sample email:
email = df.content3[df.content3.str.contains('[- ]*Original Message')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = '[- ]*Original Message[\S\s]*(Sent:[\S\t ]*|Date:[\S\t ]*)'
print(re.sub(condition, '', email).strip())

# Do it for all emails:
def deal_originals_new(row):
    condition = '[- ]*Original Message[\S\s]*(Sent:[\S\t ]*|Date:[\S\t ]*)'
    return re.sub(condition, '', str(row)).strip()
df['content4'] = df.content3.map(deal_originals_new)


# In[ ]:


# Check again:
# print(df.content4.str.contains('[- ]*Original Message').value_counts())
# for email in df.content4[df.content4.str.contains('[- ]*Original Message')]:
#     print(email)
#     print('############################################## END OF EMAIL ###############################################')


# The above few emails have varied patterns as they have different use cases of Original message. So leaving them here.

# # 5. Dealing with *From*:

# Checking content once again

# In[ ]:


# for i in range(50,90):
#     print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")
#     print(df.content4[i])


# From the above samples, we can see that there are quite a few entries with this pattern with the `From:` entry:
# ```
# From: John J Lavorato/ENRON@enronXgate on 02/19/2001 10:19 AM
# To: John Arnold/HOU/ECT@ECT
# cc:  
# Subject: 
# ```
# So let's deal with them.

# In[ ]:


# emails containing the pattern:
df.content4.str.contains('From:[\S\s]*Subject:[\S \t]*').value_counts()


# In[ ]:


# Test the deal with one sample email:
email = df.content4[df.content4.str.contains('From:[\S\s]*Subject:[\S \t]*')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = 'From:[\S\s]*Subject:[\S \t]*'
print(re.sub(condition, '', email).strip())

# Do it for all emails:
def deal_from(row):
    condition = 'From:[\S\s]*Subject:[\S \t]*'
    return re.sub(condition, '', str(row)).strip()
df['content5'] = df.content4.map(deal_from)


# Is the task complete?

# In[ ]:


df.content5.str.contains('From:').value_counts()


# Nope, so what went wrong?

# In[ ]:


# for email in df.content5.loc[df.content5.str.contains('From:')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# Let's leave at this for now, because there are no recognizable patterns here.

# # 6. Dealing with *To*:

# #### Various patters to deal with here, so brace yourselves!!!
# ```
# "Mark Sagel" <msagel@home.com> on 05/13/2001 09:23:02 PM
# To: "John Arnold" <jarnold@enron.com>
# cc:  
# Subject: Natural gas update
# ```
# ```
# Ina Rangel
# 05/10/2001 05:47 PM
# To:	John Arnold/HOU/ECT@ECT
# cc:	 
# Subject:	Question?
# ```
# ```
# Kristin Gandy@ENRON
# 09/11/2000 09:30 AM
# To: Mark Koenig/Corp/Enron@ENRON, Kevin Garland/Enron Communications@Enron 
# Communications, Jonathan Davis/HOU/ECT@ECT, Jian 
# Miao/ENRON_DEVELOPMENT@ENRON_DEVELOPMENT, Ed Wood/HOU/ECT@ECT, Jun Wang/Enron 
# Communications@Enron Communications, Miguel Vasquez/HOU/ECT@ECT, Lee 
# Jackson/HOU/ECT@ECT, Amber Hamby/Corp/Enron@Enron, Jay Hawthorn/Enron 
# Communications@Enron Communications, Joe Gordon/Corp/Enron@Enron, Susan 
# Edison/Enron Communications@Enron Communications, Vikas 
# Dwivedi/NA/Enron@Enron, Mark Courtney/HOU/ECT@ECT, Monica Rodriguez/Enron 
# Communications@enron communications
# cc: Seung-Taek Oh/NA/Enron@ENRON, John Arnold/HOU/ECT@ECT, Andy 
# Zipper/Corp/Enron@Enron, George McClellan/HOU/ECT@ECT, David 
# Oxley/HOU/ECT@ECT 
# Subject: Vanderbilt Presentation and Golf Tournament
# ```
# ```
# Date: Thu, 23 May 2002 11:03:58 -0500
# Message-ID: <C4F6659E22D8194B925933CC808802540FC15D@server4.zilkha.com>
# X-MS-Has-Attach:
# X-MS-TNEF-Correlator:
# Thread-Topic: Mill Run & Somerset Monthly reports
# Thread-Index: AcICc2zvlTjbDznqRMKWdw4R6bKDng==
# From: "Gary Verkleeren" <GVerkleeren@zilkha.com>
# To: <kurt.anderson@ps.ge.com>
# Cc: "Rick Winsor" <rwinsor@zilkha.com>, "Mark Haller" <mhaller@zilkha.com>, 
# <joseph.thorpe@ps.ge.com>
# ```

# In[ ]:


# emails containing `To`:
df.content5.str.contains('To:[\S\s]*Subject:[\S\t ]*').value_counts()


# In[ ]:


# emails containing `To` on a new line:
df.content5.str.contains('\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*').value_counts()


# In[ ]:


# emails containing zero or one line before `To`:
df.content5.str.contains('\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*').value_counts()


# In[ ]:


# emails containing zero or one or two lines before `To`:
df.content5.str.contains('\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*').value_counts()


# In[ ]:


# # Investigate +_+
# for email in df.content5.loc[df.content5.str.contains('\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# In[ ]:


# Test the deal with one sample email:
email = df.content5.loc[df.content5.str.contains('\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S\t ]*')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = '\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S \t]*'
print(re.sub(condition, '', email).strip())

# Do it for all emails:
def deal_to(row):
    condition = '\n[\S \t]*\n[\S \t]*\n[\S \t]*To:[\S\s]*Subject:[\S \t]*'
    return re.sub(condition, '', str(row)).strip()
df['content6'] = df.content5.map(deal_to)


# See if we have eliminated all `To:` to `Subject:` as well:

# In[ ]:


df.content6.str.contains('To:[\S\s]*Subject:[\S\t ]*').value_counts()


# Not yet, so what do we have here?

# In[ ]:


# for email in df.content6.loc[df.content6.str.contains('To:[\S\s]*Subject:[\S\t ]*')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# All of these weren't detected because they don't have new line characters in the begenning. But they are easy to remove now!

# In[ ]:


# Test the deal with one sample email:
email = df.content6.loc[df.content6.str.contains('To:[\S\s]*Subject:[\S\t ]*')].iloc[0]
print(email)
print("############################ END OF EMAIL ################################################################")
condition = '[\S\t ]*\nTo:[\S\s]*Subject:[\S \t]*'
print(re.sub(condition, '', email).strip())

# Do it for all emails:
def deal_to_new(row):
    condition = '[\S\t ]*\nTo:[\S\s]*Subject:[\S \t]*'
    return re.sub(condition, '', str(row)).strip()
df['content7'] = df.content6.map(deal_to_new)


# In[ ]:


df.content7.str.contains('To:').value_counts()


# In[ ]:


# for email in df.content7.loc[df.content7.str.contains('To:')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# Looking above, there's no clear pattern emerging. Hence, let's keep them in tab while we look at other problems.

# # 7. The '*=20*' problem:

# Honestly, this one is such a nuisance! I believe this character comes due to some read error/ incompatible character set (probably not UTF-8) that Enron was using

# In[ ]:


print(df.content7.str.contains('=20').value_counts())
print(df.content7.str.contains('=20|=10|=09|=01').value_counts())
print(df.content7.str.contains('=\d\d').value_counts())


# In[ ]:


# for email in df.content7.loc[df.content7.str.contains('=20|=10|=09|=01')]:
#     print(email)
#     print("############################ END OF EMAIL ################################################################")


# In[ ]:


# Test the deal with one email:
email = df.content7.loc[df.content7.str.contains('=20|=10|=09|=01')].iloc[1]
print(email)
print("############################ END OF EMAIL ################################################################")
# condition = '[=]+[\n\t =]*\d\d'
# |[=\n]*[=10]|[=\n]*[=01]|[=\n]*[=09]'
# [\w]=[\w]'
condition1 = '[=]+\d\d'
condition2 = '[=]+[ \n]+'
email = re.sub(condition1, '', email)
print("############################## AFTER COND 1 ###############################################################")
print(re.sub(condition, '', email).strip())
email = re.sub(condition2, '', email)
print("############################## AFTER COND 2 ###############################################################")
print(re.sub(condition, '', email).strip())


# Do this for all emails:
def deal_equalsto(row):
    condition1 = '[=]+\d\d'
    condition2 = '[=]+[ \n]+'
    row = re.sub(condition1, '', str(row))
    return re.sub(condition2, '', str(row)).strip()
df['content8'] = df.content7.map(deal_equalsto)


# In[ ]:


print(df.content8.str.contains('=20').value_counts())
print(df.content8.str.contains('=20|=10|=09|=01').value_counts())
# print(df.content8.str.contains('=\d\d').value_counts())


# Sweet! All intended pre-procesing has been done!!! Let's take a look at the final data :)

# In[ ]:


for i in range(1000, 1050):
    print("################################################ EMAIL CONTENT NUMBER:",i,"############################################################################")
    print(df.content8[i])


# # 8. Other patterns or problems that creep in:

# We find html links sometimes:
# 
# ```
# Planned subsidies would distort competition 
# 
# Oct. 23, 2001 (Helsingin Sanomat) -- The Finnish forest industry has severely criticised the financing arrangements of the new pulp ...
# ( Story... <http://www.forestweb.com/digest/news.control.taf?_section=liststories&_function=detail&FORESTWEBNEWS_uid1612&_UserReference=9BACFC3BFDBDF3C13BD590B2>) 
# 
#  	
# top of page <http://www.forestweb.com/digest/news.control.taf?_section=view&_function=detail&EDITION_ID96&_UserReference=9BACFC3BFDBDF3C13BD590B2#top> 	
#   _____  
# ```
# Or it has characters like these: `>`, which are mostly forwards from other email services like Yahoo!
# ```
# CECECOMCOM@aol.com <mailto:CECECOMCOM@aol.com> wrote: 
# 
# > ATTACHMENT part 2 message/rfc822 Date: Thu, 13 Dec 2001 11:48:47 -0800 (PST)
# > FYI
# > 
# > Roadside Assistance: Something all Texas women should know... especially
# > since this has received very little publicity.
# > 
# > Your Texas drivers license has a phone number in small print on the
# > back,
# > just above the bar code: 1-800-525-5555
# ```
# But these are found once every hundred emails, hence it's safe to ignore them.

# # 3. Pre-processing for NLP: 

# Processing for NLP requires us to tokenize and select emails that confirm to certain expectations that we set

# In[ ]:


# Getting the final content8 out:
df_nlp = df[['pre_info', 'content8']]
# Removing the emails that are empty:
df_nlp = df_nlp.loc[~(df_nlp.content8=='')]
df_nlp = df_nlp.rename(columns={'content8':'content'}).reset_index(drop=True)
print(df_nlp.shape)
df_nlp.head()


# In[ ]:


# # Tokenize using your fav tokenizer:
# # This step is quite time consuming: 
df_nlp['bert_tokens'] = df_nlp['content'].map(lambda r: tokenizer.tokenize(r))
df_nlp['nltk_tokens'] = df_nlp['content'].map(lambda r: word_tokenize(r))


# This is what you've been looking for!!

# In[ ]:


df_nlp.to_csv('enron-pre-processed-nlp.csv')


#  Consider only emails that have more than 10 tokens and less than 512 tokens for BERT processing purposes:

# In[ ]:


# Saving a file for BERT transformer models:
df_out_final = df_nlp.content.loc[df_nlp['bert_tokens'].map(lambda r: (len(r)>10 and len(r)<512))].reset_index(drop=True)
print(df_out_final.shape)
# Save the final contents:
df_out_final.to_csv('enron-processed.tsv',sep='\t')


# In[ ]:


token_list = [token for row in df_nlp['nltk_tokens'] for token in row]
# token_list = [token for row in df_nlp['bert_tokens'] for token in row]
token_series = pd.Series(token_list)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(token_series.value_counts())


# ## Viola! Your emails are ready for Large scale language models like BERT! Check the output directory :)
