#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
start_time=time.time()
import numpy as np
import operator
from collections import Counter
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
def satoshi_to_bitcoin(satoshi):
    return float(float(satoshi)/ float(100000000))
test_data=bq_assistant.head("transactions")
def Create_Bar_plotly(list_of_tuples, items_to_show=40, title=""):
    #list_of_tuples=list_of_tuples[:items_to_show]
    data = [go.Bar(
            x=[val[0] for val in list_of_tuples],
            y=[val[1] for val in list_of_tuples]
    )]
    layout = go.Layout(
    title=title,xaxis=dict(
        autotick=False,
        tickangle=290 ),)
    fig = go.Figure(data=data, layout=layout)
    #py.offline.iplot(data,layout=layout)
    
    py.offline.iplot(fig)


# 
# ## So Bitcoins were pretty difficult to understand at first but while digging deep into it it seems quite simple. 
# <br>
# ### So in this notebook we will first learn about bitcoins and their workings, then try to find out some interesting insights from it.
# ### In case you already know about bitcoin transactions you can click through hyperlink and go directly to visualizations.
# ## CONTENTS
# <b><a href='#1'>1:  Wallets</a></b>
# <br>
# <b><a href='#2'>2:  Transactions</a></b><br>
# <b> &nbsp;&nbsp;  <a href='#2.2'>2.2: Transaction In a nutshell</a></b><br>
# <b> &nbsp;&nbsp;  <a href='#2.3'>2.3: Sample Transaction</a></b><br>
# <b> &nbsp;&nbsp;  <a href='#2.4'>2.4: Security in Transaction</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.4.1'>2.4.1: Short Description</a></b><br>
# <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.4.2'>2.4.2: Long Description</a></b>
# <br>
# <b><a href='#3'>3: Visualizations</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.1'>3.1: Who have the most number of Bitcoins.</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.2'>3.2: TimeSeries For Highest Valued Transactions</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.3'>3.3: Average Highest value transaction per year</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.4'>3.4: Bar Chart for top transaction counts per year</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.5'>3.5: Guess who got rich in 2018??</a></b><br>
# <b>&nbsp;&nbsp;<a href='#3.6'>3.6: Who did most of the transactions</a></b><br>
# <br>
# <br>
# <br>
# <br>
# <a id='1'></a>
# ## Wallets:
# <br>
# Concept of Wallets are quite simple, Basically lets say you have 100 dollars note. Where you will put it?, in a wallet right, just like that in Bitcoin wallets you put bitcoins but difference is that bitcoin wallet is digital.
# <br>
# 
# But there is a problem, How can you ensure that ONLY you can access 100 dollar note in your wallet and no one else. 
# 
# <b> To solve the above mentioned problem lets start from the beginning and see  how bitcoin wallet is created. </b>
# <br>
# 
# 
# <b> How you got to the address, your public bitcoin address.</b>
# <br>
# <b> How a wallet is created?</b>
# <br>
# Lets understand how a wallet is created and how it ensures that only you have access to the bitcoins you have
# <br>
# Here are three keywords you need to remember Private_key, Public_key, and Bitcoin_address.
# 
# 
# <b>Step 1:</b>
# <br>
# First of all you get a fixed length of random characters its called PRIVATE_KEY. You need to keep it hidden and dont have to give it to anyone else.  
# <br>
# <br>
# <b> Step 2:</b>
# <br>
# Now that you have a private key what you need to do is pass this private_key to a function. This is a special function which will give you a new key, lets name it Public Key.  
# <br>
# <br>
# public_key=FUNCTION(private_key)
# 
# We will skip how a function is implemented. To know how it is implemented we need to understand cry.
# 
# <br>
# But the point to notice is that it is a <b>one way function</b> and there is no practical way to get  private_key from public key.
# <br>
# 
# <b> Step 3</b>
# <br>
# Now that you have two things <b>Private_key</b> and <b>public_key</b>.
# What we will do now is pass the public_key to another function which gives us another key. Lets name it <b>address</b>.
# <br>
# <b> ADDRESS= Some_other_function(public_key)</b>
# 
# Now Address is something that you can give to anyone.
# If you want anyone to transfer some money intor your wallet  you will provide him this ADDRESS.
# 
# What wallet stores is your private_address and unspent outputs, we will undertand the unspent outputs in the next section.
# 
# Now the question which arises over here how can we use private_key, public_key and address to transfer money and ensure safety of your wallet.
# 
# <b>Summary</b><br>
# So here are few points if you missed anything, you had a private key: a random characters of fixed length, you passed it to a function to get a public key which was passed again to a function to get address. These functions are one way function which means if
# a=function(b)
# then there is no practical way to find <b>b</b> if you know <b>a</b>.
# <br>
# <br>
# <br>
# <br>
# <a id='2'></a>
# ## Transactions:
# <br>
# In this section we will understand how is bitcoins put in a wallet and how can only the wallet holder spends it and how it is transferred.
# <a id='2.1'></a>
# ### How transaction works in a NUTSHELL?
# In transactions there is a concept of input and outputs. 
# 
# In bitcoins each transaction have a number of inputs and outputs.
# In this scenario there are three people: IBAD, READER(you) and amazon
# 
# Now lets say you want to buy a nice watch. 
# So you go to the ecommerce website and it says to transfer 1 bitcoin to their address to get a watch.
# 
# So you will do is open your wallet and transfer 1 btc to Amazon (ecommerce website) to get a watch.
# 
# 
# Unfortunately you have no bitcoins.
# 
# So you came to me and asked me to transfer 100btc to your address and you provide me your "address" you made in the previous section.
# RECALL:    address=Function(public_key)
#                   public_keye=Function(private key)
#                   private key= "Random Characters of Fixed Length" Must be kept hidden
# 
# What is going to happen is this that I will write a transaction and it will look something like this. 
# 
# <b>TRANSACTION#1 by Ibad:</b><br>
# <b>INPUT:</b> Hey miner, I have 1000 bitcoins, Want a proof?. look at OUTPUT#2762
# 
# <b><a id='output123'>OUTPUT#123:</a></b>
# Hey Miner,  This is for the reader, he can spent 100 Bitcoins
# 
# <b><a id='output124'>OUTPUT#124:</a></b>
# Hey Miner, This is for ME, I must be able to spent 900 Bitcoins
# 
# I had 1000 bitcoins, I transferred 100 to you, and got 900 change back to myself.
# .
# Now after writing this transaction I send it to the bitcoin network, After some times a MINER sees this transactions, Validates it(checks it) just like the banker checks your cheque for forgery and approves it. Once approved this transaction will be marked as Confirmed and you can now spend your 100 bitcoins. 
# 
# 
# Now that you have 100 bitcoins now So you want to buy that watch from amazon so what you are going to do is write a transaction.
# 
# <b>TRANSACTION#2 by READER:</b>
# <br>
# <b>INPUT:</b> Hey miner, I have 100 bitcoins, Want a proof?.  look at <a href='#output123'>OUTPUT#123</a>
# 
# 
# <b>OUTPUT#125: </b>
# Hey Miner, This is for the amazon, they can spend 1 Bitcoin.
# 
# <b><a id='output126'>OUTPUT#126:</a></b>
# Hey Miner, This is for me(reader)(you)(as it is written by you hence i am using me), I must be able to spend 99 Bitcoins.
# 
# You had 100 bitcoins you sent 1 bitcoin to amazon and got 99 back to you.
# 
# #### Now  <a href='#output123'>OUTPUT#123</a>  is  SPENT OUTPUT
# 
# ####  <a href='#output126'>OUTPUT#126</a> is unspent output
# 
# #### <a href='output124'>Output#124</a> is unspent output.
# 
# 
# For the next time if you want to do another transaction you will reference <a href='#output126'>Output#126</a> in your INPUT.
# 
# 
# 
# 
# <a id='2.3'></a>
# ## SAMPLE TRANSACTION:
# Lets look at a sample input and sample output and lets see how they help in transactions.
# 
# To make it simple assume there are two people you and me, Now you have some bitcoins with you and want to transfer some of it to me. 
# 
# <b>Scenario:</b> You want to transfer me <b>0.04</b> bitcoins to me.
# <br>
# You currently have: <b>0.06</b> bitcoins.
# 
# 
# Let take a look at sample transaction input First and understand what does it implies.
# 

# In[ ]:


x=test_data.iloc[2].inputs[0]
x["input_pubkey_base58"]="1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y"
print (x)
#MODIFIED THE INPUT BECAUSE HEAD VALUE KEEPS ON CHANGING


# There are total of 6 properties over here, we will consider only those that are important to get the basic idea.
# The first property "input_pubkey_base58" is your ADDRESS which in this case is: <b>1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y</b>
# 
# The property "input_script_string" contains the reference to previous unspent output from where you are spending.
# 
# 
# inputs are just there for referencing the previous unspent outputs.
# 
# Now lets look at the output for this transaction.

# In[ ]:


x=test_data.iloc[3].outputs[0]
x["output_satoshis"]=4000000
x["output_pubkey_base58"]="1bonesF1NYidcd5veLqy1RZgF4mpYJWXZ"
print (x)
print ("-"*0)
x["output_satoshis"]=1159000
x["output_pubkey_base58"]="1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y"
print (x)
#MODIFIED THE INPUT BECAUSE HEAD VALUE KEEPS ON CHANGING


# Recall from above that your address is: 1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y
# 
# Here we can see there are two outputs.
# 
# output_satoshis is the amount of bitcoins you are sending. Sathoshi is the smallest unit of bitcoin. just like cents in dollars.
# 
# 1 Bitcoin=100,000,000 satoshis. 
# 
# So the first output says to transfer 4000000 satoshis (or 0.004 bitcoins) to  address output_pubkey_base58': '1bonesF1NYidcd5veLqy1RZgF4mpYJWXZ' which is my address. 
# 
# 
# Now the second output:
# So the second output says to transfer 1159000 satoshis (or 0.01159 bitcoins) to  address output_pubkey_base58': '1KEH32noJFb3tiBbWzLZo9nie6C4VhNP7Y' which is YOUR address. 
# 
# 
# Heres a summary of what happened: I had 0.006 bitcoins with me, I transferred 0.004 to your account,
# Got a change of 0.001159 back to my address.
# 
# But wait I should get 0.002 bitcoins back but I just got 0.001159 back, That's because the the rest of it will be counted as the transaction fees paid to the miner, In bitcoins you can set the transaction fees yourself. There is no limit on what transaction fees you set, you can set it from 0 to (total_amount - sent_amount)
# 
# 
# <a id='2.4'></a>
# ## SECURITY IN TRANSACTION
# 
# Now the most important part is: How can we ensure safety, Since all outputs and inputs are public, so anyone can write a transaction and transfer money from someone unspent output to his address. 
# 
# To ensure security bitcoins utilizes the concept of Digital Signatures.
# Whenever you write a transaction: the input is digitally signed:  To understand what it means take a look at this picture:
# [](https://www.cryptocompare.com/media/1284/digital_signature.png)
# 
# 
# If you are interested in just getting the idea of security in bitcoin you can just read the short description, if you want to get indepth detailed idea then read the long description of security in bitcoin.
# <br><br><br><br>
# <a id='2.4.1'></a>
# <br>
# <b>Short Description:</b>
# <br>
# Lets say you create a transaction to send 100 bitcoins to me, What you are going to do before sending it to miner is this that you will sign it with your private key, YOU will generate a signature with your private key, so
# Signature= Some_Function(private_key)
# 
# But there one property of this signature which will help us ensure security
# Now the miner can verify the signature with your PUBLIC_KEY, 
# 
# isSignatureValid= Verification_function(signature, public_key)
# 
# if isSignatureValid==True then the miner will validate transaction else miner will not accept the transaction.
# <br>
# <br>
# <a id='2.4.2'></a>
# <b>Long Description: </b>
# <br>
# Recall that the proof of having a bitcoin is the unspent output on your address. 
# The output looks something like this:
# 
# 
# 
# 
# 

# In[ ]:


print (test_data.iloc[2].outputs[0])


# Look closely there is a variable named output_script_string, this is a script, an output script which puts on the condition that only the private_key holder of the specified public_key have the rights to spend this output.
# Look at the last command <b>EQUALVERIFY CHECKSIG</b>, This checks the signature whether it matches the address or not.
# 

# Recall that inputs gives reference to the previous output,  In the input you will satisfy the condition put on by the output, to use it. 
# 
# In this way every bitcoin transaction is available to everyone to see but still its secure and immutable. 

# #### There are many more factors in transactions. 
#  https://unglueit-files.s3.amazonaws.com/ebf/05db7df4f31840f0a873d6ea14dcc28d.pdf
#  #### I read few chapters of this book, It contains detailed information about how Bitcoins works.
#  ## I tried to summarize the transaction and wallet chapter of this book. Do give suggestions on where I might be wrong or how can I improve my explanation. 

# <a id='3'></a>
# ## Now lets find something interesting
# 

# 
# ### Lets find the addresses who have the most number of bitcoins

# In[ ]:


q = """
SELECT  o.output_pubkey_base58, sum(o.output_satoshis) as output_sum from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o 
    where o.output_pubkey_base58 not in (select i.input_pubkey_base58
    from UNNEST(inputs) as i)
    group by o.output_pubkey_base58 order by output_sum desc limit 1000
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
query_job = client.query(q)
iterator = query_job.result(timeout=30)
rows = list(iterator)
results2 = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
results2["output_sum"]=results2["output_sum"].apply(lambda x: float(x/100000000))


# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = results2["output_pubkey_base58"][:10]
y_pos = np.arange(len(objects))
performance = results2["output_sum"][:10]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Bitcoins')
plt.title('Bitcoins Addresses Who received Most number of bitcoins')
plt.show()


# <a id='3.1'></a>
# ## The most number of bitcoins someone got in just ONE transactions.
# ### To do this what we are going to do is we will find out top1000 outputs from the data sorted by the amount of bitcoins transacted in a single output.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,o.output_pubkey_base58, o.output_satoshis as output_max from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o order by output_max desc limit 1000
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
query_job = client.query(q)
iterator = query_job.result(timeout=30)
rows=list(iterator)
results3 = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
#CONVERT SATOSHIS TO BITCOINS
results3["output_max"]=results3["output_max"].apply(lambda x: float(x/100000000))
results3.head()


# ### On 16th December 2011  "1M8s2S5bgAzSSzVTeL7zruvMPLvzSkEAuv".        This address received 500000 bitcoins IN A SINGLE TRANSACTION !!! What a lucky person...

# <a id='3.2'></a>
# ## Lets find out the timeseries plot for highest valued transactions.

# In[ ]:


results4=results3.sort_values(by="day")
layout = go.Layout(title="Time Series of Highest single output transaction")
data = [go.Scatter(x=results4.day, y=results4.output_max)]
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# ### It is quite clear that transactions of more than 100K bitcoins stopped in 2016.
# ### Maybe because people were transferring using multiple transactions. (using multiple addresses to evade the eyes of hacker on their addresses).

# In[ ]:


results4["day"]=results4["day"].apply(lambda x: x.year)
years_output={}
years_max_output_count={}
for i,x in results4.iterrows():
    if x["day"] not in years_output:
        years_output[x["day"]]=[]
    if x["day"] not in years_max_output_count:
        years_max_output_count[x["day"]]=[]
    years_output[x["day"]].append(x["output_max"])
    years_max_output_count[x["day"]].append(x["output_pubkey_base58"])
years_output_final={}
for x in years_output.keys():
    years_output_final[str(x)]=np.mean(years_output[x])
years_max_output_count_final={}
for x in years_max_output_count.keys():
    years_max_output_count_final[str(x)]=len(years_max_output_count[x])


# <a id='3.3'></a>
# ### Lets find out the average AMOUNT of transaction per year among top 1000 transactions from the data

# In[ ]:


d=Counter(years_output_final)
d.most_common(1)
Create_Bar_plotly(d.most_common(), title="Single Highest Valued Transaction Average Per Year")


# <a id='3.4'></a>
# ## Lets find out which year have most number of top1000 transactions.

# In[ ]:


d=Counter(years_max_output_count_final)
Create_Bar_plotly(d.most_common(), title="Most number of high transaction yearwise")


# ### 2012 seems to have majority of top1000 transactions, followed by 2011.
# ## One thing to notice that high amount transaction got quite low in 2013,2014,2015 and very low in 2016. 
# 
# ## It seems that in 2017 when bitcoins price were topped at 20000 dollars the investors sold out their bitcoins resulting in a huge number of transactions.

# <a id='3.5'></a>
# ### In 2018 there was just one transaction which made up to the top 1000 transactions list 

# In[ ]:


results4[results4.day==2018]
results3[results3.index==970]


# ### On 7th January 2018, someone just earned 51044 BITCOINS in just a single transaction !!!!.
# 
# ### Lets look at the transactions in 2010 which comes under the top1000 amount transactions.

# In[ ]:


results3.iloc[list(results4[results4.day==2010].index)]


# ### Lets look at the transactions in 2016 which comes under the top1000 amount transactions.

# In[ ]:


results3.iloc[list(results4[results4.day==2016].index)]


# 

# <a id='3.6'></a>
# ### Now lets find out who did most of the transactions

# In[ ]:


QUERY = """
SELECT
    inputs.input_pubkey_base58 AS input_key, count(*)
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
WHERE inputs.input_pubkey_base58 IS NOT NULL
GROUP BY inputs.input_pubkey_base58 order by count(*) desc limit 1000
"""
bq_assistant.estimate_query_size(QUERY)
ndf=bq_assistant.query_to_pandas(QUERY)


# In[ ]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
objects = ndf["input_key"][:10]
y_pos = np.arange(len(objects))
performance = ndf["f0_"][:10] 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects, rotation=90)
plt.ylabel('Number of transactions')
plt.title('BITCOIN ADDRESSES WITH MOST NUMBER OF TRANSACTIONS')
plt.show()


# In[ ]:


ndf.iloc[0]


# <b> The highest number of transactions was 1893290 done by the above mentioned address.</b>

# ### Please give your feedback and do tell me I am an wrong anywhere conceptually/ logically
# ### Upvote if you like it.
