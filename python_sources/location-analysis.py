import networkx as nx
# I am using Stanford CoreNLP for Named Entity Recognition

import re
import os
import pandas as pd
import numpy as np
import json
df_aliases = pd.read_csv('../input/Aliases.csv', index_col=0)
df_emails = pd.read_csv('../input/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('../input/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('../input/Persons.csv', index_col=0)

from nltk.stem.porter import PorterStemmer
import nltk

from corenlp import *
corenlp = StanfordCoreNLP()  # wait a few minutes...
#result=corenlp.parse("Parse this sentence.")

def isLocation(sentenceString):
	try:	
		print(sentenceString)
		sentenceString.replace('/', '')
		#sentenceString = sentenceString.decode('utf-8').encode('ascii', 'replace')
		import re
		result=corenlp.parse(sentenceString)
		j=json.loads(result)
		for i in j['sentences']:
			for k in i['words']:
				if k[1]['NamedEntityTag']=='LOCATION':
					return k[0]
		return('NA')
	except:
		return('NA')

df_emails['Location']=df_emails['MetadataSubject'].map(isLocation)

# We will now be doing Region analysis
data=df_emails
locationCount=data['Location'].value_counts()

locationDataTop10=locationCount.head(10)
x_pos = np.arange(len(locationDataTop10.index))
plt.bar(x_pos,locationDataTop10.values)
plt.xlabel('Region')
plt.ylabel('Reference Count')
plt.xticks(x_pos,locationDataTop10.index,rotation=70)
plt.show()


