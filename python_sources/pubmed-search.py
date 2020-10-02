#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# Just a quick example of how to use [Biopython](https://biopython.org/) to search pubmed just in case you want to try augmenting your analysis with additional data that may not be in the CORD-19 subset.

# # Enter your email.  PubMed will contact you if your use gets excessive
# 
# Biopython limits the search queries for you so that it should not create any DNS problems.  If you have a key to pubmed, you can set that up and search more quickly.

# In[ ]:


email = 'put-your-email-here@needsinput.com'


# In[ ]:


pip install biopython


# In[ ]:


from Bio import Entrez, Medline
Entrez.email = email
handle = Entrez.einfo()
record = Entrez.read(handle)
handle.close()
print(record)


# In[ ]:


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
def getrecord(id, db):
    handle = Entrez.efetch(db=db, id=id, rettype='Medline', retmode='text')
    rec = handle.read()
    handle.close()
    return rec

def search(entrez_istance, terms, db='pubmed', mindate='2019/12/01'):
    handle = entrez_istance.esearch(db = db, term = terms, retmax=10, mindate=mindate)
    record = entrez_istance.read(handle)
    record_db = {}
    for id in record['IdList']:
        record = getrecord(id,db)
        recfile = StringIO(record)
        rec = Medline.read(recfile)
        record_db[int(id)] = {}
        if 'AB' in rec and 'AU' in rec and 'LID' in rec and 'TI' in rec:
            record_db[int(id)] = {}
            record_db[int(id)]['Authors'] = ' '.join(rec['AU'])
            record_db[int(id)]['doi'] = rec['LID']
            record_db[int(id)]['abstract'] = rec['AB']
            record_db[int(id)]['title'] = rec['TI']
        
    return record_db

search(Entrez, '2019-nCoV lungs')

