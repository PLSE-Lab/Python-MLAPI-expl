#!/usr/bin/env python
# coding: utf-8

# **OBJECTIVE**
# The objective is to encode the address feature better than one-hot, ie. with less columns but without information loss.
# The objective is to get a sparse matrix whose each column corresponds to a street name, an avenue name, or more generally any element of interest.
# For this purpose we create a dictionnary with all important places (Broadway, 5th street,...), and important keywords (ex : rd, st, blvd...). To build this dictionnary we begin by cleaning the address field and we select the expressions that appear the most.
# If an address contains an element of the dictionnary it gets a one in the corresponding column.
# Therefore, this encoding of the address field is not stricty one-hot, as there can be more than 1 one per row. It is cleaner than one-hot encoding (less columns) and gives the results a nice boost :)
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from scipy.sparse import csr_matrix


# Now time for cleaning the addresses :

# In[ ]:


train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")
data=pd.concat([train.drop("interest_level",1),test],0)

#To select the rows corresponding to the train and the test set
trainSel=np.array([True]*len(train)+[False]*len(test))
testSel=np.array([False]*len(train)+[True]*len(test))

data=data.reset_index(drop=True)

#We need the field of interest
disadr=data.display_address    

#What does this feature look like ? 
disadr.head(10)


# In[ ]:


def cleanAdr(adress):
    adress=adress.lower()
    adress=adress.strip()
    adress=re.sub('\.','',adress)
    adress=re.sub(',','',adress)
    adress=re.sub('\'','',adress)
    adress=re.sub('(?P<num>[0-9]+)(th|st|rd|nd)','\g<1>',adress) 
    adress=re.sub('(?P<num>[0-9]+) (th|nd)','\g<1>',adress) 
    adress=re.sub('3 rd (?P<num>(st|rd|ave))','3 \g<1>',adress) 
    adress=re.sub('^[0-9]+-[0-9]+ ','',adress)
    adress=re.sub('street','st',adress) 
    adress=re.sub(' av$',' ave',adress)
    adress=re.sub('avenue','ave',adress)
    adress=re.sub('place','pl',adress)
    adress=re.sub('boulevard','blvd',adress)
    adress=re.sub(' lane',' ln',adress)
    adress=re.sub(' road',' rd',adress)
    adress=re.sub(' parkway',' pkwy',adress)
    adress=re.sub(' square',' sq',adress)
    adress=re.sub(' drive',' dr',adress)
    adress=re.sub(' park',' pk',adress)
    adress=re.sub('east','e',adress) 
    adress=re.sub('west','w',adress) 
    adress=re.sub('north','n',adress)
    adress=re.sub('south ','s',adress)
    adress=re.sub(' +',' ',adress)
    #special case of the abcx avenues
    adress=re.sub('(?P<num>(a|b|c|x)) ave','ave \g<1>',adress) 
    #Pb with streets that contain pk or st in their name (ex : pk avenue)
    adress=re.sub('.*pk ave','pk ave',adress)
    adress=re.sub('.*pk pl','pk pl',adress)
    adress=re.sub('.*st marks','st marks',adress)
    adress=re.sub('.*(?P<num>ave (x|a|b))','\g<1>',adress)
    #famous places without their specification (avenue,street,park...)
    #We choose the most common
    adress=re.sub('ave of the [a-z]+','6 ave',adress)
    adress=re.sub('^([0-9]+ )?madison$','madison ave',adress)
    adress=re.sub('^([0-9]+ )?thompson$','thompson st',adress)
    adress=re.sub('^([0-9]+ )?columbus$','columbus ave',adress)
    adress=re.sub('^([0-9]+ )?sullivan$','sullivan st',adress)
    adress=re.sub('^([0-9]+ )?john$','john st',adress)
    adress=re.sub('^([0-9]+ )?putnam$','putnam ave',adress)
    adress=re.sub('^([0-9]+ )?union$','union sq',adress)
    adress=re.sub('^([0-9]+ )?st marks$','st marks pl',adress)
    adress=re.sub('^([0-9]+ )?saint nicholas$','saint nicholas ave',adress)
    #streets without the "st"
    adress=re.sub('(?P<num>(e|w) [0-9]+)$','\g<1> st',adress)
    adress=re.sub('^(?P<num>[0-9][0-9])$','\g<1> st',adress)
    #The "and" without specifications
    adress=re.sub('(?P<num>[0-9]+) (and|&) (?P<n>([a-z]+|[0-9]+))','\g<1> st and \g<2> ave',adress)    
    adress=adress.strip()
    return adress


# In[ ]:


#We apply this cleaning to the address column
disadr=disadr.apply(cleanAdr)

#The addresses of the training set, for the dictionnaries
disadrTr=disadr[trainSel].reset_index(drop=True)

#What does the cleaned feature look like ?
disadr.head(10)


# Now that we have the cleaned address we can build the dictionnary :

# In[ ]:


def createDico(regex,serieofad,withdrawuniques):
    sel=serieofad.apply(lambda x:True if re.search(regex,x) else False)
    d=serieofad[sel]
    d=d.apply(lambda x:re.search(regex,x).group(0))
    dico=d.value_counts()
    if withdrawuniques:
        dico=dico[dico>2]
    dico=list(dico.index)
    return dico

#dictionnary of the streets
regex=re.compile('([0-9]+|[a-z]+) st')
dicost=createDico(regex,disadr,True)
dicostTr=createDico(regex,disadrTr,False)

#dictionnary of the avenues
regex=re.compile('(([0-9]+|[a-z]+) ave|ave (a|b|c|x))')
dicoave=createDico(regex,disadr,True)
dicoaveTr=createDico(regex,disadrTr,False)

#dictionnary of the other stuffs
regex=re.compile('([0-9]+|[a-z]+)( pkwy| rd| ln| pl| blvd| sq| dr| pk| terrace| heights)')
dicorest=createDico(regex,disadr,True)
dicorestTr=createDico(regex,disadrTr,False)



#is the word only in the testing set?
stsel1=list(map(lambda x:1 if x in dicostTr else 0,dicost))
avesel1=list(map(lambda x:1 if x in dicoaveTr else 0,dicoave))
restsel1=list(map(lambda x:1 if x in dicorestTr else 0,dicorest))


#We select the elements outside the intersections
dicost2=list(map(lambda x:re.sub(' st','',x),dicost))
dicoave2=list(map(lambda x:re.sub(' ave','',x),dicoave))
dicorest2=list(map(lambda x:re.sub('( pkwy| rd| ln| pl| blvd| sq| dr| pk| terrace| heights)','',x),dicorest))
stsel2=list(map(lambda x:0 if x in dicoave2+dicorest2 else 1,dicost2))
avesel2=list(map(lambda x:0 if x in dicost2+dicorest2 else 1,dicoave2))
restsel2=list(map(lambda x:0 if x in dicoave2+dicost2 else 1,dicorest2))



for i in range(len(dicost)):
    if stsel2[i]==1:
        dicost[i]=re.sub(' st','',dicost[i])
for i in range(len(dicoave)):
    if avesel2[i]==1:
        dicoave[i]=re.sub(' ave','',dicoave[i])
for i in range(len(dicorest)):
    if restsel2[i]==1:
        dicorest[i]=re.sub('( pkwy| rd| ln| pl| blvd| sq| dr| pk| terrace| heights)','',dicorest[i])



#Now we keep in the dictionnaries only the words present in the training set
dicost=[obj for ind,obj in enumerate(dicost) if stsel1[ind]==1]
dicoave=[obj for ind,obj in enumerate(dicoave) if avesel1[ind]==1]
dicorest=[obj for ind,obj in enumerate(dicorest) if restsel1[ind]==1]


# We can add special elements to the dictionnary :

# In[ ]:


#Additional dictionnaries - you can add as many words as you want
dicosp=["central pk","flatiron","broadway"]
dicodis=["financial district","hells kitchen","midtown","upper","soho","murray hill","village",
         "chelsea","tribeca","harlem","lower","astoria","kips bay","turtle bay","williamsburg"]
diconewfeatures=["st","ave","blvd","pkwy","rd","terrace","e","w"] 

#The final dictionnary !! -> dico
dico=dicost+dicoave+dicorest+dicosp+dicodis+diconewfeatures
dico=list(set(dico))

#Control of the dictionnary (we want to withdraw abnormal words)
l=pd.Series(list(map(len,dico)))
ind=l[l==1].index  #OK
del dico[410]

#An extract of the dictionnary
dico[165:175]


# For each address we want a list of all the dictionnary words it contains :

# In[ ]:


#We check the presence of each expression of the dictionnary in each address
def searchDico(x,dico):
    out=[]
    for i in range(len(dico)):
        if " "+dico[i]+" " in x:   #spaces are here to avoid '3 st' being found in '33 st'
            out.append(i)
    return out


disadr=disadr.apply(lambda x:' '+x+' ') 
streets=disadr.apply(lambda x:searchDico(x,dico)) 


# We can finally build the sparse matrix

# In[ ]:


#Direct building of the sparse matrix - much more efficient than building a DF and convert to sparse :)
nbof1=sum(streets.apply(len))
l=list(streets)
columns=[e for liste in l for e in liste]
ind=list(streets.apply(len))
ind=list(np.cumsum(ind))
ind=[0]+ind
display_address_sparse=csr_matrix(([1]*nbof1,columns,ind))


# In[ ]:


#Final check : small cleaning of the dictionnary

das=display_address_sparse.toarray()
das=pd.DataFrame(das)
s=das.apply(sum)
badwords=s[s==0].index

dico=list(np.delete(np.array(dico),badwords))  
display_address_sparse=display_address_sparse[:,np.nonzero(display_address_sparse.sum(axis=0))[1]]


# In[ ]:


display_address_sparse


# Appendix : a good feature, the presence of the street number

# In[ ]:


r=r"^([0-9]+) "
disadr2=disadr.apply(lambda x:re.sub('[0-9]+ (st|ave|rd|blvd)','',x))
sel=disadr2.apply(lambda x:True if re.search(r,x.strip()) else False)
street_number=pd.Series(np.zeros(len(data)))
street_number[sel]=1


# A big part of the dictionary is built automatically, but you can add or remove expressions and see the effects :)
# Personally, as there are still many columns, I used this matrix to reinforce my stacking only at the second level and it gave me a cool boost !
# It was a great competition and I hope you enjoyed this kernel, :)
