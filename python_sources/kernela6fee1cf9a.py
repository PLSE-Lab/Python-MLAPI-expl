#!/usr/bin/env python
# coding: utf-8

# In[8]:


from collections import Counter
import pprint


# First, some functions to calculate probabilities of victory for various L and D values (living and dead soldiers).

# In[5]:


def gotbattle2(l,d,pdict):
	if (l,d) in pdict.keys(): return (pdict[(l,d)],pdict)
	if l==0:p=0
	elif d==0:p=1
	else:
		l_wins, pdict = gotbattle2(l,d-1,pdict)
		d_wins, pdict = gotbattle2(l-1,d+1,pdict)
		p=0.5*(l_wins+d_wins)
	pdict[(l,d)]=p
	return(p,pdict)
	

def generate_gg(armysize,pdict={}):
	for size in range(armysize):
		for l in range(size+1):
			gotbattle2(l,size+1-l,pdict)
	
def extract_lists(pdict):
	llist,dlist,plist=[],[],[]
	for l,d in pdict.keys():
		p=pdict[(l,d)]
		llist.append(l)
		dlist.append(d)
		plist.append(p)
	return llist,dlist,plist


# Now let's generate probabilities for all of the cases where L+D <= 20 (i.e. 19 living vs 1 dead, 18 living vs 2 dead, 17 living vs 3 dead, etc.)  Then we'll count all the probabilities and see how many times we got 0.5.

# In[12]:


pdict={}
generate_gg(20,pdict)
pdl=list(pdict.values())
p_counts=Counter(pdl)
pprint.pprint(p_counts)


# Sixth from the top, we see that there's only one instance of 0.5, which is 1 living vs 1 dead.  As you can see, with L+D <= 20, this is already a very long list.  Now we're going to try a a much bigger number than 20, and we're going to get a very VERY long list, so this time we'll just check to see how many 0.5 entries we've got.

# In[15]:


generate_gg(2000,pdict)
pdl=list(pdict.values())
p_counts=Counter(pdl)
print(p_counts[0.5])


# There you have it, still only that first 0.5 for 1 living vs 1 dead.

# In[ ]:




