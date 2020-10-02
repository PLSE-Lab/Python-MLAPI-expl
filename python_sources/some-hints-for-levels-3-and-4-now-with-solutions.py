#!/usr/bin/env python
# coding: utf-8

# There are already several excellent explorations of the level 1 and 2 ciphers, so I'm not going to retread old ground here: instead, here's a look at levels 3 and 4 ciphers with a few hints that may help to nudge you in the right direction if you're stuck. I've deliberately not included the full solution to either cipher because I don't want to spoil the fun for people who are still working on them.
# 
# Update September 9th: Now includes solutions!
# 
# Thank you so much to Team Kaggle for running this competition: I started using pandas literally three weeks ago and this was a really fun way to solidify that knowledge and get some hands-on experience!

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import random
import math
import string
import itertools
import copy

from fuzzywuzzy import fuzz 
from tqdm import tqdm


# Most of these packages are self-explanatory. `fuzzywuzzy` is for comparing chunks of text (too slow for the final match-up, but great for testing potential decryption algorithms on individual examples), and`tqdm` lets you seamlessly add progress counters to for-loops. It's super useful for longer computations!
# 
# Before we start, under the cut are some basic functions to do things like pad plaintexts with `*`s to get them to the correct length, crop them back down correctly, and implement the the level 1 and 2 encryption/decryption (spoilers, obviously).

# In[ ]:


# After figuring out / forgetting / re-figuring out the padding convention a few times, I just wrote a function to automate it.
def padding(s):
	if len(s)%100==0:
		return s
	else:
		pad = 100 - len(s)%100
		parity = pad%2
		offset = pad//2
		if parity == 1:
			return ("*"*(offset))+s+("*"*(offset+1))
		else:
			return ("*"*(offset))+s+("*"*offset)

# Siimlarly, we need to be able to cut a padded text back down to size in a consistent way.
crop = lambda s,l: s[len(s)//2-sum(divmod(l,2)):len(s)//2+l//2]

# Character frequency analysis on an iterable of strings (list, series, etc).
def char_freq(series):
	chars = {i:0 for i in string.ascii_letters + string.digits + string.punctuation}
	for s in series:
		for i in s:
			try:
				chars[i]+=1
			except:
				chars[i]=1
	return pd.Series(chars).sort_values(ascending=False)

# Encryption and decryption functions for levels 1 and 2. Level 2 is an example of just how weird solutions can look if you're
# not thinking about them quite right: I got thinking in terms of rectangles early on in the solving process and never quite
# recovered.

def decrypt_1(c,s=15):
	pattern = [15,24,11,4]
	res = ""
	pos = pattern.index(s)
	for i in range(len(c)):
		if c[i].isalpha() and c[i].lower()!="z":
			n = ord(c[i]) - pattern[pos]
			if (c[i].islower() and n<97) or (c[i].isupper() and n<65):
				n+=25
			res+=chr(n)
			pos = (pos+1)%4
		else:
			res+=c[i]
	return res

def encrypt_1(c,s=15):
	pattern = [15,24,11,4]
	res = ""
	pos = pattern.index(s)
	for i in range(len(c)):
		if c[i].isalpha() and c[i].lower()!="z":
			n = ord(c[i]) + pattern[pos]
			if (c[i].islower() and n>121) or (c[i].isupper() and n>89):
				n+=-25
			res+=chr(n)
			pos = (pos+1)%4
		else:
			res+=c[i]
	return res

def encrypt_2(s):
	blocks = []
	left,right= math.ceil(len(s)/40),len(s)//40
	for k in range(0,right):
		blocks+=[[i for i in s[40*k:40*k+21]]]
		blocks+=[[np.nan]+[i for i in s[40*(k+1)-1:k*40+20:-1]]+[np.nan]]
	if left-right==1:
		blocks+=[[i for i in s[40*right:]]+[np.nan]]
	return "".join(blocks[i][j] for j in range(len(blocks[0])) for i in range(len(blocks)) if not pd.isna(blocks[i][j]))

def decrypt_2(s):
	top,bottom= math.ceil(len(s)/40),len(s)//40
	blocks = [s[:top]]+[s[top + k*(top+bottom):top + (k+1)*(top+bottom)] for k in range(0,19)]+[s[-bottom:]]
	m =  "".join(["".join([blocks[0][k]]+[blocks[i][2*k] for i in range(1,20)]+[blocks[20][k]]+[blocks[i][2*k+1] for i in range(19,0,-1)]) for k in range(bottom)])
	if top - bottom ==1:
		m+="".join(blocks[k][-1] for k in range(20))
	return m


# In[ ]:


train_df = pd.read_csv('../input/ciphertext-challenge-iii/train.csv')
train_df.set_index("index",inplace=True)

# Drop all plaintexts that have already been assigned in levels 1 and 2.
assigned = pd.read_csv('../input/submission2a/submission2a.csv').rename(columns={"index":"p_index1"})
train = train_df.drop(index=list(assigned[assigned.p_index1!=-1].p_index1))


# Set up the basics: text length, padded text, and padded text length.
train['length'] = train['text'].str.len()
train['padded_text'] = train.apply(lambda x: padding(x.text),axis = 1)
train['padded_length']=train['padded_text'].str.len()

test_data = pd.read_csv('../input/ciphertext-challenge-iii/test.csv')
test_data["length"] = test_data.ciphertext.str.len()


# # Level 3
# 
# Difficulty 3 ciphertexts are long streams of 5-digit numbers, so I googled numeric ciphers but didn't find much (but I did learn about base64, which will be useful later). I spent a long time fruitlessly trying to get some sort of RSA-based approach to work, so let me save you from that particular pit of despair right now and say: it's not RSA.
# 
# (Credit here to [redstr](https://www.kaggle.com/redstr), who posted a comment somewhere I can no longer find that eventually nudged me in the correct direction.)
# 
# First some initial setup: separate out the level 3 ciphertexts, record their lengths (both the absolute character length and actual useful length, which here means splitting at the whitespaces), and match up the one longest example with its corresponding plaintext.
# 
# We can also compute the four possible level 2 encryptions of the plaintext; right now there's four options because we don't know how many alphabetic characters are in the padding, so we don't know where to start the level 1 encryption pattern.

# In[ ]:


test_3 = test_data[test_data.difficulty==3].copy()
test_3["unsplit_length"] = test_3.ciphertext.str.len()
test_3['length'] = test_3.apply(lambda x: len(x.ciphertext.split()),axis=1)

# Pair up a single example on length grounds and compute the four possible versions of its level 2 encryption.
c1 = test_3.loc[60920,"ciphertext"]
p1 = train.loc[34509,"padded_text"]
p1e = [encrypt_2(encrypt_1(p1,i)) for i in [15,24,11,4]]


# Initial questions:
# 
# * Does a given number always map to the same character? 
# * Are we looking at substitution or transposition? (Or both?!) 
# * And how can we figure any of this out when we have four possible encryptions of the corresponding plaintext to compare to?
# 
# We can start with some basic frequency analysis on the 5-digit numbers in our chosen example:

# In[ ]:


# Find some numbers in c1 which appear more than once
c1_numbers = pd.Series(c1.split()).value_counts()
print("Most common numbers in our chosen ciphertext example: \n{}".format(c1_numbers.sort_values(ascending=False).head()))


# No one number appears more than three times, so it's definitely not a one-to-one mapping!
# 
# But we can at least use those repeated numbers to identify the correct level 2 encryption for this one ciphertext example. Specifically: for each repeated number, we can look at what characters it maps to in each of the four possible level 2 texts; the correct decryption should hopefully exhibit some sort of regular pattern.
# 
# We do this for the top five numbers from above:

# In[ ]:


# Pick out the indices where those repeated numbers appear
c1_repetitions = [(j,i) for j in c1_numbers[c1_numbers>1].index.tolist()[:5] for i in range(len(c1.split())) if c1.split()[i]==j]

# Have a look at what those numbers map to in each of the four lv2-encoded plaintexts
pd.DataFrame({(j,i):[p1e[k][i] for k in range(4)] for (j,i) in c1_repetitions},index=['encrypt15','encrypt24','encrypt11','encrypt04'])


# The numbers indexing the columns are the five 5-digit numbers, and the indices at which they appear in our ciphertext. The rows show the corresponding character in each of the candidate strings. 
# 
# Look at that repetition in the fourth row!
# 
# So a good working theory is that the fourth candidate string is the correct lv2 encryption of this particular plaintext, and that a given number in the ciphertext always decrypts to the same character.
# 
# (You could also repeat this analysis with the two ciphertexts of length 500, both to verify that this wasn't just some freak coincidence and to pair them with their plaintexts.)
# 
# Okay, so what does the entire dataset look like?

# In[ ]:


# Store the information about that first pair 
test_3.loc[60920,"p_index"] = 34509
test_3.loc[60920,"lvl1_key"] = 4

# Get a list of all the numbers that appear in all of the ciphertexts
combined_numbers = pd.Series([int(i) for i in list(itertools.chain.from_iterable(test_3.ciphertext.str.split()))])

print("Min: {}, Max: {}, Count of distinct numbers: {}.\n\n".format(combined_numbers.min(), combined_numbers.max(),len(set(combined_numbers))))

print("Most common numbers and their frequencies: ")
displaydf = pd.DataFrame([combined_numbers.value_counts().sort_values(ascending=False).head(10),combined_numbers.value_counts().sort_values(ascending=False).head(10)/len(combined_numbers)]).T
displaydf.rename(columns={0:'count',1:'percent of total'}).style.format({'count': "{:.0f}",'percent of total': "{:.2%}"})


# We see that each character is encrypted to an integer between 1 and 50433, the vast majority of those numbers do appear at least once, but even the most popular numbers make up less than 1% of the total.
# 
# I decided to see if I could bruteforce my way to a number -> character decryption dictionary, or at least build enough of one to spot the pattern, so I wrote functions to:
# 
# * update the decryption dictionary based on the current list of matches,
# * apply that dictionary to the ciphertexts to obtain a partial translation (unknown characters are denoted with a `*`),
# * look for ciphertext/plaintext matches.
# 
# The big problem with looking for matches is that we only have partial information about the decrypted ciphertext *and* we only have partial information about the corresponding plaintexts (we're missing the padding characters). So my matching algorithm works on the Sherlock Holmes "once you've eliminated the impossible..." approach: for a given ciphertext, you 
# 
# * first throw out any plaintexts that are definitely not correct, i.e. disagree on a character that's known for both strings;
# * then throw out any matches with too much uncertainty: too few known characters in the ciphertext, or too many ciphertext characters mapping to plaintext padding characters.
# 
# This means we don't actually need that many characters to be confident of a match: if we only know 12 characters in a ciphertext but there's only one plaintext that doesn't disagree with those 12, then it has to be the correct match. The second step just helps to cut down on false positives. 
# 
# There's also a check built into the dictionary update function that catches any contradictions and reverts to the last known correct dictionary if needed, so even an occasional false positive isn't a disaster.
# 
# Of course you don't throw this straight at the entire dataset (27,000 ciphertexts * 54,000 plaintexts, urk): I started with the length $\geq 200$ ones and slowly expanded the search as my decryption dictionary grew.

# In[ ]:


# Function to update the decryption dictionary
# Note that the dict values are sets, which is mildly irritating when we want to actually decrypt something later, but it's
# great for catching errors: if we accidentally mismatch a ciphertext/plaintext pair, we'll end up assigning multiple 
# characters to one number and the function runs a check at the end to catch this and revert to the last safe dictionary.

def update_dict(d_dict,test_df):
    current_len = len(d_dict)
    current_dict = copy.deepcopy(d_dict) # in case we screw up
    df = test_df[~pd.isna(test_df.p_index)]
    for i,c in tqdm(df.iterrows()):
        ptext = encrypt_2(encrypt_1(train.loc[c.p_index,"padded_text"],c.lvl1_key))
        csplit = c.ciphertext.split()
        for j in range(len(csplit)):
            if ptext[j]!="*":
                try:
                    d_dict[csplit[j]].add(ptext[j])
                except:
                    d_dict[csplit[j]] = set([ptext[j]])
    # now verify that we haven't screwed up:
    mult_assign = [i for i in d_dict.keys() if len(d_dict[i])>1] # check if we've assigned two characters to one number
    long_str_assign = [i for i in d_dict.keys() if len(list(d_dict[i])[0])>1] # check if we've assigned a longer string to a number
    if mult_assign or long_str_assign:
        print("Something's gone wrong... Revert? Y/n")
        reply = input(" ")
        # default behaviour here is to revert to the previous dict: if you didn't mean to, just run the function again
        if reply not in ["N","n"]:
            d_dict = copy.deepcopy(current_dict)       
    else:
        print("Dictionary update success! We have {} new entries, for a total of {}".format(len(d_dict)-current_len,len(d_dict)))
        print("Updating the translations for the test dataset...")
        for i,c in tqdm(test_3.iterrows()):
            test_3.loc[i,"partial_translate"]=translate_3(c.ciphertext,d_dict)
        print("Done!")


# Decrypts a ciphertext from level 3 to 2, as much as is possible; missing chars are denoted by "*"
def translate_3(s,d_dict):
    res = ""
    for i in s.split():
        if i in d_dict.keys():
            res+=list(d_dict[i])[0]
        else:
            res+="*"
    return res


# Bruteforce search for ciphertext/plaintext matches based on partial decryption
# You can restrict which subset of level 3 ciphertexts to consider (df_to_test) and which subset of the plaintexts (df_train)
# to compare them to. 
# Ciphertexts with fewer than min_length known characters are skipped, and potential matches with more than fp_threshold uncertainties
# (i.e. matches to plaintext padding characters) are dropped.
def potential_match_search(df_to_test,df_train,d_dict,min_length,fp_threshold):
    print("Searching for matches for {} ciphertexts...".format(len(df_to_test)))
    res = {}
    for a,c in tqdm(df_to_test.iterrows()):
        # Pick out the indices in the ciphertext whose decryption is known
        c_pos = [s for s in range(len(c.partial_translate)) if c.partial_translate[s]!="*"]
        c_trunc = "".join([c.partial_translate[s] for s in c_pos])
        # user-set threshold requires a minimum number of known points to proceed
        if len(c_trunc)>=min_length:
            possibles = []
            for i,p in df_train[df_train.padded_length==c.length].iterrows():
                for l in ["encrypt15","encrypt24","encrypt11","encrypt04"]:
                    t = "".join([df_train.loc[i,l][j] for j in c_pos])
                    # If the corr. character in the plaintext is known, we require it to match the ciphertext.
                    # We also allow the plaintext char to be unknown (i.e. comes from the padding)
                    if (sum(t[k] in [c_trunc[k],"*"] for k in range(len(c_trunc)))>=min_length) and (t.count("*") < fp_threshold):
                        possibles+=[(i,t,int(l[-2:]))]
            res[a] = possibles[:]
    return res
  
    


# To get started, I matched the eight ciphertexts of length $\geq 300$ by hand, by the same sort of methods as the original longest example: this gives our decryption dictionary somewhere to start.

# In[ ]:


test_3["p_index"] = np.nan
test_3["lvl1_key"] = np.nan

# These examples were matched by hand.
for (i,j,k) in zip([60920,99421,75719,746, 2734, 10978, 30192, 70167],[34509,31644,76893,93461, 40234, 47443, 77656,76309],[4,4,11,11,24,15,11,24]):
	test_3.loc[i,"p_index"] = j
	test_3.loc[i,"lvl1_key"] = k


print("Performing setup... (takes a while but only has to be done once)")
train["encrypt15"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,15)),axis=1)
train["encrypt24"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,24)),axis=1)
train["encrypt11"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,11)),axis=1)
train["encrypt04"] = train.apply(lambda x: encrypt_2(encrypt_1(x.padded_text,4)),axis=1)    

decrypt_3_dict = {}
update_dict(decrypt_3_dict,test_3)


# We can start by trying to match the length 200 texts:

# In[ ]:


# Start with the low-hanging fruit: texts of length >=200. 
result = potential_match_search(test_3[test_3.length==200],train[train.padded_length==200],decrypt_3_dict,min_length = 30,fp_threshold = 30)
exacts = [i for i in result.keys() if len(result[i])==1]
print("Exact matches: {} of {}.".format(len(exacts),len(test_3[test_3.length==200])))

for i in exacts:
    test_3.loc[i,"p_index"] = result[i][0][0]
    test_3.loc[i,"lvl1_key"] = int(result[i][0][2])

update_dict(decrypt_3_dict,test_3)


# Not bad! We found unique matches for almost all of the length 200 texts, and more than doubled the size of our decryption dictionary. So let's run it again and see if we can get those last few matches...

# In[ ]:


result = potential_match_search(test_3[(test_3.length==200)&(pd.isna(test_3.p_index))],train[(train.padded_length==200)&(train.length>=90)],decrypt_3_dict,min_length = 40,fp_threshold = 40)
exacts = [i for i in result.keys() if len(result[i])==1]
print("Exact matches: {} of {}.".format(len(exacts),len(test_3[(test_3.length==200)&(pd.isna(test_3.p_index))])))

for i in exacts:
    test_3.loc[i,"p_index"] = result[i][0][0]
    test_3.loc[i,"lvl1_key"] = int(result[i][0][2])

update_dict(decrypt_3_dict,test_3)


# You can keep running the above three blocks and tweaking the thresholds until all of the length 200 texts have a match. If we ever accidentally make an incorrect match, the function has a check to catch it and revert to the last correct dictionary, so don't be afraid to experiment.
# 
# You could then start on matching some length-100 texts. Again, don't throw the entire dataset at it: pick some sensible restrictions of the ciphertext/plaintext datasets so that it finishes in a finite amount of time. Even ~20 ciphertext/plaintext matches will expand the dictionary by several thousand entries, making the next run more effective.
# 
# I'm curious about whether it's actually possible to bruteforce this entire cipher without finding the key: maybe with some more efficient code and some parallel processing you could reach a tipping point in the dictionary size that makes it possible to match the super-short texts?
# 
# Anyway, the main goal here was to get enough data about the number -> character mapping to figure out what's happening. So let's have a look...

# In[ ]:


print("Most common numbers and their mappings:")
for i in list(pd.Series(combined_numbers).value_counts().sort_values(ascending=False).index)[:20]:
    if str(i) in decrypt_3_dict.keys():
        print("{} -> {}".format(i,list(decrypt_3_dict[str(i)])[0]))


# Looks like the most common ciphertext numbers all map to uncommon characters like X, Q, and !. That's pretty strange.
# 
# But what if we're looking at it from the wrong angle? If we instead just collect all of the values in our decryption dictionary (i.e. all the characters that get mapped *to*) and count how many times each one appears, a different picture emerges...

# In[ ]:


decryption_image=pd.Series([list(decrypt_3_dict[i])[0].lower() for i in decrypt_3_dict.keys()]).value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(decryption_image.index[0:30],decryption_image[0:30],label="Ciphertext")
plt.ylabel('Character count',fontsize=16)
plt.title('Frequency of the top 30 characters in our decryption dictionary',fontsize=16)
plt.show()


# Hmm. It's not quite perfect, but that distribution looks kinda familiar...

# ## Level 3 solution
# 
# If you put enough effort into bruteforce-matching texts, you may have collected a large enough sample size that your graph matches the distribution of English letter frequencies; if you just match the length $\geq 200$ texts (as this notebook does) it'll be close-ish but not exact.
# 
# So, numbers from 0 to ~54,000, matching up to letters with approximately the frequency of standard English: we're looking at a book cipher. In other words, the key is some long text and for each number in the cipher you go grab the character from that location in the key.
# 
# Which text? Let's string together what we have and see if anything stands out (truncated to first 1500 characters for display purposes):

# In[ ]:


print("".join(list(decrypt_3_dict[str(i)])[0] if str(i) in decrypt_3_dict.keys() else "*" for i in range(1500)))


# `*HIS**ROJEC* G**E*BER****OOK` almost certainly says "THIS PROJECT GUTENBERG BOOK", and there's a `DONNE*LY` and a few things that look like "cryptogram". 
# 
# If you google "Donne`*`ly cryptogram Project Gutenberg", you'll find that Ignatius Donnelly wrote a book called "The Great Cryptogram" that purported to uncover an elaborate cipher in the works of Shakespeare, and (more relevantly to us) Joseph Pyle thoroughly debunked Donnnelly's theory in a book called "The Little Cryptogram".
# 
# "Pyle" should sound familiar: the sequence of shifts in cipher 1 was $[15,24,11,4]$, or, if you take the corresponding letters in the English alphabet: P Y L E. Bingo!
# 
# (Incidentally, if you google "pyle shakespeare", the top hit is [this tweet](https://twitter.com/nathanwpyle/status/1082069772496453632) and I'm still a little bit sad that this wasn't where the metaclue was pointing.)
# 
# Anyway! A little bit of trial and error establishes that we want the UTF-8 encoded plaintext version of the book and that '\n' should be replaced by whitespace, and we have our decryption function:

# In[ ]:


# level3key.txt is https://www.gutenberg.org/files/46464/46464-0.txt
with open('../input/level3key/level3key.txt',"r") as f:
	key = f.readlines()

key[0] = ' '+key[0]
splitkey = "".join([i for j in key for i in j])
splitkey = splitkey.replace('\n','  ')
splitkey = {i:splitkey[i] for i in range(len(splitkey))}

def decrypt_3(s,translate_3 = splitkey):
	res = ""
	for i in s.split():
		if int(i) in translate_3.keys():
			res+=list(translate_3[int(i)])[0]
		else:
			res+="*"
	return res


# In[ ]:


c1 = test_3.iloc[0,1]
print("Test it out on an example:\n\n{} \n\ndecrypts to \n\n{}".format(c1,decrypt_1(decrypt_2(decrypt_3(c1)))))


# # Level 4
# 
# This was a very satisfying level to solve. And you don't really need to have solved any of the previous levels in order to tackle level 4 (although you do need it for the final decryption/text matching step, of course) so if you're stuck on level 3, maybe give level 4 a try instead?
# 
# Let's have a look at a ciphertext example to see what we have this time:

# In[ ]:


test_4 = test_data[test_data.difficulty==4].copy()
test_4.loc[1,"ciphertext"]


# 30 seconds with google will identify this as base64-encoded text: those $=$ symbols at the end are characteristic of that format.
# 
# The [base64 wikipedia page](https://en.wikipedia.org/wiki/Base64) will tell you everything you need to know about it, but essentially, the encoding process is:
# 
# 1. Choose a mapping between your plaintext alphabet and 8-bit binary (aka the numbers $0-255$).
# 1. Map your plaintext to a sequence 8-bit binary numbers and concatenate them into one long string.
# 1. Split the string into groups of 6, giving you a new sequence of 6-bit binary numbers.
# 1. Choose a mapping from 6-bit binary numbers ($0-63$) to the alphabet {uppercase letters, lowercase letters, digits 0-9, +,/} and apply the mapping to your sequence. Pad with $=$ symbols at the end to ensure that the length of the final string is divisible by 3.
# 
# So there's two obvious places to mess with this process if you're looking to encrypt something: if you choose a non-standard mapping at either end, only people who know your choice of mapping will be able to decode it.
# 
# First we can check whether we're decrypting to individual characters of a level 3-encrypted text, or to those 5-digit blocks we were working with before:

# In[ ]:


mean3 =(test_3.unsplit_length/test_3.length).mean()
print("Average number of characters in a level 3 string per character of decrypted text: {0:.2f}\n".format(mean3))
print("Lengths of ciphertext strings after we undo the base64 encoding: \n{}\n".format(round(test_4.length*(3/4),-1).value_counts()))
print("Lengths of ciphertext strings after we also divide by the level 3 mean: \n{}".format(round(test_4.length*(3/4)/mean3,-1).value_counts()))


# So, individual characters it is! We can now fix our length statistic:

# In[ ]:


test_4['lvl4_length'] = test_4.ciphertext.str.len()
test_4['length'] = test_4.apply(lambda x: int(round(x.lvl4_length/7.75,-2)),axis=1)


# So after we decrypt the base64 portion, we'll have numbers in the range $0-255$ and we'll need to somehow map them onto the digits $0-9$ plus whitespace.
# 
# Rather than just rushing in blindly, I wanted to first try to get a feel for what that pairing might look like and which parts of the standard base64 decoding would need to be modified.
# 
# We can look at what the most common first two characters are of the level 4 ciphertexts:

# In[ ]:


lvl4_firstchars = pd.Series([x.ciphertext[:2] for i,x in test_4.iterrows()])
print("All first two character combos from level 4 ciphertexts and their frequencies (% of examples): \n{}".format(lvl4_firstchars.value_counts().sort_values(ascending=False)/len(test_4)*100))


# I didn't just display the top few examples; that's *all* of them. That's... unlikely.
# 
# In standard base 64, on the 6-bit side we have A -> 000000, B -> 000001, C -> 000010, which means that all of these two-character combos would give us very small numbers for the first 8-bit number.
# 
# Say... numbers in the range 0 - 9, maybe?
# 
# Check:

# In[ ]:


standard_b64chars = string.ascii_uppercase+string.ascii_lowercase+string.digits+"+/"
b64_to_num = {j:i for (i,j) in zip(range(64),standard_b64chars)}
num_to_b64 = {i:j for (i,j) in zip(range(64),standard_b64chars)}
to_binary = lambda x: "{0:06b}".format(x)

conversions = {s:int('0b'+(to_binary(b64_to_num[s[0]])+to_binary(b64_to_num[s[1]]))[:8],2) for s in list(lvl4_firstchars.value_counts().index)}

print("First decoded character of each level 4 string:")
for s in list(lvl4_firstchars.value_counts().index):
    print("{} -> {}".format(s,conversions[s]))


# Okay! So let's see what happens if we just apply the base64 decryption to all of our ciphertexts (skipping the final step where we translate 8-bit numbers back to characters for now).

# In[ ]:


def decrypt_base64(s,key1=b64_to_num):
	if len(s)%4!=0:
		s1 = s+"="*(len(s)%4)
	else:
		s1 = s[:]
	blocks = len(s1)//4 # now always guaranteed to divide neatly
	pad = s1[-3:].count("=")
	s1 = s1.translate({ord("="): None})
	converted = "".join(['{0:06b}'.format(key1[i]) for i in s1])
	converted = ["0b"+converted[8*i:8*(i+1)] for i in range(blocks*3)]
	if pad >0:
		converted = converted[:-pad]
	return [int(i,2) for i in converted]

test_4["base64"] = test_4.apply(lambda x: decrypt_base64(x.ciphertext),axis=1)
print("Have a look at the first few characters of some random examples...")
for i in range(10):
    print(test_4.iloc[i,-1][:20])


# It looks each decrypted character (i.e. each column) corresponds to a number in a fairly small range, although that range varies from character to character without any immediately-visible pattern. But it feels like we're on the right track; these tiny ranges for each character can't be a coincidence.
# 
# So now just have to pair each set up with the digits 0-9 (and whitespace) somehow.
# 
# Are we lucky enough to have the first character just match up directly? We can check this by doing a frequency analysis of first characters in our base64-decoded level 4 ciphertexts vs the level 3 ciphertexts from earlier:

# In[ ]:


print("Frequency of each number in the first position of the base64 decoded level 4 ciphertexts (% of examples) :\n{}".format(pd.Series([x.base64[0] for i,x in test_4.iterrows()]).value_counts().sort_values(ascending=False)/len(test_4)*100))
print("Frequency of each digit in the first character of level 3 ciphertexts (% of examples) :\n{}".format(pd.Series([x.ciphertext[0] for i,x in test_3.iterrows()]).value_counts().sort_values(ascending=False)/len(test_3)*100))


# Of course it wasn't that easy! It looks like 4 -> 5 and 5 -> 4, 0 -> 1, 7 -> 6, and 2 and 3 pair up in some order. Maybe just a simple swap between pairs of numbers? But then what happens at the next step?
# 
# You have various options at this point:
# 
# * You could repeat this analysis for the first few characters and look for a pattern in the pairings. 
# * Or you could look at every 6th character; for example, if you look at characters 0, 7, 13,... etc, these correspond (on average) to the first digit of a 5 digit number in the level 3 encrypted texts, so 1, 2, 3, 4 will be overrepresented and 0 and whitespace will be underrepresented. Similarly, 6, 12, 18,... etc will correspond (on average) to whitespace characters.
# * Remember how certain numbers (527, 540, 44280, 42985, ... ) were super popular in the level 3 ciphers? You could try to identify common sequences of digits in the base 64 decodings and pair them up.
# 
# This would be much easier if we could compare individual ciphertexts/plaintexts directly as we did in previous levels, but I never figured out the level 3 encryption algorithm (just the decryption!) so I had to rely on frequency analysis of a related but different set (the level 3 ciphers) to get a feel for what's happening.
# 
# But eventually you should be able to collect enough data to spot a pattern, and from there the whole thing falls neatly into place.
# 
# Good luck!

# ## Level 4 solution
# 
# First I took the base64-decrypted-ciphertexts (which I'll keep referring to as the "ciphertexts" even though we've done some initial decryption on them) and made a list of which numbers can appear in each position; this corresponds to the columns from a few displays ago.
# 
# Across all ciphertexts, for each given index there are exactly 11 numbers that ever appear in that position. These are the numbers that we have to decrypt to 0,1,...9 and whitespace for that position.
# 
# One thing you'll notice pretty quickly is that the number corresponding to the whitespace is always easy to identify: most of the numbers are in sequence but the whitespace is offset by some small amount. And if two positions encrypt their whitespace with the same number, you'll find that they encrypt their other numbers identically too.
# 
# The breakthrough came when I thought to display each list sorted by its whitespace value instead of its position in the ciphertext: instead of looking at the numbers that appear in position 0,1,... etc of the ciphertext, I looked at the list of numbers associated to a whitespace value of 0, 1, 2...
# 
# We can display the first few examples and look for a pattern:

# In[ ]:


# first, get a list of the numbers associated to each character position
# we use a set for now (simplifies dealing with duplicates; we'll convert to a list later)
# every example has length >= 550 so let's look at those first:
alphabet=[]
for j in range(550):
	alphabet+= [set([x.base64[j] for i,x in (test_4.iloc[:500]).iterrows()])]

# see where we didn't get enough data:
missing = [i for i in range(len(alphabet)) if len(alphabet[i])<11]

# rerun on the entire dataset to fill in the blanks
for j in missing:
	alphabet[j] = set([x.base64[j] for i,x in test_4.iterrows()])

# listify and sort
alphabet = [sorted(list(j)) for j in alphabet]

    
# now pick out the whitespace value for each character position 
# In hindsight there's probably a better way to do this, but at the time I didn't really know where I was going with this

# Whitespace is always separated from the other numbers, but sometimes 8,9 are a separate block too
# Luckily, there are only 3! ways to arrange the blocks [whitespace][numbers corr to 1 - 7][numbers corr to 8, 9]
# So associate a binary number to each pattern to identify it, convert it to an integer, then store where the whitespace is for that pattern
pattern_mapping = {514:0,640:0,6:-3,5:-1,257:-1,384: 2}


# build a dictionary that just lists what number maps to whitespace at each step
whitespace_dict = {}

# extract the whitespace character for each position
# we skip 0 for now because position 0 is never whitespace
for i in range(1,len(alphabet)):
	a = alphabet[i]
	gap_pattern = int("0b"+"".join(["0" if a[i]-a[i-1]==1 else "1" for i in range(1,11)]),2)
	if gap_pattern in pattern_mapping.keys():
		whitespace_dict[i] = a[pattern_mapping[gap_pattern]]
	elif a[1] - a[0]!=1: # so we can separate the whitespace but not the other
		whitespace_dict[i] = a[0]
	elif a[-1] - a[-2]!=1: # as previous
		whitespace_dict[i] = a[-1]

print("Truncated to first 30 values for display purposes \n\n Each line contains:\nwhitespace_value: [the numbers that decrypt to 0, 1,...,9 when whitespace decrypts to whitespace_value] \n \n")
# now print out the results (truncated to first 30 for display purposes)
for i in range(30):
	if [k for k in whitespace_dict.keys() if whitespace_dict[k]==i]!=[]:
		print(i,[j for j in alphabet[[k for k in whitespace_dict.keys() if whitespace_dict[k]==i][0]] if j!=i])
	else:
		print(i)


# There's some definite binary vibes here: look how the pattern jumps when we hit a power of 2, for example.
# 
# If you play around with these lists for a while, you should eventually spot the pattern: each list is the result of bitwise XORing the numbers $\{0,1,\dots,9\}$ with a fixed number in the range $[0,255]$. Or conversely: the key is some long sequence of integers in the range $[0,255]$ and you encrypt/decrypt your text by XORing together the text and the key.
# 
# That's enough to cobble together a decryption algorithm, but I'm fudging the details slightly here in the interests of showing *how* you might get to the solution rather than just going straight there.
# 
# What you're actually doing is XORing the ASCII values of the *strings* '0', '1',...,'9', ' ' with some key. This is a more satisfying solution (it explains why the whitespace character has the value it does) and also more general: it means you could encrypt any string if you wanted to, not just sequences of numbers.
# 
# With that in mind, generating enough of the key to perform the decryption becomes easy:

# In[ ]:


test_4["b64_length"] = test_4.apply(lambda x:len(x.base64),axis=1)

# XORing the ciphertext and the plaintext together would generate the key, but we don't know the plaintexts yet
# However, we do know that the plaintext is one of 11 characters.
# So XOR the ciphertext character in position i with all 11 possible plaintext characters: the key must be one of these 11 numbers.
# Repeat for all ciphertexts and take the intersection: that's the key value for position i.

key_bags = {}
vals = [ord(i) for i in string.digits+" "]

# Rather than doing this to the whole test set, we'll start with just the longer texts and see if that gets us enough information
for j in tqdm(range(1152)):
	res = set(range(256))
	for i,c in test_4[test_4.length>=200].iterrows():
		if c.b64_length>j:
			res &= set([c.base64[j]^k for k in vals])
	key_bags[j]=res

missing = [i for i in key_bags.keys() if len(key_bags[i])!=1]

# There's a few that didn't quite get narrowed down, so we throw the rest of the test set at those indices only:
for j in tqdm(missing):
	res = set(range(256))
	for i,c in test_4.iterrows():
		if c.b64_length>j:
			res &= set([c.base64[j]^k for k in vals])
	key_bags[j]=res

# We can figure out 0 by hand.
key_bags[0] = set([49])

# finally, collect the results together in a list.
key_list = [list(key_bags[i])[0] for i in range(1152)]


# In[ ]:


def decrypt_4(s):
	tob64 = decrypt_base64(s)
	return "".join([chr(key_list[i]^tob64[i]) for i in range(min(len(key_list),len(tob64)))])


# In[ ]:


c1 = test_4.iloc[0,1]
print("Test it out on an example:\n\n{} \n\ndecrypts to \n\n{}".format(c1,decrypt_1(decrypt_2(decrypt_3(decrypt_4(c1))))))


# Looks good! 
# 
# Of course, the question is: where does that key come from?
# 
# In a just world, it would be the text of Hamlet's "To be, or not to be" speech translated into Baconian, but also I guess that would be too obvious.
# 
# Well, I tried looking for repetition in the key and looking for repeating binary subwords of various lengths, translating from Baconian (both traditional and newer versions), trying to find a link to the numbers in either Pyle's Little Cryptogram or Donnelly's Great Cryptogram, interpreting the 8-bit numbers as pixel intensities, looking for a non-standard base64 mapping that was compatible with the cipher, interpreting the binary as Morse code (I was getting desperate by this point)...
# 
# Yeah, I'm still stumped.

# In[ ]:




