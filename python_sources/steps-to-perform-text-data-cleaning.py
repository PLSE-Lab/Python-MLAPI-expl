#!/usr/bin/env python
# coding: utf-8

# **Benifits of Twitter minning for a brand**
# 
# 1. We can discover sentimental analysis for acustomer sentiment on a brand.
# 2. We can know popularity of brand by active tweets.
# 3. Used to identity the customer needs.
# 4. Used for predictions and forecasting

# **Business Problem**
# 
# *Find the most popular  features of an Apple iphone among the fans on twitter?*
# 
# Extracting the twittes related to customer opinions on iphone.
# 
# **sample tweet:**
# "I luv my alt;3 iphone &amp;you'r awsm apple.DisplayIsAwesome,so happpppppy:)http://www.apple.com"

# **STEPS FOR DATA CLEANING**
# 
# **1. Escaping HTML characters**
# 
# **Code:**
# import HTMLParser
# html_parser= HTMLParser.HTMLParser()
# tweet=html_praser.unescape(orginal_tweet)
# 
# **output:**
# "I luv my <3 iphone & you'r awsm apple.Display Is Awesome,so happpppppy http://www.apple.com"
# 
# **2.Decoding data**
# 
# **Code:**
# tweet=original_tweet.decode("utf8").encode('ascii','ignore')
# 
# **output:**
# "I luv my <3 iphone & you'r awsm apple.DisplayIsAwesome,so happpppppy:)http://www.apple.com"
# 
# **3.Apostrophe Lookup**
# 
# APPOSTROPHES={" 's":"is",'"re":"are",....} ## Need a huge dictonary
# words=tweet.split()
# reformed=[APPOSTROPHES[word] if word in APPOSTROPHES  else word for word in words]
# reformed=" ".join(reformed)
# 
# **output:**
# "I luv my <3 iphone & you are awsm apple.Display Is Awesome,so happpppppy:) http://www.apple.com"
# 
# **4.Removal of stop-words**
# 
# When data analysis needs to be data driven at the word level,the commonly occuring words (stop words) should be removed.One can either create a long list of stop-words or one can use predefined language specific libraries.
# 
# **5.Removal of punctuations**
# 
# All the punctuations marks according to the priorites should be dealt with. For Example:" .", "," ," ?"  are important punctuations that should be retained while others need to removed.
# 
# **6.Removal of Expressions**
# 
# Textual data(usually speech transcripts )may contain human expressions like [laughing],[crying],[audience paused].These expressions are usually non relevant to content of the speech and hence need to removed.
# 
# **7.Split the attached words**
# 
# **code:**
# cleaned=" ".join(re.findall('[A-Z][^A-Z]*' , original_tweet))
# 
# **output:**
# "I luv my <3 iphone & you'r awsm apple.Display Is Awesome,so happpppppy:) http://www.apple.com"
# 
# **8.Slangs lookup**
# 
# **code:**
# tweet =_slang_loopup(tweet)
# 
# **output**
# "I love my <3 iphone & you are awesome apple.Display Is Awesome,so happpppppy:) http://www.apple.com"
# 
# **9.Standardizing word**
# 
# **code**
# tweet=".join(".join(s)[:2]for_,s in itertools.groupby(tweet))
# 
# **output**
# "I love my <3 iphone & you are awesome apple.Display Is Awesome,so happy :) http://www.apple.com"
# 
# **10.Removal URL'S**
# 
# **Output**
# "I love my  iphone & you are awesome apple.Display Is Awesome,so happy!",<3,;)
# 
# **ADVANCED DATA CLEANING**
# 
# *Grammer checking*
# 
# *Spelling correction*
# 
# 

# **Please upvote for encouragement**
