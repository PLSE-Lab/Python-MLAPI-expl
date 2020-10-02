#!/usr/bin/env python
# coding: utf-8

# # Regular expressions to match dates and times 

# If ever you want to empathise with the anger and frustration apparent in the toxic comments of this competition, my advice is to spend a morning designing regular expressions patterns.
# 
# I'm sharing my efforts at creating a pattern to match dates and times to save anyone else foolish enough to think this would be a good idea. 
# 
# This notebook ends up being a bit of a walk through regular expression design, so hopefully useful if you are less familiar with this form of pattern matching. It isn't perfect - these things never are given the number of edge cases you have to deal with - but seems to be working well enough for data cleaning purposes.

# ## Why remove dates and times?

# The rational for removing dates and timestamps is that they are unlikely to contain useful information for predicting the comment labels, and crucially they can often be used to uniquely identify a specific comment.  If a training data includes unique references, then almost any machine learning model will be prone to simply learning the mapping from these unique references to the labels, and therefore overfitting the training set and generalising poorly when applied to the test set.
# 
# It's possible that there is some value in the time and date in this competition. Perhaps people are more likely to sling insults late at night, or suffering the winter blues. However, it may be more productive to steer your model towards the content of the comment by simply removing this distracting meta data.    

# ## Options for removal

# The simplest method of removing times and dates is to strip out all numbers and punctuation, and given the task in this competition, that will probably work just fine. However, it also leaves months of the year and probably a few arbitrary 'st' 'rd' 'th' suffixes lying around. Equally, there will be cases where this approach would destroy valuable information, and so a more careful cleaning process is required.
# 
# Generally speaking, if you can clean your data non-destructively, then your downstream tasks - such as tokenisation and parsing - ought to work better.
# 
# My initial attempt was to use an off the shelf solution. After some googling I found the [datefinder python package](https://github.com/akoumjian/datefinder). This works well, but it is really designed to extract and normalise the date and time information - converting it to a datetime object. For our purposes, datefineder is a little over-eager, finding false positive matches like identifying the 'fri' in 'girl**fri**end' as a day.
# 
# So, why not build a regular expression to find dates and times? Can't be that hard can it ...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
pd.set_option('display.max_rows', 40)


# ### Finding dates which just contain numbers 

# We start by defining some examples of the date / time formats we want to find.
# 
# Dates containing just numbers can be in the format:
# * dd-mm-yyyy; 
# * mm-dd-yyyy; 
# * yyyy-mm-dd; 
# 
# and so on, with the year possibly being 2 digits rather than 4.

# We now define regular expressions for the day, month and year elements in turn. If you aren't familiar with regular expressions, then there are lots of websites with introductions and tutorials. [https://www.regular-expressions.info/](https://www.regular-expressions.info/) is a good place to start. In summary, regular expressions provide a way to search for complex patterns in text strings using a combination of wildcards and tokens that match a specific set of characters. So, for example, ```\d``` will match any digit and ```\w``` any alpha-numeric character.

# Lets start with the  'day' number in a date. It can be anything from 1 to 31, and could have a leading zero. So, in terms of a regular expression pattern we look for a 0, 1, 2, or 3 in the first character, and make this optional by adding a question mark:
# ```python
# '[0-3]?'
# ```
# And then search for any digit in the second character using the special token
# ```python
# '\d'
# ```
# We then wrap all of this in parentheses. This is to ensure that we keep track of the logic of the expression as it becomes more complex as we add alternatives options using the logical OR operator '|'.
# 
# Parentheses on their own create a capture group. This would allow us to extract the specific text matched by the pattern in the parentheses. In our case we aren't interetsed in the days part of a string on its own - we want the whole of the date. So, we create a non capturing group by putting ```?:``` after the open parenthesis. Our complete pattern to match a number from 1 to 31 with an optional leading zero becomes:
# ```python
# '(?:[0-3]?\d)'
# ```

# Using the same principles we define patterns for day, month and year:

# In[ ]:


nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 
nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero
nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 
                            # that people doon't generally use all number format for old dates, but write them out 


# We also need to define a delimiter - the bit separating the numbers - since this could be one of a numer of different characters such as . , - / \
# 
# Note that we make the delimiter optional to account for examples such as '20180702'

# In[ ]:


nDELIM = r'(?:[\/\-\._])?'  # 


# Now we can put all of these bits together according to the various date formats we are looking for.  Note that you can put comments and line breaks in you regex patterns to make them easier to read. However, you must then remeber to compile them with the VERBOSE flag.

# In[ ]:


NUM_DATE = f"""
    (?:
        # YYYY-MM-DD
        (?:{nYR}{nDELIM}{nMNTH}{nDELIM}{nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}{nDELIM}{nDAY}{nDELIM}{nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}{nDELIM}{nMNTH}{nDELIM}{nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}{nDELIM}{nDAY}{nDELIM}{nYR})
    )"""

myDate = re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE | re.UNICODE)


# In order to test our expressions, we create a small test list rather than running against the actual dataset. The following list was created using by applying the datefinder library to the comments, with a few manual edits from me. As you can see, there are quite a few false positives. 

# In[ ]:


test_dates = ['15:12', '15:30, 12th March 1972 and then 3rd May 2014.', 'May', 'April 2006',
 't 6', 'a few words before 31/12/2019 ',
 '99.10.10.8', '2016.31.55',
 'on 2012-09-11 and a few words after ',
 '1234-56-78', '2 t', '04:27, 20 Jan 2004',
 '10, 9 March 2011', '20:04, 17 Dec 2004',
 'wed at', 't 2005', '2012/56/10', '5th', 't 1600', 'on August, 2014',
 'May 25, 1992', 'on mar', '1, 4 , 5', '03:19, 1 December',
 'on 2 October', '1930', '15:59, 17 December',
 '5-5-5', 'the time is 12:12:54 am ', '17:80pm', '10:52, 11',
 '10 second', '6 of t', '17:09, 16 Jun 2005',
 '19:32', '20th', 'Dec of 04', '1978 T',
 '09/11', 't133', 't 21:31', 't on 29 October 2013 T',
 '19:53, 15', 'wed by t', '3/4 of t', '149,000', '29 August 2006', 't Mar',
 'November 2014', '7 of t', 'December 11 2006', '14:59, 16',
 'mon to', '02:08, Mar 13, 2004', '69 of', 'of nov',
 '/may-2012', '3am', 'of March', '2003',
 'on 12 Dec', 't 2', 'at mar', 't of 1993',
 '2790', 't, may', '21:51, January 11, 2016',
 '22 May 2005', '16 December 2005', 'July, 1870', 'On Dec 14, 2006',
 't 3 of t', '07:51, 2004 Jul 16', 't dec', 'April 2006 T',
 't, 2012', 'March 16, 1869', 'wed to', '20 January 2013',
 '26t', 't-may','4000']


# Let's see how the expression works againts this test set:

# In[ ]:


pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])


# So, not too bad.
# 
# It has correctly identified the dates in rows 5 and 8, but we have a we also have a match of '2016.31' from the original string '2016.31.55', and also a '2012/56' a bit futher down the list.
# 
# At first glance it looks like '2016' must be a match on year, and '31' a match on days, but this should not be possible since we have not included a pattern which consists only of nYR and nDAY.
# 
# The reason is that our delimiter is optional, and is specified independently, and so each delimiter can be different or blank. This has allowewd the pattern to match: '2016' for year, '.' for the delimiter, '3' for month or day, then blank for the delimiter and '1' for the remaining day or month.  

# To work these things out I recommend using an interactive online regex testing site such as [https://regex101.com/](https://regex101.com/).
# 
# One solution is to ensure that the delimiter is consistent, i.e. if you find '-' between the year and month, you have to also use '-' between month and day. This can be achieved through named capture groups.
# 
# We create the first delimiter as capture group called 'delim' as follows::
# ```python
# '(?P<delim>[\/\-\._]?)'
# ```
# The second delimiter pattern now refers back to the first - we just match a copy of whatever was found within the delim capture group.
# ```python
# '(?P=delim)'
# ```

# We can't use the same named group in different subsections of the regex pattern. Therefore, we need to create a different named group for each of the date patterns within the aggregate NUM_DATE pattern. Also, we will now name the overall capture group for the entire date.  

# In[ ]:


NUM_DATE = f"""
    (?P<num_date>
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
    )"""

myDate = re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE)


# Now, let's test this version:

# In[ ]:


pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])


# So, we have now fixed the second of the false positives, but we are still getting a match of '2016.31.5' based on a pattern of YYYY-MM-D.  The solution here is to require that we have either the start of end of the string, or a non-digit character at either end of our match. We can do this by adding extra tokens to the beginning and end of the NUM_DATE capture group:

# In[ ]:


NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""

myDate = re.compile(NUM_DATE, re.IGNORECASE | re.VERBOSE)


# In[ ]:


pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])


# ### Finding dates with numbers and text 

# Now that we cn find numeric dates, we move on to finding dates which are written in natural language. We adopt exactly the same approach as above. We need to cater for examples with various permutations of day-month-year, year-month-day, etc; but now our patterns for the day and month get a bit more complicated. Day needs to account for suffixes e.g. '1st', '2nd', or fully written out numbers as well as digits. 
# Note that the version below doesn't define all numbers written in English as this feels like overkill.

# In[ ]:


DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
    |
    # or just a number, but without a leading zero
    (?:[123]?\d)
)"""


# Month is more straightforward - we just have the names and their common abbreviations.

# In[ ]:


MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'


# Year can be either a 4 digit number beginning with a 1 or 2, or any 3 or 2 digit number.  We also allow for the option of a leading apostrophe if it is a 2 digit year, as in the year '98.

# In[ ]:


YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""


# We will also consider a few more alternatives for delimiter than before to account for examples like "3rd of April"

# In[ ]:


DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'


# Now we can combine all of these into a full date pattern, and see how it fares against our test list. Note that in this case we are allowing the delimiter to change within the date. This is to account for examples such as "10th March, 2012".

# In[ ]:


DATE_PATTERN = f"""(?P<wordy_date>
    (?:^|\W)
    (?:
    (?:
        # Year - Day - Month (or just Day - Month)
        (?:{YEAR}{DELIM})?(?:{DAY}{DELIM}{MONTH})
        |
        # Month - Day - Year (or just Month - Day)
        (?:{MONTH}{DELIM}{DAY})(?:{DELIM}{YEAR})?
    )
    | 
        # Just Month - Year
    (?:{MONTH}{DELIM}{YEAR})
    )
    (?:$|\W)
)"""

# Let's combine with the numbers only version from above
myDate = re.compile(f'{DATE_PATTERN}', re.IGNORECASE | re.VERBOSE | re.UNICODE)

pd.DataFrame([{'test_text': txt, 'match': '@@'.join(myDate.findall(txt))} for txt in test_dates])


# Not too bad, but we are getting some unusual matches where it is picking up part of the time, or another number precededing the actual date and treating this as a thinking this is a two digit year. For example, finding
# ```python 
# ':27, 20 Jan'
# ```
# within 
# ```python
# '04:27, 20 Jan 2004'
# ```
# This is similar to the problem we faced above, but we cannot use the same solution since we want to allow the delimiter to vary. 

# The underling issue is that regular expressions containing the OR token '|' are evaluated left to right as the target string is scanned, and once a match is found, that match is returned and the remaining alternatives are not considered. That is, the '|' operator is non-greedy - it does not give you the longest matching substring, but the first match found.  

# I think the only way to solve this thoroughly is to compile and search using separate regular expressions for each of the alternate capture groups, and then select the longest match manually. However, rather than go down that route, let's continue with regex and see how close we can get to what we want.
# 
# The problem matches are rather odd: a 2 digit year followed by day and then month in words. This doesn't seem like a likely date format.  We might see a 4 digit year followed by day and month, but probably not a 2 digit year. So, let's just create a new 4 digit only YEAR pattern for scenarios when YEAR precedes day and month, and put this into the overall date pattern.

# In[ ]:


YEAR_4D = r"""(?:[12]\d\d\d)"""


# In[ ]:


DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

myDate = re.compile(f'{DATE_PATTERN}', re.IGNORECASE | re.VERBOSE | re.UNICODE)

pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])


# That now looks good. So we now move on to the final step of adding in the time.

# ## Finding times in text

# Times are a bit more straightforward. We are looking for things like 07:25 perhaps followed by am or pm. You might also have seconds, so we need an extra optional element.
# 
# To avoid matching any old number we want to restict ourtselves to valid times, so our pattern shouldn't match something like 72:80.

# In[ ]:


TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon
:[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""


# In[ ]:


myDate = re.compile(f'{TIME}', re.IGNORECASE | re.VERBOSE | re.UNICODE)

pd.DataFrame([{'test_text': txt, 'match': '@@'.join(myDate.findall(txt))} for txt in test_dates])


# This looks OK. So let's try to combine all three elements into a single pattern and see what we get.

# ### Finding everything combined 

# We will define an overarching capture group to collect the combined match.

# In[ ]:


COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

myDate = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)


# This is now a very complicated pattern, with lots of named groups, optional components. It could do with a delimiter between the TIME bit and the rest, and I'm sure lots of other tweaks, but I have run out of steam. Let's look at what it produces:

# In[ ]:


pd.DataFrame([{'test_text': txt, 'match': myDate.findall(txt)} for txt in test_dates])


# So, it looks like it is picking up the right matches, but we are getting the same match more than once. This is because of the mess of groups we now have in the merged pattern. We could tidy it up, but perhaps we don't have to. Let's define a helper function that will find any matches in the 'combined' named group using the ```.finditer()``` function.

# In[ ]:


def findCombined(txt):
    myResults = []
    for matchGroup in myDate.finditer(txt):
        myResults.append(matchGroup.group('combined'))
    return myResults

pd.DataFrame([{'test_text': txt, 'match': findCombined(txt)} for txt in test_dates])


# That looks much better. 
# 
# And now a final step, we want to replace these dates and times with either nothing, or a special token. For this we can use the ```.sub()``` function and don't have to worry about the match groups. 

# In[ ]:


pd.DataFrame([{'test_text': txt, 'match': findCombined(txt), 'subbed': myDate.sub(' xxDATExx ', txt)} for txt in test_dates])


# Again, from these limited tests it all looks OK, so now we can try it on the actual data, creating a new column with - hopefully - most of the dates and times replaced with the token xxDATExx.

# In[ ]:


# Let's use tqdm to givec us a nice progress bar
from tqdm import tqdm
tqdm.pandas(tqdm)


# In[ ]:


for df in [train, test]:
    df['fewer_dates'] = df.comment_text.progress_apply(lambda x: myDate.sub(' xxDATExx ', x))


# And as a sanity check, let's see how many rows were affected, and have a look at some of the replacements we have made. 

# In[ ]:


# First change the display options so we can see the entire comments
pd.set_option('display.max_colwidth', -1)
train.loc[train.fewer_dates.str.contains('xxDATExx'), ['comment_text', 'fewer_dates']].head()


# In[ ]:



print('Found {} rows with dates in the training set'.format(train.fewer_dates.str.contains('xxDATExx').sum()))
print('Found {} rows with dates in the test set'.format(test.fewer_dates.str.contains('xxDATExx').sum()))
      


# It looks like it is working (although this doesn't show how many we might have missed), but after all that effort it's unlikely to make much of a difference to our downstream analyis as it only affect < 10% of the data. However, hopefully useful to some of you. 

# ### Final regex patterns 

# For ease of reference, here are the final edits of the regular expression patterns we used, all in a single cell to make it easier to copy to your own code:

# In[ ]:


nDAY = r'(?:[0-3]?\d)'  # day can be from 1 to 31 with a leading zero 
nMNTH = r'(?:11|12|10|0?[1-9])' # month can be 1 to 12 with a leading zero
nYR = r'(?:(?:19|20)\d\d)'  # I've restricted the year to being in 20th or 21st century on the basis 
                            # that people doon't generally use all number format for old dates, but write them out 
nDELIM = r'(?:[\/\-\._])?'  # 
NUM_DATE = f"""
    (?P<num_date>
        (?:^|\D) # new bit here
        (?:
        # YYYY-MM-DD
        (?:{nYR}(?P<delim1>[\/\-\._]?){nMNTH}(?P=delim1){nDAY})
        |
        # YYYY-DD-MM
        (?:{nYR}(?P<delim2>[\/\-\._]?){nDAY}(?P=delim2){nMNTH})
        |
        # DD-MM-YYYY
        (?:{nDAY}(?P<delim3>[\/\-\._]?){nMNTH}(?P=delim3){nYR})
        |
        # MM-DD-YYYY
        (?:{nMNTH}(?P<delim4>[\/\-\._]?){nDAY}(?P=delim4){nYR})
        )
        (?:\D|$) # new bit here
    )"""
DAY = r"""
(?:
    # search 1st 2nd 3rd etc, or first second third
    (?:[23]?1st|2{1,2}nd|\d{1,2}th|2?3rd|first|second|third|fourth|fifth|sixth|seventh|eighth|nineth)
    |
    # or just a number, but without a leading zero
    (?:[123]?\d)
)"""
MONTH = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)'
YEAR = r"""(?:(?:[12]?\d|')?\d\d)"""
DELIM = r'(?:\s*(?:[\s\.\-\\/,]|(?:of))\s*)'

YEAR_4D = r"""(?:[12]\d\d\d)"""
DATE_PATTERN = f"""(?P<wordy_date>
    # non word character or start of string
    (?:^|\W)
        (?:
            # match various combinations of year month and day 
            (?:
                # 4 digit year
                (?:{YEAR_4D}{DELIM})?
                    (?:
                    # Day - Month
                    (?:{DAY}{DELIM}{MONTH})
                    |
                    # Month - Day
                    (?:{MONTH}{DELIM}{DAY})
                    )
                # 2 or 4 digit year
                (?:{DELIM}{YEAR})?
            )
            |
            # Month - Year (2 or 3 digit)
            (?:{MONTH}{DELIM}{YEAR})
        )
    # non-word character or end of string
    (?:$|\W)
)"""

TIME = r"""(?:
(?:
# first number should be 0 - 59 with optional leading zero.
[012345]?\d
# second number is the same following a colon
:[012345]\d
)
# next we add our optional seconds number in the same format
(?::[012345]\d)?
# and finally add optional am or pm possibly with . and spaces
(?:\s*(?:a|p)\.?m\.?)?
)"""

COMBINED = f"""(?P<combined>
    (?:
        # time followed by date, or date followed by time
        {TIME}?{DATE_PATTERN}{TIME}?
        |
        # or as above but with the numeric version of the date
        {TIME}?{NUM_DATE}{TIME}?
    ) 
    # or a time on its own
    |
    (?:{TIME})
)"""

myDate = re.compile(COMBINED, re.IGNORECASE | re.VERBOSE | re.UNICODE)


# In[ ]:




