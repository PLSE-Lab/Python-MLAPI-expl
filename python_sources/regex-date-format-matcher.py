#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re


# Task 1 :- Read the text file which contains dates in different format, we have to correctly find the format for each dates so that at later stage the sub parts of the dates could be extracted which than will help to convert each dates into one single universal format

# In[ ]:


f = open("../input/evendates/evendates.txt", "r")
data = (f.readlines())
f.close()
dates = "".join(data).split('\n')
print(dates[:5])
print(len(dates))


# https://regexper.com/#%5E%5Cd%7B4%7D-%5Cd%7B1%2C2%7D-%5Cd%7B1%2C2%7D%5Cs%2B%5Cd%7B2%7D%28%3A%7C.%29%5Cd%7B2%7D%28%3A%7C.%29%5Cd%7B2%7D%28%3A%7C.%29%5Cd%7B2%2C3%7D
# https://regex101.com/   <br>
# Create pattern list to match the dates : The process which i followed to create the pattern list and matching dates is as below
# 1. Picked the most similar pattern dates from list dates
# 2. With the help of above two websites, checked the pattern, and added it to the patterns list
# 3. Added the matched dates in match list and non match to non match list
# 4. Modified the pattern like 2020-06-12 and 2020/05/13 both are like yyyy-dd-mm format henced clubed them together using (-|/) operator in pattern
# 5. Added another pattern in list to match the second most similar looking patterns
# 6. Follwed the same steps until all the dates from dates list matches to some valid pattern from patterns list
# 7. At last we have 2 string which are not valid dates hence they are left as non match

# In[ ]:


patterns = []
patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}$')
patterns.append(r'^\d{1,2}(-|/)\d{1,2}(-|/)\d{4}$')
patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}\s{0,2}\d{2}(:|.)\d{2}(:|.)\d{2}(:|.)\d{2,3}$')
patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}\s{0,2}\d{2}(:|.)\d{2}(:|.)\d{2}$')
patterns.append(r'^\d{4}(-|/)\d{1,2}(-|/)\d{1,2}T\d{1,2}(:|.)\d{1,2}(:|.)\d{1,2}$')

patterns.append(r'^\d{1,2}(-|/)\d{1,2}(-|/)\d{4}\s{0,2}\d{1,2}(:|.)\d{2}\s{0,2}(AM|PM)\s{0,2}EST$')
patterns.append(r'^\d{1,2}(-|/)\d{1,2}(-|/)\d{4}\s{0,2}\d{1,2}(:|.)\d{2}\s{0,2}(AM|PM)\s{0,2}EDT$')

patterns.append(r'^[A-Z]{3}\s{0,2}\d{1,2},?\s{0,2}\d{4}$')
patterns.append(r'^[A-Z]{3}\s{0,2}\d{1,2},?\s{0,2}\d{4}\s{0,2}\d{1,2}(:|.)\d{1,2}\s{0,2}(AM|PM)$')


match = []
non_match = []
        
for date in dates:
    matched = False
    for pattern in patterns:
        compiler = re.compile(pattern,flags=re.IGNORECASE)
        if compiler.search(date):
            match.append((date,pattern))
            matched = True
    if not matched:
        non_match.append(date)
            

# Let;s check the length of matched dates and non matched dates
print(len(match), len(non_match))


# In[ ]:


# print invalid dates
print(non_match)


# In[ ]:


# print the matched dates alongwith their format pattern they matched with
print(match)


# In[ ]:


import datetime 
patterns = []


patterns.append(r'(?P<year>\d{4})(-|/)(?P<month>([0][1-9]|[^0][0-2]|[1-9]))(-|/)(?P<day>\d{1,2})')
patterns.append(r'(?P<year>\d{4})(-|/)(?P<month>([0][1-9]|[^0][0-2]|[1-9]))(-|/)(?P<day>\d{1,2})\s{0,2}(?P<hr>\d{2})(:|.)(?P<min>\d{2})(:|.)(?P<sec>\d{2})(:|.)(?P<ms>\d{2,3})')
dateParts = []

for date in dates:
    for pattern in patterns:
        compiler = re.compile(pattern,flags=re.IGNORECASE)
        if compiler.search(date):
            result = compiler.search(date)
            print(date)
            if result.group("year"):
                year = (int)(result.group("year"))  
            if result.group("day"):
                month = (int)(result.group("month"))
            if result.group("month"):
                day = (int)(result.group("day"))
            #year =          
            #month = (int)(result.group("month"))
            #day = (int)(result.group("day"))
            #date = datetime.datetime(year,month,day)
            dateParts.append(str(datetime.datetime(year,month,day)))
            print(result.groupdict())
            
dateParts

