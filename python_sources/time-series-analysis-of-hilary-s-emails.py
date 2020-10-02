#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The data comes both as CSV files and a SQLite database

import pandas as pd
import numpy as np


# In[ ]:


persons = pd.read_csv("../input/Persons.csv")
emails  = pd.read_csv("../input/Emails.csv")

hilaryEmails = emails[emails.SenderPersonId == 80]
hilaryEmails = hilaryEmails.dropna(how='any')


# ## Temporal Analysis
# 
# Wrangle the time into a readable format
# 
# For the most part the times are in the following format:
#     
#     day of week, Month Day, Year, time AM|PM

# In[ ]:


date_re = r'(?P<dow>\w+,\s*?)?(?P<month>)'


# In[ ]:


hilaryEmails.ExtractedDateSent[hilaryEmails.ExtractedDateSent.str.split(" ").apply(len) == 6]


# Some dates are pretty bad and need manual readjustment.  I will not try to do that here

# In[ ]:


def fixDate(date):
    spl = date.split(" ")
    
    if len(spl) >= 6:
        try:
            if spl[0].startswith(u'\xef\xbf\xbd'):
                spl = spl[1:]
        except UnicodeEncodeError:
            spl = spl[1:]
        dow, month, day, year, time = spl[:5]
    elif len(spl) == 5:
        if spl[-1].endswith('M'):
            return np.NAN
        else:
            dow, month, day, time, year = spl[:5]
    else:
        return np.NAN
    try:
        if ':' not in time:
            time = time[:-2] + ':' + time[-2:]
        return u"{} {} {} {}".format(month, day, year, time.replace('.', ''))
    except UnicodeEncodeError as e:
        print(e)


# In[ ]:


def tryToCoerce(s):
    try:
        return pd.to_datetime(fixDate(s))
    except Exception:
        return np.NaN


# In[ ]:


pd.to_datetime(fixDate('Thu Sep 17 06:03:43 2009'))


# In[ ]:


sum(hilaryEmails.
    ExtractedDateSent
    .apply(tryToCoerce)
    .isnull())


# In[ ]:


hilaryEmails['cleanedDate'] = (hilaryEmails
                               .ExtractedDateSent.apply(tryToCoerce)
                               .dropna(how="any")
)


# In[ ]:


hilaryEmails.index = hilaryEmails.cleanedDate


# In[ ]:


hilaryEmails.sort_index(inplace=True)


# In[ ]:


minDate, maxDate = hilaryEmails.index.min(), hilaryEmails.index.max()


# In[ ]:


"Hilary's emails range from {} to {}".format(minDate.date(), maxDate.date())


# ### How many emails per day did she send?

# In[ ]:


hilaryEmails.resample('D', how='count').Id.plot()


# There is intense email activity around july 2009 to july 2010.  It then drops off. You can inspect to make sure

# In[ ]:


# before 2011 and after counts
hilaryEmails[:'2011-01'].Id.count(), hilaryEmails['2011-01':].Id.count()


# ### On Which day did she send the most emails?

# In[ ]:


hilaryEmails.Id.resample("D", how="count").sort_values(ascending=False).head(1)


# ## What happened on 8-29-2009!?

# In[ ]:


hilaryEmails['2009-08-29'][['MetadataSubject', 
                            'ExtractedBodyText']]


# Oprah...

# ### What were her email times?
# 

# In[ ]:


hilaryEmails.groupby(lambda s: s.hour).apply(len)


# Since I failed to take into account am/pm, either she sends most of her emails at 7am or 7pm.
