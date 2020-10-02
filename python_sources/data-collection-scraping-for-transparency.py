#!/usr/bin/env python
# coding: utf-8

# # Scraping Debate Transcripts

# In[ ]:


from bs4 import BeautifulSoup

import numpy as np
import pandas as pd
import re
import requests


# ## List out links to the pages containing the transcripts
# These transcripts have been created by *Rev: https://www.rev.com/*  
# **Disclaimer: I am not affiliated with this company and do not claim ownership of their data**

# In[ ]:


# All Transcript Links

transcript_links = [
    "https://www.rev.com/blog/transcripts/new-hampshire-democratic-debate-transcript",
    "https://www.rev.com/blog/transcripts/january-iowa-democratic-debate-transcript",
    "https://www.rev.com/blog/transcripts/december-democratic-debate-transcript-sixth-debate-from-los-angeles",
    "https://www.rev.com/blog/transcripts/november-democratic-debate-transcript-atlanta-debate-transcript",
    "https://www.rev.com/blog/transcripts/october-democratic-debate-transcript-4th-debate-from-ohio",
    "https://www.rev.com/blog/transcripts/democratic-debate-transcript-houston-september-12-2019",
    "https://www.rev.com/blog/transcripts/transcript-of-july-democratic-debate-2nd-round-night-2-full-transcript-july-31-2019",
    "https://www.rev.com/blog/transcripts/transcript-of-july-democratic-debate-night-1-full-transcript-july-30-2019",
    "https://www.rev.com/blog/transcripts/transcript-from-night-2-of-the-2019-democratic-debates",
    "https://www.rev.com/blog/transcripts/transcript-from-first-night-of-democratic-debates"
]


# ## Create function to extract needed info
# They all have a similar structure, and most everything we need is found in a `p` tag called `fl-callout-text`.  

# In[ ]:


# We'll be scraping some info from the structure
# Speaker: (HH:MM:SS) Text here
# Where the 'HH:' is optional
patterns = {'speaker': '^[^\(\):\[\]]+:',
            'time': '^\((\d{2}:)?\d{2}:\d{2}\)'}

def get_transcript(transcript_link):
    response = requests.get(transcript_link)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Get the name of the debate
    debate_name = soup.find('span', class_='fl-heading-text').text
    # The main content
    content = soup.find('div', class_='fl-callout-text')
    # Keep a record of the current section
    # By default, we call the section "Entire Debate"
    section = 'Entire Debate'
    data = []
    for item in content:
        # h2 or p
        item_type = item.name
        # h2 -> this is a section header
        if item_type == 'h2':
            section = item.text
        # p -> this is some speech from a candidate/moderator
        elif item_type == 'p':
            # Hold all data for current item
            item_data = {'debate': debate_name, 'section': section}
            text = item.text
            # for each pattern
            for pattern_name, pattern in patterns.items():
                # try to find the pattern
                match_obj = re.match(pattern, text)
                # if it exists, add it to `item_data` and lstrip from the string
                if match_obj:
                    match_str = match_obj.group(0)
                    item_data[pattern_name] = match_str
                    text = text.lstrip(match_str).strip()
                # add the remaining text after the patterns have been removed
                item_data['speech'] = text
            data.append(item_data)
    return data


# ## Call this function for each link, and put all of the results in `df`

# In[ ]:


transcript_data = []


# In[ ]:


for link in transcript_links:
    transcript_data += get_transcript(link)


# In[ ]:


df = pd.DataFrame(transcript_data)
# From the patterns above, our speaker name has a ':', we'll strip that out
df['speaker'] = df.speaker.apply(lambda name: name.rstrip(':').strip() if not pd.isnull(name) else name)
df.head()


# ## Basic Initial Data Quality Check

# In[ ]:


df.info()


# ### Where are the missing speakers?
# It looks like these are opening remarks or empty speech elements, so we can safely drop them

# In[ ]:


df.loc[pd.isnull(df.speaker)]


# In[ ]:


df = df.loc[~pd.isnull(df.speaker)].reset_index(drop=True)


# ## We also had some missing values in `time`
# These are all coming from `Transcript from Night 1 of the 2019 June Democratic Debates`.  
# Upon checking the link, the site did not post times for this debate

# In[ ]:


df.loc[pd.isnull(df.time)].debate.value_counts()


# ## Define Function to parse out times `(HH:MM:SS)` -> `n_seconds` (int)
# These aren't very helpful - represent them in seconds instead

# In[ ]:


def parse_time_seconds(time_string):
    if time_string and ':' in time_string:
        ord_time = time_string[1:-1].split(':')[::-1]
        n_seconds = 0
        for i, time_measurement in enumerate(ord_time):
            n_seconds += int(time_measurement)*(60**i)
        return n_seconds
    return None


# ## Parse out these times as seconds, and estimate the Speaking Time
# The times we get out of our `parse_time_seconds` are the number of seconds into the debate.  
# As an example, `(01:05:20)` would mean that we are 1 hour, 5 minutes, and 20 seconds into the debate.  
# Parsing this to seconds gives us `n_seconds = 3920`.  
# But we need the amount each speaker spends speaking...Lets do that below

# In[ ]:


# We group by debate and section, because for each section, the time resets. Sometimes,
# The sections aren't labeled, and the time just resets...we'll deal with this.
for (debate, section), debate_section_df in df.groupby(by=['debate', 'section']):
    
    # Earlier, we noted that this debate has no times, we'll skip it
    if debate != 'Transcript from Night 1 of the 2019 June Democratic Debates':
        
        # get the index of this debate section
        index = debate_section_df.index
        
        # apply the function we created above
        time_seconds = debate_section_df.time.apply(parse_time_seconds).values
        df.loc[index, 'time_seconds'] = time_seconds
        
        # find the time diff, and append a `np.nan` to the end for the final speaking time.
        # Unfortunately, we have no idea how long they're speaking for. It is always a
        # Moderator's closing statements though, so it won't affect our analysis.
        time_diff = time_seconds[1:]-time_seconds[:-1]
        
        # Above, we mentioned that sometimes sections aren't labeled...
        # This means that sometimes the time in seconds just drops
        # i.e. [... 3700 3800 25 70 ...]
        # in terms of the `time_diff`, this results in the first number
        # after the drop being negative...let's fix that
        if (time_diff < 0).any():
            
            # break into sections
            section_breaks = np.where(time_diff < 0)[0]+1
            sections = np.split(index, section_breaks)
            
            # for each section, set it individually
            for i, section in enumerate(sections):
                df.loc[section, 'section'] = 'Part {}'.format(i+1)
                time_seconds = df.loc[section, 'time_seconds'].values
                time_diff = time_seconds[1:]-time_seconds[:-1]
                total_speaking_time = np.concatenate([time_diff, np.array([np.nan])])
                df.loc[section, 'total_speaking_time'] = total_speaking_time
        else:
            total_speaking_time = np.concatenate([time_diff, np.array([np.nan])])
            df.loc[index, 'total_speaking_time'] = total_speaking_time


# ## Do some cleaning on the Speaker Names
# Sometimes, the source is inconsistent on the speaker names...  
# That's okay - we can make a quick mapping dictionary that clears up the inconsistencies

# In[ ]:


mapping_dct = {'Abby P': 'Abby Phillips',
               'Abby Phillip': 'Abby Phillips',
               'Amna': 'Amna Nawaz',
               'Amy Klobachar': 'Amy Klobuchar',
               'Bennet': 'Michael Bennet',
               'Bill De Blasio': 'Bill de Blasio',
               'Brianne P': 'Brianne P.',
               'David': 'David Muir',
               'E. Warren': 'Elizabeth Warren',
               'Elizabeth W': 'Elizabeth Warren',
               'Elizabeth W.': 'Elizabeth Warren',
               'Elizabeth Warre': 'Elizabeth Warren',
               'George S': 'George S.',
               'Gillibrand': 'Kirsten Gillibrand',
               'Kirsten G.': 'Kirsten Gillibrand',
               'Kristen Gillibr': 'Kirseten Gillibrand',
               'John H': 'John H.',
               'Jose': 'Jose D.B.',
               'Jose D. B.': 'Jose D.B.',
               'Judy': 'Judy Woodruff',
               'Lindsey': 'Linsey Davis',
               'M. Williamson': 'Marianne Williamson',
               'Marianne W.': 'Marianne Williamson',
               'Marianne Willia': 'Marianne Williamson',
               'Mayor Buttigieg': 'Pete Buttigieg',
               'Mayor de Blasio': 'Bill de Blasio',
               'Ms. Williamson': 'Marianne Williamson',
               'Savannah': 'Savannah G.',
               'Savanagh G': 'Savannah G.',
               'Sen Klobuchar': 'Amy Klobuchar',
               'Senator Bennet': 'Michael Bennet',
               'Senator Booker': 'Cory Booker',
               'Senator Warren': 'Elizabeth Warren',
               'Yamiche': 'Yamiche A.',
               'Yang': 'Andrew Yang'}


# In[ ]:


df['speaker'] = df.speaker.apply(lambda name: mapping_dct.get(name) if name in mapping_dct else name)


# ## `time` and `time_seconds` aren't so useful to me
# I'm more concerned about how much time each speaker spends speaking, which I'll rename to `speaking_time_seconds`

# In[ ]:


df = df.drop(['time', 'time_seconds'], axis=1)
df.columns = ['debate_name', 'debate_section', 'speaker', 'speech', 'speaking_time_seconds']


# In[ ]:


df.info()


# In[ ]:


df = df.loc[df.speech!='']


# In[ ]:


df.to_csv('../data/debate_transcripts.csv', encoding='cp1252', index=False)

